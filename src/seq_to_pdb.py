"""
Generating peptide pdb file with tLEaP. Requires `ambertools` from conda-forge
to be installed in the current env.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List

import hydra
import rootutils
from omegaconf import DictConfig
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Mapping from single-letter to three-letter amino acid codes
aa_321 = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
}

infile_templ = """source oldff/leaprc.ff99SBildn
peptide = sequence {%s}
savepdb peptide output.pdb
quit
"""


def translate_1letter_to_3letter(one_letter_seq: str, zwitter_ion: bool = True) -> List[str]:
    """
    Convert a one-letter amino acid sequence to three-letter codes.

    Args:
        one_letter_seq: Single-letter amino acid sequence (e.g., "ACD").
        zwitter_ion: If True, add N and C terminal labels for zwitter ion form.
            N-terminal residues will be prefixed with "N" (e.g., "NMET"),
            C-terminal residues will be prefixed with "C" (e.g., "CALA").

    Returns:
        List of three-letter amino acid codes, optionally with terminal labels.
    """
    three_letter_seq = []
    for i, one_letter in enumerate(one_letter_seq):
        my_res_name = aa_321[one_letter]
        if zwitter_ion:
            if i == 0:
                my_res_name = "N" + my_res_name
            elif i == len(one_letter_seq) - 1:
                my_res_name = "C" + my_res_name
        three_letter_seq.append(my_res_name)
    return three_letter_seq


def make_peptide_with_tleap(three_letter_seq: List[str], save_path: Path) -> None:
    """
    Generate a PDB file for a three-letter amino acid sequence using tLEaP.

    The sequence should include terminal group specifications:
    - For capping with ACE and NME residues, put ACE as the first residue
      and NME as the last residue.
    - For zwitter ion form, attach an "N" label to the front of the first
      residue (e.g., "NMET") and a "C" label to the front of the last
      residue (e.g., "CALA").

    Args:
        three_letter_seq: List of three-letter amino acid codes with terminal labels.
        save_path: Path where the generated PDB file will be saved.

    Raises:
        RuntimeError: If tleap executable is not found or if tleap execution fails.
        FileNotFoundError: If tleap does not produce the expected output file.
    """
    save_path = Path(save_path).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve absolute path to tleap to avoid partial executable path issues (S607).
    tleap_executable = shutil.which("tleap")
    if tleap_executable is None:
        raise RuntimeError(
            "Could not find 'tleap' executable on PATH; please install AmberTools and ensure 'tleap' is available.",
        )

    script = infile_templ % (" ".join(three_letter_seq))

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdir_path = Path(tmpdirname)
        temp_in_path = tmpdir_path / "temp.in"
        output_pdb_path = tmpdir_path / "output.pdb"

        # Write tleap input script
        with temp_in_path.open("w") as f:
            f.write(script)

        # Run tleap in the temporary directory.
        # The command is fixed (no shell=True) and only reads user data from temp_in_path.
        result = subprocess.run(  # noqa: S603
            [tleap_executable, "-s", "-f", str(temp_in_path)],
            cwd=tmpdir_path,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            # Optionally include a bit of stderr to help debugging.
            stderr_preview = (result.stderr or "").strip().splitlines()
            stderr_preview = "\n".join(stderr_preview[:10])
            raise RuntimeError(
                f"tleap failed with exit code {result.returncode}.\nstderr (first 10 lines):\n{stderr_preview}",
            )

        if not output_pdb_path.exists():
            raise FileNotFoundError(
                f"'output.pdb' was not created by tleap in {tmpdir_path}",
            )

        # Clean up tleap artifacts we don't need.
        leap_log_path = tmpdir_path / "leap.log"
        leap_log_path.unlink(missing_ok=True)
        temp_in_path.unlink(missing_ok=True)

        # Copy resulting PDB to the requested save_path.
        shutil.copy(output_pdb_path, save_path)
        output_pdb_path.unlink(missing_ok=True)


@hydra.main(version_base="1.3", config_path="../configs", config_name="seq_to_pdb.yaml")
def seq_to_pdb(cfg: DictConfig) -> None:
    """
    Convert amino acid sequences to PDB files using tLEaP.

    Reads sequences either from a file (seq_filename) or a single sequence (seq_name),
    then generates PDB files for each sequence using tLEaP.

    Args:
        cfg: Hydra configuration dictionary containing:
            - paths.data_dir: Base directory for output files
            - seq_filename: Optional path to file containing sequences (one per line)
            - seq_name: Optional single sequence string

    Raises:
        AssertionError: If neither or both seq_filename and seq_name are provided.
    """
    pdb_dir = Path(cfg.paths.data_dir) / "pdbs"
    pdb_dir.mkdir(parents=True, exist_ok=True)

    # Check only one of seq_filename or seq_name is provided
    assert cfg.seq_filename is not None or cfg.seq_name is not None, (
        "Either seq_filename or seq_name must be provided"
    )
    assert cfg.seq_filename is None or cfg.seq_name is None, (
        "Only one of seq_filename or seq_name must be provided"
    )

    if cfg.seq_filename is not None:
        seq_filename = cfg.seq_filename
        with Path(seq_filename).open() as f:
            sequences = [line.strip() for line in f.readlines()]
    else:
        sequences = [cfg.seq_name]

    for sequence in tqdm(sequences):
        save_path = pdb_dir / f"{sequence}.pdb"
        make_peptide_with_tleap(translate_1letter_to_3letter(sequence), save_path)


if __name__ == "__main__":
    seq_to_pdb()
