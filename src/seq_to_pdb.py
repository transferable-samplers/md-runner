"""
Generating peptide pdb file with tLEaP. Requires `ambertools` from conda-forge
to be installed in the current env.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import hydra
import rootutils
from omegaconf import DictConfig
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

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


def translate_1letter_to_3letter(one_letter_seq, zwitter_ion=True):
    """Generate 3 letter sequence with 1-letter seq.
    If `zwitter_ion`, adding N and C terminal special label for zwitter ion form.
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


def make_peptide_with_tleap(three_letter_seq, save_path):
    """Make a pdb file for a three letter amino acid sequence and save
    it to the `save_path`.

    Note:
    One is expected to specify the terminal groups with the sequence.
    E.g., for capping with ACE and NME residue, one need to put ACE
     as the first residue and NME as the last residue.
    For zwitter ion form of ending, one need to attach a N label to the front
    of the first residue, e.g., NMET; as well as a C label to the front of the
    last residue, e.g., CALA.
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
    pdb_dir = Path(cfg.paths.data_dir) / "pdbs"
    pdb_dir.mkdir(parents=True, exist_ok=True)

    # Check only one of seq_filename or seq_name is provided
    assert cfg.seq_filename is not None or cfg.seq_name is not None, "Either seq_filename or seq_name must be provided"
    assert cfg.seq_filename is None or cfg.seq_name is None, "Only one of seq_filename or seq_name must be provided"

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
