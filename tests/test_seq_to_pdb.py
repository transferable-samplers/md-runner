"""
Tests for seq_to_pdb.py - generating PDB files from sequences.
"""

import os
from pathlib import Path

import pytest
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import call
from omegaconf import DictConfig, open_dict

from tests.helpers.utils import compose_config

# Simple test sequences
TEST_SEQUENCES = ["AA", "G"]


@pytest.fixture(scope="function")
def cfg_test_seq_to_pdb(tmp_path: Path) -> DictConfig:
    """
    Hydra-composed config for seq_to_pdb tests.

    Args:
        tmp_path: pytest-provided temporary directory path.

    Returns:
        DictConfig: Composed and patched Hydra config for the test.
    """
    # Important: clear Hydra before initializing
    GlobalHydra.instance().clear()

    # Create a temporary sequence file
    seq_file = tmp_path / "test_sequences.txt"
    with seq_file.open("w") as f:
        f.write("\n".join(TEST_SEQUENCES))

    cfg = compose_config(config_name="seq_to_pdb", overrides=[f"seq_filename={seq_file}"])

    # Override config for testing purposes
    with open_dict(cfg):
        cfg.paths.data_dir = str(tmp_path / "data")
        cfg.paths.log_dir = str(tmp_path / "logs")
        cfg.paths.work_dir = os.getcwd()

    yield cfg

    # Cleanup for next param
    GlobalHydra.instance().clear()


@pytest.mark.forked  # prevents OpenMM/tLEaP issues
@pytest.mark.pipeline
def test_seq_to_pdb(cfg_test_seq_to_pdb: DictConfig) -> None:
    """
    Tests that seq_to_pdb can generate PDB files from sequences.

    Asserts:
    - PDB files are created for each sequence
    - PDB files are not empty
    """
    # Extract core logic from main function to avoid @hydra.main decorator issues
    from src.seq_to_pdb import make_peptide_with_tleap, translate_1letter_to_3letter

    pdb_dir = Path(cfg_test_seq_to_pdb.paths.data_dir) / "pdbs"
    pdb_dir.mkdir(parents=True, exist_ok=True)

    seq_filename = cfg_test_seq_to_pdb.seq_filename
    with Path(seq_filename).open() as f:
        sequences = [line.strip() for line in f.readlines()]

    for sequence in sequences:
        save_path = pdb_dir / f"{sequence}.pdb"
        make_peptide_with_tleap(translate_1letter_to_3letter(sequence), save_path)

    pdb_dir = Path(cfg_test_seq_to_pdb.paths.data_dir) / "pdbs"

    for sequence in TEST_SEQUENCES:
        pdb_path = pdb_dir / f"{sequence}.pdb"
        assert pdb_path.exists(), f"PDB file not created for sequence {sequence}"
        assert pdb_path.stat().st_size > 0, f"PDB file is empty for sequence {sequence}"

