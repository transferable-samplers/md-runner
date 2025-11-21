"""
Tests for seq_to_pdb.py - generating PDB files from sequences.
"""

from pathlib import Path

import pytest
from omegaconf import DictConfig

@pytest.mark.forked  # prevents OpenMM/tLEaP issues
def test_seq_to_pdb(cfg_test_seq_to_pdb: DictConfig) -> None:
    """
    Tests that seq_to_pdb has successfully generated PDB files from sequences.
    NOTE: This test depends on the cfg_test_seq_to_pdb fixture in conftest.py,
    which runs seq_to_pdb once per module to generate PDB files for all tests.

    Asserts:
    - PDB files are created for each sequence
    - PDB files are not empty
    """
    test_sequence = cfg_test_seq_to_pdb.seq_name
    pdb_path = Path(cfg_test_seq_to_pdb.paths.data_dir) / "pdbs" / f"{test_sequence}.pdb"
    assert pdb_path.exists(), f"PDB file for sequence {test_sequence} was not created."
    assert pdb_path.stat().st_size > 0, f"PDB file for sequence {test_sequence} is empty."
