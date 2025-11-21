"""
Tests for seq_to_pdb.py - generating PDB files from sequences.
"""

from pathlib import Path

import pytest
<<<<<<< HEAD
=======
from conftest import TEST_SEQUENCE
>>>>>>> 36f676f7857ea19203b246fb07f77ab7eb767697
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

from src.seq_to_pdb import seq_to_pdb
from tests.helpers.utils import compose_config



@pytest.fixture(scope="session")
def cfg_test_seq_to_pdb(shared_tmp_path: Path) -> DictConfig:
    """
    Hydra-composed config for seq_to_pdb tests.
    Provides config only, no side effects.

    Args:
        shared_tmp_path: Session-scoped temporary directory path.

    Returns:
        DictConfig: Composed and patched Hydra config for the test.
    """
    GlobalHydra.instance().clear()

    cfg = compose_config(config_name="seq_to_pdb", overrides=[f"seq_name={TEST_SEQUENCE}"])

    # Override config for testing purposes
    with open_dict(cfg):
        cfg.paths.data_dir = str(shared_tmp_path / "data")
        cfg.paths.log_dir = str(shared_tmp_path / "logs")
        cfg.paths.work_dir = str(Path.cwd())

    yield cfg

    GlobalHydra.instance().clear()


@pytest.mark.forked  # prevents OpenMM/tLEaP issues
def test_seq_to_pdb(cfg_test_seq_to_pdb: DictConfig) -> None:
    """
    Test that seq_to_pdb successfully generates PDB files from sequences.

    Asserts:
        - PDB files are created for each sequence
        - PDB files are not empty
    """
    # Generate PDB files
    seq_to_pdb(cfg_test_seq_to_pdb)

    # Verify PDB files were created
    test_sequence = cfg_test_seq_to_pdb.seq_name
    pdb_path = Path(cfg_test_seq_to_pdb.paths.data_dir) / "pdbs" / f"{test_sequence}.pdb"
    assert pdb_path.exists(), f"PDB file for sequence {test_sequence} was not created."
    assert pdb_path.stat().st_size > 0, f"PDB file for sequence {test_sequence} is empty."
