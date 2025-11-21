"""
Shared test fixtures.
"""

import os
from pathlib import Path

import pytest
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

from src.seq_to_pdb import seq_to_pdb
from tests.helpers.utils import compose_config

# Create report directory if it doesn't exist
report_dir = os.environ.get("PYTEST_REPORT_DIR", "tests/")
Path(report_dir).mkdir(parents=True, exist_ok=True)

TEST_SEQUENCE = "PYA"


@pytest.fixture(scope="session")
def shared_tmp_path(tmp_path_factory):
    """
    Session-scoped temporary directory shared across all tests.
    Provides a base directory for organizing test artifacts.
    """
    return tmp_path_factory.mktemp("shared_test_data")


@pytest.fixture(scope="session")
def dir_with_pdb(shared_tmp_path: Path) -> Path:
    """
    Explicitly generates PDB files for tests that need them.
    
    Args:
        shared_tmp_path: Base temporary directory for test artifacts.
        
    Returns:
        Path to directory containing generated PDB files.
    """
    GlobalHydra.instance().clear()

    cfg = compose_config(config_name="seq_to_pdb", overrides=[f"seq_name={TEST_SEQUENCE}"])

    # Override config for testing purposes
    with open_dict(cfg):
        cfg.paths.data_dir = str(shared_tmp_path / "data")
        cfg.paths.log_dir = str(shared_tmp_path / "logs")
        cfg.paths.work_dir = str(Path.cwd())

    # Generate PDB files once for all tests in this session
    seq_to_pdb(cfg)

    pdb_dir = Path(cfg.paths.data_dir) / "pdbs"

    yield pdb_dir

    GlobalHydra.instance().clear()
