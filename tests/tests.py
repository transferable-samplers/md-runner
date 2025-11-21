"""
Tests for seq_to_pdb.py - generating PDB files from sequences.
"""

import os
from pathlib import Path

import numpy as np
import pytest
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

from src.seq_to_pdb import seq_to_pdb
from tests.helpers.utils import compose_config

# Create report directory if it doesn't exist
report_dir = os.environ.get("PYTEST_REPORT_DIR", "tests/")
os.makedirs(report_dir, exist_ok=True)


@pytest.fixture(scope="module")
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

    cfg = compose_config(config_name="seq_to_pdb", overrides=["seq_filename=sequences/example_sequences.txt"])

    # Override config for testing purposes
    with open_dict(cfg):
        cfg.paths.data_dir = str(tmp_path / "data")
        cfg.paths.log_dir = str(tmp_path / "logs")
        cfg.paths.work_dir = os.getcwd()

    yield cfg

    # Cleanup for next param
    GlobalHydra.instance().clear()


@pytest.mark.forked  # prevents OpenMM/tLEaP issues
def test_seq_to_pdb(cfg_test_seq_to_pdb: DictConfig) -> None:
    """
    Tests that seq_to_pdb can generate PDB files from sequences.

    Asserts:
    - PDB files are created for each sequence
    - PDB files are not empty
    """
    seq_to_pdb(cfg_test_seq_to_pdb)

    with open(cfg_test_seq_to_pdb.seq_filename) as f:
        sequences = [line.strip() for line in f.readlines()]

    for sequence in sequences:
        pdb_path = Path(cfg_test_seq_to_pdb.paths.data_dir) / "pdbs" / f"{sequence}.pdb"
        print(pdb_path)
        assert pdb_path.exists(), f"PDB file for sequence {sequence} was not created."
        assert pdb_path.stat().st_size > 0, f"PDB file for sequence {sequence} is empty."


@pytest.fixture(scope="function")
def cfg_test_generate_md(tmp_path: Path) -> DictConfig:
    """
    Hydra-composed config for generate_md tests.

    Args:
        tmp_path: pytest-provided temporary directory path.

    Returns:
        DictConfig: Composed and patched Hydra config for the test.
    """
    # Important: clear Hydra before initializing
    GlobalHydra.instance().clear()

    # Create a test PDB file (we'll need seq_to_pdb to generate it first, or use a fixture)
    pdb_dir = tmp_path / "data" / "pdbs"
    pdb_dir.mkdir(parents=True, exist_ok=True)

    cfg = compose_config(
        config_name="generate_md",
        overrides=[
            "pdb_filename='AA.pdb'platform=CPU",  # Use CPU for tests to avoid GPU requirements
        ],
    )

    # Override config for testing purposes
    with open_dict(cfg):
        cfg.paths.data_dir = str(tmp_path / "data")
        cfg.paths.log_dir = str(tmp_path / "logs")
        cfg.paths.work_dir = os.getcwd()
        cfg.pdb_dir = str(pdb_dir)
        cfg.frame_interval = 10  # 1000 fs between frames
        cfg.frames_per_chunk = 10  # Small chunks for testing
        cfg.warmup_steps = 10  # Reduced warmup for faster tests

    yield cfg

    # Cleanup for next param
    GlobalHydra.instance().clear()


@pytest.mark.forked  # prevents OpenMM issues
@pytest.mark.pipeline
def test_generate_md_basic(cfg_test_generate_md: DictConfig) -> None:
    """
    Basic test that generate_md can run a simulation.

    Asserts:
    - Simulation runs without errors
    - Chunk files are created
    - Chunk files contain expected data
    """
    # Set a very short simulation time for basic test
    with open_dict(cfg_test_generate_md):
        cfg_test_generate_md.time_ns = 0.001  # 0.001 ns = very short simulation

    # Call the main function directly with the config
    # We need to import here to avoid issues with @hydra.main decorator
    from src.generate_md import generate_md

    generate_md(cfg_test_generate_md)

    chunks_dir = Path(cfg_test_generate_md.output_dir) / "chunks"
    assert chunks_dir.exists(), "Chunks directory not created"

    chunk_files = list(chunks_dir.glob("chunk_*.npz"))
    assert len(chunk_files) > 0, "No chunk files created"

    # Verify chunk file structure
    chunk_data = np.load(chunk_files[0])
    assert "positions" in chunk_data, "Chunk missing positions"
    assert "velocities" in chunk_data, "Chunk missing velocities"
    assert chunk_data["positions"].shape[0] > 0, "Chunk has no frames"
