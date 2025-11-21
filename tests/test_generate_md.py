"""
Tests for generate_md.py - MD simulation generation with resuming.
"""

import os
from pathlib import Path

import numpy as np
import pytest
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

from tests.helpers.utils import compose_config

# Test sequence - use a very short one for speed
TEST_SEQUENCE = "AA"
TEST_PDB_FILENAME = TEST_SEQUENCE


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
            f"pdb_filename={TEST_PDB_FILENAME}",
            "platform=CPU",  # Use CPU for tests to avoid GPU requirements
        ],
    )

    # Override config for testing purposes
    with open_dict(cfg):
        cfg.paths.data_dir = str(tmp_path / "data")
        cfg.paths.log_dir = str(tmp_path / "logs")
        cfg.paths.work_dir = os.getcwd()
        cfg.pdb_dir = str(pdb_dir)
        cfg.temperature = 310
        cfg.frame_interval = 1000  # 1000 fs between frames
        cfg.frames_per_chunk = 10  # Small chunks for testing
        cfg.warmup_steps = 1000  # Reduced warmup for faster tests
        cfg.log_freq = 100

    yield cfg

    # Cleanup for next param
    GlobalHydra.instance().clear()


@pytest.mark.forked  # prevents OpenMM issues
@pytest.mark.pipeline
def test_generate_md_basic(cfg_test_generate_md: DictConfig, test_pdb_file: Path) -> None:
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
    from src.generate_md import main

    main(cfg_test_generate_md)

    chunks_dir = Path(cfg_test_generate_md.output_dir) / "chunks"
    assert chunks_dir.exists(), "Chunks directory not created"

    chunk_files = list(chunks_dir.glob("chunk_*.npz"))
    assert len(chunk_files) > 0, "No chunk files created"

    # Verify chunk file structure
    chunk_data = np.load(chunk_files[0])
    assert "positions" in chunk_data, "Chunk missing positions"
    assert "velocities" in chunk_data, "Chunk missing velocities"
    assert chunk_data["positions"].shape[0] > 0, "Chunk has no frames"


@pytest.mark.forked  # prevents OpenMM issues
@pytest.mark.pipeline
def test_generate_md_resume(cfg_test_generate_md: DictConfig, test_pdb_file: Path) -> None:
    """
    Tests that generate_md can resume from a previous simulation.

    Strategy:
    1. Run simulation with time_ns=0.01 (short)
    2. Run simulation again with time_ns=0.02 (longer)
    3. Verify that the second run resumes from where the first left off

    Asserts:
    - First run creates chunks
    - Second run resumes and creates additional chunks
    - Total frames match expected count from second run
    """
    chunks_dir = Path(cfg_test_generate_md.output_dir) / "chunks"

    # First run: short simulation
    with open_dict(cfg_test_generate_md):
        cfg_test_generate_md.time_ns = 0.01  # 0.01 ns

    from src.generate_md import main

    main(cfg_test_generate_md)

    # Verify first run created chunks
    chunk_files_after_first = sorted(chunks_dir.glob("chunk_*.npz"))
    assert len(chunk_files_after_first) > 0, "First run did not create chunks"

    # Get frame count from first run
    first_run_frames = sum(np.load(f)["positions"].shape[0] for f in chunk_files_after_first)

    # Second run: longer simulation (should resume)
    with open_dict(cfg_test_generate_md):
        cfg_test_generate_md.time_ns = 0.02  # 0.02 ns (double the time)

    from src.generate_md import main

    main(cfg_test_generate_md)

    # Verify second run created additional chunks
    chunk_files_after_second = sorted(chunks_dir.glob("chunk_*.npz"))
    assert len(chunk_files_after_second) > len(chunk_files_after_first), "Second run did not create additional chunks"

    # Calculate expected total frames from second run
    # time_ns * 1e6 fs/ns / frame_interval = total frames
    expected_total_frames = int(cfg_test_generate_md.time_ns * 1e6 / cfg_test_generate_md.frame_interval)

    # Get actual total frames
    total_frames = sum(np.load(f)["positions"].shape[0] for f in chunk_files_after_second)

    # Total frames should match expected (within rounding)
    assert total_frames == expected_total_frames, (
        f"Total frames mismatch: expected {expected_total_frames}, got {total_frames}"
    )

    # Verify that frames from first run are still present (resuming should not delete them)
    assert len(chunk_files_after_second) >= len(chunk_files_after_first), "First run chunks were deleted"

