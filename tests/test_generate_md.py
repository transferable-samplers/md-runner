"""
Tests for seq_to_pdb.py - generating PDB files from sequences.
"""

import os
from pathlib import Path

import numpy as np
import pytest
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

from tests.conftest import TEST_SEQUENCE
from tests.helpers.utils import compose_config

from src.generate_md import generate_md

# Create report directory if it doesn't exist
report_dir = os.environ.get("PYTEST_REPORT_DIR", "tests/")
os.makedirs(report_dir, exist_ok=True)

@pytest.fixture(scope="function")
def cfg_test_generate_md(shared_tmp_path: Path, cfg_test_seq_to_pdb: DictConfig) -> DictConfig:
    """
    Hydra-composed config for generate_md tests.
    Uses the shared tmp_path where PDB files are already generated.
    
    Note: Depends on cfg_test_seq_to_pdb to ensure PDB files are generated first.

    Args:
        shared_tmp_path: Module-scoped temporary directory path with PDB files.
        cfg_test_seq_to_pdb: Config fixture that generates PDB files (ensures it runs first).

    Returns:
        DictConfig: Composed and patched Hydra config for the test.
    """
    # Important: clear Hydra before initializing
    GlobalHydra.instance().clear()

    test_sequence = cfg_test_seq_to_pdb.seq_name

    # PDB files are already in shared_tmp_path from cfg_test_seq_to_pdb fixture
    pdb_dir = shared_tmp_path / "data" / "pdbs"

    cfg = compose_config(
        config_name="generate_md",
        overrides=[
            f"pdb_filename={test_sequence}",  # Use AA sequence from example_sequences.txt
            "platform=cpu",  # Use CPU for tests to avoid GPU requirements
        ],
    )

    # Override config for testing purposes
    with open_dict(cfg):
        cfg.paths.data_dir = str(shared_tmp_path / "data")
        cfg.paths.log_dir = str(shared_tmp_path / "logs")
        cfg.paths.work_dir = os.getcwd()
        cfg.pdb_dir = str(pdb_dir)
        cfg.frame_interval = 1  # 1 fs between frames
        cfg.frames_per_chunk = 100  # Small chunks for testing
        cfg.warmup_steps = 10_000  # Reduced warmup for faster tests
        cfg.time_ns = 0.001

    yield cfg

    # Cleanup for next param
    GlobalHydra.instance().clear()

def check_contiguous_arrays(array_list: list[np.ndarray]) -> None:

    # Check that chunks form a contiguous sequence
    if len(array_list) > 1:
        # Calculate typical displacement between consecutive frames for comparison
        # Use the first chunk to establish baseline
        all_consecutive_displacements = []
        for chunk_positions in array_list:
            # Calculate displacement between consecutive frames within chunk
            for frame_idx in range(chunk_positions.shape[0] - 1):
                displacement = np.linalg.norm(
                    chunk_positions[frame_idx + 1] - chunk_positions[frame_idx], axis=-1
                )
                all_consecutive_displacements.extend(displacement)

        breakpoint()
        
        typical_displacement = np.median(all_consecutive_displacements)
        
        # Check displacement across chunk boundaries
        for i in range(len(array_list) - 1):
            last_frame_prev = array_list[i][-1]  # Last frame of chunk i
            first_frame_next = array_list[i + 1][0]  # First frame of chunk i+1
            
            # Calculate the displacement between consecutive frames (across chunk boundary)
            boundary_displacement = np.linalg.norm(first_frame_next - last_frame_prev, axis=-1)
            max_boundary_displacement = np.max(boundary_displacement)
            
            # The displacement across chunk boundaries should be similar to typical within-chunk displacements
            # If there's a discontinuity (e.g., simulation was restarted), the displacement would be much larger
            # Allow 20x tolerance to account for occasional large movements, but catch major discontinuities
            assert max_boundary_displacement < typical_displacement * 1.5, (
                f"Chunk {i} and {i+1} are not contiguous: "
                "Dispalce"


                f"max displacement across boundary = {max_boundary_displacement:.6f}, "
                f"typical displacement = {typical_displacement:.6f}"
            )
 
def check_chunks(chunk_dir: Path):

    positions_list = []
    velocities_list = []
    chunk_files = list(chunk_dir.glob("chunk_*.npz"))
    # Sort chunk files by numeric index in filename: chunk_<index>.npz
    chunk_files.sort(key=lambda p: int(p.stem.split("_")[-1]))
    for chunk_file in chunk_files:
        chunk_data = np.load(chunk_file)
        assert "positions" in chunk_data, "Chunk missing positions"
        assert "velocities" in chunk_data, "Chunk missing velocities"
        assert chunk_data["positions"].shape[0] > 0, "Chunk has no position frames"
        assert chunk_data["velocities"].shape[0] > 0, "Chunk has no velocity frames"
        positions_list.append(chunk_data["positions"])
        velocities_list.append(chunk_data["velocities"])
    check_contiguous_arrays(positions_list)
    check_contiguous_arrays(velocities_list)

@pytest.mark.forked  # prevents OpenMM issues
def test_generate_md_basic(cfg_test_generate_md: DictConfig) -> None:
    """
    Basic test that generate_md can run a simulation.

    Asserts:
    - Simulation runs without errors
    - Chunk files are created
    - Chunk files contain expected data
    """
    generate_md(cfg_test_generate_md)

    chunks_dir = Path(cfg_test_generate_md.output_dir) / "chunks"
    assert chunks_dir.exists(), "Chunks directory not created"

    chunk_files = list(chunks_dir.glob("chunk_*.npz"))
    assert len(chunk_files) > 0, "No chunk files created"

    check_chunks(chunks_dir) # Ensure chunks can be collated without errors

@pytest.mark.forked  # prevents OpenMM issues
def test_generate_md_resume(cfg_test_generate_md: DictConfig) -> None:
    """
    Basic test that generate_md can run a simulation.

    Asserts:
    - Simulation runs without errors
    - Chunk files are created
    - Chunk files contain expected data
    """

    generate_md(cfg_test_generate_md)

    cfg_test_generate_md.time_ns += 0.001  # Extend simulation by another 1 ps
    generate_md(cfg_test_generate_md)

    chunks_dir = Path(cfg_test_generate_md.output_dir) / "chunks"

    check_chunks(chunks_dir)  # Ensure chunks can be collated without errors