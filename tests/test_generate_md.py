"""
Tests for generate_md.py - generating MD simulation data.
"""

import os
from pathlib import Path

import numpy as np
import pytest
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

from src.generate_md import generate_md
from tests.conftest import TEST_SEQUENCE
from tests.helpers.utils import compose_config

# Create report directory if it doesn't exist
report_dir = os.environ.get("PYTEST_REPORT_DIR", "tests/")
Path(report_dir).mkdir(parents=True, exist_ok=True)


@pytest.fixture
def cfg_test_generate_md(shared_tmp_path: Path, dir_with_pdb: Path) -> DictConfig:
    """
    Hydra-composed config for generate_md tests.
    Uses explicitly generated PDB files from dir_with_pdb fixture.

    Args:
        shared_tmp_path: Session-scoped temporary directory path for test artifacts.
        dir_with_pdb: Path to directory containing generated PDB files.

    Returns:
        DictConfig: Composed and patched Hydra config for the test.
    """
    GlobalHydra.instance().clear()

    cfg = compose_config(
        config_name="generate_md",
        overrides=[
            f"seq_name={TEST_SEQUENCE}",
            "platform=cpu",  # Use CPU for tests to avoid GPU requirements
        ],
    )

    # Override config for testing purposes
    with open_dict(cfg):
        cfg.paths.data_dir = str(shared_tmp_path / "data")
        cfg.paths.log_dir = str(shared_tmp_path / "logs")
        cfg.paths.work_dir = str(Path.cwd())
        cfg.pdb_dir = str(dir_with_pdb)
        cfg.frame_interval = 1000  # Must be large enough for possible inter-chunk discontinuities
        cfg.frames_per_chunk = 5  # Small chunk size for testing
        cfg.warmup_steps = 10_000  # Reduced warmup for faster tests
        cfg.time_ns = 0.050  # Must be large enough to get multiple chunks

    yield cfg

    GlobalHydra.instance().clear()


def check_chunk_indexes(chunk_files: list[Path]) -> None:
    """
    Verify that chunk file indices are integers and form a contiguous sequence.

    Args:
        chunk_files: List of chunk file paths to check.

    Raises:
        AssertionError: If chunk filenames don't follow the expected pattern,
            are not sorted, contain duplicates, or are not contiguous.
    """
    # Verify chunk file indices are integer and contiguous
    try:
        indices = [int(p.stem.split("_")[-1]) for p in chunk_files]
    except ValueError as err:
        raise AssertionError("Chunk filename(s) do not follow pattern 'chunk_<index>.npz'") from err

    # Ensure sorted order and uniqueness
    if indices != sorted(indices):
        raise AssertionError("Chunk files are not sorted by index")
    if len(set(indices)) != len(indices):
        raise AssertionError("Duplicate chunk indices found")

    start = indices[0]
    expected = list(range(start, start + len(indices)))
    if indices != expected:
        missing = sorted(set(expected) - set(indices))
        extra = sorted(set(indices) - set(expected))
        parts = []
        if missing:
            parts.append(f"missing indices {missing}")
        if extra:
            parts.append(f"unexpected indices {extra}")
        raise AssertionError("Chunk indices are not contiguous: " + "; ".join(parts))


def check_contiguous_arrays(array_list: list[np.ndarray], alpha: float = 2.0) -> None:
    """
    Verify that arrays in the list form a contiguous trajectory.

    Checks that the displacement between consecutive chunks is not significantly
    larger than the typical displacement within chunks.

    Args:
        array_list: List of numpy arrays, each representing a chunk of frames.
            Each array should have shape (num_frames, num_points, dim).
        alpha: Multiplier for typical displacement threshold. Boundary displacements
            exceeding alpha * typical_displacement are considered non-contiguous.

    Raises:
        AssertionError: If boundary displacement exceeds the threshold.
    """
    # Calculate typical displacement between consecutive frames for comparison
    all_consecutive_displacements = []
    for chunk_array in array_list:
        # chunk_positions: shape (num_frames, num_points, dim)
        for frame_idx in range(chunk_array.shape[0] - 1):
            # Flatten frames so distance is not pointwise, but global
            f0 = chunk_array[frame_idx].reshape(-1)
            f1 = chunk_array[frame_idx + 1].reshape(-1)

            # Euclidean distance between full frames
            frame_dist = np.linalg.norm(f1 - f0)

            all_consecutive_displacements.append(frame_dist)

    typical_displacement = np.median(all_consecutive_displacements)

    # Check displacement across chunk boundaries
    for i in range(len(array_list) - 1):
        last_frame_prev = array_list[i][-1]  # Last frame of chunk i
        first_frame_next = array_list[i + 1][0]  # First frame of chunk i+1

        # Flatten frames for global distance
        f0 = last_frame_prev.reshape(-1)
        f1 = first_frame_next.reshape(-1)

        boundary_dist = np.linalg.norm(f1 - f0)
        assert boundary_dist < alpha * typical_displacement, (
            f"Non-contiguous frames detected between chunks {i} and {i + 1}: "
            f"distance {boundary_dist:.3f} exceeds threshold of {alpha * typical_displacement:.3f}."
        )


def check_chunks(
    chunk_files: list[Path],
    time_per_frame: float,
    expected_time_length_ns: float,
) -> None:
    """
    Validate chunk files for correctness and consistency.

    Checks that:
    - Chunk files contain expected data (positions and velocities)
    - Total simulation time matches expected duration
    - All chunks have the same number of frames
    - Trajectories are contiguous across chunk boundaries

    Args:
        chunk_files: List of paths to chunk .npz files.
        time_per_frame: Time duration per frame in nanoseconds.
        expected_time_length_ns: Expected total simulation time in nanoseconds.

    Raises:
        AssertionError: If any validation check fails.
    """
    positions_list = []
    velocities_list = []
    assert len(chunk_files) > 1, "Cannot thoroughly check chunks with <= 1 chunk file."

    # Sort chunk files by numeric index in filename: chunk_<index>.npz
    chunk_files.sort(key=lambda p: int(p.stem.split("_")[-1]))

    check_chunk_indexes(chunk_files)  # Verify chunk file indices are formatted correctly and contiguous

    total_frames = 0
    unique_num_frames = set()
    for chunk_file in chunk_files:
        chunk_data = np.load(chunk_file)
        assert "positions" in chunk_data, "Chunk missing positions"
        assert "velocities" in chunk_data, "Chunk missing velocities"
        assert chunk_data["positions"].shape[0] > 0, "Chunk has no position frames"
        assert chunk_data["velocities"].shape[0] > 0, "Chunk has no velocity frames"
        assert chunk_data["positions"].shape == chunk_data["velocities"].shape, (
            "Positions and velocities shape mismatch in chunk"
        )
        positions_list.append(chunk_data["positions"])
        velocities_list.append(chunk_data["velocities"])

        frames_in_chunk = chunk_data["positions"].shape[0]
        total_frames += frames_in_chunk
        unique_num_frames.add(frames_in_chunk)

    assert np.isclose(
        total_frames * time_per_frame,
        expected_time_length_ns,
        rtol=0.001,
    )
    assert len(unique_num_frames) == 1, (
        "Inconsistent number of frames across chunks; expected all chunks to have the same number of frames "
        f"but found {unique_num_frames}"
    )

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

    check_chunks(
        chunk_files,
        cfg_test_generate_md.frame_interval / 1e6,
        cfg_test_generate_md.time_ns,
    )  # Ensure chunks can be collated without errors


@pytest.mark.forked  # prevents OpenMM issues
def test_generate_md_resume(cfg_test_generate_md: DictConfig) -> None:
    """
    Test that generate_md can resume a simulation from existing chunks.

    Asserts:
        - Initial simulation runs and creates chunk files
        - Resumed simulation continues from existing chunks
        - All chunk files are valid and contiguous
    """
    generate_md(cfg_test_generate_md)

    chunks_dir = Path(cfg_test_generate_md.output_dir) / "chunks"
    assert chunks_dir.exists(), "Chunks directory not created"

    chunk_files = list(chunks_dir.glob("chunk_*.npz"))
    assert len(chunk_files) > 0, "No chunk files created before resuming"

    initial_time_ns = cfg_test_generate_md.time_ns
    check_chunks(
        chunk_files,
        cfg_test_generate_md.frame_interval / 1e6,
        initial_time_ns,
    )  # Ensure chunks can be collated without errors before resuming

    cfg_test_generate_md.time_ns *= 2  # Double simulation time to force resuming
    generate_md(cfg_test_generate_md)

    chunk_files = list(chunks_dir.glob("chunk_*.npz"))
    assert len(chunk_files) > 0, "No chunk files created after resuming"

    check_chunks(
        chunk_files,
        cfg_test_generate_md.frame_interval / 1e6,
        cfg_test_generate_md.time_ns,
    )  # Ensure chunks can be collated without errors after resuming
