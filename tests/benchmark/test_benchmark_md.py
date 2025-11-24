from pathlib import Path

import pytest

from tests.benchmark.helpers.mps import mps_context
from tests.benchmark.helpers.run_n_md_parallel import run_n_md_parallel
from tests.conftest import TEST_SEQUENCE
from tests.helpers.utils import get_project_root

# NOTE: There is some startup overhead in launching each MD process.
# However, this is fairly negligible compared to the total simulation time,
# so we ignore it for now.

SIMULATION_TIME_NS = 0.2  # 200 ps

# Fixed value of total runs to normalize across parallelism configs.
TOTAL_RUNS = 8
assert TOTAL_RUNS >= 1
assert (TOTAL_RUNS & (TOTAL_RUNS - 1)) == 0, "TOTAL_RUNS must be a power of 2"
PARALLEL_PROC_VALUES = [2**i for i in range(1, int(TOTAL_RUNS).bit_length())]  # [2, ..., TOTAL_RUNS]


@pytest.mark.forked
@pytest.mark.benchmark
def test_benchmark_md_sequential(
    benchmark,
    dir_with_pdb: Path,
    tmp_path: Path,
):
    project_root = get_project_root()

    def run():
        for batch_id in range(TOTAL_RUNS):
            # By running without MPS env, we benchmark normal CUDA behavior.
            run_n_md_parallel(
                tmp_path=tmp_path / f"batch_{batch_id}",
                project_root=project_root,
                seq_name=TEST_SEQUENCE,
                pdb_dir=dir_with_pdb,
                simulation_time_ns=SIMULATION_TIME_NS,
                n=1,  # Sequential
            )

    benchmark(run)


@pytest.mark.forked
@pytest.mark.parametrize("num_parallel_procs", PARALLEL_PROC_VALUES)
@pytest.mark.benchmark
def test_benchmark_md_mps(
    benchmark,
    dir_with_pdb: Path,
    tmp_path: Path,
    num_parallel_procs: int,
):
    project_root = get_project_root()
    total_runs = TOTAL_RUNS
    num_batches = total_runs // num_parallel_procs

    def run():
        for batch_id in range(num_batches):
            with mps_context() as env:
                run_n_md_parallel(
                    tmp_path=tmp_path / f"batch_{batch_id}",
                    env=env,
                    project_root=project_root,
                    seq_name=TEST_SEQUENCE,
                    pdb_dir=dir_with_pdb,
                    simulation_time_ns=SIMULATION_TIME_NS,
                    n=num_parallel_procs,
                )

    benchmark(run)
