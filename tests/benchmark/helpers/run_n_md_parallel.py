import subprocess
from pathlib import Path

from typing import Optional


def run_n_md_parallel(
    tmp_path: Path,
    project_root: Path,
    seq_name: str,
    pdb_dir: Path,
    simulation_time_ns: float = 0.010,
    n: int = 1,
    env: Optional[dict[str, str]] = None,
):
    """
    Launch n parallel MD processes under MPS.
    Each process gets its own output directory: base_output_dir/proc_{i}
    """

    jobs = []

    for i in range(n):
        data_dir = tmp_path / f"parallel_md_{seq_name}" / f"proc_{i}"

        cmd = [
            "python",
            "src/generate_md.py",
            # Dynamic config options
            f"seq_name={seq_name}",
            f"paths.data_dir={data_dir}",
            f"pdb_dir={pdb_dir}",
            # Static MD parameters
            "warmup_steps=0",  # The way we benchmark, we don't need an equilibration phase
            "frame_interval=1000",  # 1ps per frame
            f"time_ns={simulation_time_ns}",  # Total simulation time
            "frames_per_chunk=100",  # Save 100 frames at a time
        ]

        # python is expected to be in PATH
        p = subprocess.Popen(  # noqa: S603
            cmd,
            cwd=str(project_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        jobs.append(p)

    # Wait for all jobs
    for p in jobs:
        stdout, stderr = p.communicate()
        if p.returncode != 0:
            raise RuntimeError(
                f"MD job failed (exit {p.returncode}).\nSTDERR:\n{stderr}\nSTDOUT:\n{stdout}",
            )
