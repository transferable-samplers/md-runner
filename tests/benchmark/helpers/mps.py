import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def mps_context():
    """
    Start CUDA MPS, yield environment, then clean up.
    """

    # Create directories
    mps_dir = Path(tempfile.mkdtemp(prefix="mps_"))
    pipe_dir = mps_dir / "pipe"
    log_dir = mps_dir / "log"

    pipe_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Set the environment for subprocesses
    env = os.environ.copy()
    env["CUDA_MPS_PIPE_DIRECTORY"] = str(pipe_dir)
    env["CUDA_MPS_LOG_DIRECTORY"] = str(log_dir)

    # Start the MPS control daemon
    # nvidia-cuda-mps-control is a system command expected to be in PATH
    subprocess.run(["nvidia-cuda-mps-control", "-d"], env=env, check=True)  # noqa: S603, S607

    try:
        yield env  # This environment is used by all child processes
    finally:
        # Tell MPS server to quit
        # Using subprocess.Popen to avoid shell=True while piping
        quit_process = subprocess.Popen(  # noqa: S603
            ["nvidia-cuda-mps-control"],  # noqa: S607
            stdin=subprocess.PIPE,
            env=env,
        )
        quit_process.communicate(input=b"quit\n")
        shutil.rmtree(mps_dir)
