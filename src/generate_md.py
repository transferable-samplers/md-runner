"""

Script to generate MD data as in timewarp.

Benchmarking on Mila Cluster March 10 2025

on AL6 (uncapped)
A100: 0.99 s/it
l40s: 0.59 s/it
rtx8000: 0.66 s/it
CPU (cn-h001):
    1x: 8.05 s/it
    2x: 5.95 s/it
    4x: 6.22 s/it
    8x: 6.40 s/it
    16x: 8.00 s/it
    32x: 18.49 s/it

on AL8 (uncapped)
A100: 1.22s / it
CPU
    1x: 11.9 s/it
    2x 7.5 s/it

"""

import logging
from pathlib import Path

import hydra
import numpy as np
import openmm
import rootutils
from omegaconf import DictConfig
from openmm import Platform, XmlSerializer
from openmm.app import ForceField, PDBFile, Simulation, StateDataReporter

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="generate_md.yaml")
def generate_md(cfg: DictConfig) -> None:  # noqa: C901
    """
    Generate molecular dynamics simulation data from a PDB file.

    This function runs an MD simulation using OpenMM, saving trajectory data in chunks.
    It supports resuming from previous chunks if the simulation was interrupted.

    Args:
        cfg: Hydra configuration dictionary containing:
            - seq_name: Name of the sequence
            - pdb_dir: Directory containing PDB files
            - temperature: Simulation temperature in Kelvin
            - frame_interval: Number of integration steps between saved frames
            - frames_per_chunk: Number of frames to save per chunk file
            - time_ns: Total simulation time in nanoseconds
            - warmup_steps: Number of integration steps for warmup
            - log_freq: Frequency of logging (in integration steps)
            - platform_name: OpenMM platform name (e.g., "CPU", "CUDA")
            - platform_properties: Optional platform-specific properties

    Returns:
        None: Function exits early if final chunk already exists.
    """

    assert cfg.frame_interval > 0
    assert cfg.frames_per_chunk > 0
    assert cfg.time_ns > 0

    assert cfg.get("pdb_dir") is not None or (cfg.get("seq_filename") is not None and cfg.get("seq_idx") is not None), (
        "Either 'pdb_dir' or both 'seq_filename' and 'seq_idx' must be specified in the config"
    )

    if cfg.get("seq_name") is not None:
        pdb_path = Path(cfg.pdb_dir) / f"{cfg.seq_name}.pdb"
        sequence = cfg.seq_name
    else:
        with Path(cfg.seq_filename).open() as f:
            sequences = f.read().strip().splitlines()
        sequence = sequences[cfg.seq_idx]
        pdb_path = Path(cfg.pdb_dir) / f"{sequence}.pdb"

    output_dir = f"{cfg.paths.data_dir}/md/{sequence}_{cfg.temperature}_{cfg.frame_interval}_{cfg.frames_per_chunk}"

    # Calculate number of frames from time period
    # Each integration step is 1 fs, frame_interval steps between frames
    # time_ns * 1e6 fs/ns = total time in fs = num_frames * frame_interval
    num_frames = int(cfg.time_ns * 1e6 / cfg.frame_interval)

    # Calculate number of chunks needed
    num_chunks = (num_frames + cfg.frames_per_chunk - 1) // cfg.frames_per_chunk
    logger.info(f"Simulating system {pdb_path} at {cfg.temperature}K")
    logger.info(
        f"Total frames to generate: {num_frames} "
        f"(calculated from {cfg.time_ns} ns / {cfg.frame_interval} fs per saved frame).",
    )
    logger.info(
        f"Chunking: {cfg.frames_per_chunk} frames per chunk -> {num_chunks} chunk(s) "
        f"(last chunk may contain fewer frames).",
    )

    chunks_dir = Path(output_dir) / "chunks"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    final_chunk_path = chunks_dir / f"chunk_{num_chunks - 1}.npz"
    if final_chunk_path.exists():
        logger.info(f"Final chunk already exists at {final_chunk_path}, skipping simulation.")
        return

    pdb = PDBFile(str(pdb_path))
    topology = pdb.getTopology()
    positions = pdb.getPositions(asNumpy=True)

    platform_properties = {}
    if hasattr(cfg, "platform_properties") and cfg.platform_properties is not None:
        platform_properties = dict(cfg.platform_properties)
        if "Threads" in platform_properties:
            platform_properties["Threads"] = str(platform_properties["Threads"])

    forcefield = ForceField("amber14-all.xml", "implicit/obc1.xml")
    system = forcefield.createSystem(
        topology,
        nonbondedMethod=openmm.app.CutoffNonPeriodic,
        nonbondedCutoff=2.0 * openmm.unit.nanometer,
        constraints=None,
    )
    integrator = openmm.LangevinMiddleIntegrator(
        cfg.temperature * openmm.unit.kelvin,
        0.3 / openmm.unit.picosecond,
        1.0 * openmm.unit.femtosecond,
    )
    platform = Platform.getPlatform(cfg.platform_name)
    simulation = Simulation(
        topology,
        system,
        integrator,
        platform=platform,
        platformProperties=platform_properties,
    )
    logger.info(f"Platform name: {cfg.platform_name} properties: {platform_properties}")
    simulation.reporters.append(
        StateDataReporter(
            str(Path(output_dir) / "output.txt"),
            cfg.log_freq * cfg.frame_interval,
            step=True,
            potentialEnergy=True,
            temperature=True,
            progress=True,
            totalSteps=num_frames * cfg.frame_interval,
            remainingTime=True,
            speed=True,
            elapsedTime=True,
            append=True,
        ),
    )

    # Determine starting chunk and initialize simulation
    found_chunk_files = list(chunks_dir.iterdir())
    start_chunk = 0

    if found_chunk_files:
        # NOTE: We had issues with loading checkpoints on different machines.
        # Hence when resuming simply load the positions and velocities from the last saved chunk.
        # This means the trajectories are not deterministic when resuming.
        # If anyone wants to use the checkpoint files instead, they are still being saved every chunk.
        # A PR for deterministic resuming would be welcome! :)

        logger.info("Found existing trajectory chunks, loading previous chunk as checkpoint...")

        chunk_indexes = []
        for filepath in found_chunk_files:
            filename = filepath.name
            if filename.startswith("chunk_"):
                chunk_index = filename.split("_")[-1][:-4]  # remove '.npz'
                assert chunk_index.isdigit(), f"Unexpected chunk filename format: {filename}"
                chunk_indexes.append(int(chunk_index))
        if not chunk_indexes:
            raise FileNotFoundError("Found npz files but none matched expected chunk filename format.")

        last_chunk_idx = max(chunk_indexes)
        assert sorted(chunk_indexes) == list(range(last_chunk_idx + 1)), (
            "Missing chunk files, cannot resume simulation."
        )

        start_chunk = last_chunk_idx + 1

        # Load positions and velocities from the last saved chunk
        chunk_filename = chunks_dir / f"chunk_{last_chunk_idx}.npz"
        data = np.load(chunk_filename)
        pos = data["positions"][-1]
        vel = data["velocities"][-1]
        simulation.context.setPositions(pos * openmm.unit.nanometer)
        simulation.context.setVelocities(vel * openmm.unit.nanometer / openmm.unit.picosecond)

        # Set simulation state to continue from where we left off
        frames_completed = start_chunk * cfg.frames_per_chunk
        simulation.context.setTime(frames_completed * integrator.getStepSize() * cfg.frame_interval)
        simulation.currentStep = frames_completed * cfg.frame_interval + cfg.warmup_steps

        logger.info(f"Resuming from chunk {start_chunk} (frame {frames_completed})...")
    else:
        logger.info("No existing chunks found, starting new simulation...")
        simulation.context.setPositions(positions)
        simulation.minimizeEnergy()
        logger.info("Minimized. Running warmup...")
        simulation.step(cfg.warmup_steps)
        logger.info(f"Warmup done, ({cfg.warmup_steps} steps)")

    logger.info(f"Running simulation: {num_chunks} chunks, starting from chunk {start_chunk}...")

    # Iterate over chunks
    for chunk_idx in range(start_chunk, num_chunks):
        # Calculate how many frames are in this chunk
        frames_in_chunk = min(cfg.frames_per_chunk, num_frames - chunk_idx * cfg.frames_per_chunk)

        chunk_positions = []
        chunk_velocities = []

        # Iterate over frames within the chunk
        for _ in range(frames_in_chunk):
            simulation.step(cfg.frame_interval)
            st = simulation.context.getState(getPositions=True, getVelocities=True)
            coords = st.getPositions(asNumpy=True) / openmm.unit.nanometer
            velocities = st.getVelocities(asNumpy=True).value_in_unit(
                openmm.unit.nanometer / openmm.unit.picosecond,
            )
            chunk_positions.append(coords)
            chunk_velocities.append(velocities)

        # Save checkpointing files.
        # NOTE: We had issues with loading checkpoints on different devices,
        # Hence when resuming simply load the positions and velocities from the last saved chunk.
        # These are left here for completeness and in case someone wants to use them in the future.
        simulation.saveCheckpoint(str(Path(output_dir) / "checkpoint.chk"))
        simulation.saveState(str(Path(output_dir) / "state.xml"))
        system_xml_path = Path(output_dir) / "system.xml"
        with system_xml_path.open("w") as output:
            output.write(XmlSerializer.serialize(system))

        # Save the chunk to npz files.
        chunk_filename = f"chunk_{chunk_idx}.npz"
        chunk_path = chunks_dir / chunk_filename
        chunk_positions_array = np.array(chunk_positions, dtype=np.float32)
        chunk_velocities_array = np.array(chunk_velocities, dtype=np.float32)
        np.savez_compressed(chunk_path, positions=chunk_positions_array, velocities=chunk_velocities_array)
        logger.info(f"Saved chunk {chunk_idx} with {frames_in_chunk} frames to {chunk_path}")


if __name__ == "__main__":
    generate_md()
