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
import os
from typing import Optional

import hydra
import numpy as np
import openmm
import rootutils
from omegaconf import DictConfig
from openmm import Platform, XmlSerializer
from openmm.app import ForceField, Simulation, StateDataReporter, PDBFile

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="generate_md.yaml")
def main(cfg: DictConfig) -> Optional[float]:

    assert cfg.pdb_filename is not None, "pdb_filename must be specified in the config"
    
    # Calculate number of frames from time period
    # Each integration step is 1 fs, frame_interval steps between frames
    # time_ns * 1e6 fs/ns = total time in fs = num_frames * frame_interval
    num_frames = int(cfg.time_ns * 1e6 / cfg.frame_interval)
    logger.info(f"Calculated {num_frames} frames for {cfg.time_ns}ns simulation time")
    
    chunks_dir = os.path.join(cfg.output_dir, "chunks")
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(chunks_dir, exist_ok=True)

    final_chunk_path = os.path.join(
        chunks_dir, f"positions_{(num_frames - 1) // cfg.frames_per_chunk}.npz"
    )
    if os.path.exists(final_chunk_path):
        logger.info(f"Chunks already exist at {chunks_dir}, skipping simulation.")
        return

    pdb_path = os.path.join(cfg.pdb_dir, f"{cfg.pdb_filename}.pdb")
    pdb = PDBFile(pdb_path)
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
    logger.info(f"Platform name: {cfg.platform_name} {platform_properties}")
    logger.info(f"Simulating system {pdb_path} at {cfg.temperature}K")
    logger.info(f"Interval between saved frames: {cfg.frame_interval}fs")
    logger.info(f"Number of frames to generate: {num_frames}")
    logger.info(f"Total simulation time: {cfg.time_ns}ns")
    platform = Platform.getPlatform(cfg.platform_name)
    simulation = Simulation(
        topology,
        system,
        integrator,
        platform=platform,
        platformProperties=platform_properties,
    )
    simulation.reporters.append(
        StateDataReporter(
            os.path.join(cfg.output_dir, "output.txt"),
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
        )
    )

    # Calculate number of chunks needed
    num_chunks = (num_frames + cfg.frames_per_chunk - 1) // cfg.frames_per_chunk

    # Determine starting chunk and initialize simulation
    found_chunk_files = os.listdir(chunks_dir)
    start_chunk = 0

    if found_chunk_files:
        # NOTE: We had issues with loading checkpoints on different machines.
        # Hence when resuming simply load the positions and velocities from the last saved chunk.
        # This means the trajectories are not deterministic when resuming.
        # If anyone wants to use the checkpoint files instead, they are still being saved every chunk.
        # A PR for deterministic resuming would be welcome! :)

        logger.info("Found existing trajectory chunks, loading previous chunk as checkpoint...")

        chunk_indexes = []
        for filename in found_chunk_files:
            if filename.startswith("positions_"):
                chunk_index = os.path.basename(filename).split("_")[-1][:-4]  # remove '.npz'
                assert chunk_index.isdigit(), f"Unexpected chunk filename format: {filename}"
                assert filename.replace("positions_", "velocities_") in found_chunk_files, f"Missing corresponding velocities chunk for {filename}"
                chunk_indexes.append(int(chunk_index))
        if not chunk_indexes:
            raise FileNotFoundError("Found npz files but none matched expected chunk filename format.")
        
        last_chunk_idx = max(chunk_indexes)
        start_chunk = last_chunk_idx + 1
        
        # Load positions and velocities from the last saved chunk
        pos_filename = os.path.join(chunks_dir, f"positions_{last_chunk_idx}.npz")
        vel_filename = os.path.join(chunks_dir, f"velocities_{last_chunk_idx}.npz")
        pos = np.load(pos_filename)["positions"][-1]
        vel = np.load(vel_filename)["velocities"][-1]
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
        logger.info("minimized. running warmup...")
        simulation.step(cfg.warmup_steps)
        logger.info(f"warmup done, ({cfg.warmup_steps} steps)")

    logger.info(f"Running simulation: {num_chunks} chunks, starting from chunk {start_chunk}...")

    # Outer loop: iterate over chunks
    for chunk_idx in range(start_chunk, num_chunks):
        # Calculate how many frames are in this chunk
        frames_in_chunk = min(cfg.frames_per_chunk, num_frames - chunk_idx * cfg.frames_per_chunk)
        
        chunk_positions = []
        chunk_velocities = []
        
        # Inner loop: iterate over frames within this chunk
        for frame_in_chunk in range(frames_in_chunk):
            simulation.step(cfg.frame_interval)
            st = simulation.context.getState(getPositions=True, getVelocities=True)
            coords = st.getPositions(asNumpy=True) / openmm.unit.nanometer
            velocities = st.getVelocities(asNumpy=True).value_in_unit(openmm.unit.nanometer / openmm.unit.picosecond)
            chunk_positions.append(coords)
            chunk_velocities.append(velocities)

        # Save the chunk
        # NOTE: We had issues with loading checkpoints on different devices,
        # Hence when resuming simply load the positions and velocities from the last saved chunk
        # These are left here for completeness and in case someone wants to use them in the future
        simulation.saveCheckpoint(os.path.join(cfg.output_dir, "checkpoint.chk"))
        simulation.saveState(os.path.join(cfg.output_dir, "state.xml"))
        with open(f"{cfg.output_dir}/system.xml", "w") as output:
            output.write(XmlSerializer.serialize(system))

        position_chunk_filename = f"positions_{chunk_idx}.npz"
        velocity_chunk_filename = f"velocities_{chunk_idx}.npz"

        position_chunk_path = os.path.join(chunks_dir, position_chunk_filename)
        velocity_chunk_path = os.path.join(chunks_dir, velocity_chunk_filename)

        chunk_positions = np.array(chunk_positions, dtype=np.float32)
        chunk_velocities = np.array(chunk_velocities, dtype=np.float32)

        logger.info(f"saving positions to {position_chunk_path} with shape {chunk_positions.shape}")
        np.savez_compressed(position_chunk_path, positions=chunk_positions)
        logger.info(f"saving velocities to {velocity_chunk_path} with shape {chunk_velocities.shape}")
        np.savez_compressed(velocity_chunk_path, velocities=chunk_velocities)


if __name__ == "__main__":
    main()
