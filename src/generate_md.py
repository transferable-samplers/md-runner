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

import glob

import hydra
import numpy as np
import openmm
import rootutils
import tqdm
from omegaconf import DictConfig
from openmm import Platform, XmlSerializer
from openmm.app import CheckpointReporter, ForceField, Simulation, StateDataReporter, PDBFile

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="generate_md.yaml")
def main(cfg: DictConfig) -> Optional[float]:

    assert cfg.pdb_filename is not None, "pdb_filename must be specified in the config"
    chunks_dir = os.path.join(cfg.output_dir, "chunks")
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(chunks_dir, exist_ok=True)

    final_chunk_path = os.path.join(
        chunks_dir, f"positions_{(cfg.num_frames - 1) // cfg.frames_per_chunk}.npz"
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
    logger.info(f"Number of frames to generate: {cfg.num_frames}")
    logger.info(f"Total simulation time: {cfg.num_frames * cfg.frame_interval / 1e6} microseconds")
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
            totalSteps=cfg.num_frames * cfg.frame_interval,
            remainingTime=True,
            speed=True,
            elapsedTime=True,
            append=True,
        )
    )

    found_chunk_files = os.listdir(chunks_dir)

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
        highest_index = max(chunk_indexes)
        highest_value = highest_index + 1
        pos_filename = os.path.join(chunks_dir, f"positions_{highest_index}.npz")
        vel_filename = os.path.join(chunks_dir, f"velocities_{highest_index}.npz")
        pos = np.load(pos_filename)["positions"][-1]
        vel = np.load(vel_filename)["velocities"][-1]
        simulation.context.setPositions(pos * openmm.unit.nanometer)
        simulation.context.setVelocities(vel * openmm.unit.nanometer/openmm.unit.picosecond)
        simulation.context.setTime(highest_value * integrator.getStepSize() * cfg.frame_interval)
        simulation.currentStep = highest_value * cfg.frame_interval + cfg.warmup_steps
        current_step_md = simulation.currentStep
        current_step_md = max(0, current_step_md - cfg.warmup_steps)
        start_step = current_step_md // cfg.frame_interval
        logger.info(f"Loaded checkpoint at chunk {start_step}. Resuming…")
    else:
        logger.info("No existing chunks found, starting new simulation...")
        simulation.context.setPositions(positions)
        simulation.minimizeEnergy()
        logger.info("minimized. running warmup...")
        simulation.step(cfg.warmup_steps)
        logger.info(f"warmup done, ({cfg.warmup_steps} steps)")
        start_step = 0

    chunk_positions = []
    chunk_velocities = []
    logger.info(f"running simulation...")

    for step in range(start_step, cfg.num_frames): # removed tqdm due to admin complaints about excessive unbuffered stdout
        simulation.step(cfg.frame_interval)
        st = simulation.context.getState(getPositions=True, getVelocities=True)
        coords = st.getPositions(asNumpy=True) / openmm.unit.nanometer
        velocities = st.getVelocities(asNumpy=True).value_in_unit(openmm.unit.nanometer / openmm.unit.picosecond)
        chunk_positions.append(coords)
        chunk_velocities.append(velocities)

        if not (step + 1) % cfg.frames_per_chunk or (step + 1) == cfg.num_frames:

            # Save the state of the simulation
            # NOTE: We had issues with loading checkpoints on different devices,
            # Hence when resuming simply load the positions and velocities from the last saved chunk
            # These are left here for completeness and in case someone wants to use them in the future
            simulation.saveCheckpoint(os.path.join(cfg.output_dir, "checkpoint.chk"))
            simulation.saveState(os.path.join(cfg.output_dir, "state.xml"))
            with open(f"{cfg.output_dir}/system.xml", "w") as output:
                output.write(XmlSerializer.serialize(system))

            position_chunk_filename = f"positions_{step // cfg.frames_per_chunk}.npz"
            velocity_chunk_filename = f"velocities_{step // cfg.frames_per_chunk}.npz"

            position_chunk_path = os.path.join(chunks_dir, position_chunk_filename)
            velocity_chunk_path = os.path.join(chunks_dir, velocity_chunk_filename)

            chunk_positions = np.array(chunk_positions, dtype=np.float32)
            chunk_velocities = np.array(chunk_velocities, dtype=np.float32)

            logger.info(f"saving positions to {position_chunk_path} with shape {chunk_positions.shape}")
            np.savez_compressed(position_chunk_path, positions=chunk_positions)
            logger.info(f"saving velocities to {velocity_chunk_path} with shape {chunk_velocities.shape}")
            np.savez_compressed(velocity_chunk_path, velocities=chunk_velocities)

            chunk_positions = []
            chunk_velocities = []


if __name__ == "__main__":
    main()
