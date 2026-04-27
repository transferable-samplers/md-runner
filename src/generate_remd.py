import hashlib
import logging
from pathlib import Path

import hydra
import numpy as np
import openmm
import rootutils
from omegaconf import DictConfig
from openmm import CustomCentroidBondForce, LangevinMiddleIntegrator, Platform, unit
from openmm.app import ForceField, PDBFile, Simulation
from openmmtools import mcmc, multistate, states
from openmmtools.cache import global_context_cache


def stable_hash(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


N_STATES_DICT = {
    2: (4, 4),
    4: (4, 4),
    6: (5, 5),
    8: (5, 5),
    9: (5, 6),
    10: (6, 7),
    11: (6, 7),
    12: (6, 7),
    13: (7, 8),
    14: (7, 8),
    15: (7, 8),
    16: (7, 8),
    17: (7, 8),
    18: (8, 9),
    19: (8, 9),
    20: (8, 9),
    24: (10, 10),
}


SEQUENCE_N_STATES_OVERRIDES = {
    # 14-mer polyA: auto-hash picked 8, exceeding the 16-mer's 7.
    "AAAAAAAAAAAAAA": 7,
    # Bumped +1 from auto-picked values due to low swap acceptance rates.
    "RPKPQQFFGLM": 7,
    "RPPGFSPFR": 6,
}


def get_n_states(sequence: str, mode: str = "auto") -> int:
    if sequence in SEQUENCE_N_STATES_OVERRIDES:
        return SEQUENCE_N_STATES_OVERRIDES[sequence]
    seq_len = len(sequence)
    if seq_len in N_STATES_DICT:
        possible_states = N_STATES_DICT[seq_len]
        if mode == "auto-max":
            return max(possible_states)
        if stable_hash(sequence) % 2 == 0:
            return possible_states[0]
        else:
            return possible_states[1]
    else:
        raise ValueError(f"Sequence length {seq_len} not in N_STATES_DICT")


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def geometric_temps(min_temp: float, max_temp: float, n_states: int) -> unit.Quantity:
    """Generate geometrically spaced temperatures between min_temp and max_temp in Kelvin."""
    return unit.Quantity(
        np.geomspace(min_temp.value_in_unit(unit.kelvin), max_temp.value_in_unit(unit.kelvin), n_states),
        unit.kelvin,
    )


def add_com_restraint(
    system,
    topology,
    r0=2.0 * unit.nanometer,
    k=100.0 * unit.kilojoules_per_mole / unit.nanometer**2,
):
    """Add flat-bottomed COM restraint between first two protein chains.
    See: https://cbc-univie.github.io/transformato/_modules/transformato/restraints.html#Restraint._add_flatbottom_parameters

    Arguments
    ---------
    system : openmm.System
        OpenMM system to add force to.
    topology : openmm.app.Topology
        Topology containing chain information.
    r0 : openmm.unit.Quantity
        Flat-bottom radius (default: 2.0 nm).
    k : openmm.unit.Quantity
        Spring constant for distances > r0 (default: 100 kJ/mol/nm^2).

    Raises
    ------
    ValueError
        If topology has more than 2 chains.
    """
    chains = list(topology.chains())

    if len(chains) < 2:
        return

    if len(chains) > 2:
        raise ValueError(f"COM restraint only supports 2 chains, found {len(chains)}")

    force = CustomCentroidBondForce(2, "step(distance(g1,g2)-r0) * 0.5*k*(distance(g1,g2)-r0)^2")
    force.addPerBondParameter("r0")
    force.addPerBondParameter("k")

    chain1_atoms = [atom.index for atom in chains[0].atoms()]
    chain2_atoms = [atom.index for atom in chains[1].atoms()]

    force.addGroup(chain1_atoms)
    force.addGroup(chain2_atoms)
    force.addBond([0, 1], [r0, k])

    system.addForce(force)


CONSTRAINT_MAP = {
    None: None,
    "None": None,
    "HBonds": openmm.app.HBonds,
    "AllBonds": openmm.app.AllBonds,
    "HAngles": openmm.app.HAngles,
}


def resolve_constraints(constraints):
    if constraints in CONSTRAINT_MAP:
        return CONSTRAINT_MAP[constraints]
    raise ValueError(f"Unknown constraints value {constraints!r}; expected one of {list(CONSTRAINT_MAP)}")


def get_system(topology, forcefield_files, com_restraint=False, constraints="HBonds"):
    forcefield = ForceField(*forcefield_files)
    system = forcefield.createSystem(
        topology,
        nonbondedMethod=openmm.app.CutoffNonPeriodic,
        nonbondedCutoff=2.0 * unit.nanometer,
        constraints=resolve_constraints(constraints),
    )
    if com_restraint:
        add_com_restraint(system, topology)
    return system


def setup_platform(cfg):
    platform_properties = {}
    if hasattr(cfg, "platform_properties") and cfg.platform_properties is not None:
        platform_properties = dict(cfg.platform_properties)
        if "Threads" in platform_properties:
            platform_properties["Threads"] = str(platform_properties["Threads"])
    platform = Platform.getPlatform(cfg.platform_name)
    global_context_cache.set_platform(platform, platform_properties)
    logger.info(f"Platform name: {cfg.platform_name} properties: {platform_properties}")
    return platform, platform_properties


def thermal_scramble(
    topology,
    system,
    positions,
    platform,
    platform_properties,
    cfg,
):
    """Heat-cool cycle to produce a decorrelated starting structure before REMD.

    Protocol:
      1. Minimize, assign velocities at target_temp with scramble_seed.
      2. Ramp target_temp -> scramble_high_temp over scramble_ramp_up_ps.
      3. Hold at scramble_high_temp for scramble_hold_ps to cross barriers.
      4. Ramp back to target_temp over scramble_ramp_down_ps (gradual, not a quench).
      5. Re-equilibrate at target_temp for scramble_equilibrate_ps.

    The scrambling trajectory is not equilibrium sampling; only the final snapshot is returned.
    """
    target_T = cfg.min_temp * unit.kelvin
    high_T = cfg.scramble_high_temp * unit.kelvin
    dt = cfg.timestep_fs * unit.femtosecond
    steps_per_ps = int(round(1000.0 / cfg.timestep_fs))
    update_steps = max(1, int(round(cfg.scramble_ramp_update_ps * steps_per_ps)))

    integrator = LangevinMiddleIntegrator(target_T, 1.0 / unit.picosecond, dt)
    integrator.setRandomNumberSeed(int(cfg.scramble_seed))

    simulation = Simulation(
        topology,
        system,
        integrator,
        platform=platform,
        platformProperties=platform_properties,
    )
    simulation.context.setPositions(positions)
    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(target_T, int(cfg.scramble_seed))

    def ramp(t_start, t_end, duration_ps):
        total_steps = int(round(duration_ps * steps_per_ps))
        if total_steps <= 0:
            return
        n_updates = max(1, total_steps // update_steps)
        t0 = t_start.value_in_unit(unit.kelvin)
        t1 = t_end.value_in_unit(unit.kelvin)
        steps_done = 0
        for i in range(n_updates):
            frac = (i + 1) / n_updates
            T = (t0 + frac * (t1 - t0)) * unit.kelvin
            integrator.setTemperature(T)
            n_steps = update_steps if i < n_updates - 1 else total_steps - steps_done
            simulation.step(n_steps)
            steps_done += n_steps

    logger.info(
        f"Thermal scramble (seed={cfg.scramble_seed}): "
        f"{cfg.min_temp}K -> {cfg.scramble_high_temp}K over {cfg.scramble_ramp_up_ps} ps, "
        f"hold {cfg.scramble_hold_ps} ps, cool over {cfg.scramble_ramp_down_ps} ps, "
        f"equilibrate {cfg.scramble_equilibrate_ps} ps.",
    )

    ramp(target_T, high_T, cfg.scramble_ramp_up_ps)

    integrator.setTemperature(high_T)
    hold_steps = int(round(cfg.scramble_hold_ps * steps_per_ps))
    if hold_steps > 0:
        simulation.step(hold_steps)

    ramp(high_T, target_T, cfg.scramble_ramp_down_ps)

    integrator.setTemperature(target_T)
    equib_steps = int(round(cfg.scramble_equilibrate_ps * steps_per_ps))
    if equib_steps > 0:
        simulation.step(equib_steps)

    state = simulation.context.getState(getPositions=True)
    final_positions = state.getPositions(asNumpy=True)
    logger.info("Thermal scramble complete.")
    return final_positions


def save_swap_rates(reporter, output_dir: Path, sequence: str):
    """Compute and save swap acceptance rates between neighboring thermodynamic states."""
    analysis = reporter._storage_analysis
    accepted = np.array(analysis.variables["accepted"][:])  # (iter, state_i, state_j)
    proposed = np.array(analysis.variables["proposed"][:])
    n_states = accepted.shape[1]

    rates = np.zeros(n_states - 1)
    for i in range(n_states - 1):
        n_acc = accepted[:, i, i + 1].sum()
        n_prop = proposed[:, i, i + 1].sum()
        rates[i] = n_acc / n_prop if n_prop > 0 else 0.0

    seq_len = len(sequence)

    logger.info(
        f"Swap rates {sequence} ({seq_len}) : " + ", ".join(f"{i}<->{i + 1}: {r:.4f}" for i, r in enumerate(rates)),
    )
    np.savetxt(output_dir / "swap_rates.txt", rates)


def demultiplex_trajectories(reporter):
    """Demultiplex trajectories by thermodynamic state (e.g. temperature).

    Returns
    -------
    positions : np.ndarray
        Shape (n_states, n_iter, n_atoms, 3), float32.
    velocities : np.ndarray
        Shape (n_states, n_iter, n_atoms, 3), float32.
    """
    checkpoint = reporter._storage_checkpoint
    analysis = reporter._storage_analysis
    checkpoint_interval = int(checkpoint.CheckpointInterval)

    positions = np.array(checkpoint.variables["positions"][:])  # (n_ckpt, replica, atom, 3)
    velocities = np.array(checkpoint.variables["velocities"][:])  # (n_ckpt, replica, atom, 3)
    all_states = np.array(analysis.variables["states"][:])  # (n_iter, replica)

    n_ckpt, n_states, n_atoms, _ = positions.shape

    # Get state assignments at checkpoint (coord/vel save) frames only
    ckpt_indices = np.arange(n_ckpt) * checkpoint_interval
    states = all_states[ckpt_indices]  # (n_ckpt, replica)

    # Reorder by state using argsort: order[state] = replica holding that state
    demux_pos = np.zeros((n_states, n_ckpt, n_atoms, 3), dtype=np.float32)
    demux_vel = np.zeros((n_states, n_ckpt, n_atoms, 3), dtype=np.float32)

    for frame in range(n_ckpt):
        order = np.argsort(states[frame])
        demux_pos[:, frame] = positions[frame, order]
        demux_vel[:, frame] = velocities[frame, order]

    return demux_pos, demux_vel


def save_state_trajectories(reporter, output_dir: Path, temperatures: np.ndarray):
    """Demultiplex trajectories by thermodynamic state and save to npz file."""
    positions, velocities = demultiplex_trajectories(reporter)
    n_states, n_iter = positions.shape[:2]

    np.savez_compressed(
        output_dir / "trajectories.npz",
        positions=positions,  # (n_states, n_ckpt, n_atoms, 3)
        velocities=velocities,  # (n_states, n_ckpt, n_atoms, 3)
        temperatures=temperatures.astype(np.float32),  # (n_states,)
    )
    logger.info(f"Saved trajectories: {n_states} states, {n_iter} frames each")


@hydra.main(version_base="1.3", config_path="../configs", config_name="generate_remd.yaml")
def generate_remd(cfg: DictConfig) -> None:  # noqa: C901
    assert cfg.frame_interval > 0
    assert cfg.time_ns > 0
    assert cfg.timestep_fs > 0

    assert cfg.get("pdb_dir") is not None or (cfg.get("seq_filename") is not None and cfg.get("seq_idx") is not None), (
        "Either 'pdb_dir' or both 'seq_filename' and 'seq_idx' must be specified in the config"
    )

    if cfg.get("seq_name") is not None:
        pdb_path = Path(cfg.pdb_dir) / f"{cfg.seq_name}.pdb"
    else:
        with Path(cfg.seq_filename).open() as f:
            sequences = f.read().strip().splitlines()
        if cfg.seq_idx < 0 or cfg.seq_idx >= len(sequences):
            raise ValueError(f"seq_idx {cfg.seq_idx} out of range for {len(sequences)} sequences in {cfg.seq_filename}")
        sequence = sequences[cfg.seq_idx]
        pdb_path = Path(cfg.pdb_dir) / f"{sequence}.pdb"
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found at {pdb_path}")

    if str(cfg.n_states).lower() in ("auto", "auto-max"):
        mode = str(cfg.n_states).lower()
        n_states = get_n_states(sequence, mode)
        logger.info(f"{mode}-selected n_states={n_states} for sequence length {len(sequence)}")
    else:
        n_states = int(cfg.n_states)
        assert n_states > 1

    run_name = f"{sequence}_{int(cfg.min_temp)}K-{int(cfg.max_temp)}K_{n_states}_{cfg.timestep_fs}_{cfg.frame_interval}"
    if cfg.get("scramble", False):
        run_name += f"_scramble{int(cfg.scramble_seed)}"
    output_dir = Path(cfg.paths.data_dir) / "remd" / run_name

    if cfg.get("scramble", False):
        # Seed numpy (used by openmmtools for swap-acceptance decisions and other stochastic
        # choices). OpenMM integrators have per-instance seeds; the scramble integrator is
        # seeded explicitly below, and the REMD sampler's integrators fall back to OpenMM's
        # default (OS-random), which we leave alone.
        np.random.seed(int(cfg.scramble_seed))

    platform, platform_properties = setup_platform(cfg)

    pdb = PDBFile(str(pdb_path))
    topology = pdb.getTopology()
    positions = pdb.getPositions(asNumpy=True)

    # Calculate number of frames from time period
    # Each integration step is timestep_fs fs, frame interval steps between frames
    # time_ns * 1e6 fs/ns = total time in fs = num_frames * frame_interval * timestep_fs
    num_frames = int(cfg.time_ns * 1e6 / (cfg.frame_interval * cfg.timestep_fs))

    system = get_system(topology, cfg.forcefield_files, cfg.com_restraint, cfg.get("constraints", "HBonds"))

    temperatures = geometric_temps(cfg.min_temp * unit.kelvin, cfg.max_temp * unit.kelvin, n_states)
    logger.info(
        f"Simulating system {pdb_path} with {n_states} replicas at temperatures: "
        f"{', '.join([f'{t.value_in_unit(unit.kelvin):.1f} K' for t in temperatures])}",
    )
    logger.info(
        f"Total simulation frames per replica to generate: {num_frames} "
        f"calculated from {cfg.time_ns:,} ns / ({cfg.frame_interval:,} * {cfg.timestep_fs} fs per saved frame).",
    )
    thermodynamic_states = [states.ThermodynamicState(system=system, temperature=temp) for temp in temperatures]
    sampler_state = states.SamplerState(
        positions=positions,
        box_vectors=system.getDefaultPeriodicBoxVectors() if system.usesPeriodicBoundaryConditions() else None,
    )

    move = mcmc.LangevinDynamicsMove(
        timestep=cfg.timestep_fs * unit.femtosecond,
        collision_rate=0.3 / unit.picosecond,
        n_steps=cfg.frame_interval,
    )

    sampler = multistate.ReplicaExchangeSampler(
        mcmc_moves=move,
        number_of_iterations=num_frames,
        # Non-reversible parallel tempering (DEO)
        replica_mixing_scheme="swap-neighbors",
        deterministic_swap_order=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    nc_path = output_dir / "remd.nc"
    ckpt_path = output_dir / "remd_checkpoint.nc"

    exit_early_at_ns = cfg.get("exit_early_at_ns", None)
    if exit_early_at_ns is not None and nc_path.exists() and ckpt_path.exists():
        probe = multistate.MultiStateReporter(
            str(nc_path),
            checkpoint_storage=str(ckpt_path),
            open_mode="r",
        )
        try:
            existing_iter = probe._storage_analysis.variables["states"].shape[0]
        finally:
            for s in (probe._storage_analysis, probe._storage_checkpoint, probe._storage):
                try:
                    s.close()
                except Exception:
                    pass
        existing_ns = existing_iter * cfg.frame_interval * cfg.timestep_fs / 1e6
        if existing_ns >= float(exit_early_at_ns):
            logger.info(
                f"Existing run at {existing_iter} iterations ({existing_ns:.2f} ns) "
                f">= exit_early_at_ns={exit_early_at_ns} ns; exiting without resuming.",
            )
            return

    reporter = multistate.MultiStateReporter(
        nc_path,
        checkpoint_interval=1,  # Save coords and velocities every swap attempt
        checkpoint_storage=ckpt_path,
    )

    if nc_path.exists() and ckpt_path.exists():
        logger.info(f"Resuming from existing simulation files: {nc_path}, {ckpt_path}")
        sampler = multistate.ReplicaExchangeSampler.from_storage(reporter)
        is_minimized = bool(getattr(reporter._storage_checkpoint, "is_minimized", 0))
        is_equilibrated = bool(getattr(reporter._storage_checkpoint, "is_equilibrated", 0))
    else:
        logger.info("Starting new REMD simulation from scratch")
        if cfg.get("scramble", False):
            scrambled_positions = thermal_scramble(
                topology,
                system,
                positions,
                platform,
                platform_properties,
                cfg,
            )
            PDBFile.writeFile(topology, scrambled_positions, str(output_dir / "scramble_snapshot.pdb"))
            sampler_state = states.SamplerState(
                positions=scrambled_positions,
                box_vectors=sampler_state.box_vectors,
            )
        sampler.create(thermodynamic_states, [sampler_state] * n_states, reporter)
        reporter._storage_checkpoint.is_minimized = int(cfg.get("scramble", False))
        reporter._storage_checkpoint.is_equilibrated = 0
        reporter.sync()
        is_minimized = bool(cfg.get("scramble", False))
        is_equilibrated = False

    if not is_equilibrated:
        if not is_minimized:
            sampler.minimize()
            reporter._storage_checkpoint.is_minimized = 1
            reporter.sync()
            logger.info("Minimized, running warmup/equilibration...")

        # Determine number of equilibration iterations from warmup time (preferred)
        # Each iteration corresponds to `frame_interval * timestep_fs` femtoseconds
        # convert ns -> fs then to iterations
        warmup_iterations = int(cfg.warmup_time_ns * 1e6 / (cfg.frame_interval * cfg.timestep_fs))
        warmup_iterations = max(1, int(warmup_iterations))

        sampler.equilibrate(warmup_iterations)
        reporter._storage_checkpoint.is_equilibrated = 1
        reporter.sync()
        n_equib_ps = warmup_iterations * cfg.frame_interval * cfg.timestep_fs / 1e3
        n_equib_ns = n_equib_ps / 1e3
        logger.info(
            f"Warmup done, {warmup_iterations} iterations ({n_equib_ps:.2f} ps / {n_equib_ns:.3f} ns)",
        )

    if exit_early_at_ns is not None:
        target_iter = int(float(exit_early_at_ns) * 1e6 / (cfg.frame_interval * cfg.timestep_fs))
        current_iter = sampler.iteration or 0
        remaining = max(0, target_iter - current_iter)
        logger.info(
            f"exit_early_at_ns={exit_early_at_ns} ns -> target_iter={target_iter}; "
            f"current_iter={current_iter}, running up to {remaining} more iterations.",
        )
        sampler.run(n_iterations=remaining)
    else:
        sampler.run()

    save_swap_rates(reporter, output_dir, sequence)
    # NOTE: Demultiplexing is CPU-bound and may waste GPU time; some may prefer to do this as a post-processing step
    if cfg.get("demultiplex", False):
        logger.info("Demultiplexing and saving state trajectories...")
        temps = np.array([t.value_in_unit(unit.kelvin) for t in temperatures])
        save_state_trajectories(reporter, Path(output_dir), temps)
        logger.info("Demultiplexing complete.")


if __name__ == "__main__":
    generate_remd()
