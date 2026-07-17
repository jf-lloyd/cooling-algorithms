"""
Simulation runner for cooling protocols.

Created by Jerome Lloyd on 5th June 2026.
"""

import os
import multiprocessing as _mp
import numpy as np
import pandas as pd
import cirq
import qsimcirq as qsim
from tqdm import tqdm

from .measurements import DefaultMeasurement1

# ── Parallel worker (module-level so it's picklable by spawn) ────────────────

def _parallel_worker(args):
    (protocol, circuit_fn, R, k_worker, measure_every, seed,
     circuit_memoization_size, measurement_cls) = args
    # measurement_cls is a class, not an instance: classes pickle by reference,
    # so this crosses the spawn boundary where a built Measurement could not.
    sim = Simulation(protocol, measurement=measurement_cls)
    if circuit_memoization_size is not None and not hasattr(circuit_fn, 'sim_options'):
        sim.set_memoization_size(circuit_memoization_size)
    return sim.run(
        circuit_fn, R, k_worker,
        measure_every=measure_every,
        seed=seed,
        circuit_memoization_size=circuit_memoization_size,
        _save=False,
    )


class Simulation:
    """
    Runs cooling protocol simulations.

    Parameters
    ----------
    protocol : Protocol
    
    assume_trajectories : bool (default True)
        The channel always contains a reset. Normally qsim rescans the whole circuit
        on every simulate() call to discover trajectory MC is needed.
        When True, we short-circuit qsim's _needs_trajectories to always return
        True, skipping that scan.

    measurement : Measurement, or a Measurement subclass (default DefaultMeasurement1)
        Default measurement for run(). Pass either an instance, or the class itself
        (e.g. cooling.DefaultMeasurement2), which is instantiated as
        cls(device, model). run(measurement=...) still overrides per call.

    """

    def __init__(self, protocol: "Protocol", assume_trajectories: bool = True,
                 measurement=None):
        self.protocol    = protocol
        self.device      = protocol.device
        self.model       = protocol.model

        # Keep the class as well as the instance: run_parallel can only ship a
        # class to spawned workers (a built Measurement holds unpicklable PauliSums).
        self._measurement_cls = measurement if isinstance(measurement, type) else None
        if isinstance(measurement, type):
            measurement = measurement(self.device, self.model)
        self.measurement = measurement

        # System qubits first: get_system_state reshape assumes this layout.
        self.qubit_order = self.device.qubits

        self.assume_trajectories = assume_trajectories
        if assume_trajectories:
            # Module-global patch: every circuit here needs trajectories.
            qsim.qsim_simulator._needs_trajectories = lambda circuit: True

        self.options = {'max_fused_gate_size': 4, 'use_gpu': False, 't': 2}
        self.memoization = 1
        self.set_memoization_size()

    def set_memoization_size(self, size:int=2):
        ## memoization sets how many circuits qsim stores in memory
        self.memoization = size
        self.simulator = qsim.QSimSimulator(self.options, circuit_memoization_size=size)

    # ── State helpers ─────────────────────────────────────────────────────────

    def get_system_state(self, state_vector):
        """
        System state after the final bath reset layer.

        With qubit_order = system + bath, system occupies the high-order bits.
        In the noiseless case the bath is |0...0>, so column 0 is the system
        state. With reset noise, the bath can be flipped to another
        computational basis state; select the occupied bath sector instead.
        """
        psi = np.asarray(state_vector, dtype=np.complex128).reshape(
            2 ** self.device.Ns, 2 ** self.device.Nb
        )
        bath_probs = np.sum(np.abs(psi) ** 2, axis=0)
        occupied = np.flatnonzero(bath_probs > 1e-10)

        if len(occupied) != 1:
            raise ValueError(
                "Expected the bath to occupy one basis state after reset, "
                f"found probabilities {bath_probs}."
            )

        col = int(occupied[0])
        system_state = psi[:, col]
        system_norm = np.linalg.norm(system_state)
        if not np.isclose(system_norm, 1.0, atol=1e-4):
            raise ValueError(
                "System state norm drifted too far from 1 after bath projection: "
                f"{system_norm}."
            )
        return system_state / system_norm

    def _initial_state(self, rng):
        """Random computational basis state with bath reset to |0…0>."""
        dim  = 2 ** (self.device.Ns + self.device.Nb)
        init = int(rng.integers(0, dim))
        return self.simulator.simulate(
            cirq.Circuit(self.protocol.reset_layer),
            initial_state=init,
            qubit_order=self.qubit_order,
        ).state_vector()

    # ── Simulation methods ────────────────────────────────────────────────────

    @property
    def _default_path(self):
        return os.path.join(os.path.dirname(__file__), "..", "data", "simulations")

    @staticmethod
    def _build_fname(schedule, R: int, K: int, tag: str = None) -> str:
        fname = f"{schedule.fname}_R{R}K{K}"
        if tag:
            fname += f"_{tag}"
        return fname

    def get_path(self, schedule, R: int, K: int, tag: str = None, save_path: str = None) -> str:
        """Return the full path of a saved simulation file."""
        path = save_path or self._default_path
        return os.path.join(path, self._build_fname(schedule, R, K, tag) + ".parquet")

    def load(self, schedule, R: int, K: int, tag: str = None, save_path: str = None) -> pd.DataFrame:
        """Load a saved simulation record."""
        return pd.read_parquet(self.get_path(schedule, R, K, tag, save_path))

    def save(self, record: pd.DataFrame, fname: str = "test_data", path: str = None):
        """Save a simulation record as a parquet file."""
        if path is None:
            path = self._default_path
        os.makedirs(path, exist_ok=True)
        record.to_parquet(os.path.join(path, fname + ".parquet"))

    def run(self, circuit_fn, R: int, K: int = 1, measurement=None, measure_every: int = 1, seed=None,
            circuit_memoization_size=None, tag: str = None, save_path: str = None,
            overwrite: bool = False, _save: bool = True) -> pd.DataFrame:
        """
        Run K independent trajectories of R cooling steps each.

        circuit_fn   : Schedule, FrozenCircuit, or callable int → FrozenCircuit.
        R            : number of cooling steps per trajectory
        K            : number of independent trajectories
        measurement  : Measurement instance or subclass; overrides the one set on
                       the Simulation. Defaults to that, else DefaultMeasurement1.
        measure_every : measure observables every this many steps (default 1).
        seed         : RNG seed
        circuit_memoization_size : override qsim memoization. If None and a
                       Schedule is passed, defaults to the schedule cache size.
        tag          : appended to filename when auto-saving
        save_path    : override default save directory.
        overwrite    : if False (default), abort before running if the output file already exists.

        Returns a DataFrame with columns {repeat, t, ...observables}.
        """
        self._validate_run_args(R, K, measure_every)
        schedule, circuit_fn = self._unwrap_schedule(circuit_fn, circuit_memoization_size)

        if schedule is not None and _save:
            fname = self._build_fname(schedule, R, K, tag)
            path = save_path or self._default_path
            full_path = os.path.join(path, fname + ".parquet")
            if os.path.exists(full_path) and not overwrite:
                raise FileExistsError(
                    f"Output file already exists: {full_path}\n"
                    f"change file tag or set overwrite=True to overwrite."
                )
        if measurement is None:
            measurement = self.measurement or DefaultMeasurement1(self.device, self.model)
        elif isinstance(measurement, type):
            measurement = measurement(self.device, self.model)
        circuit_fn = self._wrap(circuit_fn)

        # Derive three independent sub-seeds so all randomness is reproducible.
        # SeedSequence(None) gives a fresh random seed when seed=None.
        ss = np.random.SeedSequence(seed)
        rng_seed, qsim_seed, schedule_seed = ss.spawn(3)
        rng = np.random.default_rng(rng_seed)
        # Rebuild simulator with a fixed qsim seed (controls trajectory collapses).
        self.simulator = qsim.QSimSimulator(
            self.options, circuit_memoization_size=self.memoization,
            seed=int(np.random.default_rng(qsim_seed).integers(2**31))
        )
        # Reseed the schedule's circuit-selection RNG so repeated run() calls
        # with the same seed draw the same sequence of circuits from the cache.
        if schedule is not None:
            selection_seed, build_seed = schedule_seed.spawn(2)
            schedule._rng = np.random.default_rng(selection_seed)
            if hasattr(schedule, '_build_rng'):
                schedule._build_rng = np.random.default_rng(build_seed)

        rows = []

        for k in range(K):
            if schedule is not None and schedule.sim_options.get('resample_trajectories'):
                schedule.build_cache(_warn=False)
            state = self._initial_state(rng)
            rows.append({"repeat": k, "t": 0,
                         **measurement.measure_from_state_vector(self.get_system_state(state))})

            for t in range(1, R + 1):
                state = self._step(circuit_fn(t), state)
                if t % measure_every == 0:
                    rows.append({"repeat": k, "t": t,
                                 **measurement.measure_from_state_vector(self.get_system_state(state))})

        record = pd.DataFrame(rows)

        if schedule is not None and _save:
            self.save(record, fname=fname, path=path)

        return record

    def run_parallel(self, circuit_fn, R: int, K: int, n_workers: int = None,
                     measure_every: int = 1, seed=None, circuit_memoization_size=None,
                     tag: str = None, save_path: str = None, overwrite: bool = False,
                     measurement=None) -> pd.DataFrame:
        """
        Run K trajectories split across n_workers processes.

        Same API as run(). n_workers defaults to min(K, cpu_count).

        measurement : Measurement *subclass* (not an instance), e.g.
                      cooling.DefaultMeasurement2. Defaults to the one set on the
                      Simulation, else DefaultMeasurement1. Workers are spawned, so
                      only a class can be shipped: a built Measurement holds cirq
                      PauliSums, which are not picklable. Each worker instantiates
                      it as cls(device, model).
        """
        self._validate_run_args(R, K, measure_every)
        # Resolve per-call arg, else the class given to __init__, else the instance
        # given to __init__ (which cannot be shipped -> explicit error below).
        measurement_cls = (measurement if measurement is not None
                           else self._measurement_cls if self._measurement_cls is not None
                           else self.measurement)
        if measurement_cls is not None and not isinstance(measurement_cls, type):
            raise TypeError(
                "run_parallel needs a Measurement subclass, not an instance "
                f"(got {type(measurement_cls).__name__}): spawned workers cannot "
                "unpickle a built Measurement. Pass e.g. measurement=cooling.DefaultMeasurement2, "
                "or use run() for a pre-built instance."
            )
        schedule, raw_circuit_fn = self._unwrap_schedule(circuit_fn, circuit_memoization_size)

        if schedule is not None:
            fname = self._build_fname(schedule, R, K, tag)
            path = save_path or self._default_path
            full_path = os.path.join(path, fname + ".parquet")
            if os.path.exists(full_path) and not overwrite:
                raise FileExistsError(
                    f"Output file already exists: {full_path}\n"
                    f"Change file tag or set overwrite=True to overwrite."
                )

        if n_workers is None:
            n_workers = min(K, _mp.cpu_count())
        if n_workers <= 0:
            raise ValueError(f"n_workers must be positive, got {n_workers!r}.")
        n_workers = min(n_workers, K)

        ss = np.random.SeedSequence(seed)
        worker_seeds = [int(np.random.default_rng(s).integers(2**31))
                        for s in ss.spawn(n_workers)]

        base, rem = divmod(K, n_workers)
        k_splits = [base + (1 if i < rem else 0) for i in range(n_workers)]

        # Workers receive the Schedule itself so each process can reseed its
        # local schedule copy and honor resample_trajectories. Worker-side
        # saving is disabled in _parallel_worker.
        worker_circuit_fn = schedule if schedule is not None else raw_circuit_fn
        worker_args = [
            (self.protocol, worker_circuit_fn, R, k, measure_every, ws,
             circuit_memoization_size, measurement_cls)
            for k, ws in zip(k_splits, worker_seeds)
        ]

        ctx = _mp.get_context('spawn')
        with ctx.Pool(n_workers) as pool:
            dfs = pool.map(_parallel_worker, worker_args)

        # Reassign repeat indices to be globally unique across workers.
        offset, out = 0, []
        for df in dfs:
            df = df.copy()
            df['repeat'] += offset
            offset += int(df['repeat'].max()) + 1
            out.append(df)

        record = pd.concat(out, ignore_index=True)

        if schedule is not None:
            self.save(record, fname=fname, path=path)

        return record

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _step(self, circuit, state):
        """Advance state by one circuit application."""
        return self.simulator.simulate(
            circuit, initial_state=state, qubit_order=self.qubit_order
        ).state_vector()

    def _unwrap_schedule(self, circuit_fn, circuit_memoization_size=None):
        """If circuit_fn is a Schedule, extract its circuit_fn and set memoization.

        Memoization size: circuit_memoization_size if given, else the schedule
        cache_size (one slot per distinct circuit). +1 for the reset circuit.
        """
        if hasattr(circuit_fn, 'sim_options'):
            schedule = circuit_fn
            memo = circuit_memoization_size if circuit_memoization_size is not None else schedule.cache_size
            self.set_memoization_size(memo + 1)
            return schedule, schedule.circuit_fn
        return None, circuit_fn

    @staticmethod
    def _validate_run_args(R: int, K: int, measure_every: int):
        if R < 0:
            raise ValueError(f"R must be non-negative, got {R!r}.")
        if K <= 0:
            raise ValueError(f"K must be positive, got {K!r}.")
        if measure_every <= 0:
            raise ValueError(f"measure_every must be positive, got {measure_every!r}.")

    @staticmethod
    def _wrap(circuit_fn):
        """Coerce a FrozenCircuit to a callable int → FrozenCircuit."""
        if isinstance(circuit_fn, cirq.FrozenCircuit):
            return lambda t: circuit_fn
        return circuit_fn
