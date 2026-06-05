"""
Simulation runner for cooling protocols.

Created by Jerome Lloyd on 5th June 2026.
"""

import os
import numpy as np
import pandas as pd
import cirq
import qsimcirq as qsim
from tqdm import tqdm

from .measurements import DefaultMeasurement1


class Simulation:
    """
    Runs cooling protocol simulations.

    Parameters
    ----------
    protocol : Protocol
    """

    def __init__(self, protocol: "Protocol"):
        self.protocol    = protocol
        self.device      = protocol.device
        self.model       = protocol.model

        # System qubits first: get_system_state reshape assumes this layout.
        self.qubit_order = self.device.qubits

        options = {'max_fused_gate_size': 5, 'use_gpu': False, 't': 8}
        ## memoization sets how many circuits qsim stores in memory
        ## for single repeated circuit we want circuit_memoization_size=1
        ## for K repeated circuits we may want circuit_memoization_size=K
        self.simulator = qsim.QSimSimulator(options, circuit_memoization_size=2)

    # ── State helpers ─────────────────────────────────────────────────────────

    def get_system_state(self, state_vector):
        """
        Reduced state on system qubits (valid immediately after a reset layer).

        With qubit_order = system + bath, system occupies the high-order bits.
        Reshaping to (2**Ns, 2**Nb) and taking column 0 selects bath=|0…0>.
        """
        return state_vector.reshape(2 ** self.device.Ns, 2 ** self.device.Nb)[:, 0]

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

    def save(self, record: pd.DataFrame, fname: str = "test_data",
             path: str = None):
        """Save a simulation record as a pickle file."""
        if path is None:
            path = os.path.join(os.path.dirname(__file__), "..", "data", "simulations")
        os.makedirs(path, exist_ok=True)
        record.to_pickle(os.path.join(path, fname + ".pkl"))

    def run(self, circuit_fn, R: int, K: int = 1, measurement=None, seed=None) -> pd.DataFrame:
        """
        Run K independent trajectories of R cooling steps each.

        circuit_fn  : FrozenCircuit or callable int → FrozenCircuit
        R           : number of cooling steps per trajectory
        K           : number of independent trajectories
        measurement : Measurement; defaults to DefaultMeasurement1
        seed        : RNG seed

        Returns a DataFrame with columns {repeat, t, ...observables}.
        """
        if measurement is None:
            measurement = DefaultMeasurement1(self.device, self.model)
        circuit_fn = self._wrap(circuit_fn)
        rng  = np.random.default_rng(seed)
        rows = []

        for k in tqdm(range(K)):
            state = self._initial_state(rng)
            rows.append({"repeat": k, "t": 0,
                         **measurement.measure_from_state_vector(self.get_system_state(state))})

            for t in range(1, R + 1):
                state = self._step(circuit_fn(t), state)
                rows.append({"repeat": k, "t": t,
                             **measurement.measure_from_state_vector(self.get_system_state(state))})

        return pd.DataFrame(rows)

    def expectation_values(self, circuit_fn, R: int, K: int = 1, measurement=None, seed=None) -> pd.DataFrame:
        """
        Run K trajectories of R cooling steps, using a concatenated circuit.

        Concatenates the R channel applications into one big circuit and uses
        simulate_moment_steps once per trajectory, reducing translation overhead.
        Measures at every Nth moment (end of each channel, after the reset layer).

        circuit_fn  : FrozenCircuit or callable int → FrozenCircuit
        R           : number of cooling steps per trajectory
        K           : number of independent trajectories
        measurement : Measurement; defaults to DefaultMeasurement1
        seed        : RNG seed

        Returns a DataFrame with columns {repeat, t, ...observables}.
        """
        if measurement is None:
            measurement = DefaultMeasurement1(self.device, self.model)
        circuit_fn = self._wrap(circuit_fn)

        # Build concatenated circuit once — reused for all K trajectories
        big_circuit = cirq.Circuit()
        for t in range(1, R + 1):
            big_circuit += cirq.Circuit(circuit_fn(t))
        N = len(big_circuit) // R
        # short-circuit cirq's per-op parameterisation scan — our circuit is
        # always concrete (no sympy symbols), so resolve_parameters returns early
        big_circuit._is_parameterized_ = lambda: False

        # Observables at each channel boundary moment (last moment of each channel)
        obs_list  = [op for op, _ in measurement._measures.values()]
        obs_names = list(measurement._measures.keys())
        indexed_obs = {k * N - 1: obs_list for k in range(1, R + 1)}

        rng  = np.random.default_rng(seed)
        rows = []

        for k in tqdm(range(K)):
            state = self._initial_state(rng)
            rows.append({"repeat": k, "t": 0,
                         **measurement.measure_from_state_vector(self.get_system_state(state))})

            results = self.simulator.simulate_moment_expectation_values(
                big_circuit, indexed_obs, cirq.ParamResolver(),
                qubit_order=self.qubit_order, initial_state=state,
            )
            for step_idx, obs_values in enumerate(results):
                rows.append({"repeat": k, "t": step_idx + 1,
                             **dict(zip(obs_names, obs_values))})

        return pd.DataFrame(rows)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _step(self, circuit, state):
        """Advance state by one circuit application."""
        return self.simulator.simulate(
            circuit, initial_state=state, qubit_order=self.qubit_order
        ).state_vector()

    @staticmethod
    def _wrap(circuit_fn):
        """Coerce a FrozenCircuit to a callable int → FrozenCircuit."""
        if isinstance(circuit_fn, cirq.FrozenCircuit):
            return lambda t: circuit_fn
        return circuit_fn
