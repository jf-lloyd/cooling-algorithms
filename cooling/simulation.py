from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import pickle
import cirq
import qsimcirq as qsim
from tqdm import tqdm


class Simulation():

    def __init__(self, protocol: "Protocol"):

        self.protocol = protocol
        self.device = protocol.device
        self.model = protocol.model

        # Canonical qubit order: system qubits first, then bath qubits.
        # The state-vector reshape in get_system_state assumes this layout
        # (system = high-order bits, bath = low-order bits), so we pass it
        # explicitly to every simulate() call rather than relying on cirq's
        # default sort. The default sorts NamedQubits alphabetically, which
        # would place 'bath_*' before 'system_*' and silently misalign the
        # reshape.
        self.qubit_order = list(self.device.system_qubits) + list(self.device.bath_qubits)

        self.simulator_options = {'max_fused_gate_size': 5, 'use_gpu': False, 't': 8, 'r': 1}
        self.simulator = qsim.QSimSimulator(self.simulator_options)

    def get_system_state(self, state_vector):
        '''
        Reduced state on the system.

        Valid only when the bath is in product with the system and reset to
        |0...0> (true immediately after a reset layer). With qubit_order =
        system + bath, system occupies the high-order bits, so reshaping to
        (2**Ns, 2**Nb) and taking column 0 selects the bath=|0...0> slice.
        '''
        return (state_vector.reshape(2 ** self.device.Ns, 2 ** self.device.Nb))[:, 0]

    def save_record(self, record, fname, path):
        record.to_pickle(path + fname + ".pkl")

    def record_from_trajectory(self, measurement: "Measurement", N_resets: int, k: int, seed: int | None = None,
                               save=True, fname=None, path=None):

        rng = np.random.default_rng(seed)

        simulator = self.simulator
        Ns = self.device.Ns
        Nb = self.device.Nb
        dim = 2 ** (Ns + Nb)

        rows = []

        for r in tqdm(range(k)):
            # initial state index, randomly drawn from computational basis
            init = int(rng.integers(0, dim))

            state = simulator.simulate(
                cirq.Circuit(self.protocol.reset_layer),
                initial_state=init,
                qubit_order=self.qubit_order,
            ).state_vector()

            # t = 0 measurement
            record_0 = measurement.measure_from_state_vector(self.get_system_state(state))
            record_0.update({"repeat": r, "t": 0})
            rows.append(record_0)

            # subsequent steps
            for t in range(1, N_resets + 1):
                circuit = self.protocol.cycle_circuit(t)
                state = simulator.simulate(
                    circuit,
                    initial_state=state,
                    qubit_order=self.qubit_order,
                ).state_vector()

                record_t = measurement.measure_from_state_vector(self.get_system_state(state))
                record_t.update({"repeat": r, "t": t})
                rows.append(record_t)

                # periodic checkpoint: only build the dataframe when we save
                if save and t % 20 == 0:
                    self.save_record(pd.DataFrame(rows), fname, path)

        df = pd.DataFrame(rows)
        if save:
            self.save_record(df, fname, path)

        return df