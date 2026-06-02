from abc import ABC, abstractmethod  
import numpy as np
import pandas as pd
import pickle
import cirq
import qsimcirq as qsim
from tqdm import tqdm


class Simulation():
    
    def __init__(self, protocol:"Protocol"):
    
        self.protocol = protocol
        self.device = protocol.device
        self.model = protocol.model

        self.simulator_options = {'max_fused_gate_size':5, 'use_gpu':False, 't':8, 'r':1}
        self.simulator = qsim.QSimSimulator(self.simulator_options)

    def get_system_state(self, state_vector):
        ''' get reduced state on system (assumed SB in product) '''
        return (state_vector.reshape(2**self.device.Ns, 2**self.device.Nb))[:,0]

    def save_record(self, record, fname, path):
        record.to_pickle(path+fname+".pkl")

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
    
            state = simulator.simulate(cirq.Circuit(self.protocol.reset_layer), initial_state=init).state_vector()
    
            # t = 0 measurement
            record_0 = measurement.measure_from_state_vector(self.get_system_state(state))
            record_0.update({"repeat": r, "t": 0})
            rows.append(record_0)
    
            # subsequent steps
            for t in range(1, N_resets + 1):
                circuit = self.protocol.cycle_circuit(t)
                state = simulator.simulate(circuit, initial_state=state).state_vector()
    
                record_t = measurement.measure_from_state_vector(self.get_system_state(state))
                record_t.update({"repeat": r, "t": t})
                rows.append(record_t)

                if save:
                    df = pd.DataFrame(rows)
                    if t%20==0:
                        self.save_record(df, fname, path)

    
        df = pd.DataFrame(rows)
        if save:
            self.save_record(df, fname, path)
    
        return df
            


