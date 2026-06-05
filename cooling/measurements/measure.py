import numpy as np
import cirq

class Measurement:
    """
    Collects observables and evaluates them from a system state vector.
    """
    def __init__(self, device:"Device"):
        self._measures = {} # name : function(wavefunction) -> value
        self.device = device

    def add_observable(self, name, operator, herm=True):
        """
        add cirq operator to be measured: operator should be a cirq.PauliSum object
        """
        if name in self._measures:
            raise ValueError(f"Measure '{name}' already exists.")
        self._measures[name] = (operator, herm)
        return self

    def measure_from_state_vector(self, state):
        measurement = {}
        for name, (operator, herm) in self._measures.items():
            val = operator.expectation_from_state_vector(state, qubit_map = self.device.qubit_index_map)
            if herm:
                assert abs(val.imag)<1e-5
                val = val.real
            measurement[name] = val
        return measurement

    def add_Hamiltonian(self, model:"Model"):
        operator = model.hamiltonian
        self.add_observable('H0', operator)
        return self

    def add_local_Sops(self):
        for k, q in enumerate(self.device.system_qubits):
            self.add_observable(f'Z_{k}', cirq.Z(q))
            self.add_observable(f'Y_{k}', cirq.Y(q))
            self.add_observable(f'X_{k}', cirq.X(q))
        return self

    def add_total_spin(self):
        Ztot = sum(cirq.Z(q) for q in self.device.system_qubits)/2
        self.add_observable('total_Z', Ztot)
        Xtot = sum(cirq.X(q) for q in self.device.system_qubits)/2
        self.add_observable('total_X', Xtot)
        Ytot = sum(cirq.Y(q) for q in self.device.system_qubits)/2
        self.add_observable('total_Y', Ytot)
        Stot2 = (Xtot)**2+(Ytot)**2+(Ztot)**2
        self.add_observable('total_S2', Stot2)
        return self


class DefaultMeasurement1(Measurement):

    """
    simple measurement containing total spin and (zero-order) Hamiltonian
    """
    def __init__(self, device:"Device", model:"Model"):
        super().__init__(device) 
        self.add_Hamiltonian(model)
        self.add_total_spin()

