from abc import ABC, abstractmethod
import cirq
import numpy as np


class Model(ABC):
    """
    Abstract base class for quantum spin models. A model is basically a list of couplings and on-site terms.
    From this the model builds Hamiltonian and ordered list of system gates.

    Subclasses must implement:
        - name : string identifier encoding model type and geometry
        - build_coupling_lists : ordered dict {operator_str: [(strength, s[, t]), ...]} where
            operator strings are: 'XX', 'YY', 'ZZ' for two-site interactions; 'X', 'Y', 'Z' for single-site fields.
            Insertion order controls gate application order in build_system_layer.

    The base class provides default implementations of:
        - build_system_layer  : translates coupling_lists to Cirq gates via _GATE_MAP.
            Two-site gates are applied layer-by-layer using lattice.bond_colouring().
            Override if XX,YY,ZZ decomposition not required (e.g. Heisenberg gates, iSWAP).
        - build_hamiltonian : translates coupling_lists to a cirq.PauliSum and
            commuting components dict, keyed by operator string.

    After initialisation, the following attributes are available:
        model.hamiltonian            : full Hamiltonian as cirq.PauliSum
        model.hamiltonian_components : dict of commuting pieces {op_str: cirq.PauliSum}
        model.coupling_lists         : coupling data to define Hamiltonian, gates
        model.system_layer           : list of Cirq gates for one system evolution step
    """

    # Maps operator strings to Cirq gate constructors.
    # Two-site: gate(strength) returns a gate implementing exp(-i strength PP).
    # One-site:  gate(strength) returns a gate implementing exp(-i strength P).
    # If other gates are needed (e.g. iSWAP), then build_system_layer can be overwritten in model implementation
    _GATE_MAP = {
        'XX': lambda J: cirq.XXPowGate(exponent=J / (np.pi / 2), global_shift=-0.5),
        'YY': lambda J: cirq.YYPowGate(exponent=J / (np.pi / 2), global_shift=-0.5),
        'ZZ': lambda J: cirq.ZZPowGate(exponent=J / (np.pi / 2), global_shift=-0.5),
        'X':  lambda g: cirq.rx(2 * g),
        'Y':  lambda g: cirq.ry(2 * g),
        'Z':  lambda g: cirq.rz(2 * g),
    }

    # Maps operator strings to Cirq Pauli operators for building PauliSum.
    _PAULI_MAP = {
        'XX': lambda q1, q2: cirq.X(q1) * cirq.X(q2),
        'YY': lambda q1, q2: cirq.Y(q1) * cirq.Y(q2),
        'ZZ': lambda q1, q2: cirq.Z(q1) * cirq.Z(q2),
        'X':  lambda q: cirq.X(q),
        'Y':  lambda q: cirq.Y(q),
        'Z':  lambda q: cirq.Z(q),
    }

    def __init__(self, device: "CoolingDevice"):
        self._device = device
        self._lattice = device.lattice
        self._Ns = device.Ns

        if hasattr(self, '_ACCEPTED_PARAMS'):
            unknown = set(self.params) - self._ACCEPTED_PARAMS
            if unknown:
                raise ValueError(f"Unknown parameter(s) {unknown}. Accepted: {self._ACCEPTED_PARAMS}")

        self._coupling_lists = self.build_coupling_lists()
        self._system_layer = self.build_system_layer()
        self.hamiltonian, self.hamiltonian_components = self.build_hamiltonian()

    @property
    def system_layer(self):
        return self._system_layer

    @property
    def coupling_lists(self):
        return self._coupling_lists

    @property
    def lattice(self):
        return self._lattice

    @property
    def device(self):
        return self._device

    @property
    def Ns(self):
        return self._Ns

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def build_coupling_lists(self) -> dict:
        """
        Return an ordered dict {operator_str: [(strength, s[, t]), ...]}.
        Operator strings: 'XX', 'YY', 'ZZ' for two-site; 'X', 'Y', 'Z' for one-site.
        Insertion order controls gate application order in build_system_layer.
        """
        pass

    def build_system_layer(self) -> list:
        """
        Build Cirq gate list from coupling_lists using _GATE_MAP.
        Two-site gates are applied layer-by-layer via bond_colouring().
        Override for non-standard gate sets (e.g. Heisenberg gates).
        """
        cl = self._coupling_lists
        qubits = self._device.system_qubits
        gates = []

        for op_str, terms in cl.items():
            if len(op_str) == 2:  # two-site
                bond_strength = {(s, t): J for J, s, t in terms}
                for layer in self._lattice.bond_colouring():
                    for s, t in layer:
                        J = bond_strength.get((s, t), 0.)
                        if J != 0.:
                            gates.append(self._GATE_MAP[op_str](J)(qubits[s], qubits[t]))
            else:  # one-site
                for strength, s in terms:
                    if strength != 0.:
                        gates.append(self._GATE_MAP[op_str](strength)(qubits[s]))

        return gates

    def build_hamiltonian(self) -> tuple:
        """
        Build Cirq PauliSum and commuting components from coupling_lists.
        Returns (H0, components) where components is a dict keyed by operator string.
        """
        cl = self._coupling_lists
        qubits = self._device.system_qubits
        components = {}

        for op_str, terms in cl.items():
            if len(op_str) == 2:  # two-site
                components[op_str] = sum(
                    J * self._PAULI_MAP[op_str](qubits[s], qubits[t])
                    for J, s, t in terms
                )
            else:  # one-site
                components[op_str] = sum(
                    g * self._PAULI_MAP[op_str](qubits[s])
                    for g, s in terms
                )

        H0 = sum(components.values(), cirq.PauliSum())
        return H0, components

    def __getstate__(self):
        # cirq.PauliSum contains a LinearDict with an unpicklable local lambda.
        # Exclude hamiltonian; rebuild it on unpickling from coupling_lists.
        state = self.__dict__.copy()
        del state['hamiltonian']
        del state['hamiltonian_components']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.hamiltonian, self.hamiltonian_components = self.build_hamiltonian()

    def draw_model(self):
        C = cirq.Circuit(self.system_layer)
        print(C)
