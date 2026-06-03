"""
Original Ising model (pre-refactor) -- kept for comparison only.
"""

import cirq
import numpy as np
from .modelbase import Model

class IsingModel(Model):
    """
    Transverse-field Ising model (with optional longitudinal X field),
    defined on an arbitrary Lattice and Device.

        H_J  = J * sum_{<i,j>} X_i X_j
        H_g  = -g * sum_i Z_i
        H_gx = -gx * sum_i X_i

        H0 = H_J + H_g + H_gx

    """

    def __init__(self, device, params:dict):
        super(IsingModel, self).__init__(device)

        self._Ns = self.lattice.Ns
        self.system_qubits = device.system_qubits
        self.params = params

        self._system_layer = self.get_system_layer()
        self.hamiltonian, self.hamiltonian_components, self.local_Hops = self.build_hamiltonian()

    @property
    def name(self):
        dim = self.lattice.Dim
        if dim == 1:
            Lx = self.lattice.L
            Ly = 1
            pbc = self.lattice.pbc
        elif dim == 2:
            Lx = self.lattice.Lx
            Ly = self.lattice.Ly
            pbc_x = self.lattice.pbc_x
            pbc_y = self.lattice.pbc_y
        J = self.params["J"]
        g = self.params["g"]
        gx = self.params["gx"]
        my_name = f"IsingModel{dim}D_Lx{Lx}Ly{Ly}J{J:.3f}g{g:.3f}gx{gx:.3f}"
        if dim == 1:
            if pbc:
                my_name += "_pbc"
        elif dim == 2:
            if pbc_x:
                my_name += "_pbcx"
            if pbc_y:
                my_name += "_pbcy"
        return my_name

    @property
    def Ns(self):
        return self._Ns

    def build_hamiltonian(self):
        J = self.params.get('J', 0.0)
        g = self.params.get('g', 0.0)
        gx = self.params.get('gx', 0.0)
        lattice = self.lattice
        qubits = self.system_qubits

        local_Hops = []
        for (s, t) in lattice.nearest_neighbour_pairs():
            q1 = qubits[s]
            q2 = qubits[t]
            H_local = 0.
            if J != 0.0:
                H_local += J * (cirq.X(q1) * cirq.X(q2))
            if g != 0.0:
                H_local += -g/2 * cirq.Z(q1)
                H_local += -g/2 * cirq.Z(q2)
            if gx != 0.0:
                H_local += -gx/2 * cirq.X(q1)
                H_local += -gx/2 * cirq.X(q2)
            local_Hops.append(H_local)

        HJ = 0
        if J != 0.0:
            for (s, t) in lattice.nearest_neighbour_pairs():
                HJ += J * (cirq.X(qubits[s]) * cirq.X(qubits[t]))
        Hg = 0
        if g != 0.0:
            for i in range(self._Ns):
                Hg += -g * cirq.Z(qubits[i])
        Hgx = 0
        if gx != 0.0:
            for i in range(self._Ns):
                Hgx += -gx * cirq.X(qubits[i])

        H0 = HJ + Hg + Hgx
        return H0, [HJ, Hg, Hgx], local_Hops

    def build_floquet_hamiltonians(self, order=1):
        h1, h2, h3 = self.hamiltonian_components
        cm = lambda A, B: A*B - B*A
        H1 = 1/2j * (cm(h3, h2) + cm(h3, h1) + cm(h2, h1))
        self.floquet_hamiltonian1 = H1
        if order > 1:
            print("not implemented order>1 yet!")

    def get_system_layer(self):
        J = self.params.get('J', 0.0)
        g = self.params.get('g', 0.0)
        gx = self.params.get('gx', 0.0)
        qubits = self.system_qubits
        lattice = self.lattice
        dim = self.lattice.Dim
        bonds = lattice.nearest_neighbour_pairs()

        if not bonds:
            layers = [[]]
        elif dim == 1:
            Lx = lattice.L
            pbc = lattice.pbc
            even, odd = [], []
            for ix in range(0, Lx - (1 - pbc)):
                if ix % 2 == 0:
                    even.append((lattice.index(ix), lattice.index((ix+1) % Lx)))
                else:
                    odd.append((lattice.index(ix), lattice.index((ix+1) % Lx)))
            layers = [even, odd]
        elif dim == 2:
            Lx, Ly = lattice.Lx, lattice.Ly
            pbc_x, pbc_y = lattice.pbc_x, lattice.pbc_y
            X_even, X_odd, Y_even, Y_odd = [], [], [], []
            for ix in range(0, Lx - (1 - pbc_x)):
                for iy in range(Ly):
                    if ix % 2 == 0:
                        X_even.append((lattice.index(ix, iy), lattice.index((ix+1) % Lx, iy)))
                    else:
                        X_odd.append((lattice.index(ix, iy), lattice.index((ix+1) % Lx, iy)))
            for iy in range(0, Ly - (1 - pbc_y)):
                for ix in range(Lx):
                    if iy % 2 == 0:
                        Y_even.append((lattice.index(ix, iy), lattice.index(ix, (iy+1) % Ly)))
                    else:
                        Y_odd.append((lattice.index(ix, iy), lattice.index(ix, (iy+1) % Ly)))
            layers = [X_even, X_odd, Y_even, Y_odd]
        else:
            layers = [bonds]

        system_gates = []
        if J != 0.0:
            XXgate = cirq.XXPowGate(exponent=J/(np.pi/2), global_shift=-0.5)
            for layer in layers:
                for (s, t) in layer:
                    system_gates.append(XXgate(qubits[s], qubits[t]))
        if g != 0.0:
            Zgate = cirq.rz(-g * 2)
            for i in range(self._Ns):
                system_gates.append(Zgate(qubits[i]))
        if gx != 0.0:
            Xgate = cirq.rx(-gx * 2)
            for i in range(self._Ns):
                system_gates.append(Xgate(qubits[i]))

        return system_gates
