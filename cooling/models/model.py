from abc import ABC, abstractmethod    
import cirq
import numpy as np

class Model(ABC):
    """
    Quantum model defined on a Device. The 'model' consists of list of gates applied in one evolution step. 
    An effective Hamiltonian is defined from the gate list.
    """

    def __init__(self, device: "Device"):
        self._lattice = device.lattice
        self._device = device
        self._system_layer = None

    @property
    def system_layer(self):
        return self._system_layer

    @property
    def lattice(self):
        return self._lattice
    
    @abstractmethod
    def get_system_layer(self):
        '''
        Returns a list of Cirq gates, representing full update step.
        The gates correspond to the Hamiltonian as exp(-iH): when called in protocol, 
        we apply for set trotter angle as exp(-iH)**delta
        '''
        pass

    def draw_model(self):
        '''
        Draw circuit defined by gates in self._gate_list.
        '''
        C = cirq.Circuit(gate for gate in self.system_layer)
        print(C)
        return

        
    @abstractmethod
    def build_hamiltonian(self):
        """
        Returns (zero order Floquet) Hamiltonian as Cirq operator, 
        and non-commuting pieces H1, H2, ... for constructing Floquet Hamiltonians
        """
        pass

    @abstractmethod
    def build_floquet_hamiltonians(self, order=2):
        """
        Builds higher order Floquet Hamiltonians up to given order as Cirq operators.
        """
        pass


class IsingModel(Model):
    """
    Transverse-field Ising model (with optional longitudinal X field),
    defined on an arbitrary Lattice and Device.

        H_J  = J * sum_{<i,j>} X_i X_j
        H_g  = -g * sum_i Z_i
        H_gx = -gx * sum_i X_i

        H0 = H_J + H_g + H_gx

    """

    def __init__(self, device:"Device", params:dict):
        """
        params :
            'J'   : Ising coupling (float)
            'g'   : transverse Z field (float)
            'gx'  : longitudinal X field (float, default 0.)
        """
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
                my_name+="_pbc"
        elif dim == 2:
            if pbc_x:
                my_name+="_pbcx"
            if pbc_y:
                my_name+="_pbcy"
        return my_name

    @property
    def Ns(self):
        return self._Ns

    def build_hamiltonian(self):
        """
        Return the base Hamiltonian H0 = H_J + H_g + H_gx, and components
        """
        J = self.params.get('J', 0.0)
        g = self.params.get('g', 0.0)
        gx = self.params.get('gx', 0.0)

        lattice = self.lattice
        qubits = self.system_qubits

        local_Hops = [] # doesn't capture boundary energy for obc
        boundary = lattice.boundary()
        for (s, t) in lattice.nearest_neighbour_pairs():
            q1 = qubits[s]
            q2 = qubits[t]
            H_local = 0.
            if J != 0.0:
                H_local += J * (cirq.X(q1) * cirq.X(q2))
            if g != 0.0:
                H_local += -g/2 * cirq.Z(q1)
                H_local += -g/2 * cirq.Z(q2)
                # if s in boundary: # full field if site on bry
                #     H_local += -g/2 * cirq.Z(q1)
                # if t in boundary:
                #     H_local += -g/2 * cirq.Z(q2)
            if gx != 0.0:
                H_local += -gx/2 * cirq.X(q1)
                H_local += -gx/2 * cirq.X(q2)
                # if s in boundary: # full field if site on bry
                #     H_local += -gx/2 * cirq.X(q1)
                # if t in boundary:
                #     H_local += -gx/2 * cirq.X(q2)
            local_Hops.append(H_local)
        
        # Two-site XX term
        HJ = 0
        if J != 0.0:
            for (s, t) in lattice.nearest_neighbour_pairs():
                q1 = qubits[s]
                q2 = qubits[t]
                H_bond = J * (cirq.X(q1) * cirq.X(q2))
                HJ += H_bond

        # Transverse field
        Hg = 0
        if g != 0.0:
            for i in range(self._Ns):
                Hg += -g * cirq.Z(qubits[i])

        # X field
        Hgx = 0
        if gx != 0.0:
            for i in range(self._Ns):
                Hgx += -gx * cirq.X(qubits[i])

        H0 = HJ + Hg + Hgx
        return H0, [HJ, Hg, Hgx], local_Hops

    def build_floquet_hamiltonians(self, order=1):
        h1, h2, h3 = self.hamiltonian_components 
        cm = lambda A, B : A*B-B*A
        H1 = 1/2j*(cm(H3,H2)+cm(H3,H1)+cm(H2,H1))
        self.floquet_hamiltonian1 = H1
        if order>1:
            print("not implemented order>1 yet!")
        return 

    def get_system_layer(self):
        """
        Return a list of gates applied in one full update.
        We apply the gates in the order: U(gx)U(g)U(J)|psi> 
        For two-qubit gates the following ordering is applied:
        In 2D: 1) X-even 2) X-odd 3) Y-even 4) Y-odd
        In 1D: 1) even 2) odd
        The gates correspond to the Hamiltonian as exp(-iH): when called in protocol, 
        we apply for set trotter angle as exp(-iH)**delta
        
        """
        
        J = self.params.get('J', 0.0)
        g = self.params.get('g', 0.0)
        gx = self.params.get('gx', 0.0)

        qubits = self.system_qubits
        lattice = self.lattice
        dim = self.lattice.Dim
        bonds = lattice.nearest_neighbour_pairs()

        # first build list of 2-qubit gate layers

        # No bonds: trivial
        if not bonds:
            layers = [[]]

        # 1D chain: even/odd bond partition by left site index

        elif dim == 1:

            Lx = lattice.L
            pbc = lattice.pbc
            even = []
            odd = []
    
            for ix in range(0, Lx- (1-pbc), 1):
                if ix %2 == 0:
                    even.append((lattice.index(ix), lattice.index((ix+1)%Lx)))
                else:
                    odd.append((lattice.index(ix), lattice.index((ix+1)%Lx)))
            layers = [even, odd]
            
        # 2D rectangular: X-even, X-odd, Y-even, Y-odd
        elif dim == 2:

            Lx = lattice.Lx
            Ly = lattice.Ly
            pbc_x = lattice.pbc_x
            pbc_y = lattice.pbc_y
            X_even, X_odd, Y_even, Y_odd = [], [], [], []

            for ix in range(0, Lx- (1-pbc_x), 1):
                for iy in range(0, Ly, 1):
                    if ix %2 == 0:
                        X_even.append((lattice.index(ix, iy), lattice.index((ix+1)%Lx, iy)))
                    else:
                        X_odd.append((lattice.index(ix, iy), lattice.index((ix+1)%Lx, iy)))

            for iy in range(0, Ly- (1-pbc_y), 1):
                for ix in range(0, Lx, 1):
                    if iy %2 == 0:
                        Y_even.append((lattice.index(ix, iy), lattice.index(ix, (iy+1)%Ly)))
                    else:
                        Y_odd.append((lattice.index(ix, iy), lattice.index(ix, (iy+1)%Ly)))

            layers = [X_even, X_odd, Y_even, Y_odd]

        # all other cases return bond list
        else:
            layers = [bonds]
            
        system_gates = []

        # --- two-qubit XX couplings exp(-iJXX) ---
        # cirq: XXPowGate(theta) = exp(-i π theta/2 XX)
        if J != 0.0:
            XXgate = cirq.XXPowGate(exponent=J/(np.pi/2), global_shift=-0.5)
            for layer in layers:
                for (s,t) in layer:
                    q1 = qubits[s]
                    q2 = qubits[t]
                    system_gates.append(XXgate(q1, q2))

        # --- single-qubit Z field gates exp(-igZ) ---
        # cirq: rz(theta) = exp(-i theta/2 Z)
        if g != 0.0:
            Zgate = cirq.rz(-g * 2)
            for i in range(self._Ns):
                system_gates.append(Zgate(qubits[i]))

        # --- single-qubit X field gates exp(-igxX) (optional) ---
        # cirq: rx(theta) = exp(-i theta/2 X)
        if gx != 0.0:
            Xgate = cirq.rx(-gx * 2)
            for i in range(self._Ns):
                system_gates.append(Xgate(qubits[i]))

        return system_gates


class HeisModel(Model):
    """
    anisotropic Heisenberg model defined on an arbitrary Lattice and Device.

        H_x  = Jx * sum_{<i,j>} X_i X_j
        H_y  = Jy * sum_{<i,j>} Y_i Y_j
        H_z  = Jz * sum_{<i,j>} Z_i Z_j

        H0 = H_x + H_y + H_z

    """

    def __init__(self, device:"Device", params:dict):
        """
        params :
            'Jx'   : XX coupling (float)
            'Jy'   : YY coupling (float)
            'Jz'   : ZZ coupling (float)
        """
        super(HeisModel, self).__init__(device)

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
        Jx = self.params["Jx"]
        Jy = self.params["Jy"]
        Jz = self.params["Jz"]
        
        my_name = f"HeisModel{dim}D_Lx{Lx}Ly{Ly}Jx{Jx:.3f}Jy{Jy:.3f}Jz{Jz:.3f}"
        if dim == 1:
            if pbc:
                my_name+="_pbc"
        elif dim == 2:
            if pbc_x:
                my_name+="_pbcx"
            if pbc_y:
                my_name+="_pbcy"
        return my_name

    @property
    def Ns(self):
        return self._Ns

    def build_hamiltonian(self):
        """
        Return the base Hamiltonian H0 = H_J + H_g + H_gx, and components
        """
        Jx = self.params.get('Jx', 0.0)
        Jy = self.params.get('Jy', 0.0)
        Jz = self.params.get('Jz', 0.0)

        lattice = self.lattice
        qubits = self.system_qubits

        local_Hops = [] # doesn't capture boundary energy for obc
        H_x = 0.
        H_y = 0.
        H_z = 0.
        boundary = lattice.boundary()

        for (s, t) in lattice.nearest_neighbour_pairs():
            q1 = qubits[s]
            q2 = qubits[t]
            H_local = 0.
            if Jx != 0.0:
                H_local += Jx * (cirq.X(q1) * cirq.X(q2))
                H_x += Jx * (cirq.X(q1) * cirq.X(q2))
            if Jy != 0.0:
                H_local += Jy * (cirq.Y(q1) * cirq.Y(q2))
                H_y += Jy * (cirq.Y(q1) * cirq.Y(q2))
            if Jz != 0.0:
                H_local += Jz * (cirq.Z(q1) * cirq.Z(q2))
                H_z += Jz * (cirq.Z(q1) * cirq.Z(q2))
                  
            local_Hops.append(H_local)
        
        H0 = H_x + H_y + H_z
        return H0, [H_x, H_y, H_z], local_Hops

    def build_floquet_hamiltonians(self, order=1):
        h1, h2, h3 = self.hamiltonian_components 
        cm = lambda A, B : A*B-B*A
        H1 = 1/2j*(cm(H3,H2)+cm(H3,H1)+cm(H2,H1))
        self.floquet_hamiltonian1 = H1
        if order>1:
            print("not implemented order>1 yet!")
        return 

    def get_system_layer(self):
        """
        Return a list of gates applied in one full update.
        We apply the gates in the order: U(Jz)U(Jy)U(Jx)|psi> 
        For two-qubit gates the following ordering is applied:
        In 2D: 1) X-even 2) X-odd 3) Y-even 4) Y-odd
        In 1D: 1) even 2) odd
        The gates correspond to the Hamiltonian as exp(-iH): when called in protocol, 
        we apply for set trotter angle as exp(-iH)**delta
        
        """
        
        Jx = self.params.get('Jx', 0.0)
        Jy = self.params.get('Jy', 0.0)
        Jz = self.params.get('Jz', 0.0)

        qubits = self.system_qubits
        lattice = self.lattice
        dim = self.lattice.Dim
        bonds = lattice.nearest_neighbour_pairs()

        # first build list of 2-qubit gate layers

        # No bonds: trivial
        if not bonds:
            layers = [[]]

        # 1D chain: even/odd bond partition by left site index

        elif dim == 1:

            Lx = lattice.L
            pbc = lattice.pbc
            even = []
            odd = []
    
            for ix in range(0, Lx- (1-pbc), 1):
                if ix %2 == 0:
                    even.append((lattice.index(ix), lattice.index((ix+1)%Lx)))
                else:
                    odd.append((lattice.index(ix), lattice.index((ix+1)%Lx)))
            layers = [even, odd]
            
        # 2D rectangular: X-even, X-odd, Y-even, Y-odd
        elif dim == 2:

            Lx = lattice.Lx
            Ly = lattice.Ly
            pbc_x = lattice.pbc_x
            pbc_y = lattice.pbc_y
            X_even, X_odd, Y_even, Y_odd = [], [], [], []

            for ix in range(0, Lx- (1-pbc_x), 1):
                for iy in range(0, Ly, 1):
                    if ix %2 == 0:
                        X_even.append((lattice.index(ix, iy), lattice.index((ix+1)%Lx, iy)))
                    else:
                        X_odd.append((lattice.index(ix, iy), lattice.index((ix+1)%Lx, iy)))

            for iy in range(0, Ly- (1-pbc_y), 1):
                for ix in range(0, Lx, 1):
                    if iy %2 == 0:
                        Y_even.append((lattice.index(ix, iy), lattice.index(ix, (iy+1)%Ly)))
                    else:
                        Y_odd.append((lattice.index(ix, iy), lattice.index(ix, (iy+1)%Ly)))

            layers = [X_even, X_odd, Y_even, Y_odd]

        # all other cases return bond list
        else:
            layers = [bonds]
            
        system_gates = []

        # --- two-qubit XX couplings exp(-iJxXX) ---
        # cirq: XXPowGate(theta) = exp(-i π theta/2 XX)
        if Jx != 0.0:
            XXgate = cirq.XXPowGate(exponent=Jx/(np.pi/2), global_shift=-0.5)
            for layer in layers:
                for (s,t) in layer:
                    q1 = qubits[s]
                    q2 = qubits[t]
                    system_gates.append(XXgate(q1, q2))

        # --- two-qubit YY couplings exp(-iJyYY) ---
        # cirq: YYPowGate(theta) = exp(-i π theta/2 YY)
        if Jy != 0.0:
            YYgate = cirq.YYPowGate(exponent=Jy/(np.pi/2), global_shift=-0.5)
            for layer in layers:
                for (s,t) in layer:
                    q1 = qubits[s]
                    q2 = qubits[t]
                    system_gates.append(YYgate(q1, q2))

        # --- two-qubit ZZ couplings exp(-iJzZZ) ---        
        # cirq: ZZPowGate(theta) = exp(-i π theta/2 ZZ)
        if Jz != 0.0:
            ZZgate = cirq.ZZPowGate(exponent=Jz/(np.pi/2), global_shift=-0.5)
            for layer in layers:
                for (s,t) in layer:
                    q1 = qubits[s]
                    q2 = qubits[t]
                    system_gates.append(ZZgate(q1, q2))
                    
        return system_gates

