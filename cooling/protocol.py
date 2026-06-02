from abc import ABC, abstractmethod  
import numpy as np
import cirq
import scipy.linalg

class Protocol(ABC):
    """
    Abstract cooling protocol.
    """

    def __init__(self, device:"Device", model:"Model"):
        self.device = device
        self.lattice = device.lattice
        self.model = model

        self._couplings = None
        self._coupling_operator = None

        self._reset_layer = []
        for i in range(self.device.Ns):
            self._reset_layer.append(cirq.I(self.device.system_qubits[i]))
        for mu in range(self.device.Nb): 
            self._reset_layer.append(cirq.ResetChannel(2)(self.device.bath_qubits[mu]))

    @property
    def reset_layer(self):
        return self._reset_layer

    def set_couplings(self, couplings):
        """ 
            set system-bath couplings via user-defined list 
            couplings[i] returns system qubit index coupled to i-th bath qubit
        """
        assert len(couplings) == self.device.Nb
        assert np.all(np.array(couplings) < self.device.Ns)
        self._couplings = couplings

    def set_random_couplings(self):
        couplings = [np.random.randint(0, self.device.Ns) for i in range(self.device.Nb)]
        self._couplings = couplings

    def set_coupling_operator(self, sys_op=None):
        X = np.array([[0,1],[1,0]], dtype=complex)
        Y = np.array([[0,-1j],[1j,0]], dtype=complex)
        Z = np.array([[1,0],[0,-1]], dtype=complex)
        if sys_op is None: # default Os = X
            sys_op = Y
        op = np.kron(sys_op, Y)
        self._coupling_operator = op

    def set_coupling_iSWAP(self):
        X = np.array([[0,1],[1,0]], dtype=complex)
        Y = np.array([[0,-1j],[1j,0]], dtype=complex)
        Z = np.array([[1,0],[0,-1]], dtype=complex)

        op = (np.kron(Y, Y)+np.kron(X, X))/2
        self._coupling_operator = op

    def random_pauli_coupling_operator(self):
        X = np.array([[0,1],[1,0]], dtype=complex)
        Y = np.array([[0,-1j],[1j,0]], dtype=complex)
        Z = np.array([[1,0],[0,-1]], dtype=complex)
        r = np.random.randint(2)
        sys_op = [X, Y][r]
        op = np.kron(sys_op, Y)
        self._coupling_operator = op
        
    @abstractmethod
    def cycle_circuit(self, t=1, **controls):
        '''
            Return cirq circuit for t-th channel application. Note t is a "dummy" variable for homogeneous protocol.
            Each cycle consists of alternating system gates, bath gates, and system-bath gates. 
            Finally a reset layer is applied. 
            The simulation will build circuit as [cycle_circuit[t=1], cycle_circuit[t=2], ...]
        '''
        pass



class SimpleThermalProtocol(Protocol):

    """
    'Simple' thermal cooling protocol : 
    defines a single repeated cooling cycle with fixed temperature and coupling parameters, as defined in arxiv:2506.21318 
    each cycle has depth ~ sqrt(beta) and uses time-dependenent filter function to fix temperature of final state
    """

    def __init__(self, device:"Device", model:"Model", params:dict, frozen_circuit:bool=False):
        """
        params :
            'beta' : target temperature
            'theta' : system-bath coupling strength
            'h' : bath splitting
            'delta' : trotter angle applied to all gates
            'NT': reset time MT = int(NT/a) [a is frequency-space Gaussian width]
            'randomize_couplings': if SB couplings are randomized at start of each cycle
            'randomize_Pauli_ops': if SB coupling operators are random choice of [X,Y,Z] at start of each cycle
            'randomization_time': if R.T.>0, extra system evolution for random duration ~R.T., measured in units of cycle depth
        """
        super().__init__(device=device, model=model)        
        self._beta = params.get("beta", 1.)
        self._theta = params.get("theta", 0.5)
        self._h = params.get("h", 1.)
        self._delta = params.get("delta", 0.2*np.pi/2)
        self._NT = params.get("NT", 5)
        self._randomize_couplings = params.get("randomize_couplings", True)
        self._random_Pauli_ops = params.get("random_Pauli_ops", True)
        self._randomization_time = params.get("randomization_time", 0)
        self._couplings = None 
        self._params = params
        self._frozen_circuit = frozen_circuit
        self._cycle = None

        self.gamma = params.get("gamma", 0.)

        self._a = self._delta*np.sqrt(abs(4*self._h/self._beta))
        self._MT = max(self._NT, int(self._NT/self._a))

        self._filter_f = self.get_filter_f()
        # self.set_coupling_operator()
        self.set_random_couplings()
        self._bath_layer = self.get_bath_layer()
        self._noise_layer = self.get_noise_layer()

    @property
    def name(self):
        my_name = f"Thermal_Nb{self.device.Nb}b{self._beta:.3f}th{self._theta:.3f}h{self._h:.3f}d{self._delta:.3f}"
        return my_name
    
    @property
    def params(self): return self._params

    @property
    def a(self): return self._a

    @property
    def MT(self): return self._MT

    @property
    def T(self): return self._MT*self._delta

    def get_filter_f(self):
        ''' thermal filter function'''
        
        delta = self._delta
        a = self.a
        MT = self.MT
        f = np.array([np.exp(-a**2*t**2/2) for t in np.arange(-MT, MT+1)])
        f /= delta*np.sum(np.abs(f)) # normalisation delta Sum_i |f_i| = 1
        
        return f

    def get_bath_layer(self):

        # --- single-qubit Z field gates exp(-ihZ) ---
        # cirq: rz(h) = exp(-i h/2 Z)
        h = self._h
        bath_gate = cirq.rz(-h) 
        auxs = self.device.bath_qubits
        return [bath_gate(a) for a in auxs]
        
    def get_coupling_layer(self, j):

        couplings = self._couplings

        # --- two-qubit OY couplings exp(-itheta OY)---
            
        op = self._coupling_operator
        theta = self._theta
        filter_f = self._filter_f
        U = scipy.linalg.expm(-1j*theta*filter_f[j]*op)
        coupling_gate = cirq.MatrixGate(U)

        auxs = self.device.bath_qubits
        qubits = self.device.system_qubits
        
        coupling_layer = []
        for i in range(self.device.Nb):
            coupling_layer.append( coupling_gate(qubits[couplings[i]], auxs[i]) )

        return coupling_layer

    def get_noise_layer(self):
        qubits = self.device.system_qubits
        gamma = self.gamma 
        return [cirq.depolarize(p=gamma).on(q) for q in qubits]

    
    def cycle_circuit(self, t=1):
        '''
            build cirq circuit for t-th channel application
        '''
        if self._frozen_circuit:
            if self._cycle is not None:
                return self._cycle

        # set system-bath coupling at start of cycle
        if self._randomize_couplings is False:
            if self._couplings is None:
                raise ValueError("Please set system-bath couplings or set randomize_couplings=True")
        else:
            self.set_random_couplings() 
            couplings = self._couplings

        if self._random_Pauli_ops is True:
            self.random_pauli_coupling_operator()

        if self._randomization_time > 0.:
            MTrand = int(np.random.exponential((2*self.MT+1)*self._randomization_time))

        system_layer = self.model.system_layer
        bath_layer = self._bath_layer
        reset_layer = self._reset_layer
        if self.gamma != 0.:
            noise_layer = self._noise_layer

        delta = self._delta
        cycle = cirq.Circuit()
        
        for j in range(2*self.MT+1):
            cycle.append(u**delta for u in system_layer)
            cycle.append(u**delta for u in bath_layer)
            cycle.append(u**delta for u in self.get_coupling_layer(j))
            if self.gamma != 0.:
                cycle.append(d for d in noise_layer)

        # extra system evolution for randomized time
        if self._randomization_time > 0.:
            for d in range(MTrand):
                cycle.append(u**delta for u in system_layer)
                if self.gamma != 0.:
                    cycle.append(d for d in noise_layer)

        # reset
        cycle.append(reset_layer)
        if self._frozen_circuit:  
            self._cycle = cirq.FrozenCircuit(cycle)

        return cycle
        
class SimpleGroundProtocol(Protocol):

    """
    'Simple' ground state cooling protocol, using the MPC of the QP cooling paper
    """

    def __init__(self, device:"Device", model:"Model", params:dict, frozen_circuit:bool=False):
        """
        params :
            'beta' : target temperature (acts as thermal broadening of step)
            'theta' : system-bath coupling strength
            'delta' : trotter angle applied to all gates
            'NT': reset time MT = int(NT/a) [a is frequency-space Gaussian width]
            'randomize_couplings': if SB couplings are randomized at start of each cycle
            'randomize_Pauli_ops': if SB coupling operators are random choice of [X,Y,Z] at start of each cycle
            'randomization_time': if R.T.>0, extra system evolution for random duration ~R.T., measured in units of cycle depth
        """
        super().__init__(device=device, model=model)        
        self._beta = params.get("beta", 1.)
        self._theta = params.get("theta", 0.5)
        self._delta = params.get("delta", 0.2*np.pi/2)
        self._h =  np.pi/2 ## no delta applied 
        self._NT = params.get("NT", 2)
        self._randomize_couplings = params.get("randomize_couplings", True)
        self._random_Pauli_ops = params.get("random_Pauli_ops", True)
        self._randomization_time = params.get("randomization_time", 0)
        self._couplings = None 
        self._params = params
        self._frozen_circuit = frozen_circuit
        self._cycle = None

        self.gamma = params.get("gamma", 0.)

        self._MT = max(10, int(self._NT*self._beta/self._delta))

        self._filter_f = self.get_filter_f()
        # self.set_coupling_operator()
        self.set_random_couplings()
        self._bath_layer = self.get_bath_layer()
        self._noise_layer = self.get_noise_layer()

    @property
    def name(self):
        my_name = f"Ground_Nb{self.device.Nb}b{self._beta:.3f}th{self._theta:.3f}h{self._h:.3f}d{self._delta:.3f}"
        return my_name
    
    @property
    def params(self): return self._params


    @property
    def MT(self): return self._MT

    @property
    def T(self): return self._MT*self._delta

    def get_filter_f(self):
        ''' thermal filter function'''
        
        delta = self._delta
        h = self._h
        beta = self._beta
        MT = self.MT

        f = []
        for t in np.arange(-MT, MT+1):
            if t == 0:
                f.append(1/2)
            else:
                f.append(np.sin(np.pi*t/2)/np.sinh(delta*np.pi*t/beta)*delta/beta)
        f /= delta*np.sum(np.abs(f)) # normalisation delta Sum_i |f_i| = 1
        
        return f

    def get_bath_layer(self):

        # --- single-qubit Z field gates exp(-ihZ) ---
        # cirq: rz(h) = exp(-i h/2 Z)
        h = self._h
        bath_gate = cirq.rz(-h) 
        auxs = self.device.bath_qubits
        return [bath_gate(a) for a in auxs]
        
    def get_coupling_layer(self, j):

        couplings = self._couplings

        # --- two-qubit OY couplings exp(-itheta OY)---
            
        op = self._coupling_operator
        theta = self._theta
        filter_f = self._filter_f
        U = scipy.linalg.expm(-1j*theta*filter_f[j]*op)
        coupling_gate = cirq.MatrixGate(U)

        auxs = self.device.bath_qubits
        qubits = self.device.system_qubits
        
        coupling_layer = []
        for i in range(self.device.Nb):
            coupling_layer.append( coupling_gate(qubits[couplings[i]], auxs[i]) )

        return coupling_layer

    def get_noise_layer(self):
        qubits = self.device.system_qubits
        gamma = self.gamma 
        return [cirq.depolarize(p=gamma).on(q) for q in qubits]

    
    def cycle_circuit(self, t=1):
        '''
            build cirq circuit for t-th channel application
        '''
        if self._frozen_circuit:
            if self._cycle is not None:
                return self._cycle

        # set system-bath coupling at start of cycle
        if self._randomize_couplings is False:
            if self._couplings is None:
                raise ValueError("Please set system-bath couplings or set randomize_couplings=True")
        else:
            self.set_random_couplings() 
            couplings = self._couplings

        if self._random_Pauli_ops is True:
            self.random_pauli_coupling_operator()

        if self._randomization_time > 0.:
            MTrand = int(np.random.exponential((2*self.MT+1)*self._randomization_time))

        system_layer = self.model.system_layer
        bath_layer = self._bath_layer
        reset_layer = self._reset_layer
        if self.gamma != 0.:
            noise_layer = self._noise_layer

        delta = self._delta
        cycle = cirq.Circuit()
        
        for j in range(2*self.MT+1):
            cycle.append(u**delta for u in system_layer)
            cycle.append(u for u in bath_layer)
            cycle.append(u**delta for u in self.get_coupling_layer(j))
            if self.gamma != 0.:
                cycle.append(d for d in noise_layer)

        # extra system evolution for randomized time
        if self._randomization_time > 0.:
            for d in range(MTrand):
                cycle.append(u**delta for u in system_layer)
                if self.gamma != 0.:
                    cycle.append(d for d in noise_layer)

        # reset
        cycle.append(reset_layer)
        if self._frozen_circuit:  
            self._cycle = cirq.FrozenCircuit(cycle)

        return cycle
