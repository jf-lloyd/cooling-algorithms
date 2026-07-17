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
            val = operator.expectation_from_state_vector(state, qubit_map = self.device.qubit_index_map, check_preconditions=False)
            if herm:
                if abs(val.imag) > 1e-5:
                    raise ValueError(f"Observable '{name}' returned non-negligible imaginary part {val.imag:.2e}; check operator is Hermitian.")
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


class DefaultMeasurement2(DefaultMeasurement1):

    """
    as DefaultMeasurement1, plus the magnetisation-resolved energy decomposition.
    For every total-Sz sector M it records
        f"p_M{M:+.1f}"   ->  p_M  = <P_M>          weight of the state in sector M
        f"pE_M{M:+.1f}"  ->  pE_M = <P_M H P_M>    energy carried by sector M
    which decomposes the total energy exactly:  H0 = sum_M pE_M.

    Both are recorded UNNORMALISED, and deliberately so: p_M and pE_M are linear
    in the state, so they average correctly over trajectories and over time, and
    are never NaN. Recover the energy within a sector at analysis time as

        E_M = <pE_M> / <p_M>

    which is automatically weighted by how much of each trajectory actually sits
    in the sector. Do NOT average a per-shot E_M = pE_M/p_M directly: it is a
    ratio, so trajectories with p_M ~ 0 contribute wild values with equal weight,
    and sum_M <p_M><E_M> != <H0> (it misses the p_M-E_M covariance).

    P_M = sum_{b : Sz(b) = M} |b><b| is diagonal in the computational basis
    (cirq: |0> -> Sz=+1/2, |1> -> Sz=-1/2), so it is applied as a mask.
    Requires [H, Sz_tot] = 0 (true for XY/XXZ-type models).

    M = (Ns - 2k)/2 for k = 0..Ns down spins: integer M for even Ns,
    half-integer M for odd Ns.

    Costs one application of H per measurement, not one per sector: since
    [H, Sz_tot] = 0, P_M H P_M = H P_M, so a single phi = H|psi> is sliced by
    each sector mask.
    """
    def __init__(self, device:"Device", model:"Model"):
        super().__init__(device, model)
        Ns = device.Ns
        pc = np.array([bin(i).count("1") for i in range(2 ** Ns)])
        self._H_mres  = model.hamiltonian
        self._sectors = [((Ns - 2 * k) / 2, pc == k) for k in range(Ns + 1)]

    def _apply_pauli_sum(self, operator, psi):
        """phi = O|psi> for a cirq.PauliSum, applied term-by-term (no dense matrix)."""
        Ns  = self.device.Ns
        idx = self.device.qubit_index_map
        psi_t = np.asarray(psi).reshape((2,) * Ns).astype(np.complex128)
        phi   = np.zeros_like(psi_t)
        for ps in operator:
            c = complex(ps.coefficient)
            if not ps.qubits:                      # identity term
                phi += c * psi_t
                continue
            out = cirq.apply_unitary(
                ps / c,                            # unit coefficient -> unitary
                cirq.ApplyUnitaryArgs(
                    target_tensor=psi_t.copy(),
                    available_buffer=np.zeros_like(psi_t),
                    axes=tuple(idx[q] for q in ps.qubits),
                ),
            )
            phi += c * out
        return phi.reshape(-1)

    def measure_from_state_vector(self, state):
        measurement = super().measure_from_state_vector(state)
        psi = np.asarray(state).ravel().astype(np.complex128)
        phi = self._apply_pauli_sum(self._H_mres, psi)   # H|psi>, once for all sectors
        w   = np.conjugate(psi) * phi                    # per-basis-state energy density
        for M, mask in self._sectors:
            measurement[f"p_M{M:+.1f}"]  = float(np.sum(np.abs(psi[mask]) ** 2))
            measurement[f"pE_M{M:+.1f}"] = float(np.sum(w[mask]).real)
        return measurement

