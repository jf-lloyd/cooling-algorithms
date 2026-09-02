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

    def add_JW_majorana_correlators(self):
        """
        Add the real 2L-Majorana covariance matrix for the Jordan-Wigner
        fermionization of the qubit chain (system_qubits order fixes the JW
        site order), defining per-site Majoranas
            a_i = (prod_{l<i} Z_l) X_i,   b_i = (prod_{l<i} Z_l) Y_i
        and measuring M_{mu,nu} = i<gamma_mu gamma_nu> for every pair mu<nu
        in the global ordering (a_0,b_0,a_1,b_1,...). Each entry reduces to a
        single real Hermitian Pauli string (no complex bookkeeping needed):
            M^aa_ij =  <Y_i S_ij X_j>
            M^ab_ij =  <Y_i S_ij Y_j>
            M^ba_ij = -<X_i S_ij X_j>
            M^bb_ij = -<X_i S_ij Y_j>
            M^ab_ii = -<Z_i>            (M^aa_ii = M^bb_ii = 0, not stored)
        with S_ij = Z_{i+1}...Z_{j-1} (i<j). Validated against direct
        Majorana-matrix construction for L<=6 (exact to machine precision,
        see validate_jw_majorana.py).

        The standard fermion correlators <c_i^dag c_j>, <c_i c_j> are
        recovered losslessly at analysis time via
            <c_i^dag c_j> = ((Mab_ij - Mba_ij) - i(Maa_ij + Mbb_ij)) / 4
            <c_i c_j>     = ((Mab_ij + Mba_ij) + i(Mbb_ij - Maa_ij)) / 4
        """
        q = self.device.system_qubits
        Ns = len(q)
        for i in range(Ns):
            self.add_observable(f'Mab_{i}_{i}', -cirq.Z(q[i]))
            for j in range(i + 1, Ns):
                s = cirq.PauliString({q[l]: cirq.Z for l in range(i + 1, j)})
                self.add_observable(f'Maa_{i}_{j}',  cirq.Y(q[i]) * s * cirq.X(q[j]))
                self.add_observable(f'Mab_{i}_{j}',  cirq.Y(q[i]) * s * cirq.Y(q[j]))
                self.add_observable(f'Mba_{i}_{j}', -cirq.X(q[i]) * s * cirq.X(q[j]))
                self.add_observable(f'Mbb_{i}_{j}', -cirq.X(q[i]) * s * cirq.Y(q[j]))
        return self

    def add_spinspin_correlators(self):
        """
        Add all-pairs two-site correlators <X_iX_j>, <Y_iY_j>, <Z_iZ_j> 
        """
        q = self.device.system_qubits
        Ns = len(q)
        for i in range(Ns):
            for j in range(i + 1, Ns):
                self.add_observable(f'XX_{i}_{j}', cirq.X(q[i]) * cirq.X(q[j]))
                self.add_observable(f'YY_{i}_{j}', cirq.Y(q[i]) * cirq.Y(q[j]))
                self.add_observable(f'ZZ_{i}_{j}', cirq.Z(q[i]) * cirq.Z(q[j]))
        return self


class DefaultMeasurement1(Measurement):

    """
    simple measurement containing total spin, (zero-order trotter) Hamiltonian,
    and <H^2> (for the energy variance Var(H) = <H^2> - <H>^2).
    """
    def __init__(self, device:"Device", model:"Model"):
        super().__init__(device)
        self.add_Hamiltonian(model)
        self.add_total_spin()
        self._H_hsq = model.hamiltonian

    def measure_from_state_vector(self, state):
        measurement = super().measure_from_state_vector(state)
        psi = np.asarray(state).ravel().astype(np.complex128)
        phi = self._apply_pauli_sum(self._H_hsq, psi)   # H|psi>
        measurement['Hsq'] = float(np.sum(np.abs(phi) ** 2))   # <psi|H^2|psi> = ||H|psi>||^2
        return measurement


class DefaultMeasurement2(DefaultMeasurement1):

    """
    as DefaultMeasurement1, plus local single-site <X_k>, <Y_k>, <Z_k> for every
    system qubit k, and spin-spin correlators <X_kX_j>, <Y_kY_j>, <Z_kZ_j>.
    """
    def __init__(self, device:"Device", model:"Model"):
        super().__init__(device, model)
        self.add_local_Sops()
        self.add_spinspin_correlators()



class DefaultMeasurement4(DefaultMeasurement1):

    """
    as DefaultMeasurement1 (total spin, H0, Hsq), plus the full real
    2L-Majorana JW covariance matrix (see add_JW_majorana_correlators),
    for reconstructing free-fermion quasiparticle occupations <n_k> in
    models with a known JW mapping (e.g. TFIM).
    """
    def __init__(self, device:"Device", model:"Model"):
        super().__init__(device, model)
        self.add_JW_majorana_correlators()


class DefaultMeasurement3(DefaultMeasurement1):

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

    def measure_from_state_vector(self, state):
        measurement = super().measure_from_state_vector(state)
        psi = np.asarray(state).ravel().astype(np.complex128)
        phi = self._apply_pauli_sum(self._H_mres, psi)   # H|psi>, once for all sectors
        w   = np.conjugate(psi) * phi                    # per-basis-state energy density
        for M, mask in self._sectors:
            measurement[f"p_M{M:+.1f}"]  = float(np.sum(np.abs(psi[mask]) ** 2))
            measurement[f"pE_M{M:+.1f}"] = float(np.sum(w[mask]).real)
        return measurement

