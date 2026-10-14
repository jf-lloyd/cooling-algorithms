import itertools
import numpy as np
import scipy.linalg as sl
import scipy.sparse as sp
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



# ── Commutator diagnostics:  <i[K, O]>  for K = H, H_F^(1), H_F (exact) ─────────

class _PauliTable:
    """
    Fast evaluation of many Pauli-string expectations on one state vector.

    A Pauli string P on n qubits (cirq big-endian: qubit i <-> bit n-1-i of the
    basis index) acts on a basis state as
        P|b> = i^{n_Y} (-1)^{popcount(b & zmask)} |b XOR xmask>,
    with xmask the X/Y sites and zmask the Z/Y sites. So
        <phi|P|psi> = i^{n_Y} sum_b conj(phi[b ^ xmask]) (-1)^{popcount(b & zmask)} psi[b],
    a gather plus a sign, vectorised over the 2^n basis states.
    Each distinct string is stored once; observables are linear combinations,
    combined with a dense (n_obs x n_strings) coefficient matrix.
    """
    def __init__(self, qubits):
        self.qubits = list(qubits); self.n = len(self.qubits)
        self.pos = {q: i for i, q in enumerate(self.qubits)}
        b = np.arange(2 ** self.n)
        self._bits = ((b[:, None] >> (self.n - 1 - np.arange(self.n))[None, :]) & 1).astype(np.int8)
        self.keys, self._index, self.xidx, self.zsgn, self.ph = [], {}, [], [], []

    def key_of(self, ps: cirq.PauliString):
        return tuple(sorted((self.pos[q], str(p)) for q, p in ps.items()))

    def add_string(self, ps: cirq.PauliString) -> int:
        k = self.key_of(ps)
        if k in self._index:
            return self._index[k]
        xm, zsites, ny = 0, [], 0
        for i, p in k:
            bit = 1 << (self.n - 1 - i)
            if p in ("X", "Y"): xm |= bit
            if p in ("Z", "Y"): zsites.append(i)
            if p == "Y": ny += 1
        sgn = 1 - 2 * (self._bits[:, zsites].sum(1) & 1) if zsites else np.ones(2 ** self.n, dtype=np.int8)
        self._index[k] = len(self.keys); self.keys.append(k)
        self.xidx.append(np.arange(2 ** self.n) ^ xm); self.zsgn.append(sgn.astype(np.float64)); self.ph.append(1j ** ny)
        return self._index[k]

    def finalize(self):
        self.xidx = np.array(self.xidx); self.zsgn = np.array(self.zsgn); self.ph = np.array(self.ph)

    def overlaps(self, phi, psi):
        """<phi|P_s|psi> for every stored string s (vector of length n_strings)."""
        return self.ph * np.einsum("sb,sb->s", np.conj(phi)[self.xidx], self.zsgn * psi[None, :])


def pauli_generator(op, delta):
    """
    Hermitian generator G (as a cirq.PauliSum) of a gate op with op = exp(-i delta G),
    from the principal matrix log on the gate's own qubits, expanded in Pauli strings
    (identity component dropped). Assumes the eigenphases of op lie in (-pi, pi).
    """
    qs = list(op.qubits); k = len(qs)
    G = 1j * sl.logm(cirq.unitary(op)) / delta
    out = cirq.PauliSum()
    for labels in itertools.product("IXYZ", repeat=k):
        if all(l == "I" for l in labels):
            continue
        ps = cirq.PauliString({q: getattr(cirq, l) for q, l in zip(qs, labels) if l != "I"})
        c = np.trace(ps.matrix(qubits=qs) @ G) / 2 ** k
        if abs(c) > 1e-12:
            out += float(np.real(c)) * ps
    return out


def floquet1_hamiltonian(model, delta, check=True):
    """
    First-order BCH Floquet Hamiltonian of ONE first-order Trotter system layer,
        U_F = U_n ... U_1,   U_j = exp(-i delta G_j)   (layer order = model.get_system_layer(order=1)),
        H_F^(1) = sum_j G_j - (i delta / 2) sum_{j<k} [G_k, G_j],
    returned as a local cirq.PauliSum. With check=True, asserts sum_j G_j == model.hamiltonian.
    """
    layer = [u ** delta for u in model.get_system_layer(order=1)]
    Gs = [pauli_generator(op, delta) for op in layer]
    H0 = cirq.PauliSum()
    for G in Gs:
        H0 += G
    if check:
        diff = H0 - model.hamiltonian
        resid = max([abs(t.coefficient) for t in diff if t.qubits] + [0.0])
        if resid > 1e-8:
            raise ValueError(f"sum of layer generators differs from model.hamiltonian (max |coeff| {resid:.2e}); "
                             "check the system layer / delta convention")
    HF = H0
    for k in range(len(Gs)):
        for j in range(k):
            HF += -0.5j * delta * (Gs[k] * Gs[j] - Gs[j] * Gs[k])
    return _clean_pauli_sum(HF)


def floquet_exact_matrix(model, delta, qubits, margin=0.3):
    """
    Exact Floquet Hamiltonian H_F = (i/delta) log U_F of one first-order Trotter system layer, as a dense
    matrix on `qubits` (big-endian), with the branch cut in the largest gap of the eigenphases and the
    trace fixed to that of H. Raises if spectral_width * delta >= 2 pi - margin (no unique branch).
    """
    H = model.hamiltonian.matrix(qubits=qubits); w = np.linalg.eigvalsh(H)
    if (w[-1] - w[0]) * delta >= 2 * np.pi - margin:
        raise ValueError(f"spectral width * delta = {(w[-1] - w[0]) * delta:.2f} >= 2pi - {margin}: "
                         "the Floquet Hamiltonian has no unique branch; use the first-order (BCH) H_F instead")
    UF = cirq.Circuit([u ** delta for u in model.get_system_layer(order=1)]).unitary(qubit_order=qubits)
    ev, V = np.linalg.eig(UF); ph = np.angle(ev); sp = np.sort(ph)
    gp = np.diff(np.r_[sp, sp[0] + 2 * np.pi]); k = int(np.argmax(gp)); cut = sp[k] + gp[k] / 2
    th = np.mod(ph - cut, 2 * np.pi) + cut
    HF = V @ np.diag(-th / delta) @ np.linalg.inv(V); HF = (HF + HF.conj().T) / 2
    return HF + (np.trace(H) - np.trace(HF)).real / len(H) * np.eye(len(H))


def _clean_pauli_sum(ps, tol=1e-12):
    out = cirq.PauliSum()
    for t in ps:
        if t.qubits and abs(t.coefficient) > tol:
            out += t
    return out


class CommutatorMeasurement(DefaultMeasurement2):
    """
    DefaultMeasurement2 plus commutator diagnostics of 'diagonality' of the state.

    If rho commutes with K, then C_O = <i[K, O]> = -i Tr([K, rho] O) = 0 for EVERY O; this needs
    no temperature fit, and since it is linear in rho the trajectory average is the ensemble value.
    Equivalently <i[K, O]> = d<O>/dt under K alone, so in a boundary-cooled steady state nonzero
    values near the ancillas are the Hamiltonian currents that balance the dissipation.

    O runs over the full local set: all single-site Paulis P_k (3 Ns) and all two-site products
    P_i Q_j, P, Q in {X, Y, Z}, i < j (9 Ns(Ns-1)/2). (For a real H and O with no Y, i[H, O] is
    imaginary and only probes Im rho; the Y-containing O probe the real off-diagonals.)

    Columns, with Oname = f"{P}_{k}" or f"{P}{Q}_{i}_{j}":
        cH_<Oname>     K = model.hamiltonian
        cHF1_<Oname>   K = first-order BCH Floquet Hamiltonian of one Trotter system layer at DELTA
        cHFx_<Oname>   K = exact Floquet Hamiltonian (dense; only if EXACT_FLOQUET, small Ns only,
                       and only while spectral width * DELTA < 2 pi)

    Configure by subclassing at MODULE level (so run_parallel can pickle it to spawned workers):
        class M(cooling.CommutatorMeasurement):
            DELTA = 0.1             # Trotter step of the system layer (GS protocol: 2T/N; DB: delta)
            EXACT_FLOQUET = True    # optional dense exact-H_F commutators
        Simulation(protocol, measurement=M)
    Assumes the protocol uses the first-order (trotter_order=1) system layer for H_F.
    """
    DELTA = None
    BCH_FLOQUET = True
    EXACT_FLOQUET = False

    def __init__(self, device: "Device", model: "Model"):
        super().__init__(device, model)
        q = list(device.system_qubits); Ns = len(q)
        self._obs = [(f"{a}_{k}", getattr(cirq, a)(q[k])) for k in range(Ns) for a in "XYZ"]
        self._obs += [(f"{a}{b}_{i}_{j}", cirq.PauliString({q[i]: getattr(cirq, a), q[j]: getattr(cirq, b)}))
                      for i in range(Ns) for j in range(i + 1, Ns) for a in "XYZ" for b in "XYZ"]
        self._tab = _PauliTable(q)
        Ks = [("cH", model.hamiltonian)]
        if self.BCH_FLOQUET or self.EXACT_FLOQUET:
            if self.DELTA is None:
                raise ValueError("CommutatorMeasurement: set the class attribute DELTA (Trotter step) in a subclass")
        if self.BCH_FLOQUET:
            Ks.append(("cHF1", floquet1_hamiltonian(model, self.DELTA)))
        self._blocks = []
        for label, K in Ks:
            rows = []
            for name, O in self._obs:
                C = _clean_pauli_sum(1j * (K * O - O * K))
                terms = []
                for t in C:
                    c = complex(t.coefficient)
                    assert abs(c.imag) < 1e-9, f"i[K,O] not Hermitian for {label}, {name}"
                    terms.append((self._tab.add_string(t / c), c.real))
                rows.append(terms)
            self._blocks.append((label, rows))
        self._obs_strings = [self._tab.add_string(cirq.PauliString(O) if not isinstance(O, cirq.PauliString) else O)
                             for _, O in self._obs]
        self._tab.finalize()
        ns = len(self._tab.keys)
        self._coef = {}                                              # sparse (n_obs x n_strings), ~1% filled
        for label, rows in self._blocks:
            r_, c_, v_ = [], [], []
            for r, terms in enumerate(rows):
                for s, c in terms:
                    r_.append(r); c_.append(s); v_.append(c)
            self._coef[label] = (sp.csr_matrix((v_, (r_, c_)), shape=(len(rows), ns)),
                                 [f"{label}_{name}" for name, _ in self._obs])
        self._HFx = floquet_exact_matrix(model, self.DELTA, q) if self.EXACT_FLOQUET else None
        self._HFx_names = [f"cHFx_{name}" for name, _ in self._obs]

    def measure_from_state_vector(self, state):
        measurement = super().measure_from_state_vector(state)
        psi = np.asarray(state).ravel().astype(np.complex128)
        ev = np.real(self._tab.overlaps(psi, psi))                  # <psi|P_s|psi>, all strings at once
        for M, names in self._coef.values():
            measurement.update(zip(names, (M @ ev).tolist()))
        if self._HFx is not None:                                   # <i[H_F,O]> = -2 Im <H_F psi | O psi>
            a = self._tab.overlaps(self._HFx @ psi, psi)[self._obs_strings]
            measurement.update(zip(self._HFx_names, (-2 * np.imag(a)).tolist()))
        return measurement
