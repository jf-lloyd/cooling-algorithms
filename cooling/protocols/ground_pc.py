"""
Ground state protocol -- choice between different filter functions
Created by Jerome Lloyd on 19th June 2026.
"""

import numpy as np
import cirq
from .protocolbase import Protocol

class GroundStateProtocol(Protocol):

    """
    Ground state cooling protocol. Targets the ground state, with different filter functions.
    Currently supported filter functions: 'gaussian', 'constant', 'super_gaussian'.
    Supported system-bath coupling gates: 'XX', 'YX', 'ZX', 'iSWAP' (default not used).
    """

    _COUPLING_GATE_MAP = {k: v for k, v in Protocol._COUPLING_GATE_MAP.items()}

    def __init__(self, device:"CoolingDevice", model:"Model", params:dict=None,
                 noise_model:"cirq.NoiseModel|None"=None, function="gaussian",
                 trotter_order=1, merge_single_qubit_gates: bool = True,
                 drop_negligible_operations: bool = True, verbose: bool = False, allow_iSWAP = False):

        super().__init__(device, model, params, noise_model)

        self.function = function
        if function == "gaussian":
            self.filter_function = self.gaussian_filter_function
        elif function == "constant":
            self.filter_function = self.constant_filter_function
        elif function == "super_gaussian":
            self.filter_function = self.super_gaussian_filter_function
        else:
            raise ValueError(f"Unknown filter function {function!r}. Choose 'gaussian', 'constant', or 'super_gaussian'.")

        if trotter_order not in (1, 2):
            raise ValueError(f"trotter_order must be 1 or 2, got {trotter_order!r}.")
        self.trotter_order = trotter_order
        self.merge_single_qubit_gates = merge_single_qubit_gates
        self.drop_negligible_operations = drop_negligible_operations
        self.verbose = verbose

        # Instance-level map shadows the class default (which excludes iSWAP).
        # allow_iSWAP=True re-admits 'iSWAP' to the allowed coupling ops.
        self.allow_iSWAP = allow_iSWAP
        self._COUPLING_GATE_MAP = (
            dict(Protocol._COUPLING_GATE_MAP) if allow_iSWAP
            else {k: v for k, v in Protocol._COUPLING_GATE_MAP.items() if k != 'iSWAP'}
        )

    @property
    def name(self):
        p = self.params
        parts = []
        for key, fmt in [('delta', 'd{:.4f}'), ('h', 'h{:.2f}'), ('sigma', 's{:.3f}'), ('theta', 'th{:.3f}')]:
            if key in p:
                parts.append(fmt.format(p[key]))
        parts.append(self.function)
        if self.trotter_order == 2:
            parts.append("o2")
        return "_".join(parts)

    @property
    def print_channel_description(self):
        print(f"Using filter function: {self.function}")
        print(self.channel.__doc__)

    # ── Filter functions ──────────────────────────────────────────────────────
    # All filters share the signature (sigma, delta, NT) and return a normalised
    # array f of length 2*NT+1.  sigma is the time-domain width (continuous time
    # units: the filter decays / is truncated over |δt| ~ sigma).

    def gaussian_filter_function(self, sigma: float, delta: float, NT: int):
        """Gaussian filter — f[t] = exp(-(δt)²/(2σ²))."""
        a = delta/sigma
        MT = max(int(NT), int(NT / a))
        f = [np.exp(-(a*t)**2 / 2) for t in np.arange(-MT, MT + 1)]
        f /= delta * np.sum(np.abs(f))
        return f

    def constant_filter_function(self, delta: float, NT: int):
        """Constant (flat) filter — f[t] = 1."""
        MT = int(NT)
        f = np.ones(2 * MT + 1)
        f /= delta * np.sum(np.abs(f))
        return f

    def super_gaussian_filter_function(self, sigma: float, delta: float, NT: int, n: int = 2):
        """Freq-domain super-Gaussian: f̃(Ω) ∝ exp(-(|Ω|·sigma)^{2n}).
        n=1: Gaussian; n=2: flat-topped; larger n → near-rectangular.
        Time tails decay as exp(-c|t|^{2n/(2n-1)}), enabling accurate truncation at small NT.
        n passed via params as 'SG_N' (default 2)."""
        tlist     = np.arange(-NT, NT + 1)
        tau       = tlist
        N_fft     = max(4096, 32 * (2 * NT + 1))
        omega_max = 6.0 / sigma
        omega     = np.linspace(-omega_max, omega_max, N_fft)
        dw        = omega[1] - omega[0]
        f_tilde   = np.exp(-(np.abs(omega) * sigma) ** (2 * n))
        f         = np.real(np.exp(1j * np.outer(tau, omega)) @ f_tilde) * dw / (2 * np.pi)
        f        /= delta * np.sum(np.abs(f))
        return f

    def fourier_filter_function(self, omega: float, flist, h: float, delta: float):
        """Fourier-transformed filter function."""
        MT    = len(flist) // 2
        tlist = np.arange(-MT, MT + 1)
        return np.sum([flist[t] * np.exp(1j * (h - omega) * delta * tlist[t]) for t in range(len(tlist))])

    # ── Circuit building helpers ──────────────────────────────────────────────

    def _get_bath_layer(self, h: float):
        """Uniform Zeeman splitting on bath qubits. cirq: rz(-h) = exp(ih/2 Z)."""
        return [cirq.rz(-h)(b) for b in self.device.bath_qubits]

    def _get_coupling_layer(self, coupling_geometry: dict, coupling_ops: dict, theta: float):
        """
        Coupling gates for all system-bath pairs.
        coupling_ops : {bath_idx: op_string} — 'X', 'Y', 'Z' per bath qubit.
        exponent = 2*theta/π so that gate**delta**f[j] → exp(-i·theta·delta·f[j]·OP).
        """
        S  = self.device.system_qubits
        B  = self.device.bath_qubits
        sb = self.coupling_gates(coupling_ops)
        return [sb[bi](exponent=2 / np.pi * theta)(S[si], B[bi]) for bi, si in coupling_geometry.items()]

    @staticmethod
    def _gate_count(circuit: cirq.Circuit) -> int:
        return sum(1 for _ in circuit.all_operations())

    # ── Main channel builder ──────────────────────────────────────────────────

    def channel(self, coupling_geometry: dict, coupling_ops: dict, params: dict = None, compile: bool = True) -> cirq.FrozenCircuit:
        """
        GroundStateProtocol channel. Supported coupling gates (SB): 'XX', 'YX', 'ZX'

        coupling_geometry : dict {bath_idx: sys_idx}
        coupling_ops      : dict {bath_idx: op_string}, op_string in {'X', 'Y', 'Z'}
        params:
            Required:
                delta  : float — Trotter angle
                h      : float — bath splitting
                sigma  : float — filter time-domain width (continuous time)
                theta  : float — coupling strength
            Optional:
                NT     : int (default 5) — filter half-length (circuit depth ~ 2*NT+1 steps)
                SG_N   : int (default 2, super_gaussian only) — super-Gaussian order n

        trotter_order (set in __init__, default 1):
            1 — first-order (Lie-Trotter) split: sys(δ) -> bath(δ) -> coupling(δ·f[j])
                per step, error O(δ) global.
            2 — second-order (Strang) split with merged half-steps ("leapfrog"):
                sys(δ/2) -> [bath(δ/2) -> coupling(δ·f[j]) -> bath(δ/2) -> sys(δ)]*
                -> ... -> sys(δ/2), error O(δ²) global, ~4/3x gates per step.
        """
        self.validate_geometry(coupling_geometry, coupling_ops)

        params = {**self.params, **(params or {})}
        delta = self.require_real(params, "delta")
        h     = self.require_real(params, "h")
        sigma = self.require_real(params, "sigma")
        NT    = self.require_real(params, "NT", default=5)
        theta = self.get_param(params, "theta")

        filter_kwargs = {}
        if self.function == "super_gaussian":
            filter_kwargs['n'] = self.require_int(params, "SG_N", default=2)
        filter_f = self.filter_function(sigma, delta, NT, **filter_kwargs)
        MT       = len(filter_f) // 2

        c_ops       = [u**delta for u in self._get_coupling_layer(coupling_geometry, coupling_ops, theta)]
        reset_layer = self._reset_layer

        cycle = cirq.Circuit()

        if self.trotter_order == 1:
            sys_ops  = [u**delta for u in self.model.get_system_layer(order=1)]
            bath_ops = [u**delta for u in self._get_bath_layer(h)]
            for j in range(2 * MT + 1):
                cycle.append(sys_ops)
                cycle.append(bath_ops)
                cycle.append(u**filter_f[j] for u in c_ops)

        else:  # trotter_order == 2: Strang split with merged half-steps
            sys_layer = self.model.get_system_layer(order=2)
            sys_full  = [u**delta       for u in sys_layer]
            sys_half  = [u**(delta / 2) for u in sys_layer]
            bath_half = [u**(delta / 2) for u in self._get_bath_layer(h)]

            N = 2 * MT + 1
            cycle.append(sys_half)
            for j in range(N):
                cycle.append(bath_half)
                cycle.append(u**filter_f[j] for u in c_ops)
                cycle.append(bath_half)
                cycle.append(sys_full if j < N - 1 else sys_half)

        cycle.append(reset_layer)

        if self.merge_single_qubit_gates:
            n_before = self._gate_count(cycle)
            cycle = cirq.merge_single_qubit_gates_to_phxz(cycle)
            n_after = self._gate_count(cycle)
            if self.verbose:
                print(f"single gate merging -- removed {n_before - n_after} gates")
        if self.drop_negligible_operations:
            n_before = self._gate_count(cycle)
            cycle = cirq.drop_negligible_operations(cycle)
            n_after = self._gate_count(cycle)
            if self.verbose:
                print(f"drop negligible operations -- removed {n_before - n_after} gates")
        if compile:
            gateset = cirq.CZTargetGateset(allow_partial_czs=True)
            cycle = cirq.optimize_for_target_gateset(cirq.Circuit(cycle), gateset=gateset)
            n_before = self._gate_count(cycle)
            cycle = cirq.eject_phased_paulis(cycle)
            cycle = cirq.eject_z(cycle)
            n_after = self._gate_count(cycle)
            if self.verbose:
                print(f"ejecting phased gates -- removed {n_before - n_after} gates")
            cycle = cirq.align_left(cycle)
        cycle = self.apply_noise(cycle)

        return cirq.FrozenCircuit(cycle)
