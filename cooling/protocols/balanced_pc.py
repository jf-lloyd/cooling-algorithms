"""
Detailed balance protocol -- choice between different filter functions
Created by Jerome Lloyd on 4th June 2026.
"""

import numpy as np
import cirq
from .protocolbase import Protocol

class DetailedBalanceProtocol(Protocol):

    """
    Detailed balance protocol, using either gaussian filter (arxiv/2506.21318) or
    "modulated coupling protocol" (step-like) filter (arxiv/2404.12175).
    The channel is returned as a FrozenCircuit for a given Device, Model, coupling
    geometry, coupling operators and parameters (see channel() for details).

    Supported system-bath coupling gates: 'XX', 'YX', 'ZX'.
    Note iSWAP gate is not compatible with detailed balance (non-hermitian sigma^-) so we exclude it.
    """


    _COUPLING_GATE_MAP = {k: v for k, v in Protocol._COUPLING_GATE_MAP.items() if k != 'iSWAP'}

    def __init__(self, device:"CoolingDevice", model:"Model", params:dict=None,
                 noise_model:"cirq.NoiseModel|None"=None, function="gaussian",
                 trotter_order=1, merge_single_qubit_gates: bool = True,
                 drop_negligible_operations: bool = True, verbose: bool = False):

        super().__init__(device, model, params, noise_model)

        self.function = function
        if function == "gaussian":
            self.filter_function = self.gaussian_filter_function
        elif function == "mcp":
            self.filter_function = self.mcp_filter_function
        else:
            raise ValueError(f"Unknown filter function {function!r}. Choose 'gaussian' or 'mcp'.")

        if trotter_order not in (1, 2):
            raise ValueError(f"trotter_order must be 1 or 2, got {trotter_order!r}.")
        self.trotter_order = trotter_order
        self.merge_single_qubit_gates = merge_single_qubit_gates
        self.drop_negligible_operations = drop_negligible_operations
        self.verbose = verbose

    @property
    def name(self):
        p = self.params
        parts = []
        for key, fmt in [('beta', 'b{:.2f}'), ('delta', 'd{:.4f}'),
                         ('N', 'N{:.0f}'), ('h', 'h{:.2f}'), ('theta', 'th{:.3f}')]:
            if key in p:
                parts.append(fmt.format(p[key]))
        p2 = getattr(self.noise_model, 'p2', 0.)
        parts.append(f"p{p2:.2e}")
        parts.append(self.function)
        parts.append(f"o{self.trotter_order}")
        return "_".join(parts)

    @property
    def print_channel_description(self):
        """Print the channel description (inc parameters required by channel) for this protocol."""
        print(f"Using filter function: {self.function}")
        print(self.channel.__doc__)

    # ── Filter functions ──────────────────────────────────────────────────────

    # All filters share the signature (beta, delta, h, N) and return a normalised
    # array f of length N exactly, centered symmetrically about t=0: integer-spaced
    # (t=-M..M) for odd N, half-integer-spaced (no sample exactly at t=0) for even N.
    # This matches GroundStateProtocol, where N is likewise the circuit depth.
    #
    # N is the depth, not a truncation seed: it replaces the old NT, for which the
    # depth was the derived quantity max(NT, NT*beta/delta) and so could not be set
    # directly.  Truncating a filter to N samples is an approximation -- for mcp in
    # particular the sinc/sinh tail is cut, which rounds the frequency-domain step
    # and, once N falls well short of beta/delta, introduces ringing.

    @staticmethod
    def _tlist(N:int):
        return np.arange(N) - (N - 1) / 2

    def gaussian_filter_function(self, beta:float, delta:float, h:float, N:int):
        """Gaussian detailed balance filter — width ~ sqrt(beta/h). Length N."""
        a  = delta * np.sqrt(abs(4 * h / beta))
        f  = np.exp(-a**2 * self._tlist(N)**2 / 2)
        f /= delta * np.sum(np.abs(f))
        return f

    def mcp_filter_function(self, beta:float, delta:float, h:float, N:int):
        """Modulated coupling pulse (sinc/sinh step-function filter). Length N."""
        if not np.isclose(h, np.pi / 2):
            raise ValueError(f"mcp requires h=π/2, got h={h:.4f}")
        f = []
        for t in self._tlist(N):
            if t == 0:  # 0/0 limit, only hit for odd N
                f.append(0.5)
            else:
                f.append(np.sin(np.pi * t / 2) / np.sinh(delta * np.pi * t / beta) * delta / beta)
        f  = np.array(f)
        f /= delta * np.sum(np.abs(f))
        return f

    def fourier_filter_function(self, omega:float, flist, h, delta):
        """Fourier-transformed filter function."""
        n     = len(flist)
        tlist = self._tlist(n)
        if self.function == "mcp": # don't multiply h by delta
            return np.sum([flist[t] * np.exp(1j * (h - omega * delta ) * tlist[t]) for t in range(n)])
        else:
            return np.sum([flist[t] * np.exp(1j * (h - omega) * delta * tlist[t]) for t in range(n)])

    # ── Circuit building helpers ──────────────────────────────────────────────

    def _get_bath_layer(self, h:float):
        """Uniform Zeeman splitting on bath qubits. cirq: rz(-h) = exp(ih/2 Z)."""
        return [cirq.rz(-h)(b) for b in self.device.bath_qubits]

    def _get_coupling_layer(self, coupling_geometry:dict, coupling_ops:dict, theta:float):
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

    def channel(self, coupling_geometry:dict, coupling_ops:dict, params:dict=None, compile:bool=True) -> cirq.FrozenCircuit:
        """
        DetailedBalanceProtocol channel. Supported coupling gates (SB) 'XX', 'YX', 'ZX'

        coupling_geometry : dict {bath_idx: sys_idx}
        coupling_ops      : dict {bath_idx: op_string}, op_string in {'X', 'Y', 'Z'}
        params:
            Required:
                beta   : float — inverse target temperature
                delta  : float — Trotter angle
                h      : float — bath splitting (mcp requires h = π/2)
                N      : int — circuit depth (number of filter layers); the filter
                         is returned with exactly N samples
                theta  : float — coupling strength

        N is the depth directly, matching GroundStateProtocol. It replaces the old
        NT, for which the depth was the derived quantity max(NT, NT*beta/delta)
        (mcp) or max(NT, NT/a) (gaussian) and so could not be set independently of
        beta and delta -- passing an old NT value as N is not equivalent.

        trotter_order (set in __init__, default 1):
            1 — first-order (Lie-Trotter) split: sys(δ) -> bath(δ) -> coupling(δ·f[j])
                per step, error O(δ) global.
            2 — second-order (Strang) split with merged half-steps ("leapfrog"):
                sys(δ/2) -> [bath(δ/2) -> coupling(δ·f[j]) -> bath(δ/2) -> sys(δ)]*
                -> ... -> sys(δ/2), error O(δ²) global, ~4/3x gates per step.

        Use Protocol.draw_channel(coupling_geometry, coupling_ops, params) to visualise.
        """
        self.validate_geometry(coupling_geometry, coupling_ops)

        params = {**self.params, **(params or {})}
        beta  = self.require_real(params, "beta")
        delta = self.require_real(params, "delta")
        h     = self.require_real(params, "h")
        N     = self.require_int(params, "N")
        theta = self.get_param(params, "theta")

        filter_f = self.filter_function(beta, delta, h, N)
        n_layers = len(filter_f)

        c_ops = [u**delta for u in self._get_coupling_layer(coupling_geometry, coupling_ops, theta)]
        reset_layer = self._reset_layer

        cycle = cirq.Circuit()

        if self.trotter_order == 1:
            sys_ops  = [u**delta for u in self.model.get_system_layer(order=1)]
            if self.function == "mcp":
                bath_ops = list(self._get_bath_layer(h))
            else:
                bath_ops = [u**delta for u in self._get_bath_layer(h)]

            for j in range(n_layers):
                cycle.append(sys_ops)
                cycle.append(bath_ops)
                cycle.append(u**filter_f[j] for u in c_ops)

        else:  # trotter_order == 2: Strang split with merged half-steps
            sys_layer = self.model.get_system_layer(order=2)
            sys_full = [u**delta for u in sys_layer]
            sys_half = [u**(delta / 2) for u in sys_layer]
            if self.function == "mcp":
                bath_half = [u**0.5 for u in self._get_bath_layer(h)]
            else:
                bath_half = [u**(delta / 2) for u in self._get_bath_layer(h)]

            cycle.append(sys_half)
            for j in range(n_layers):
                cycle.append(bath_half)
                cycle.append(u**filter_f[j] for u in c_ops)
                cycle.append(bath_half)
                cycle.append(sys_full if j < n_layers - 1 else sys_half)

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
        if compile and getattr(self.model, 'allow_compile', True):
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
