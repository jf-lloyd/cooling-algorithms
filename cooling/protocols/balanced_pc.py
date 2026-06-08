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

    Supported system-bath coupling gates: 'XX', 'YX', 'ZX', 'iSWAP'.
    """

    def __init__(self, device:"CoolingDevice", model:"Model", params:dict=None, gamma:float=0., function="gaussian"):

        super().__init__(device, model, params, gamma)

        self.function = function
        if function == "gaussian":
            self.filter_function = self.gaussian_filter_function
        elif function == "mcp":
            self.filter_function = self.mcp_filter_function
        else:
            raise ValueError(f"Unknown filter function {function!r}. Choose 'gaussian' or 'mcp'.")

    @property
    def name(self):
        p = self.params
        parts = []
        for key, fmt in [('beta', 'b{:.2f}'), ('delta', 'd{:.4f}'),
                         ('h', 'h{:.2f}'), ('theta', 'th{:.3f}')]:
            if key in p:
                parts.append(fmt.format(p[key]))
        parts.append(self.function)
        return "_".join(parts)

    @property
    def print_channel_description(self):
        """Print the channel description (inc parameters required by channel) for this protocol."""
        print(f"Using filter function: {self.function}")
        print(self.channel.__doc__)

    # ── Filter functions ──────────────────────────────────────────────────────

    def gaussian_filter_function(self, beta:float, delta:float, h:float, NT:int):
        """Gaussian detailed balance filter — width ~ sqrt(beta/h)."""
        a  = delta * np.sqrt(abs(4 * h / beta))
        MT = max(NT, int(NT / a))
        f  = np.array([np.exp(-a**2 * t**2 / 2) for t in np.arange(-MT, MT + 1)])
        f /= delta * np.sum(np.abs(f))
        return f

    def mcp_filter_function(self, beta:float, delta:float, h:float, NT:int):
        """Modulated coupling pulse (sinc/sinh step-function filter)."""
        if not np.isclose(h, np.pi / 2):
            raise ValueError(f"mcp requires h=π/2, got h={h:.4f}")
        MT = max(NT, int(NT * beta / delta))
        f  = []
        for t in np.arange(-MT, MT + 1):
            if t == 0:
                f.append(0.5)
            else:
                f.append(np.sin(np.pi * t / 2) / np.sinh(delta * np.pi * t / beta) * delta / beta)
        f  = np.array(f)
        f /= delta * np.sum(np.abs(f))
        return f

    def fourier_filter_function(self, omega:float, beta:float, delta:float, h:float, NT:int):
        """Fourier-transformed filter function."""
        flist = self.filter_function(beta, delta, h, NT)
        MT    = len(flist) // 2
        tlist = np.arange(-MT, MT + 1)
        return np.sum([flist[t] * np.exp(1j * (h - omega) * tlist[t]) for t in range(len(tlist))])

    # ── Circuit building helpers ──────────────────────────────────────────────

    def _get_bath_layer(self, h:float):
        """Uniform Zeeman splitting on bath qubits. cirq: rz(-h) = exp(ih/2 Z)."""
        return [cirq.rz(-h)(b) for b in self.device.bath_qubits]

    def _get_coupling_layer(self, coupling_geometry:dict, coupling_ops:dict, theta:float):
        """
        Coupling gates for all system-bath pairs.
        coupling_ops : {bath_idx: op_string} — 'X', 'Y', 'Z', 'iSWAP' per bath qubit.
        exponent = 2*theta/π so that gate**delta**f[j] → exp(-i·theta·delta·f[j]·OP).
        """
        S  = self.device.system_qubits
        B  = self.device.bath_qubits
        sb = self.coupling_gates(coupling_ops)
        return [sb[bi](exponent=2 / np.pi * theta)(S[si], B[bi]) for bi, si in coupling_geometry.items()]

    # ── Main channel builder ──────────────────────────────────────────────────

    def channel(self, coupling_geometry:dict, coupling_ops:dict, params:dict=None) -> cirq.FrozenCircuit:
        """
        DetailedBalanceProtocol channel. Supported coupling gates (SB) 'XX', 'YX', 'ZX', 'iSWAP'

        coupling_geometry : dict {bath_idx: sys_idx}
        coupling_ops      : dict {bath_idx: op_string}, op_string in {'X', 'Y', 'Z', 'iSWAP'}
        params:
            Required (structural — set circuit depth, must be real):
                beta   : float — inverse target temperature
                delta  : float — Trotter angle
                h      : float — bath splitting
                theta  : float — coupling strength
            Optional:
                NT     : int (default 5) — filter truncation / circuit depth

        Use Protocol.draw_channel(coupling_geometry, coupling_ops, params) to visualise.
        """
        self.validate_geometry(coupling_geometry, coupling_ops)

        params = {**self.params, **(params or {})}
        beta  = self.require_real(params, "beta")
        delta = self.require_real(params, "delta")
        h     = self.require_real(params, "h")
        NT    = self.require_int(params,  "NT", default=5)
        theta = self.get_param(params, "theta")

        filter_f = self.filter_function(beta, delta, h, NT)
        MT       = len(filter_f) // 2

        sys_ops  = [u**delta for u in self.model.system_layer]
        if self.function == "mcp":
            bath_ops = list(self._get_bath_layer(h))
        else:
            bath_ops = [u**delta for u in self._get_bath_layer(h)]
        c_ops = [u**delta for u in self._get_coupling_layer(coupling_geometry, coupling_ops, theta)]

        noise_layer = self._noise_layer
        reset_layer = self._reset_layer

        cycle = cirq.Circuit()
        for j in range(2 * MT + 1):
            cycle.append(sys_ops)
            cycle.append(bath_ops)
            cycle.append(u**filter_f[j] for u in c_ops)
            if noise_layer is not None:
                cycle.append(noise_layer)
        cycle.append(reset_layer)
        
        return cirq.FrozenCircuit(cycle)
