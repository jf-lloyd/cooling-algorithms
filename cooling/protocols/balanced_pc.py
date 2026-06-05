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
    geometry and parameters (see channel() for required parameters).

    The system coupling operator (X, Y, or Z) is selected per bath qubit via single-qubit
    rotations Rz(-az)·Ry(-ay)·XX·Ry(ay)·Rz(az). Standard angles are in pauli_angles.
    """

    # Rotation angles (az, ay) to select system Pauli: Rz(az)·Ry(ay)·X·Ry(-ay)·Rz(-az) = target Pauli.
    pauli_angles = {'X': (0.0, 0.0), 'Y': (np.pi / 2, 0.0), 'Z': (0.0, -np.pi / 2)}

    def __init__(self, device:"CoolingDevice", model:"Model", gamma:float=0., function="gaussian"):

        super().__init__(device, model, gamma)

        ## switch between gaussian and mcp filter functions
        self.function = function
        if function == "gaussian":
            self.filter_function = self.gaussian_filter_function
        elif function == "mcp":
            self.filter_function = self.mcp_filter_function
        else:
            raise ValueError(f"Unknown filter function {function!r}. Choose 'gaussian' or 'mcp'.")

        self._system_layer = self.model.system_layer

    @property
    def name(self):
        return 'no_name'

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
        f /= delta * np.sum(np.abs(f))   # normalise: delta * Sum|f| = 1
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
        f /= delta * np.sum(np.abs(f))   # normalise: delta * Sum|f| = 1
        return f

    def fourier_filter_function(self, omega:float, beta:float, delta:float, h:float, NT:int):
        """ return the Fourier transformed filter function -- need to check quasienergy conventions! """
        flist = self.filter_function(beta, delta, h, NT)
        MT = len(flist)//2
        tlist = np.arange(-MT, MT+1)
        return np.sum([flist[t]*np.exp(1j*(h-omega)*tlist[t]) for t in range(len(tlist))])
            

    # ── Circuit building helpers ──────────────────────────────────────────────

    def _get_bath_layer(self, h:float):
        """Uniform Zeeman splitting on bath qubits. cirq: rz(-h) = exp(ih/2 Z)."""
        return [cirq.rz(-h)(b) for b in self.device.bath_qubits]

    def _get_coupling_layer(self, coupling_geometry:dict, theta):
        """
        Default XX coupling gates for all SB pairs.
        The system Pauli can be rotated to Y,Z via single-qubit rotations: see _get_coupling_rotations.
        exponent = 2*theta/π so that gate**delta**f[j] → exp(-i·theta·delta·f[j]·XX).
        """
        S = self.device.system_qubits
        B = self.device.bath_qubits
        return [cirq.XXPowGate(exponent=2 / np.pi * theta)(S[si], B[bi])
            for bi, si in coupling_geometry.items()]

    def _get_coupling_rotations(self, coupling_geometry:dict, aPauli:dict):
        """
        Pauli selection rotations: Rz(-az)·Ry(-ay) before XX, Ry(ay)·Rz(az) after.

        aPauli: dict {bath_idx: (az, ay)} — keyed by bath qubit index.
                System qubit retrieved from coupling_geometry.
                Use pauli_angles for standard X/Y/Z choices.

        Returns (pre_ops, post_ops). Ops grouped by gate type so each sub-list
        acts on distinct qubits → cirq packs each into one moment.
        """
        def _nonzero(val):
            return not np.isclose(val, 0)

        S = self.device.system_qubits
        pre_rz, pre_ry, post_ry, post_rz = [], [], [], []
        for bi, si in coupling_geometry.items():
            az, ay = aPauli[bi]
            if _nonzero(az):
                pre_rz.append(cirq.rz(-az)(S[si]))
                post_rz.append(cirq.rz(az)(S[si]))
            if _nonzero(ay):
                pre_ry.append(cirq.ry(-ay)(S[si]))
                post_ry.append(cirq.ry(ay)(S[si]))
        pre  = pre_rz  + pre_ry    # Rz(-az) then Ry(-ay) — distinct qubits, one moment each
        post = post_ry + post_rz   # Ry(ay)  then Rz(az)
        return pre, post

    def validate_geometry(self, coupling_geometry: dict):
        """
        Validate coupling_geometry = {bath_idx: sys_idx}.

        - Every bath qubit in the device must appear exactly once as a key.
        - Every sys_idx value must be a valid system qubit index.
        - Each system qubit appears at most once (max 1 bath per system qubit
          per layer). Multiple baths per system qubit not yet supported.
        """
        Nb, Ns = self.device.Nb, self.device.Ns
        expected_bath = set(range(Nb))
        given_bath    = set(coupling_geometry.keys())

        if given_bath != expected_bath:
            missing = expected_bath - given_bath
            extra   = given_bath - expected_bath
            msg = "coupling_geometry must include all bath qubits."
            if missing: msg += f" Missing bath indices: {missing}."
            if extra:   msg += f" Unknown bath indices: {extra}."
            raise ValueError(msg)

        sys_indices = list(coupling_geometry.values())
        invalid = {si for si in sys_indices if not (0 <= si < Ns)}
        if invalid:
            raise ValueError(
                f"coupling_geometry contains invalid system qubit indices {invalid}. "
                f"Valid range: 0 to {Ns - 1}."
            )

        if len(sys_indices) != len(set(sys_indices)):
            duplicates = {si for si in sys_indices if sys_indices.count(si) > 1}
            raise ValueError(
                f"coupling_geometry assigns multiple bath qubits to system qubit(s) {duplicates}. "
                f"Multiple baths per system qubit are not yet supported."
            )

    # ── Main channel builder ──────────────────────────────────────────────────

    def channel(self, coupling_geometry:dict, params:dict) -> cirq.FrozenCircuit:
        """
        DetailedBalanceProtocol channel.

        Requires:
        coupling_geometry   : dict {bath_idx: sys_idx}, system bath coupling geometry
        params              : dict {param_name: value}, channel parameters
            Required params (structural — set circuit depth, must be real):
                beta            : float — inverse target temperature
                delta           : float — Trotter angle
                h               : float — bath splitting
                theta           : float — coupling strength
            Optional params:
                NT              : int (default 5) — filter truncation / circuit depth
                aPauli          : dict {bath_idx: (az, ay)}, defaults to XX coupling

        Use Protocol.draw_channel(coupling_geometry, params) to visualise the circuit.
        """
        self.validate_geometry(coupling_geometry)

        # structural params — must be real valued (determine circuit depth)
        beta  = self.require_real(params, "beta")
        delta = self.require_real(params, "delta")
        h     = self.require_real(params, "h")
        NT    = self.require_int(params,  "NT", default=5)

        if self.function == "mcp" and not np.isclose(h, np.pi / 2):
            print(f"mcp requires h=π/2; overriding supplied h={h:.4f}")
            h = np.pi / 2

        theta  = self.get_param(params, "theta")
        default_aPauli = {bi: self.pauli_angles['X'] for bi in coupling_geometry.keys()}
        aPauli = {**default_aPauli, **self.get_param(params, "aPauli", default={})}

        # compute filter and derived structure
        filter_f = self.filter_function(beta, delta, h, NT)
        MT = len(filter_f) // 2

        # pre-build scaled layers (delta fixed)
        sys_ops  = [u**delta for u in self._system_layer]
        if self.function == "mcp": # don't apply delta -- requires sign-changing rotations
            bath_ops = list(self._get_bath_layer(h))
        else:
            bath_ops = [u**delta for u in self._get_bath_layer(h)]
        coupling_ops = [u**delta for u in self._get_coupling_layer(coupling_geometry, theta)]
        # get single qubit rotations to switch coupling operator
        coupling_pre, coupling_post = self._get_coupling_rotations(coupling_geometry, aPauli)

        noise_layer = self._noise_layer
        reset_layer = self._reset_layer

        # assemble cycle
        cycle = cirq.Circuit()
        for j in range(2 * MT + 1):
            cycle.append(sys_ops)
            cycle.append(bath_ops)
            cycle.append(coupling_pre)
            cycle.append(u**filter_f[j] for u in coupling_ops)
            cycle.append(coupling_post)
            if noise_layer is not None:
                cycle.append(noise_layer)

        cycle.append(reset_layer)
        return cirq.FrozenCircuit(cycle)
