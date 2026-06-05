"""
Detailed balance protocol -- choice between different filter functions
Created by Jerome Lloyd on 4th June 2026.
"""

import numpy as np
import cirq
from .protocolbase import Protocol

"""
The default coupling gate is XXPow i.e. exp(-i·angle·X_sys X_bath)
To switch the system operator to a different Pauli, we apply single-qubit rotations to the system qubit
Rz(az) · Ry(ay) · X · Ry(-ay) · Rz(-az)
PAULI_ALPHAS maps target Pauli name → (az, ay) rotation angles. 
"""
PAULI_ALPHAS = {'X': (0.0, 0.0), 'Y': (np.pi/2, 0.0), 'Z': (0.0, -np.pi/2)}

class DetailedBalanceProtocol(Protocol):

    """ 
    Detailed balance protocol, using either gaussian filter (arxiv/2506.21318) or 
    "modulated coupling protocol" (step-like) filter (arxiv/2404.12175).
    The code constructs the reset channel for a given Device, Model, coupling geometry and parameters
    (see channel for required parameters). The channel is returned as a FrozenCircuit that can then 
    be used by the Simulator; several channel parameters may be symbolic, which allows for time-dependent 
    scheduling (see Scheduler). 

    For mixing purposes it is useful to be able to randomly select single Paulis as coupling operators at start
    of each cycle. To avoid recreating the circuit every time, we instead parameterise the random Paulis by
    single qubit rotations with parameterised angles (see above re. PAULI_ALPHAS). The Scheduler can then pass
    random choices of the PAULI_ALPHAS at each new cycle, without recreating the circuit.

    Changes to circuit geometry (e.g. random coupling geometry) requires redrawing the circuit. To avoid overhead,
    caching of random circuits can be handled in the Scheduler.
    """

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

    # ── Circuit building helpers ──────────────────────────────────────────────

    def _get_bath_layer(self, h:float):
        """Uniform Zeeman splitting on bath qubits. cirq: rz(-h) = exp(ih/2 Z)."""
        return [cirq.rz(-h)(b) for b in self.device.bath_qubits]

    def _get_coupling_layer(self, coupling_geometry:dict, theta):
        """
        Default XX coupling gates for all SB pairs. 
        The system Pauli can be rotated to Y,Z via single-qubit rotations: see _get_coupling_rotations       
        theta: coupling strength — float or sympy symbol.
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
                Values are floats or sympy symbols. Use PAULI_ALPHAS for standard choices.

        Returns (pre_ops, post_ops). Ops grouped by gate type so each sub-list
        acts on distinct qubits → cirq packs each into one moment.
        """
        S = self.device.system_qubits
        pre_rz, pre_ry, post_ry, post_rz = [], [], [], []
        for bi, si in coupling_geometry.items():
            az, ay = aPauli[bi]
            pre_rz.append(cirq.rz(-az)(S[si]))   # U†: Rz(-az)
            post_rz.append(cirq.rz(az)(S[si]))   # U:  Rz(az)
            pre_ry.append(cirq.ry(-ay)(S[si]))   # U†: Ry(-ay)
            post_ry.append(cirq.ry(ay)(S[si]))   # U:  Ry(ay)
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
        Build one cooling cycle.

        coupling_geometry   : dict {bath_idx: sys_idx}, system bath coupling geometry

        Required params (S = structural, sets circuit depth and must be real parameter):
            beta            : float (S) - inverse target temperature
            delta           : float (S) - trotter angle
            h               : float (S) - bath splitting
            theta           : float or sympy  — coupling strength

        Optional params:
            NT              : int (S, default 5) — filter truncation / circuit depth parameter
            aPauli          : dict {bath_idx: (az, ay)}, optional, float or sympy, defaults to XX coupling

        Returns a FrozenCircuit. If theta or aPauli carry sympy symbols the
        circuit is parameterised and resolved before simulation (allows Scheduling).
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

        # non-structural params — may be sympy symbols
        theta  = self.allow_symbolic(params, "theta")
        default_aPauli = {bi: PAULI_ALPHAS['X'] for bi in coupling_geometry.keys()} # default to XX coupling
        aPauli = self.allow_symbolic(params, "aPauli", default=default_aPauli)

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
