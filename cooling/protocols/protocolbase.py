"""
Abstract base class for cooling protocols.

The Protocol returns the cooling channel for given physical parameters. Minimally it defines a reset layer, noise layer, and unitary system-bath layer, which are combined into the channel: the channel is returned as a FrozenCircuit. Different subclasses define different channels.

Created by Jerome Lloyd on 4th June 2026.
"""

from abc import ABC, abstractmethod
import cirq
import numbers
from ..gates import YXPowGate, ZXPowGate

class Protocol(ABC):
    """
    Abstract base class for cooling protocols.

    Parameters
    ----------
    device : CoolingDevice
    model  : Model
    gamma  : float - depolarising noise strength. 0 = off (default).
    """

    ## override in subclass if needed
    _COUPLING_GATE_MAP = {'X': cirq.XXPowGate,'Y': YXPowGate,'Z': ZXPowGate, 'iSWAP':  cirq.ISwapPowGate}


    def __init__(self, device:"CoolingDevice", model:"Model", params:dict=None, gamma:float=0.):
        self.device = device
        self.model = model
        self.params = params if params is not None else {}

        # Reset layer: identity on system qubits, reset on bath qubits.
        self._reset_layer = ([cirq.I(q) for q in device.system_qubits]
            + [cirq.reset(q) for q in device.bath_qubits])

        # Noise layer: only non-trivial if gamma != 0
        self._noise_layer = None
        if gamma != 0.0:
            self.make_noise_layer(gamma)
        
    def make_noise_layer(self, gamma):
        # default noise layer: depolarising on all qubits. Subclass can overwrite
        self._noise_layer = [cirq.depolarize(gamma).on(q) for q in self.device.qubits]

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    def noise_layer(self):
        return self._noise_layer

    @property
    def reset_layer(self):
        return self._reset_layer

    @property
    def allowed_coupling_gates(self) -> dict:
        return self._COUPLING_GATE_MAP
    
    def coupling_gates(self, coupling_ops: dict) -> dict:
        """
        Return {bath_idx: gate} for the given {bath_idx: op_str} dict.
        Valid op strings: 'X', 'Y', 'Z', 'iSWAP'.
        Override in subclass if needed.
        """
        invalid = set(coupling_ops.values()) - self._COUPLING_GATE_MAP.keys()
        if invalid:
            raise ValueError(f"Unknown coupling ops {invalid}. Choose from {list(self._COUPLING_GATE_MAP)}.")
        return {bi: self._COUPLING_GATE_MAP[op] for bi, op in coupling_ops.items()}

    def validate_geometry(self, coupling_geometry: dict, coupling_ops: dict):
        """
        Validate coupling_geometry and coupling_ops.

        - coupling_geometry and coupling_ops have the same keys.
        - Keys equal the full set of bath qubit indices {0, ..., Nb-1}.
        - All system qubit indices in coupling_geometry are valid (0 to Ns-1).

        TODO: add validate option for cirq device fixed geometry.
        """
        Nb, Ns = self.device.Nb, self.device.Ns
        expected = set(range(Nb))
        geom_keys = set(coupling_geometry.keys())
        ops_keys  = set(coupling_ops.keys())

        if geom_keys != expected or ops_keys != expected:
            missing_geom = expected - geom_keys
            missing_ops  = expected - ops_keys
            extra_geom   = geom_keys - expected
            extra_ops    = ops_keys  - expected
            msg = "coupling_geometry and coupling_ops must both contain all bath qubit indices."
            if missing_geom: msg += f" coupling_geometry missing: {missing_geom}."
            if missing_ops:  msg += f" coupling_ops missing: {missing_ops}."
            if extra_geom:   msg += f" coupling_geometry unknown: {extra_geom}."
            if extra_ops:    msg += f" coupling_ops unknown: {extra_ops}."
            raise ValueError(msg)

        invalid_sys = {si for si in coupling_geometry.values() if not (0 <= si < Ns)}
        if invalid_sys:
            raise ValueError(
                f"coupling_geometry contains invalid system qubit indices {invalid_sys}. "
                f"Valid range: 0 to {Ns - 1}."
            )

    @abstractmethod
    def channel(self, coupling_geometry: dict, coupling_ops: dict, params: dict) -> cirq.FrozenCircuit:
        """
        Build and return one cooling-cycle.
        """
        pass

    @property
    def print_channel_description(self):
        """Print the channel description (inc parameters required by channel) for this protocol."""
        print(self.channel.__doc__)

    def draw_channel(self, coupling_geometry: dict, coupling_ops: dict, params: dict, save: str = None):
        """Draw the cooling channel circuit. Pass save='filename.svg' to save."""
        C = cirq.Circuit(self.channel(coupling_geometry, coupling_ops, params))
        print(f"Circuit: {len(C) - 1} moments + reset")
        try:
            from IPython.display import display
            from cirq.contrib.svg import SVGCircuit
            display(C)
            if save is not None:
                svg = SVGCircuit(C)._repr_svg_()
                with open(save, 'w') as f:
                    f.write(svg)
                print(f"Saved to {save}")
        except ImportError:
            print(C)

    # ______ helpers for getting parameter values _____ 
    
    def require_real(self, params: dict, key: str, default=None):
        """Get parameter from dict and check that it is a real number."""
        x = params.get(key, default)
        if x is None:
            raise KeyError(f"Missing required parameter {key!r}")

        if isinstance(x, bool) or not isinstance(x, numbers.Real):
            raise TypeError(
                f"Parameter {key!r} must be a real number "
                f"got {type(x).__name__}: {x!r}"
            )

        return float(x)

    def require_int(self, params: dict, key: str, default=None, positive=True):
        """Get parameter and require a concrete integer (not bool)."""
        x = params.get(key, default)
        if x is None:
            raise KeyError(f"Missing required parameter {key!r}")

        if isinstance(x, bool) or not isinstance(x, numbers.Integral):
            raise TypeError(
                f"Parameter {key!r} must be an integer "
                f"got {type(x).__name__}: {x!r}"
            )

        if positive and x <= 0:
            raise ValueError(
                f"Parameter {key!r} must be positive, "
                f"got {type(x).__name__}: {x!r}"
            )
    
        return int(x)

    def get_param(self, params: dict, key: str, default=None):
        """Get parameter from dict without type restriction."""
        x = params.get(key, default)
        if x is None:
            raise KeyError(f"Missing required parameter {key!r}")
        return x


