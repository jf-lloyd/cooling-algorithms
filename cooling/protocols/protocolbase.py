"""
Abstract base class for cooling protocols.

The Protocol returns the cooling channel for given physical parameters. Minimally it defines a reset layer, noise layer, and unitary system-bath layer, which are combined into the channel: the channel is returned as a FrozenCircuit. Some parameters passed to the channel constructor may be symbolic (for use in Scheduler). Different subclasses define different channels. 

Created by Jerome Lloyd on 4th June 2026.
"""

from abc import ABC, abstractmethod
import cirq
import numbers

class Protocol(ABC):
    """
    Abstract base class for cooling protocols.

    Parameters
    ----------
    device : CoolingDevice
    model  : Model
    gamma  : float - depolarising noise strength. 0 = off (default).
    """

    def __init__(self, device:"CoolingDevice", model:"Model", gamma:float=0.):
        self.device = device
        self.model = model

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
        
        
    @abstractmethod
    def channel(self, coupling_geometry: dict, params: dict) -> cirq.FrozenCircuit:
        """
        Build and return one cooling-cycle.
        """
        pass

    @property
    def print_channel_description(self):
        """Print the channel description (inc parameters required by channel) for this protocol."""
        print(self.channel.__doc__)

    def draw_channel(self, coupling_geometry: dict, params: dict):
        """Draw the cooling channel circuit."""
        C = cirq.Circuit(self.channel(coupling_geometry, params))
        print(f"Circuit: {len(C) - 1} moments + reset")
        try:
            from IPython.display import display
            display(C)
        except ImportError:
            print(C)


    # ______ helpers for getting parameter values _____ 
    
    def require_real(self, params: dict, key: str, default=None):
        """Get parameter from dict and check that it is a real number (excludes symbolic)."""
        x = params.get(key, default)
        if x is None:
            raise KeyError(f"Missing required parameter {key!r}")

        if isinstance(x, bool) or not isinstance(x, numbers.Real):
            raise TypeError(
                f"Parameter {key!r} must be a real number "
                f"(circuit structure depends on it), "
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
                f"(circuit structure depends on it), "
                f"got {type(x).__name__}: {x!r}"
            )

        if positive and x <= 0:
            raise ValueError(
                f"Parameter {key!r} must be positive, "
                f"got {type(x).__name__}: {x!r}"
            )
    
        return int(x)

    def allow_symbolic(self, params: dict, key: str, default=None):
        """Get parameter without type restriction."""
        x = params.get(key, default)
        if x is None:
            raise KeyError(f"Missing required parameter {key!r}")
        return x


