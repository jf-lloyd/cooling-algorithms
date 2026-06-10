"""
Basic gate-count-based noise model.

Created by Jerome Lloyd on 10th June 2026
"""

import cirq

try:
    from ..gates import CachedDepolarizingChannel, CachedBitFlipChannel
except ImportError:
    CachedDepolarizingChannel = None
    CachedBitFlipChannel = None


def _depolarize(p: float, n_qubits: int = 1):
    """CachedDepolarizingChannel if available, else cirq.depolarize."""
    if CachedDepolarizingChannel is not None:
        return CachedDepolarizingChannel(p, n_qubits=n_qubits)
    return cirq.depolarize(p, n_qubits=n_qubits)


def _bit_flip(p: float):
    """CachedBitFlipChannel if available, else cirq.bit_flip."""
    if CachedBitFlipChannel is not None:
        return CachedBitFlipChannel(p)
    return cirq.bit_flip(p)


class BasicNoiseModel(cirq.NoiseModel):
    """
    Depolarizing noise after gates plus optional bit-flip error after reset. 
    Single and two qubit gates have different depolarizing strengths.

    Applied via `circuit.with_noise(noise_model)`

    Parameters
    ----------
    device       : CoolingDevice — used to classify operations as sys/bath/coupling.
    noise_params : dict, with keys (all optional):
        p1       : depolarizing strength applied after each 1-qubit gate (default 0)
        p2       : depolarizing strength applied after each 2-qubit gate (joint
                   2-qubit depolarizing channel, cirq.depolarize(p2, n_qubits=2))
                   (default 0)
        p_reset  : bit-flip probability applied after cirq.reset; 0 (default) =
                   no reset error
        sys      : if False, no noise on gates acting only on system qubits
                   (default True)
        bath     : if False, no noise on gates acting only on bath qubits
                   (includes reset) (default True)
        coupling : if False, no noise on gates acting on both system and bath
                   qubits (default True)
    """

    def __init__(self, device: "CoolingDevice", noise_params: dict = None):
        noise_params = noise_params or {}
        self._system_qubits = set(device.system_qubits)
        self._bath_qubits   = set(device.bath_qubits)

        self.p1      = noise_params.get('p1', 0.)
        self.p2      = noise_params.get('p2', 0.)
        self.p_reset = noise_params.get('p_reset', 0.)

        self._enabled = {
            'sys':      noise_params.get('sys', True),
            'bath':     noise_params.get('bath', True),
            'coupling': noise_params.get('coupling', True),
        }

        # Single shared channel instance per noise type, reused for every
        # operation -- avoids rebuilding the (probability, matrix) mixture
        # for every noisy gate (see cooling/gates.py CachedDepolarizingChannel).
        self._p1_channel    = _depolarize(self.p1, n_qubits=1) if self.p1 else None
        self._p2_channel    = _depolarize(self.p2, n_qubits=2) if self.p2 else None
        self._reset_channel = _bit_flip(self.p_reset) if self.p_reset else None

    def _subsystem(self, qubits) -> str:
        """Classify an operation's qubits as 'sys', 'bath', or 'coupling'."""
        in_sys  = any(q in self._system_qubits for q in qubits)
        in_bath = any(q in self._bath_qubits for q in qubits)
        if in_sys and in_bath:
            return 'coupling'
        if in_sys:
            return 'sys'
        if in_bath:
            return 'bath'
        raise ValueError(f"Operation on unknown qubits {qubits}")

    def noisy_operation(self, operation: cirq.Operation) -> cirq.OP_TREE:
        if isinstance(operation.gate, cirq.IdentityGate):
            return operation

        qubits = operation.qubits
        if not self._enabled[self._subsystem(qubits)]:
            return operation

        if isinstance(operation.gate, cirq.ResetChannel):
            if self._reset_channel is not None:
                return [operation, self._reset_channel.on(*qubits)]
            return operation

        n = len(qubits)
        if n == 1 and self._p1_channel is not None:
            return [operation, self._p1_channel.on(*qubits)]
        if n == 2 and self._p2_channel is not None:
            return [operation, self._p2_channel.on(*qubits)]
        return operation
