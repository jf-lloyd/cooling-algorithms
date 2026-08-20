"""
Amplitude-damping noise model acting on bath qubits only.

Created by Jerome Lloyd.
"""

import cirq

try:
    from ..gates import CachedAmplitudeDampingChannel
except ImportError:
    CachedAmplitudeDampingChannel = None


def _amp_damp(gamma: float):
    """CachedAmplitudeDampingChannel if available, else cirq.amplitude_damp."""
    if CachedAmplitudeDampingChannel is not None:
        return CachedAmplitudeDampingChannel(gamma)
    return cirq.amplitude_damp(gamma)


class BathAmplitudeDampingNoiseModel(cirq.NoiseModel):
    """
    Amplitude damping (T1 relaxation towards |0>) applied to bath qubits only.

    cirq convention: |0> is the spin-up computational-basis state, so the
    channel relaxes each bath qubit towards |0> with probability gamma. System
    qubits are left noiseless.

    Following the gate-count philosophy of BasicNoiseModel, the damping channel
    is appended after every operation that touches a bath qubit -- the bath
    field (Zeeman) gates and the bath leg of the system-bath coupling gates.
    Only the bath qubit(s) of an operation are damped; the system qubit of a
    coupling gate is untouched.

    Applied via `circuit.with_noise(noise_model)`.

    Parameters
    ----------
    device       : CoolingDevice -- provides the bath-qubit set.
    noise_params : dict, with keys (all optional):
        gamma          : amplitude-damping probability applied to a bath qubit
                         after each operation acting on it (default 0 -> noiseless)
        include_reset  : if True, also damp bath qubits after cirq.reset. Default
                         False -- reset leaves the bath in |0>, the fixed point of
                         the channel, so damping there has no effect.
    """

    def __init__(self, device: "CoolingDevice", noise_params: dict = None):
        noise_params = noise_params or {}
        self._bath_qubits  = set(device.bath_qubits)
        self.gamma         = noise_params.get('gamma', 0.)
        self.include_reset = noise_params.get('include_reset', False)

        # Single shared channel instance reused for every damped bath qubit.
        self._channel = _amp_damp(self.gamma) if self.gamma else None

    def noisy_operation(self, operation: cirq.Operation) -> cirq.OP_TREE:
        if self._channel is None:
            return operation
        if isinstance(operation.gate, cirq.IdentityGate):
            return operation
        if isinstance(operation.gate, cirq.ResetChannel) and not self.include_reset:
            return operation

        bath = [q for q in operation.qubits if q in self._bath_qubits]
        if not bath:
            return operation
        return [operation, *[self._channel.on(q) for q in bath]]
