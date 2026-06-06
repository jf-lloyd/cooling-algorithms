"""
Custom gates live here.

Created by Jerome Lloyd on 5th June 2026.
"""

import numpy as np
import cirq

# YXPowGate and ZXPowGate follow the same exponent convention as cirq.XXPowGate:
# gate(exponent=e) implements exp(-i·π/2·e·P⊗X), gate**t scales the exponent by t

class YXPowGate(cirq.EigenGate):
    """
    exp(-i·π/2·exponent·Y⊗X) on (system, bath) qubits.
    Same exponent convention as cirq.XXPowGate.
    """

    def __init__(self, *, exponent: float = 1.0, global_shift: float = -0.5):
        super().__init__(exponent=exponent, global_shift=global_shift)

    def _num_qubits_(self) -> int:
        return 2

    def _eigen_components(self):
        yx = np.array([[0,  0,   0, -1j],
                       [0,  0, -1j,   0],
                       [0, 1j,   0,   0],
                       [1j, 0,   0,   0]])
        return [(0, (np.eye(4) + yx) / 2),
                (1, (np.eye(4) - yx) / 2)]

    def _circuit_diagram_info_(self, args):
        return cirq.CircuitDiagramInfo(wire_symbols=['Y', 'X'], exponent=args.exponent)


class ZXPowGate(cirq.EigenGate):
    """
    exp(-i·π/2·exponent·Z⊗X) on (system, bath) qubits.
    Same exponent convention as cirq.XXPowGate.
    """

    def __init__(self, *, exponent: float = 1.0, global_shift: float = -0.5):
        super().__init__(exponent=exponent, global_shift=global_shift)

    def _num_qubits_(self) -> int:
        return 2

    def _eigen_components(self):
        zx = np.array([[0, 1,  0,  0],
                       [1, 0,  0,  0],
                       [0, 0,  0, -1],
                       [0, 0, -1,  0]])
        return [(0, (np.eye(4) + zx) / 2),
                (1, (np.eye(4) - zx) / 2)]

    def _circuit_diagram_info_(self, args):
        return cirq.CircuitDiagramInfo(wire_symbols=['Z', 'X'], exponent=args.exponent)
