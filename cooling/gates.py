"""
Custom gates live here.

Created by Jerome Lloyd on 5th June 2026.
"""

import functools
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


# ── Cached noise channels ────────────────────────────────────────────────────
#
# cirq.depolarize(p, n_qubits)._mixture_() and cirq.bit_flip(p)._mixture_()
# rebuild their (probability, Pauli matrix) list via np.kron from scratch on
# every call, with no caching. qsim's circuit translation calls
# cirq.mixture() once per noise op per distinct circuit, so for circuits with
# many noise ops (one per gate, via BasicNoiseModel) this dominates
# translation time. The matrices depend only on (p, n_qubits) -- not on
# anything circuit-specific -- so precompute once and cache by (p, n_qubits).

_PAULI = {
    'I': np.array([[1, 0], [0, 1]], dtype=complex),
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.array([[1, 0], [0, -1]], dtype=complex),
}


@functools.lru_cache(maxsize=None)
def _depolarizing_mixture(p: float, n_qubits: int) -> tuple:
    """((prob, matrix), ...) for an n-qubit depolarizing channel: identity
    with probability 1-p, each of the 4**n_qubits - 1 non-identity Pauli
    strings with probability p / (4**n_qubits - 1)."""
    n_paulis = 4 ** n_qubits
    p_each = p / (n_paulis - 1)
    labels = 'IXYZ'
    mixture = []
    for idx in range(n_paulis):
        mat = np.array([[1.]], dtype=complex)
        x = idx
        for _ in range(n_qubits):
            mat = np.kron(mat, _PAULI[labels[x % 4]])
            x //= 4
        prob = (1.0 - p) if idx == 0 else p_each
        mixture.append((prob, mat))
    return tuple(mixture)


@functools.lru_cache(maxsize=None)
def _bit_flip_mixture(p: float) -> tuple:
    """((1-p, I), (p, X))."""
    return ((1.0 - p, _PAULI['I'].copy()), (p, _PAULI['X'].copy()))


class CachedDepolarizingChannel(cirq.Gate):
    """
    Depolarizing channel with a precomputed, cached mixture.

    Equivalent to cirq.depolarize(p, n_qubits=n_qubits), but _mixture_
    returns a precomputed tuple instead of rebuilding the Pauli matrices
    via np.kron on every call.
    """

    def __init__(self, p: float, n_qubits: int = 1):
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.p = float(p)
        self.n_qubits = int(n_qubits)
        self._mix = _depolarizing_mixture(self.p, self.n_qubits)

    def _num_qubits_(self) -> int:
        return self.n_qubits

    def _mixture_(self):
        return self._mix

    def _circuit_diagram_info_(self, args):
        return cirq.CircuitDiagramInfo(wire_symbols=(f"D({self.p:.2g})",) * self.n_qubits)

    def __repr__(self):
        return f"CachedDepolarizingChannel(p={self.p!r}, n_qubits={self.n_qubits!r})"


class CachedBitFlipChannel(cirq.Gate):
    """
    Bit-flip channel with a precomputed, cached mixture.

    Equivalent to cirq.bit_flip(p), but _mixture_ returns a precomputed
    tuple instead of rebuilding the matrices on every call.
    """

    def __init__(self, p: float):
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.p = float(p)
        self._mix = _bit_flip_mixture(self.p)

    def _num_qubits_(self) -> int:
        return 1

    def _mixture_(self):
        return self._mix

    def _circuit_diagram_info_(self, args):
        return cirq.CircuitDiagramInfo(wire_symbols=(f"BF({self.p:.2g})",))

    def __repr__(self):
        return f"CachedBitFlipChannel(p={self.p!r})"
