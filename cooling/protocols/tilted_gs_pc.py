"""
Tilted staggered coupling frame for the ground-state protocol.

TiltedGroundStateProtocol is a GroundStateProtocol whose extraction axis on
system site i is rotated to cos(phi) z + s_i sin(phi) x, with the sublattice
sign s_i from a 2-colouring of the bond graph. The bath field and the bath
preparation are co-rotated, so the pumped quantity is the staggered order
parameter rather than the total z magnetisation.

Motivation: the exchange coupling plus per-cycle reset is a one-way pump whose
target is set by the quantisation axis. In the paramagnet the all-|0> target is
near the ground state; deep in the ordered phase it is not, and half the
coupling weight becomes diagonal in the ordered background.

The frame is applied by single-qubit conjugation only, so the compiled
two-qubit gate count is unchanged. phi = 0 reproduces GroundStateProtocol gate
for gate.

Physics and implementation by Yuxuan Zhang (originally as a modification of
GroundStateProtocol itself); refactored here into a subclass so that the base
protocol is left untouched.
"""

import numpy as np
import cirq

from .ground_pc import GroundStateProtocol


def _sublattice_signs(lattice, Ns):
    """s_i = +/-1 from a 2-colouring of the bond graph (BFS). Raises if not bipartite."""
    adj = {i: [] for i in range(Ns)}
    for a, b in lattice.nearest_neighbour_pairs():
        adj[a].append(b); adj[b].append(a)
    col = {}
    for root in range(Ns):
        if root in col: continue
        col[root] = 0; stack = [root]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in col:
                    col[v] = 1 - col[u]; stack.append(v)
                elif col[v] == col[u]:
                    raise ValueError(
                        "Lattice is not bipartite (odd cycle through sites "
                        f"{u},{v}); a staggered frame (phi != 0) is undefined. "
                        "Note e.g. an odd-by-odd periodic square lattice is NOT bipartite.")
    return [1 if col[i] == 0 else -1 for i in range(Ns)]


def _frame_rotation(phi, sign=1):
    """R = exp(-i * sign * (phi/2) Y): rotates the quantisation axis to
    cos(phi) z + sign*sin(phi) x."""
    c, s = np.cos(sign * phi / 2), np.sin(sign * phi / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


class _FramedGate(cirq.Gate):
    """A gate conjugated by a fixed local basis change, U -> R U R^dag.

    Powers commute with conjugation, (R U R^dag)^t = R U^t R^dag, so this stays a
    drop-in replacement wherever the protocol writes ``gate ** delta``.
    """

    def __init__(self, base: cirq.Gate, rot: np.ndarray):
        self._base, self._rot = base, rot

    def _num_qubits_(self):
        return cirq.num_qubits(self._base)

    def _unitary_(self):
        u = cirq.unitary(self._base)
        return self._rot @ u @ self._rot.conj().T

    def __pow__(self, t):
        b = self._base ** t
        return NotImplemented if b is NotImplemented else _FramedGate(b, self._rot)

    def _circuit_diagram_info_(self, args):
        n = cirq.num_qubits(self._base)
        return cirq.CircuitDiagramInfo(wire_symbols=tuple(f"F[{self._base}]" for _ in range(n)))

    def __repr__(self):
        return f"_FramedGate({self._base!r})"


class TiltedGroundStateProtocol(GroundStateProtocol):
    """
    GroundStateProtocol in a tilted staggered coupling frame.

    Extra param (in addition to everything GroundStateProtocol takes):
        phi : float (default 0) — frame angle in radians. The extraction axis on
              system site i becomes cos(phi) z + s_i sin(phi) x with s_i the
              sublattice sign; the bath field and bath preparation are co-rotated.
              phi = 0 reproduces GroundStateProtocol exactly.

    Useful in symmetry-broken phases, where pumping along the local order
    parameter rather than along z is what the dissipator should do.

    Note phi is read from the *resolved* params, so a per-call override passed to
    channel(params={'phi': ...}) takes effect -- this is what lets a schedule
    build both sectors (phi and -phi) from one protocol instance.
    """

    _PHI_TOL = 1e-12

    # phi lands next to the other protocol params in the filename (after theta),
    # matching the naming of data generated before this was split into a subclass.
    _NAME_KEYS = GroundStateProtocol._NAME_KEYS + [('phi', 'phi{:.3f}')]

    @property
    def sublattice_signs(self):
        """s_i = +/-1 per system site, from a 2-colouring of the bond graph (cached).
        Flipping all signs selects the other Neel sector."""
        if getattr(self, "_sub_signs", None) is None:
            self._sub_signs = _sublattice_signs(self.model.lattice, self.device.Ns)
        return self._sub_signs

    def _phi(self, params):
        """Frame angle from the resolved params (0 if absent)."""
        if not params:
            return float(self.params.get("phi", 0.))
        return float(params.get("phi", self.params.get("phi", 0.)))

    def _pre_cycle_layer(self, params: dict):
        """Prepare the bath in the ground state of the tilted field. Done at the START
        of the cycle (not after the reset) so that the bath is left in a definite
        computational state, which Simulation.get_system_state requires."""
        phi = self._phi(params)
        if abs(phi) < self._PHI_TOL:
            return []
        Rb = _frame_rotation(phi)
        return [cirq.MatrixGate(Rb)(b) for b in self.device.bath_qubits]

    def _get_bath_layer(self, h: float, params: dict = None):
        """Zeeman splitting on the bath qubits, quantised along the frame axis
        cos(phi) z + sin(phi) x."""
        phi = self._phi(params)
        if abs(phi) < self._PHI_TOL:
            return super()._get_bath_layer(h, params)
        R = _frame_rotation(phi)
        return [_FramedGate(cirq.rz(-h), R)(b) for b in self.device.bath_qubits]

    def _get_coupling_layer(self, coupling_geometry: dict, coupling_ops: dict, theta: float,
                            params: dict = None):
        """
        Coupling gates with each site's extraction axis rotated to
        cos(phi) z + s_i sin(phi) x (sublattice sign s_i), so the pumped quantity is
        the staggered order parameter. The conjugation is single-qubit only: the
        compiled two-qubit gate count is unchanged.
        """
        phi = self._phi(params)
        if abs(phi) < self._PHI_TOL:
            return super()._get_coupling_layer(coupling_geometry, coupling_ops, theta, params)
        S  = self.device.system_qubits
        B  = self.device.bath_qubits
        sb = self.coupling_gates(coupling_ops)
        signs = self.sublattice_signs
        Rb = _frame_rotation(phi)
        ops = []
        for bi, si in coupling_geometry.items():
            RR = np.kron(_frame_rotation(phi, signs[si]), Rb)
            ops.append(_FramedGate(sb[bi](exponent=2 / np.pi * theta), RR)(S[si], B[bi]))
        return ops
