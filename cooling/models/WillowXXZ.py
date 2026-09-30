"""
XXZ model expressed natively in Google's Willow two-qubit gate.

    H = J sum_{<ij>} ( X_i X_j + Y_i Y_j + Delta Z_i Z_j )
        + hx sum_i X_i + hy sum_i Y_i + hz sum_i Z_i

The bond term is not an independent choice here: it is whatever the hardware
entangler generates. cirq_google.ops.WILLOW is FSimGate(theta_W, phi_W) with
theta_W = pi/2, phi_W = pi/9, and exactly

    FSim(theta, phi) = exp( -i [ (theta/2)(XX+YY) + (phi/4)(ZZ - Z1 - Z2 + I) ] )

the two generators commuting because XX+YY lives entirely in the {|01>,|10>}
subspace, where ZZ = -1 and Z1 + Z2 = 0. Powers therefore act as
FSim(theta, phi)**s = FSim(s*theta, s*phi), so the anisotropy

    Delta = (phi/4) / (theta/2) = phi_W / (2 theta_W) = 1/9

is FIXED for the whole one-parameter family: the Trotter angle can be dialled
down freely, the anisotropy cannot. Delta = +1/9 with J > 0 is an easy-plane
antiferromagnet, well inside the gapless XY phase in 1D.

A bond of strength J is realised as WILLOW**(2J/theta_W), with the gate's
built-in single-qubit Z rotations undone by explicit rz gates (they commute
with the FSim exactly, so this is not a Trotter approximation). The only
two-qubit gate in the system layer is Willow.

Because the layer is already native, it must NOT be recompiled to a CZ target
gateset: this model sets allow_compile = False, which protocol channels honour
regardless of their own compile argument.

Created by Jerome Lloyd on 30th September 2026
"""

import cirq
import numpy as np

from .modelbase import Model

# Nominal Willow parameters. The cirq_google docstring warns that phi drifts
# from processor to processor and that the gate is retargeted to an
# "ISWAP-like" gate on hardware, so Delta = 1/9 is nominal, not guaranteed.
#
# cirq_google.ops.WILLOW is exactly FSimGate(pi/2, pi/9), so the gate is built
# from plain cirq here and cirq_google is not a dependency of this module --
# importing it drags in the google-cloud engine stack, which is not installed
# (or not importable) in every environment we simulate in. Where it IS
# importable we take the values from it, so that a future change to the
# nominal phi propagates.
WILLOW_THETA = np.pi / 2
WILLOW_PHI   = np.pi / 9
try:  # pragma: no cover - depends on local cirq_google install
    import cirq_google as _cg
    WILLOW_THETA = float(_cg.ops.WILLOW.theta)
    WILLOW_PHI   = float(_cg.ops.WILLOW.phi)
except Exception:
    pass

WILLOW_DELTA = WILLOW_PHI / (2 * WILLOW_THETA)       # 1/9

#: The Willow entangler, equal to cirq_google.ops.WILLOW.
WILLOW = cirq.FSimGate(theta=WILLOW_THETA, phi=WILLOW_PHI)


class WillowXXZModel(Model):
    """
    params:
        'J'  : bond strength, multiplying (XX + YY + Delta ZZ). Default 1.
        'hx' : transverse X field (default 0.)
        'hy' : transverse Y field (default 0.)
        'hz' : longitudinal Z field (default 0.)

    Delta is not a parameter -- it is fixed at WILLOW_DELTA = 1/9 by the gate.
    """

    _ACCEPTED_PARAMS = {'J', 'hx', 'hy', 'hz'}

    # native gate layer; protocol channels must not retarget it to CZs
    allow_compile = False

    def __init__(self, device: "CoolingDevice", params: dict):
        self.params = params
        self.J     = params.get('J',  1.)
        self.hx    = params.get('hx', 0.)
        self.hy    = params.get('hy', 0.)
        self.hz    = params.get('hz', 0.)
        self.Delta = WILLOW_DELTA
        super().__init__(device)

    @property
    def name(self):
        return (f"WillowXXZModel_{self.lattice.name}_J{self.J:.3f}"
                f"_hx{self.hx:.3f}_hy{self.hy:.3f}_hz{self.hz:.3f}")

    def build_coupling_lists(self):
        pairs = self._lattice.nearest_neighbour_pairs()
        couplings = {}
        if self.J != 0.:
            couplings['XX'] = [(self.J, s, t) for s, t in pairs]
            couplings['YY'] = [(self.J, s, t) for s, t in pairs]
            couplings['ZZ'] = [(self.J * self.Delta, s, t) for s, t in pairs]
        if self.hx != 0.: couplings['X'] = [(self.hx, s) for s in range(self._Ns)]
        if self.hy != 0.: couplings['Y'] = [(self.hy, s) for s in range(self._Ns)]
        if self.hz != 0.: couplings['Z'] = [(self.hz, s) for s in range(self._Ns)]
        return couplings

    @staticmethod
    def willow_bond_exponent(J: float) -> float:
        """Power s such that WILLOW**s generates J(XX + YY + Delta ZZ) + local Zs."""
        return 2 * J / WILLOW_THETA

    def build_system_layer(self, order: int = 1) -> list:
        """
        Emit one Willow gate per bond plus the single-qubit rz gates that cancel
        the -(phi/4)(Z1 + Z2) piece the gate carries. The rz commutes with the
        FSim (which conserves total Z), so bond gate and correction together
        implement exp(-i J (XX + YY + Delta ZZ)) exactly, up to a global phase.

        Field terms use the base class _GATE_MAP. Group structure and the
        second-order symmetric split follow the base class.
        """
        if order not in (1, 2):
            raise ValueError(f"System Trotter order must be 1 or 2, got {order!r}.")

        cl = self._coupling_lists
        qubits = self._device.system_qubits
        groups = []

        if self.J != 0.:
            s = self.willow_bond_exponent(self.J)
            # cancels exp(+i s (phi/4)(Z1 + Z2)); _GATE_MAP['Z'](g) is exp(-i g Z)
            g_comp = s * WILLOW_PHI / 4
            for layer in self._lattice.bond_colouring():
                group = []
                for u, v in layer:
                    group.append(WILLOW(qubits[u], qubits[v])**s)
                    group.append(self._GATE_MAP['Z'](g_comp)(qubits[u]))
                    group.append(self._GATE_MAP['Z'](g_comp)(qubits[v]))
                if group:
                    groups.append(group)

        for op_str in ('X', 'Y', 'Z'):
            group = [self._GATE_MAP[op_str](strength)(qubits[s])
                     for strength, s in cl.get(op_str, []) if strength != 0.]
            if group:
                groups.append(group)

        if order == 1 or len(groups) <= 1:
            return [gate for group in groups for gate in group]

        return (
            [gate**0.5 for group in groups[:-1] for gate in group]
            + list(groups[-1])
            + [gate**0.5 for group in reversed(groups[:-1]) for gate in group]
        )
