"""
Transverse-field Ising model with optional longitudinal field.
    H = J sum_{<ij>} X_i X_j  -  g sum_i Z_i  -  gx sum_i X_i

Created by Jerome Lloyd on 3rd June 2026
"""

from .modelbase import Model


class IsingModel(Model):
    """
    params:
        'J'  : XX coupling strength
        'g'  : transverse Z field
        'gx' : longitudinal X field (default 0.)
    """

    _ACCEPTED_PARAMS = {'J', 'g', 'gx'}

    def __init__(self, device: "CoolingDevice", params: dict):
        self.params = params
        self.J  = params.get('J',  0.)
        self.g  = params.get('g',  0.)
        self.gx = params.get('gx', 0.)
        super().__init__(device)

    @property
    def name(self):
        return f"IsingModel_{self.lattice.name}_J{self.J:.3f}_g{self.g:.3f}_gx{self.gx:.3f}"

    def build_coupling_lists(self):
        """ couplings of Hamiltonian. Note that the build order of couplings controls the order gates are run 
        (if base build_system_layer is not overwritten) -- so here we apply XX gates, then Z, then X """
        pairs = self._lattice.nearest_neighbour_pairs()
        couplings = {}
        if self.J  != 0.: couplings['XX'] = [(self.J,   s, t) for s, t in pairs]
        if self.g  != 0.: couplings['Z']  = [(-self.g,  s) for s in range(self._Ns)]
        if self.gx != 0.: couplings['X']  = [(-self.gx, s) for s in range(self._Ns)]
        return couplings

    def build_system_layer(self, order: int = 1) -> list:
        """
        Build the Ising system layer.

        For gx=0, use the cheaper second-order split
            Z(t/2) XX(t) Z(t/2)
        because all XX bond terms commute with each other. This avoids the
        generic symmetric split duplicating the expensive XX bond-colour layers.
        """
        if order not in (1, 2):
            raise ValueError(f"System Trotter order must be 1 or 2, got {order!r}.")

        if order == 1 or self.gx != 0. or self.J == 0. or self.g == 0.:
            return super().build_system_layer(order=order)

        qubits = self._device.system_qubits
        cl = self._coupling_lists

        z_half = [
            self._GATE_MAP['Z'](strength)(qubits[s])**0.5
            for strength, s in cl.get('Z', [])
            if strength != 0.
        ]

        bond_strength = {}
        for J, s, t in cl.get('XX', []):
            bond_strength[(s, t)] = J
            bond_strength[(t, s)] = J

        xx_ops = []
        for layer in self._lattice.bond_colouring():
            for s, t in layer:
                J = bond_strength.get((s, t), 0.)
                if J != 0.:
                    xx_ops.append(self._GATE_MAP['XX'](J)(qubits[s], qubits[t]))

        return list(z_half) + xx_ops + list(z_half)
