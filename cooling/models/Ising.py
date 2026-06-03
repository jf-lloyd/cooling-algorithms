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
