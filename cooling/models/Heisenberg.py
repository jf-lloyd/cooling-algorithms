"""
Anisotropic Heisenberg model.
    H = Jxx sum_{<ij>} X_i X_j + Jyy sum_{<ij>} Y_i Y_j + Jzz sum_{<ij>} Z_i Z_j

Created by Jerome Lloyd on 3rd June 2026
"""

from .modelbase import Model


class HeisenbergModel(Model):
    """
    params:
        'Jxx' : XX coupling strength
        'Jyy' : YY coupling strength
        'Jzz' : ZZ coupling strength
    """

    _ACCEPTED_PARAMS = {'Jxx', 'Jyy', 'Jzz'}

    def __init__(self, device: "CoolingDevice", params: dict):
        self.params = params
        self.Jxx = params.get('Jxx', 0.)
        self.Jyy = params.get('Jyy', 0.)
        self.Jzz = params.get('Jzz', 0.)
        super().__init__(device)

    @property
    def name(self):
        return f"HeisenbergModel_{self.lattice.name}_Jxx{self.Jxx:.3f}_Jyy{self.Jyy:.3f}_Jzz{self.Jzz:.3f}"

    def build_coupling_lists(self):
        pairs = self._lattice.nearest_neighbour_pairs()
        couplings = {}
        if self.Jxx != 0.: couplings['XX'] = [(self.Jxx, s, t) for s, t in pairs]
        if self.Jyy != 0.: couplings['YY'] = [(self.Jyy, s, t) for s, t in pairs]
        if self.Jzz != 0.: couplings['ZZ'] = [(self.Jzz, s, t) for s, t in pairs]
        return couplings
