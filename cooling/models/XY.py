"""
XY model.
    H = Jxx sum_{<ij>} X_i X_j + Jyy sum_{<ij>} Y_i Y_j

Created by Jerome Lloyd on 6th June 2026
"""

from .modelbase import Model


class XYModel(Model):
    """
    params:
        'Jxx' : XX coupling strength
        'Jyy' : YY coupling strength
    """

    _ACCEPTED_PARAMS = {'Jxx', 'Jyy'}

    def __init__(self, device: "CoolingDevice", params: dict):
        self.params = params
        self.Jxx = params.get('Jxx', 0.)
        self.Jyy = params.get('Jyy', 0.)
        super().__init__(device)

    @property
    def name(self):
        return f"XYModel_{self.lattice.name}_Jxx{self.Jxx:.3f}_Jyy{self.Jyy:.3f}"

    def build_coupling_lists(self):
        pairs = self._lattice.nearest_neighbour_pairs()
        couplings = {}
        if self.Jxx != 0.: couplings['XX'] = [(self.Jxx, s, t) for s, t in pairs]
        if self.Jyy != 0.: couplings['YY'] = [(self.Jyy, s, t) for s, t in pairs]
        return couplings
