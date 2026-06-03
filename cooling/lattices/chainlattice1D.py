"""
A 1D lattice.

Created by Jerome Lloyd on 3rd June 2026
"""

from .latticebase import Lattice

class ChainLattice1D(Lattice):
    """
    1D chain of length L. Periodic boundary conditions optional.
    Sites labelled 0,...,L−1.
    """

    def __init__(self, L: int, pbc: bool = False):
        self._L = L
        self._Ns = L
        self.pbc = pbc

    @property
    def L(self):
        return self._L

    @property
    def name(self):
        return f"chain1D_L{self._L}"

    def coords(self, s):
        return (s,)

    def index(self, *coords):
        (x,) = coords
        return x

    def nearest_neighbours(self, s):
        # +1 neighbour
        if s + 1 < self.L:
            yield s + 1
        elif self.pbc:
            yield 0

        # -1 neighbour
        if s - 1 >= 0:
            yield s - 1
        elif self.pbc:
            yield self.L - 1

    def is_wrap_bond(self, s: int, t: int) -> bool:
        return self.pbc and abs(s - t) > 1

    def boundary(self):
        if self.pbc:
            return []
        else:
            return [0, self.L-1]

    def bond_colouring(self):
        """Even/odd 2-colouring.
    
        - open chain or even-length ring: 2 layers (minimal).
        - odd-length ring: an odd cycle is not 2-colourable, so the single
          bond that would collide is placed in a third layer.
    
        """
        n_bonds = self._L if (self.pbc and self._L > 2) else self._L - 1
        even, odd, seam = [], [], []
        for ix in range(n_bonds):
            s, t = ix, (ix + 1) % self._L
            if s > t:
                s, t = t, s
            bond = (s, t)
            if self.pbc and self._L % 2 == 1 and ix == self._L - 1:
                seam.append(bond)          # wrap bond on an odd ring
            elif ix % 2 == 0:
                even.append(bond)
            else:
                odd.append(bond)
        return [layer for layer in (even, odd, seam) if layer]
