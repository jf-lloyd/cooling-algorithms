"""
A 2D square lattice.

Created by Jerome Lloyd on 3rd June 2026
"""

from .latticebase import Lattice

class SquareLattice2D(Lattice):
    """
    Lx × Ly rectangular lattice with optional PBC in x and y.
    Site numbering: s = x + y*Lx
    """

    def __init__(self, Lx, Ly, pbc_x=False, pbc_y=False):
        self._Lx = Lx
        self._Ly = Ly
        self._Ns = Lx * Ly
        self.pbc_x = pbc_x
        self.pbc_y = pbc_y

    @property
    def Lx(self): return self._Lx

    @property
    def Ly(self): return self._Ly

    def coords(self, s):
        x = s % self._Lx
        y = s // self._Lx
        return (x, y)

    def index(self, *coords):
        x, y = coords
        return x + y * self._Lx

    def nearest_neighbours(self, s):
        x, y = self.coords(s)

        # +x neighbour
        if x + 1 < self._Lx:
            yield self.index(x + 1, y)
        elif self.pbc_x:
            yield self.index(0, y)

        # -x neighbour
        if x - 1 >= 0:
            yield self.index(x - 1, y)
        elif self.pbc_x:
            yield self.index(self._Lx - 1, y)

        # +y neighbour
        if y + 1 < self._Ly:
            yield self.index(x, y + 1)
        elif self.pbc_y:
            yield self.index(x, 0)

        # -y neighbour
        if y - 1 >= 0:
            yield self.index(x, y - 1)
        elif self.pbc_y:
            yield self.index(x, self._Ly - 1)

    def boundary(self):
        seen = set()
        bry = []
        def add(s):
            if s not in seen:
                seen.add(s)
                bry.append(s)
        if self.pbc_x is False:
            for l in range(self.Ly):
                add(self.index(0, l))
                add(self.index(self.Lx - 1, l))
        if self.pbc_y is False:
            for l in range(self.Lx):
                add(self.index(l, 0))
                add(self.index(l, self.Ly - 1))
        return bry

    def bond_colouring(self):
        """Per-direction even/odd colouring.
 
        X-bonds and Y-bonds never share a layer (so they cannot collide with
        each other); within each direction the bonds form parallel 1D chains
        coloured even/odd by traversal index, with a third "seam" layer for an
        odd-length periodic direction (odd cycle, not 2-colourable).

        """
        Lx, Ly = self._Lx, self._Ly
 
        X_even, X_odd, X_seam = [], [], []
        for y in range(Ly):
            n_bonds = Lx if self.pbc_x else Lx - 1
            for ix in range(n_bonds):
                s, t = self.index(ix, y), self.index((ix + 1) % Lx, y)
                bond = (min(s, t), max(s, t))
                if self.pbc_x and Lx % 2 == 1 and ix == Lx - 1:
                    X_seam.append(bond)
                elif ix % 2 == 0:
                    X_even.append(bond)
                else:
                    X_odd.append(bond)
 
        Y_even, Y_odd, Y_seam = [], [], []
        for x in range(Lx):
            n_bonds = Ly if self.pbc_y else Ly - 1
            for iy in range(n_bonds):
                s, t = self.index(x, iy), self.index(x, (iy + 1) % Ly)
                bond = (min(s, t), max(s, t))
                if self.pbc_y and Ly % 2 == 1 and iy == Ly - 1:
                    Y_seam.append(bond)
                elif iy % 2 == 0:
                    Y_even.append(bond)
                else:
                    Y_odd.append(bond)
 
        layers = [X_even, X_odd, X_seam, Y_even, Y_odd, Y_seam]
        return [layer for layer in layers if layer]