from .base import Lattice

class SquareLattice2D(Lattice):
    """
    Lx × Ly rectangular lattice with optional PBC in x and y.
    Site numbering: s = x + y*Lx
    """

    def __init__(self, Lx, Ly, pbc_x=False, pbc_y=False):
        self._Lx = Lx
        self._Ly = Ly
        self._Ns = Lx * Ly
        self._Dim = 2
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
        bry = []
        if self.pbc_x is False:
            for l in range(self.Ly):
                bry.append(self.index(0, l))
                bry.append(self.index(self.Lx-1,l))
        if self.pbc_y is False:
            for l in range(self.Lx):
                bry.append(self.index(l, 0))
                bry.append(self.index(l, self.Ly-1))
        return bry