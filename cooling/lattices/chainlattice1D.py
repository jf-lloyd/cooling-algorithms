from .base import Lattice

class ChainLattice1D(Lattice):
    """
    1D chain of length L. Periodic boundary conditions optional.
    Sites labelled 0,...,L−1.
    """

    def __init__(self, L, pbc=False):
        self._L = L
        self._Ns = L
        self._Dim = 1
        self.pbc = pbc

    @property
    def L(self):
        return self._L

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

    def boundary(self):
        if self.pbc:
            return []
        else:
            return [0, self.L-1]
