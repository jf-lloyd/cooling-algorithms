"""
A 2D triangular lattice.

Created by Jerome Lloyd on 3rd June 2026
"""

import numpy as np
from .latticebase import Lattice


class TriangularLattice2D(Lattice):
    """
    Lx × Ly triangular lattice with optional PBC in x and y.
    Site numbering: s = x + y*Lx

    6 neighbours per interior site, in directions:
        (±1, 0),  (0, ±1),  (+1, -1),  (-1, +1)

    Drawing uses equilateral-triangle Cartesian coords:
        draw_coords(x, y) = (x + 0.5*y,  y*sqrt(3)/2)
    """

    def __init__(self, Lx: int, Ly: int, pbc_x: bool = False, pbc_y: bool = False):
        if Lx < 2 or Ly < 2:
            raise ValueError(f"TriangularLattice2D requires Lx, Ly ≥ 2 (got {Lx}×{Ly}).")
        self._Lx = Lx
        self._Ly = Ly
        self._Ns = Lx * Ly
        self.pbc_x = pbc_x
        self.pbc_y = pbc_y

    @property
    def Lx(self): return self._Lx

    @property
    def Ly(self): return self._Ly

    @property
    def name(self):
        return f"triangular2D_Lx{self._Lx}Ly{self._Ly}"

    def coords(self, s: int):
        return (s % self._Lx, s // self._Lx)

    def draw_coords(self, s: int):
        x, y = self.coords(s)
        return (x + 0.5 * y, y * np.sqrt(3) / 2)

    def index(self, *coords):
        x, y = coords
        return x + y * self._Lx

    def nearest_neighbours(self, s: int):
        x, y = self.coords(s)
        Lx, Ly = self._Lx, self._Ly
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]:
            nx, ny = x + dx, y + dy
            if self.pbc_x:
                nx = nx % Lx
            elif not (0 <= nx < Lx):
                continue
            if self.pbc_y:
                ny = ny % Ly
            elif not (0 <= ny < Ly):
                continue
            yield self.index(nx, ny)

    def is_wrap_bond(self, s: int, t: int) -> bool:
        xs, ys = self.coords(s)
        xt, yt = self.coords(t)
        return (self.pbc_x and abs(xt - xs) > 1) or (self.pbc_y and abs(yt - ys) > 1)

    def boundary(self):
        seen = set()
        bry = []
        def add(s):
            if s not in seen:
                seen.add(s)
                bry.append(s)
        if not self.pbc_x:
            for y in range(self._Ly):
                add(self.index(0, y))
                add(self.index(self._Lx - 1, y))
        if not self.pbc_y:
            for x in range(self._Lx):
                add(self.index(x, 0))
                add(self.index(x, self._Ly - 1))
        return bry

    def bond_colouring(self):
        """
        3-direction even/odd colouring: horizontal (A), vertical (B), diagonal (C).
        Each direction splits into even/odd layers, giving up to 6 layers total
        (plus seam layers for odd-length periodic directions).
        """
        Lx, Ly = self._Lx, self._Ly

        # Direction A: (x, y) -> (x+1, y)
        A_even, A_odd, A_seam = [], [], []
        for y in range(Ly):
            n_bonds = Lx if (self.pbc_x and Lx > 2) else Lx - 1
            for ix in range(n_bonds):
                s = self.index(ix, y)
                t = self.index((ix + 1) % Lx, y)
                bond = (min(s, t), max(s, t))
                if self.pbc_x and Lx % 2 == 1 and ix == Lx - 1:
                    A_seam.append(bond)
                elif ix % 2 == 0:
                    A_even.append(bond)
                else:
                    A_odd.append(bond)

        # Direction B: (x, y) -> (x, y+1)
        B_even, B_odd, B_seam = [], [], []
        for x in range(Lx):
            n_bonds = Ly if (self.pbc_y and Ly > 2) else Ly - 1
            for iy in range(n_bonds):
                s = self.index(x, iy)
                t = self.index(x, (iy + 1) % Ly)
                bond = (min(s, t), max(s, t))
                if self.pbc_y and Ly % 2 == 1 and iy == Ly - 1:
                    B_seam.append(bond)
                elif iy % 2 == 0:
                    B_even.append(bond)
                else:
                    B_odd.append(bond)

        # Direction C: (x, y) -> (x+1, y-1).  Odd periodic x-rings need a
        # separate seam layer, just like the horizontal/vertical directions.
        C_even, C_odd, C_seam = [], [], []
        for x in range(Lx):
            for y in range(Ly):
                if not self.pbc_x and x + 1 >= Lx:
                    continue
                if not self.pbc_y and y - 1 < 0:
                    continue
                s = self.index(x, y)
                t = self.index((x + 1) % Lx, (y - 1) % Ly)
                bond = (min(s, t), max(s, t))
                if self.pbc_x and Lx % 2 == 1 and x == Lx - 1:
                    C_seam.append(bond)
                elif x % 2 == 0:
                    C_even.append(bond)
                else:
                    C_odd.append(bond)

        layers = [A_even, A_odd, A_seam, B_even, B_odd, B_seam, C_even, C_odd, C_seam]
        return [layer for layer in layers if layer]
