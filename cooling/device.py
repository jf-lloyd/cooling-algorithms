'''
defines classes:
Lattice - defines lattice object, dimensionality, graph connectivity. Used by Device and Hamiltonian
Device - add qubit structure on top of lattice (qubits, auxiliaries) 
'''

from abc import ABC, abstractmethod    
import cirq

class Lattice(ABC):
    """
    Abstract lattice of Ns sites.

    Required:
      - Ns
      - coords(s)
      - index(coords)
      - nearest_neighbours(s)
    """

    def __init__(self):
        self._Ns = None
        self._Dim = None

    @property
    def Ns(self):
        """Total number of sites."""
        return self._Ns

    @property
    def Dim(self):
        """Spatial dimension."""
        return self._Dim

    @abstractmethod
    def coords(self, s):
        """Return coordinates (x,y,z..) for site index s."""
        pass

    @abstractmethod
    def index(self, *coords):
        """Return site index corresponding to coordinates."""
        pass

    @abstractmethod
    def nearest_neighbours(self, s):
        """Yield all nearest neighbours of site s."""
        pass

    def nearest_neighbour_pairs(self):
        """Return undirected neighbour pairs (s, t) with s < t."""
        pairs = []
        for s in range(self.Ns):
            for t in self.nearest_neighbours(s):
                if s < t:
                    pairs.append((s, t))
        return pairs

    @abstractmethod
    def boundary(self):
        """Return list of boundary sites (empty list if pbc)"""
        pass

    def draw(self):
        """Basic lattice graph visualisation."""

        import networkx as nx
        G = nx.Graph()
        pos = {}

        for s in range(self.Ns):
            pos[s] = list(self.coords(s))
            G.add_node(s)

        for s, t in self.nearest_neighbour_pairs():
            G.add_edge(s, t)

        nx.draw(G, pos=pos, node_color="k")


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


class RectLattice2D(Lattice):
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

class Device(cirq.Device):
    """
    Cirq device built on Lattice.
    """

    def __init__(self, lattice, Nb):
        self.lattice = lattice
        self._Ns = lattice.Ns
        self._Nb = Nb

        # System qubits
        system_qubits = []
        for s in range(self._Ns):
            coords = lattice.coords(s)
            if len(coords) == 1:
                # 1D: coords = (x,)
                x = coords[0]
                y = 0
            elif len(coords) == 2:
                # 2D: coords = (x, y,)
                x, y = coords[:2]
            else:
                # 3D: need to choose embedding, not implemented
                raise ValueError("LatticeDevice only supports 1D or 2D embeddings so far.")
            
            system_qubits.append(cirq.GridQubit(y, x))
        self._system_qubits = system_qubits

        # Map qubits → indices
        self._qubit_index_map = {q: s for s, q in enumerate(self._system_qubits)}

        # Bath qubits 
        self._bath_qubits = list(cirq.LineQubit.range(Nb))

    @property
    def Ns(self): return self._Ns

    @property
    def Nb(self): return self._Nb
    
    @property
    def system_qubits(self):
        return self._system_qubits

    @property
    def bath_qubits(self):
        return self._bath_qubits

    @property
    def qubit_index_map(self):
        return self._qubit_index_map

    def draw(self):
        """
        Draw system qubits at lattice coords and bath qubits to the right.
        """
        import networkx as nx

        G = nx.Graph()
        pos = {}
        colors = []

        # System qubits
        for s, q in enumerate(self._system_qubits):
            coords = self.lattice.coords(s)
            if len(coords) == 1:
                x = coords[0]
                y = 0
            else:
                x, y = coords[:2]

            G.add_node(q)
            pos[q] = [x, y]
            colors.append("k")

        # Edges between system qubits according to lattice structure
        for s, t in self.lattice.nearest_neighbour_pairs():
            qs = self._system_qubits[s]
            qt = self._system_qubits[t]
            G.add_edge(qs, qt)

        # Bath qubits placed at x = max_x + 1
        max_x = max(pos[q][0] for q in self._system_qubits)

        for i, a in enumerate(self._bath_qubits):
            G.add_node(a)
            pos[a] = [max_x + 1, i]
            colors.append("r")

        nx.draw(G, pos=pos, node_color=colors)
        
