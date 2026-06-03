"""
Base class Lattice object. Defines geometry and qubit-qubit connectivity for system. 
Provides a "bond colouring" which partitions system bonds into commuting/disjoint sets (used later for evolution)
"""

from abc import ABC, abstractmethod    

class Lattice(ABC):
    """
    write
    """

    def __init__(self):
        self._Ns = None

    @property
    def Ns(self):
        """Total number of sites."""
        return self._Ns

    @abstractmethod
    def coords(self, s):
        """Return coordinates (x,y,z..) for site index s."""
        pass

    @abstractmethod
    def index(self, *coords):
        """Return site index s corresponding to coordinates (x,y,z..)."""
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
    def bond_colouring(self):
        """Partition bonds into layers where no two bonds share a site
        (proper edge-colouring). Returns list[list[(s, t)]] with s < t."""
        pass

    def _greedy_colouring(self):
        """Generic greedy edge-colouring from connectivity alone."""
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(self.Ns))
        G.add_edges_from(self.nearest_neighbour_pairs())
        colour = nx.greedy_color(nx.line_graph(G), strategy="largest_first")
        layers = {}
        for edge, c in colour.items():
            s, t = sorted(edge)
            layers.setdefault(c, []).append((s, t))
        return [layers[c] for c in sorted(layers)]

