from abc import ABC, abstractmethod    

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
