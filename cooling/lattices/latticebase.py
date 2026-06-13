"""
Base class Lattice object. Defines geometry and qubit-qubit connectivity for system.
Provides a "bond colouring" which partitions system bonds into commuting/disjoint sets (used later for evolution)

Created by Jerome Lloyd on 3rd June 2026
"""

from abc import ABC, abstractmethod    

class Lattice(ABC):
    """Abstract base class defining qubit geometry and nearest-neighbour connectivity."""

    def __init__(self):
        self._Ns = None

    @property
    def Ns(self):
        """Total number of sites."""
        return self._Ns

    @property
    @abstractmethod
    def name(self):
        """Return a string identifier for this lattice (e.g. 'chain1D_L8')."""
        pass

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

    def is_wrap_bond(self, s: int, t: int) -> bool:
        """Return True if bond (s, t) crosses a periodic boundary. Default: False."""
        return False

    def draw_coords(self, s: int):
        """Cartesian coordinates for visualisation. Defaults to coords(s).
        Override for non-rectangular geometries (e.g. triangular lattice)."""
        return self.coords(s)

    def stub_endpoints(self, s: int, t: int, stub_length: float) -> tuple:
        """
        For a wrap (PBC) bond (s, t), return stub endpoint positions (gs, gt)
        for visualisation. Stubs point outward from each node in the direction
        the bond exits the unit cell.

        The base implementation works for lattices where draw_coords equals
        lattice coords (rectangular geometries). Override for lattices with
        coordinate offsets (e.g. triangular).
        """
        raw_s = self.draw_coords(s)
        raw_t = self.draw_coords(t)
        # Pad to 2D so callers can always index [0] and [1].
        cs = (raw_s[0], raw_s[1] if len(raw_s) > 1 else 0)
        ct = (raw_t[0], raw_t[1] if len(raw_t) > 1 else 0)
        gs = list(cs)
        gt = list(ct)
        for d in range(2):
            diff = ct[d] - cs[d]
            if abs(diff) > 1:
                sign = 1 if diff > 0 else -1
                gs[d] = cs[d] - sign * stub_length
                gt[d] = ct[d] + sign * stub_length
        return tuple(gs), tuple(gt)

    def nearest_neighbour_pairs(self):
        """Return undirected neighbour pairs (s, t) with s < t, each bond listed once.

        Some lattices (e.g. a 2-site PBC chain) yield the same neighbour twice
        from nearest_neighbours(s) since the "+1" and "-1" directions coincide;
        dedupe here so each physical bond is counted once in the Hamiltonian."""
        pairs = []
        seen = set()
        for s in range(self.Ns):
            for t in self.nearest_neighbours(s):
                if s < t and (s, t) not in seen:
                    seen.add((s, t))
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

