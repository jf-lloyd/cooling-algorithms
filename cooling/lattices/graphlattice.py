"""
A lattice build from a networkx graph -- used to build lattice for pre-built cirq device.

Created by Jerome Lloyd on 3rd June 2026
"""

from .latticebase import Lattice

class GraphLattice(Lattice):
    """
    Lattice defined by a pure connectivity graph.

    Holds a networkx.Graph whose nodes are integer site indices 0..Ns-1 and
    whose edges are the nearest-neighbour bonds. 
    """

    def __init__(self, graph, coords=None):
        """
        graph  : networkx.Graph with integer nodes 0..Ns-1.
        coords : optional {site_index: (x, y)} for drawing only.
        """
        self._graph = graph
        self._Ns = graph.number_of_nodes()
        self._coords = coords

        # sanity: nodes must be exactly 0..Ns-1 so site index == node
        nodes = set(graph.nodes())
        if nodes != set(range(self._Ns)):
            raise ValueError(
                "GraphLattice expects integer nodes 0..Ns-1; "
                "relabel the graph (e.g. nx.convert_node_labels_to_integers) "
                "before constructing."
            )

    @property
    def graph(self):
        return self._graph

    def coords(self, s):
        if self._coords is None:
            raise NotImplementedError(
                "DeviceLattice has no spatial coords unless supplied at construction."
            )
        return self._coords[s]

    def index(self, *coords):
        # Graph lattices have no coordinate system; index is the site label itself.
        (s,) = coords
        return s

    def nearest_neighbours(self, s):
        yield from self._graph.neighbors(s)

    def bond_colouring(self):
        """General connectivity graph: use the base greedy edge-colouring."""
        return self._greedy_colouring()