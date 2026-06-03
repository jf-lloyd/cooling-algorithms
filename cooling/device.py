'''
defines classes:
Lattice - defines lattice object, dimensionality, graph connectivity. Used by Device and Hamiltonian
Device - add qubit structure on top of lattice (qubits, auxiliaries) 
'''

from abc import ABC, abstractmethod    
import cirq

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
        
