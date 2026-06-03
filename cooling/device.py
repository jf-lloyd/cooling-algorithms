"""
The CoolingDevice defines the system and bath qubit geometry in cirq. It is initialised either with a "Lattice" object (see lattices) or with a pre-build cirq "Device".
"""

import cirq

class CoolingDevice():
    
    def __init__(self, system_qubits, bath_qubits, *, cirq_device=None, lattice=None):

        self._system_qubits = list(system_qubits)
        self._bath_qubits   = list(bath_qubits)
        self._Ns = len(self._system_qubits) # number of sys. qubits
        self._Nb = len(self._bath_qubits)  # number of bath qubits
        self._qubit_index_map = {q: i for i, q in enumerate(self._system_qubits)}

        self._cirq_device = cirq_device
        self._lattice     = lattice
        self._check_disjoint()

    @classmethod
    def from_lattice(cls, lattice, Nb):
        """Build system qubits from a Lattice object; Nb bath qubits without geometry."""
        system = [cirq.NamedQubit(f"s{s}") for s in range(lattice.Ns)]
        bath   = [cirq.NamedQubit(f"b{i}")   for i in range(Nb)]
        return cls(system, bath, lattice=lattice)

    @classmethod
    def from_cirq_device(cls, cirq_device, system_qubits, bath_qubits, lattice=None):
        """Build directly from a cirq Device. The partition of Device qubits into
        system and bath qubits must be given. If lattice is not provided, builds a
        GraphLattice reflecting device geometry.

        Unused device qubits (neither system nor bath) are silently ignored."""
        device_qubits = set(cirq_device.metadata.qubit_set)
        joint = set(system_qubits) | set(bath_qubits)
        stray = joint - device_qubits
        if stray:
            raise ValueError(f"Qubits not on the device: {stray}")
    
        if lattice is None:
            lattice = cls._lattice_from_device(cirq_device, system_qubits)
    
        return cls(system_qubits, bath_qubits, cirq_device=cirq_device, lattice=lattice)
    
    @staticmethod
    def _lattice_from_device(cirq_device, system_qubits):
        """Build a GraphLattice from the system-qubit connectivity."""
        import networkx as nx
        from .lattices import GraphLattice
    
        index_of = {q: i for i, q in enumerate(system_qubits)}
        sys_set = set(system_qubits)
    
        G = nx.Graph()
        G.add_nodes_from(range(len(system_qubits)))
        for a, b in cirq_device.metadata.nx_graph.edges():
            if a in sys_set and b in sys_set:
                G.add_edge(index_of[a], index_of[b])
    
        # optional drawing coords from grid qubits, if available
        coords = None
        if all(hasattr(q, "col") and hasattr(q, "row") for q in system_qubits):
            coords = {i: (q.col, q.row) for i, q in enumerate(system_qubits)}
    
        return GraphLattice(G, coords=coords)

    @property
    def lattice(self): 
        """underlying lattice object"""
        return self._lattice
    
    @property
    def Ns(self):
        """number of system qubits"""
        return self._Ns

    @property
    def Nb(self): 
        """number of bath qubits"""
        return self._Nb
    
    @property
    def system_qubits(self):
        """list of system qubits"""
        return self._system_qubits

    @property
    def bath_qubits(self):
        """list of bath qubits"""
        return self._bath_qubits

    @property
    def qubit_index_map(self):
        """map between system qubit names and index"""
        return self._qubit_index_map
    
    def _check_disjoint(self):
        """ guard against system/bath overlap or duplicated qubit indices """
        sys_set, bath_set = set(self._system_qubits), set(self._bath_qubits)
        if len(sys_set) != len(self._system_qubits):
            raise ValueError("Duplicate system qubits.")
        if len(bath_set) != len(self._bath_qubits):
            raise ValueError("Duplicate bath qubits.")
        overlap = sys_set & bath_set
        if overlap:
            raise ValueError(f"System and bath qubits overlap: {overlap}")

    def draw(self, ax=None, stub_length=0.4):
        """
        Visualise the device as a graph. System qubits are drawn black, bath
        qubits red. Each bond-colouring layer is drawn in its own colour.
        
        """
        import networkx as nx
        import matplotlib.pyplot as plt
        from matplotlib import cm
        
        if ax is None:
            _, ax = plt.subplots()
        
        G = nx.Graph()
        pos = {}
        color = {}
        
        # --- system qubit positions ---
        for s, q in enumerate(self._system_qubits):
            try:
                c = self._lattice.coords(s)
                pos[q] = (c[0], c[1] if len(c) > 1 else 0)
            except NotImplementedError:
                pos[q] = (s, 0)
            G.add_node(q)
            color[q] = "k"
        
        # --- bath qubit positions ---
        if self._cirq_device is not None:
            for q in self._bath_qubits:
                pos[q] = (q.col, q.row)
                G.add_node(q); color[q] = "r"
        else:
            max_x = max((p[0] for p in pos.values()), default=0)
            for i, q in enumerate(self._bath_qubits):
                pos[q] = (max_x + 1, i)
                G.add_node(q); color[q] = "r"
        
        # --- nodes ---
        nx.draw_networkx_nodes(G, pos=pos,
                               node_color=[color[q] for q in G.nodes()],
                               node_size=150, ax=ax)
        nx.draw_networkx_labels(G, pos=pos, font_size=7, ax=ax)
        
        # --- bonds per colouring layer ---
        palette = plt.colormaps["tab10"]
        
        def is_wrap(s, t):
            """True if bond (s,t) crosses a periodic boundary."""
            cs = self._lattice.coords(s)
            ct = self._lattice.coords(t)
            return any(abs(cs[d] - ct[d]) > 1 for d in range(len(cs)))
        
        def stub_ghosts(s, t):
            """Return (ghost_s_pos, ghost_t_pos) for a wrap bond."""
            cs = list(self._lattice.coords(s))
            ct = list(self._lattice.coords(t))
            gs, gt = list(cs), list(ct)
            for d in range(len(cs)):
                diff = ct[d] - cs[d]
                if abs(diff) > 1:
                    gs[d] = cs[d] - stub_length if diff > 0 else cs[d] + stub_length
                    gt[d] = ct[d] + stub_length if diff > 0 else ct[d] - stub_length
            return tuple(gs), tuple(gt)
        
        bond_layers = self._lattice.bond_colouring()
        for li, layer in enumerate(bond_layers):
            c = palette(li % 10)
            regular = []
            for (s, t) in layer:
                qs, qt = self._system_qubits[s], self._system_qubits[t]
                try:
                    wrap = is_wrap(s, t)
                except NotImplementedError:
                    wrap = False
        
                if wrap:
                    # draw two short dashed stubs instead of a long wrap edge
                    gs, gt = stub_ghosts(s, t)
                    ax.plot([pos[qs][0], gs[0]], [pos[qs][1], gs[1] if len(gs) > 1 else 0],
                            color=c, linewidth=2, linestyle="-.", alpha=1)
                    ax.plot([pos[qt][0], gt[0]], [pos[qt][1], gt[1] if len(gt) > 1 else 0],
                            color=c, linewidth=2, linestyle="-.", alpha=1)
                else:
                    regular.append((qs, qt))
        
            if regular:
                nx.draw_networkx_edges(G, pos=pos, edgelist=regular,
                                       edge_color=[c] * len(regular),
                                       width=2, ax=ax, label=f"layer {li}")
        
        # build legend entries for all layers (including stub-only layers)
        n_layers = len(bond_layers)
        handles = [plt.Line2D([0], [0], color=palette(li % 10), linewidth=2,
                               label=f"layer {li}")
                   for li in range(n_layers)]
        ax.legend(handles=handles, fontsize=7, loc="best")
        # ax.set_aspect("equal")
        return ax

