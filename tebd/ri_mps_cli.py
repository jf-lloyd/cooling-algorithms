"""Repeated-interaction (RI) cooling of 1D spin models on an MPS backend. Executes the
RI cooling circuits of the `cooling-algorithms` package (DetailedBalanceProtocol,
Randomized schedule, BasicNoiseModel) on a matrix-product state, so the protocol /
Trotter / noise convention matches the reference state-vector simulation exactly while
scaling beyond exact state vectors. Handles Nb<=L (baths spread evenly along the chain
to keep bond low; the couplings roam, so swap+split routes them). Unravels depolarizing
channels stochastically (one trajectory). CSV columns: cyc, E, E/L (col 2 = E/L).

Works for ANY 1D model defined in cooling-algorithms (--model {ising, heisenberg, xy}):
the energy is the model's own Hamiltonian, <model.hamiltonian> (a cirq.PauliSum), measured
generically term-by-term on the MPS (OBC), matching cooling's DefaultMeasurement1 H0.

Author: Yuxuan Zhang <yuxuanzhang@utexas.edu>
Copyright (c) 2026 Yuxuan Zhang."""
import sys, os, argparse, csv
p = argparse.ArgumentParser()
p.add_argument("--L", type=int, default=10); p.add_argument("--NB", type=int, default=10)
p.add_argument("--model", choices=("ising", "heisenberg", "xy"), default="ising")
p.add_argument("--J", type=float, default=1.0); p.add_argument("--g", type=float, default=1.0)
p.add_argument("--gx", type=float, default=0.0)
p.add_argument("--Jxx", type=float, default=None); p.add_argument("--Jyy", type=float, default=None)
p.add_argument("--Jzz", type=float, default=None)
p.add_argument("--beta", type=float, default=1.0)
p.add_argument("--trotter_order", type=int, choices=(1, 2), default=2)
p.add_argument("--p1", type=float, default=1e-4); p.add_argument("--p2", type=float, default=1e-3)
p.add_argument("--ncycles", type=int, default=500); p.add_argument("--sample_every", type=int, default=1)
p.add_argument("--chi", type=int, default=256); p.add_argument("--cutoff", type=float, default=1e-9)
p.add_argument("--seed", type=int, default=1); p.add_argument("--out", type=str, default="jmps")
a = p.parse_args(); sys.argv = ['t']
import numpy as np, quimb.tensor as qtn
import types as _t; sys.modules.setdefault('qsimcirq', _t.ModuleType('qsimcirq'))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cooling-algorithms"))
sys.path.insert(0, os.environ.get("COOLING_ALGS", "cooling-algorithms"))
import cooling, cirq
sys.path.insert(0, HERE); import quimb_traj as qt
X = qt.X; P0 = qt.P0; P1 = qt.P1
L, Nb = a.L, a.NB
NOISE = a.p2 > 0 or a.p1 > 0

lat = cooling.ChainLattice1D(L=L, pbc=False); dev = cooling.CoolingDevice.from_lattice(lat, Nb=Nb)
_jx = a.Jxx if a.Jxx is not None else a.J
_jy = a.Jyy if a.Jyy is not None else a.J
_jz = a.Jzz if a.Jzz is not None else a.J
if a.model == "ising":
    model = cooling.IsingModel(dev, {"J": a.J, "g": a.g, "gx": a.gx})
elif a.model == "heisenberg":
    model = cooling.HeisenbergModel(dev, {"Jxx": _jx, "Jyy": _jy, "Jzz": _jz})
else:  # xy
    model = cooling.XYModel(dev, {"Jxx": _jx, "Jyy": _jy})
HAM = model.hamiltonian   # cirq.PauliSum -> measured generically on the MPS (any cooling model)
nm = cooling.BasicNoiseModel(dev, {"p1": a.p1, "p2": a.p2, "p_reset": 0.0, "sys": True, "bath": True, "coupling": True}) if NOISE else None
prot = cooling.DetailedBalanceProtocol(dev, model, params={"beta": a.beta, "delta": 0.25, "h": 4.0, "theta": 1.0, "NT": 2},
                                       function="gaussian", noise_model=nm, trotter_order=a.trotter_order)
sched = cooling.Randomized(prot, n_cache=50, seed=a.seed, allowed_ops=["X", "Y", "Z"])

# chain ordering: system sites in order, baths spread evenly between them (low bond for roaming couplings)
sysq = list(dev.qubits[:L]); bathq = list(dev.qubits[L:])
bath_after = {}
for k in range(Nb):
    bath_after.setdefault(min(L - 1, int((k + 0.5) * L / Nb)), []).append(k)
order = []
for si in range(L):
    order.append(('s', si))
    for bk in bath_after.get(si, []): order.append(('b', bk))
qmap = {}
for pos, (kind, idx) in enumerate(order):
    qmap[(sysq if kind == 's' else bathq)[idx]] = pos
sys_sites = [qmap[sysq[i]] for i in range(L)]
n = L + Nb
rng = np.random.default_rng(1000 + a.seed)
psi = qtn.MPS_computational_state('0' * n)

def apply_unitary(U, sites):
    psi.gate_(U, sites, contract='swap+split' if len(sites) > 1 else True, max_bond=a.chi, cutoff=a.cutoff)

def reset_site(s):
    z = qt._local_z_expectation(psi, s); p0 = max(0.0, min(1.0, (1 + z) / 2))
    out = 0 if rng.random() < p0 else 1
    psi.gate_(P0 if out == 0 else P1, (s,), contract=True); psi.normalize()
    if out == 1: psi.gate_(X, (s,), contract=True)

def is_channel(gate):
    try: cirq.unitary(gate); return False
    except Exception: return True

PMAP = {'X': qt.X, 'Y': qt.Y, 'Z': qt.Z}
def pauli_expect(sites, ops):
    if len(sites) == 1:
        return float(np.real(psi.local_expectation_exact(PMAP[ops[0]], (sites[0],))))
    if len(sites) == 2:
        return qt._two_site_expectation(psi, PMAP[ops[0]], sites[0], PMAP[ops[1]], sites[1])
    O = PMAP[ops[0]]
    for o in ops[1:]: O = np.kron(O, PMAP[o])
    return float(np.real(psi.local_expectation_exact(O, tuple(sites))))
def measure_H():
    """<model.hamiltonian> on the MPS: sum_i coeff_i * <Pauli-string_i>. Model-agnostic."""
    e = 0.0
    for term in HAM:
        items = list(term.items())
        c = float(np.real(complex(term.coefficient)))
        if not items:
            e += c; continue
        pairs = sorted((qmap[q], str(pl)) for q, pl in items)
        e += c * pauli_expect([s for s, _ in pairs], [o for _, o in pairs])
    return e

rows, peak = [], 1
for cyc in range(a.ncycles):
    circ = sched.circuit_fn(cyc + 1)
    for op in circ.all_operations():
        gate = op.gate; sites = tuple(qmap[q] for q in op.qubits)
        if isinstance(gate, cirq.ResetChannel):
            reset_site(sites[0]); continue
        if NOISE and is_channel(gate):
            pp = float(getattr(gate, 'p', 0.0))
            if rng.random() < pp:
                if len(sites) == 1: apply_unitary(qt.PAULI_1Q_ERRORS[rng.integers(3)], sites)
                else:
                    l, r = qt.PAULI_2Q_ERRORS[rng.integers(15)]; apply_unitary(np.kron(l, r), sites)
            continue
        apply_unitary(cirq.unitary(gate), sites)
    peak = max(peak, int(psi.max_bond()))
    if (cyc + 1) % a.sample_every == 0 or cyc == a.ncycles - 1:
        e = measure_H()
        rows.append((cyc + 1, e, e / L))
        if (cyc + 1) % max(a.sample_every, 25) == 0:
            print(f"  cyc {cyc+1:3d}: E/L={e/L:.5f} bond={psi.max_bond()} peak={peak}", flush=True)
with open(a.out + ".csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["# cyc", "E", "E_per_L"])
    for r in rows: w.writerow(r)
print(f"DONE {a.out}: L={L} NB={Nb} beta={a.beta} p2={a.p2} -> {len(rows)} samples, peak_bond={peak}", flush=True)
