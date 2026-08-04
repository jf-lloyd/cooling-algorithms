"""Compress ONE faithful cooling CYCLE (channel: system+bath+coupling+reset) at 3x4+1 etc.
EXACT protocol/params from ../2d_ising_gpu_example/{a_device,b_model,c_protocol}.py.

H = J sum_<ij> X_iX_j - g sum_i Z_i   (J=-1, g=1, gx=0, J2=0).
Per micro-step j in [0, 2MT]:  system XX (exp(-i J delta XX) per bond) + system Z (exp(+i g delta Z))
+ bath Z (exp(+i h delta/2 Z)) + coupling (exp(-i theta delta f[j] (Z+Y)/sqrt2 ⊗ Y)).  Then RESET bath.
schedule: a=delta*sqrt(4h/beta), MT=max(NT,int(NT/a)), f[t]=exp(-a^2 t^2/2) normalised delta*sum|f|=1.
params: theta=0.6, delta=0.4*(0.1*pi/2)=0.0628, NT=5, h(bath)=4, nb=1, randomize_couplings (fixed here).

nb=1 -> channel output rho_sys (rank<=2) = A A^H with A=[a0,a1] the two bath branches; channel-level
similarity = HS overlap of marginals ||A_nat^H A_ans||_F^2/(...).  Compress with a shallow k-layer
ansatz (free angles) + coupling; report 2q-gate count vs native at each depth.
  usage: cc_cycle_compress.py [Lx Ly beta]
"""
import sys, json, numpy as np
import scipy.linalg as spla
from scipy.optimize import minimize

Lx = int(sys.argv[1]) if len(sys.argv) > 1 else 3
Ly = int(sys.argv[2]) if len(sys.argv) > 2 else 4
beta = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
J, g = -1.0, 1.0
theta = 0.6
delta = 0.4 * (0.1 * np.pi / 2)          # 0.0628
NT = 5
h = 4.0                                   # bath splitting
nq = Lx * Ly; nb = 1; n = nq + nb; bq = nq
cpl_site = nq // 2                        # fixed system qubit the bath couples to (randomize off for 1 cycle)

def idx(x, y): return x + y * Lx
bonds = []
for y in range(Ly):
    for x in range(Lx):
        if x + 1 < Lx: bonds.append((idx(x, y), idx(x + 1, y)))
        if y + 1 < Ly: bonds.append((idx(x, y), idx(x, y + 1)))
nB = len(bonds)
a_s = delta * np.sqrt(abs(4 * h / beta)); MT = max(NT, int(NT / a_s)); steps = 2 * MT + 1
ff = np.array([np.exp(-a_s**2 * tt**2 / 2) for tt in np.arange(-MT, MT + 1)]); ff /= delta * np.sum(np.abs(ff))
native2q = steps * (nB + 1)
print(f"{Lx}x{Ly} nq={nq}+1bath n={n} | J={J} g={g} theta={theta} delta={delta:.4f} h={h} beta={beta}", flush=True)
print(f"bonds={nB} a={a_s:.3f} MT={MT} micro-steps={steps} within-cycle T={MT*delta:.3f} | "
      f"native 2q/cycle = {native2q} ({steps*nB} XX + {steps} coupling)", flush=True)

X = np.array([[0, 1], [1, 0]], complex); Y = np.array([[0, -1j], [1j, 0]]); Z = np.array([[1, 0], [0, -1]], complex)
def Gxx(th): return spla.expm(-1j * th * np.kron(X, X))      # exp(-i th XX)
OP = np.kron((Z + Y) / np.sqrt(2), Y)
def Gcpl(al): return spla.expm(-1j * al * OP)

def ap2(psi, G, q1, q2):
    G = G.reshape(2, 2, 2, 2)
    return np.moveaxis(np.tensordot(psi, G, axes=([q1, q2], [2, 3])), [-2, -1], [q1, q2])
def af(psi, ph, q):                                          # exp(+i ph Z)
    sh = [1] * psi.ndim; sh[q] = 2
    return psi * np.array([np.exp(1j * ph), np.exp(-1j * ph)]).reshape(sh)

def native_cycle(psi):
    for j in range(steps):
        for (i, k) in bonds: psi = ap2(psi, Gxx(J * delta), i, k)
        for i in range(nq): psi = af(psi, g * delta, i)
        psi = af(psi, h * delta / 2, bq)
        psi = ap2(psi, Gcpl(theta * delta * ff[j]), cpl_site, bq)
    return psi

def branches(psi): return psi.reshape(2 ** nq, 2)           # columns = bath 0/1

def hs(Anat, A):
    M = Anat.conj().T @ A
    return (np.linalg.norm(M) ** 2) / (np.linalg.norm(Anat.conj().T @ Anat) * np.linalg.norm(A.conj().T @ A))

def rand_in(seed):
    rng = np.random.default_rng(seed); st = np.ones((1,), complex)
    for q in range(nq):
        a = rng.normal(size=2) + 1j * rng.normal(size=2); a /= np.linalg.norm(a); st = np.kron(st, a)
    return np.kron(st, np.array([1, 0], complex)).reshape((2,) * n)

NS = 6
ins = [rand_in(s) for s in range(NS)]
Anat = [branches(native_cycle(p.copy())) for p in ins]
print(f"native channel built; mean Tr(rho_out)={np.mean([np.linalg.norm(A)**2 for A in Anat]):.4f} (want 1)", flush=True)

# ansatz: k layers, each = system XX(theta_l)+Z(phi_l) then coupling(alpha_l); all angles free
def ansatz(psi, k, x):
    th, ph, al = x[:k], x[k:2 * k], x[2 * k:3 * k]
    for l in range(k):
        for (i, kk) in bonds: psi = ap2(psi, Gxx(th[l]), i, kk)
        for i in range(nq): psi = af(psi, ph[l], i)
        psi = af(psi, h * delta / 2, bq)
        psi = ap2(psi, Gcpl(al[l]), cpl_site, bq)
    return psi
def warm(k):                                                # coarse k-step version of the cycle
    return np.concatenate([np.full(k, J * delta * steps / k),
                           np.full(k, g * delta * steps / k),
                           np.full(k, theta * delta * np.sum(ff) / k)])

import os
KLO = int(os.environ.get("KLO", "1")); KHI = int(os.environ.get("KHI", "9"))
results = {}
for k in range(KLO, KHI):
    x0 = warm(k)
    def loss(x): return 1 - np.mean([hs(Anat[i], branches(ansatz(ins[i].copy(), k, x))) for i in range(NS)])
    res = minimize(loss, x0, method='Nelder-Mead',
                   options={'maxiter': min(2500, 250 * (3 * k)), 'xatol': 1e-6, 'fatol': 1e-10, 'adaptive': True})
    results[k] = dict(g2q=k * (nB + 1), infid=float(res.fun), start=float(loss(x0)))
    print(f"depth k={k} ({k*(nB+1):3d} 2q gates, native {native2q}, ratio {native2q/(k*(nB+1)):.1f}x): "
          f"channel infid {res.fun:.3e}  (start {results[k]['start']:.2e})", flush=True)

json.dump(dict(Lx=Lx, Ly=Ly, beta=beta, steps=steps, native2q=native2q, nB=nB, T=MT*delta, results=results),
          open(f'cc_cycle_compress_{Lx}x{Ly}_b{beta:g}.json', 'w'), indent=2)
print("saved json", flush=True)
