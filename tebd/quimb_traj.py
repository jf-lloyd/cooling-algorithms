#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RI thermalization with TEBD in Quimb (1D TFIM) — energy only (MPI, no plots)

Author: Yuxuan Zhang <yuxuanzhang@utexas.edu>
Copyright (c) 2026 Yuxuan Zhang.

- Stochastic 'trace + reset' on bath qubits (quantum trajectories)
- First- or second-order Trotterization (second-order Strang by default)
- Gate-resolved stochastic depolarizing noise and optional reset bit-flip errors
- Energy via MPO expectation on the *system only*
- MPI: each rank runs a disjoint subset of trajectories; Allreduce to average
- Robustness: single-threaded math, numba disabled, non-root output silenced (optional)
- Stores results to outbase.npz and outbase.csv on rank 0

Example (SLURM srun):
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
  srun -n 8 python ri_tebd_energy_vs_infinite_mpi.py \
    --L 20 --J 1.0 --g 1.0 --beta 0.5 --NB 3 \
    --ncycles 100 --n_traj 64 --sample_every 5 \
    --chi 256 --cutoff 1e-6 --delta 0.07853981633974483 \
    --seed 1 --nk_inf 50000 --out out/fig4_like
"""

# ---------------- Environment hardening (MUST be before heavy imports) ---------------
import os, sys

import faulthandler; faulthandler.enable()

# ------------------------------------ MPI ------------------------------------------
try:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
except Exception:
    COMM = None
    RANK = 0
    SIZE = 1

# Silence non-root stdio to avoid rare libc/MPI stdio races (disable by setting RI_QUIET_NONROOT=0)
if os.environ.get("RI_QUIET_NONROOT", "1") != "0":
    if "ipykernel" not in sys.modules and RANK != 0:
        try:
            sys.stdout = open(os.devnull, "w", buffering=1)
            sys.stderr = open(os.devnull, "w", buffering=1)
        except Exception:
            pass

def barrier():
    if COMM is not None:
        COMM.Barrier()

def split_count(total: int, size: int, rank: int) -> int:
    q, r = divmod(total, size)
    return q + (1 if rank < r else 0)

# ---------------------------------- Imports ----------------------------------------
import math
import time
import json
import argparse
import numpy as np
from scipy.linalg import expm

import quimb as qu
try:
    qu.set_numba(False)  # belt & braces
except Exception:
    pass
import quimb.tensor as qtn
from quimb.tensor import SpinHam1D

# --------------------------------- Pauli -------------------------------------------
I2 = np.eye(2, dtype=complex)
X  = np.array([[0, 1], [1, 0]], dtype=complex)
Y  = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z  = np.array([[1, 0], [0, -1]], dtype=complex)
P0 = np.array([[1, 0], [0, 0]], dtype=complex)
P1 = np.array([[0, 0], [0, 1]], dtype=complex)
PAULI_1Q_ERRORS = (X, Y, Z)
PAULI_2Q_ERRORS = tuple(
    (left, right)
    for left in (I2, X, Y, Z)
    for right in (I2, X, Y, Z)
    if not (left is I2 and right is I2)
)

# ---------------- Thermodynamic-limit u_inf(β) -------------------------------------
def _eps_k(k, J, g):
    # ε_k = 2 * sqrt(J^2 + g^2 - 2 J g cos k)
    return 2.0 * np.sqrt(J*J + g*g - 2.0*J*g*np.cos(k))

def _k_integral(func, J, g, beta, nk=50_000):
    ks = (np.arange(nk) + 0.5) * (np.pi / nk)   # midpoint rule
    return (np.pi / nk) * np.sum(func(ks, J, g, beta), dtype=np.float64)

def u_infinite(beta, J, g, nk=50_000):
    """u_inf(β) = -(1/(2π)) ∫_0^π dk ε_k tanh(β ε_k / 2)."""
    def integrand(ks, J, g, beta):
        ek = _eps_k(ks, J, g)
        return ek * np.tanh(0.5 * beta * ek)
    I = _k_integral(integrand, J, g, beta, nk=nk)
    return - I / (2.0 * np.pi)

# ---------------- RI protocol params ------------------------------------------------
def gaussian_window(delta: float, MT: int, a_true: float) -> np.ndarray:
    """Discrete Gaussian f_tau with δ * Σ|f| = 1 (L1 normalize)."""
    taus = np.arange(-MT, MT + 1)
    a_pix = a_true * delta
    f = np.exp(-0.5 * (a_pix * taus) ** 2)
    return f / (delta * np.sum(np.abs(f)))

def gate_noise_budget(L: int, NB: int, J: float, g: float, beta: float,
                      delta: float, T_factor: float, trotter_order: int,
                      p1: float, p2: float, NT=None, h_override=None) -> dict:
    """Analytical primitive-gate and stochastic-fault counts for one reset.
    Mirrors ri_cycle_trajectory's window: with NT override, MT=max(NT,int(NT/(delta*a_true)));
    h_override (e.g. Jerome's h=4) replaces the default max(2g,4J)."""
    h = float(h_override) if h_override is not None else max(2 * g, 4 * J)
    a_true = math.sqrt(4.0 * h / max(beta, 1e-12))
    if NT is not None:
        a_jer = delta * a_true
        MT = max(int(NT), int(int(NT) / a_jer)) if a_jer > 0 else int(NT)
    else:
        MT = max(2, int(math.ceil((T_factor / a_true) / delta)))
    n_slices = 2 * MT + 1
    n_couplings = min(NB, L)

    if trotter_order == 1:
        n_1q = n_slices * (L + NB)
        n_2q = n_slices * ((L - 1) + n_couplings)
    else:
        n_system_steps = n_slices + 1
        # system 1q noised once per step (leapfrog-merged Z), + 2 bath-precession halves/slice
        n_1q = n_system_steps * L + 2 * n_slices * NB
        n_2q = n_system_steps * (L - 1) + n_slices * n_couplings

    expected_faults = n_1q * p1 + n_2q * p2
    no_fault_probability = (1.0 - p1) ** n_1q * (1.0 - p2) ** n_2q
    return dict(
        MT=MT,
        filter_slices=n_slices,
        one_qubit_gates=n_1q,
        two_qubit_gates=n_2q,
        expected_faults_per_reset=expected_faults,
        probability_any_fault_per_reset=1.0 - no_fault_probability,
    )

# ---------------- System & Bath gates ----------------------------------------------
def build_system_unis(J: float, g: float, delta: float):
    """Return (U1, U2):
       U1 = e^{+i δ g Z} (2x2), U2 = e^{+i δ J X⊗X} (4x4)."""
    U1 = expm(+1j * g * delta * Z)
    U2 = expm(+1j * J * delta * np.kron(X, X))
    return U1, U2

def build_bath_precession(h: float, delta: float):
    """U_b = e^{-i δ (-h/2) Z} (2x2)."""
    return expm(-1j * delta * (-h / 2.0) * Z)

def build_sb_coupling(theta_ft: float):
    """U_SB = e^{-i θ_ft (A⊗Y)}, A=(Z+Y)/√2. θ_ft already includes δ·θ·f_τ."""
    A = (Z + Y) / np.sqrt(2.0)
    return expm(-1j * theta_ft * np.kron(A, Y))

# ---------------- MPS helpers -------------------------------------------------------
def make_initial_mps(L_sys: int, NB: int, dtype='complex128', rng=None, init='cold'):
    """Product-state init. 'cold' = |0..0> (low-entanglement, default, MPS-cheap).
    'random' = random computational-basis bitstring per call (Jerome-style hot init,
    decorrelates trajectories). NOTE: cooling FROM a random bitstring builds volume-law
    entanglement transiently -> use a noise-matched cutoff (~1e-6) or the bond explodes."""
    n = L_sys + NB
    if init == 'random' and rng is not None:
        bits = ''.join(str(int(b)) for b in rng.integers(0, 2, n))
    else:
        bits = '0' * n
    return qtn.MPS_computational_state(bits, dtype=dtype, tags='KET')

def build_site_layout(L_sys: int, NB: int, site_ordering: str):
    """Map logical system and bath labels to positions along the MPS chain."""
    if site_ordering == "auto":
        site_ordering = "interleaved" if NB == L_sys else "distributed"
    if site_ordering == "interleaved":
        if NB != L_sys:
            raise ValueError(
                "interleaved ordering currently requires NB == L so every "
                "bath can remain adjacent to its coupled system site."
            )
        system_sites = tuple(2 * site for site in range(L_sys))
        bath_sites = tuple(2 * bath + 1 for bath in range(NB))
    elif site_ordering == "distributed":
        # NB baths spread evenly: bath b homed at the middle of its system region
        # [edges[b], edges[b+1]) and placed right after that site -> coupling stays local, low bond.
        edges = np.linspace(0, L_sys, NB + 1).astype(int)
        homes = [min(L_sys - 1, (int(edges[b]) + max(int(edges[b]) + 1, int(edges[b + 1])) - 1) // 2)
                 for b in range(NB)]
        after = {}
        for b, hh in enumerate(homes):
            after.setdefault(hh, []).append(b)
        sys_sites = [0] * L_sys; bath_sites_l = [0] * NB; pos = 0
        for i in range(L_sys):
            sys_sites[i] = pos; pos += 1
            for b in after.get(i, []):
                bath_sites_l[b] = pos; pos += 1
        system_sites = tuple(sys_sites); bath_sites = tuple(bath_sites_l)
    elif site_ordering == "blocked":
        system_sites = tuple(range(L_sys))
        bath_sites = tuple(range(L_sys, L_sys + NB))
    else:
        raise ValueError(f"Unknown site_ordering={site_ordering!r}.")
    return site_ordering, system_sites, bath_sites

def validate_probability(name: str, probability: float):
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {probability}.")

def apply_sampled_1q_depolarizing(mps: qtn.MatrixProductState, site: int, probability: float,
                                  rng: np.random.Generator, chi, cutoff) -> bool:
    """Sample the 1-qubit channel: I with 1-p, otherwise uniform X/Y/Z."""
    if probability <= 0.0 or rng.random() >= probability:
        return False
    error = PAULI_1Q_ERRORS[int(rng.integers(len(PAULI_1Q_ERRORS)))]
    mps.gate_(error, where=site, contract='swap+split', max_bond=chi, cutoff=cutoff)
    return True

def apply_sampled_2q_depolarizing(mps: qtn.MatrixProductState, sites, probability: float,
                                  rng: np.random.Generator, chi, cutoff) -> bool:
    """Sample the 2-qubit channel: II with 1-p, otherwise a uniform non-II Pauli."""
    if probability <= 0.0 or rng.random() >= probability:
        return False
    left_error, right_error = PAULI_2Q_ERRORS[int(rng.integers(len(PAULI_2Q_ERRORS)))]
    for site, error in zip(sites, (left_error, right_error)):
        if error is not I2:
            mps.gate_(error, where=site, contract='swap+split',
                      max_bond=chi, cutoff=cutoff)
    return True

def apply_one_site_all_sys(mps: qtn.MatrixProductState, U1, system_sites, chi, cutoff,
                           rng: np.random.Generator, p1: float):
    for logical_site, mps_site in enumerate(system_sites):
        mps.gate_(U1, where=mps_site, tags=f'U1@S{logical_site}',
                  propagate_tags='sites',
                  contract='swap+split')
        apply_sampled_1q_depolarizing(mps, mps_site, p1, rng, chi, cutoff)
    mps.compress(method='svd', max_bond=chi, cutoff=cutoff)

def apply_two_site_even_odd_sys(mps: qtn.MatrixProductState, U2, system_sites, chi, cutoff,
                                rng: np.random.Generator, p2: float):
    L_sys = len(system_sites)
    for i in range(0, L_sys - 1, 2):
        sites = (system_sites[i], system_sites[i + 1])
        mps.gate_(U2, where=sites, tags=f'U2@S{i}-S{i+1}',
                  propagate_tags='sites', contract='swap+split',
                  max_bond=chi, cutoff=cutoff)
        apply_sampled_2q_depolarizing(mps, sites, p2, rng, chi, cutoff)
    for i in range(1, L_sys - 1, 2):
        sites = (system_sites[i], system_sites[i + 1])
        mps.gate_(U2, where=sites, tags=f'U2@S{i}-S{i+1}',
                  propagate_tags='sites', contract='swap+split',
                  max_bond=chi, cutoff=cutoff)
        apply_sampled_2q_depolarizing(mps, sites, p2, rng, chi, cutoff)

def apply_sys_bath_gate(mps: qtn.MatrixProductState, i_sys: int, i_bath: int, U_sb,
                        chi, cutoff, rng: np.random.Generator, p2: float):
    mps.gate_(U_sb, where=[i_sys, i_bath], tags=f'USB@{i_sys}-{i_bath}',
              propagate_tags='sites', contract='swap+split',
              max_bond=chi, cutoff=cutoff)
    apply_sampled_2q_depolarizing(mps, (i_sys, i_bath), p2, rng, chi, cutoff)

def apply_bath_precession(mps: qtn.MatrixProductState, Ub, bath_sites, chi, cutoff,
                          rng: np.random.Generator, p1: float):
    for logical_bath, mps_site in enumerate(bath_sites):
        mps.gate_(Ub, where=mps_site, tags=f'Ub@B{logical_bath}',
                  propagate_tags='sites',
                  contract='swap+split', max_bond=chi, cutoff=cutoff)
        apply_sampled_1q_depolarizing(mps, mps_site, p1, rng, chi, cutoff)
    mps.compress(method='svd', max_bond=chi, cutoff=cutoff)

def apply_system_strang(mps: qtn.MatrixProductState, system_sites, U_z_half, U_xx,
                        chi, cutoff, rng: np.random.Generator, p1: float, p2: float):
    """Z(duration/2)-XX(duration)-Z(duration/2) system step. Consecutive steps' boundary
    Z half-layers merge into ONE physical Z per qubit (leapfrog), so the 1q gate noise is
    applied only ONCE per step (on the 2nd half) to avoid double-counting it."""
    apply_one_site_all_sys(mps, U_z_half, system_sites, chi, cutoff, rng, 0.0)
    apply_two_site_even_odd_sys(mps, U_xx, system_sites, chi, cutoff, rng, p2)
    apply_one_site_all_sys(mps, U_z_half, system_sites, chi, cutoff, rng, p1)

# ---- Generic 1D spin model: system Hamiltonian terms + evolution -------------------
def model_system_terms(model, J=1.0, g=1.0, gx=0.0, Jxx=None, Jyy=None, Jzz=None):
    """H_sys = -Σ field - Σ coupling, returned as
    (fields=[(coeff, P 2x2)], couplings=[(coeff, P_left 2x2, P_right 2x2)]).
    ising: field -g Z (and -gx X), coupling -J XX.  xy: -Jxx XX -Jyy YY.
    heisenberg: -Jxx XX -Jyy YY -Jzz ZZ.  (Jxx/Jyy/Jzz default to J.)"""
    jx = J if Jxx is None else Jxx
    jy = J if Jyy is None else Jyy
    jz = J if Jzz is None else Jzz
    if model == "ising":
        fields = [(g, Z)] + ([(gx, X)] if gx else [])
        couplings = [(J, X, X)]
    elif model == "xy":
        fields = []
        couplings = [(jx, X, X), (jy, Y, Y)]
    elif model == "heisenberg":
        fields = []
        couplings = [(jx, X, X), (jy, Y, Y), (jz, Z, Z)]
    else:
        raise ValueError(f"unknown model {model!r} (ising|xy|heisenberg)")
    return fields, couplings

def apply_sys_step_generic(mps, system_sites, fields, couplings, dur,
                           chi, cutoff, rng: np.random.Generator, p1, p2):
    """2nd-order Strang of H_sys over time `dur`: field(dur/2) - [symmetric couplings(dur)]
    - field(dur/2). e^{-i H_sys dur} with H_sys = -Σ field - Σ coupling, so each factor is
    e^{+i coeff·t·P}. For a single coupling (Ising) reduces to field(dur/2)-coupling(dur)-field(dur/2)."""
    fhalf = [expm(+1j * c * (dur / 2.0) * P) for c, P in fields]
    chalf = [expm(+1j * c * (dur / 2.0) * np.kron(Pl, Pr)) for c, Pl, Pr in couplings]
    cfull = [expm(+1j * c * dur * np.kron(Pl, Pr)) for c, Pl, Pr in couplings]
    for U in fhalf:
        apply_one_site_all_sys(mps, U, system_sites, chi, cutoff, rng, p1)
    n = len(couplings)
    if n == 1:
        apply_two_site_even_odd_sys(mps, cfull[0], system_sites, chi, cutoff, rng, p2)
    elif n >= 2:
        for U in chalf[:-1]:
            apply_two_site_even_odd_sys(mps, U, system_sites, chi, cutoff, rng, p2)
        apply_two_site_even_odd_sys(mps, cfull[-1], system_sites, chi, cutoff, rng, p2)
        for U in reversed(chalf[:-1]):
            apply_two_site_even_odd_sys(mps, U, system_sites, chi, cutoff, rng, p2)
    for U in fhalf:
        apply_one_site_all_sys(mps, U, system_sites, chi, cutoff, rng, p1)

# ---- 1- and 2-site expectations from reduced RDMs (robust) ------------------------
def _local_z_expectation(mps: qtn.MatrixProductState, idx: int,
                         max_bond=256, cutoff=1e-10) -> float:
    """⟨Z_idx⟩ via exact MPS contraction (dense fallback)."""
    try:
        return float(np.real(mps.local_expectation_exact(Z, (idx,))))
    except Exception:
        vec = mps.to_dense()
        N = mps.nsites
        psi = vec.reshape([2] * N)
        sl0 = [slice(None)] * N; sl0[idx] = 0
        sl1 = [slice(None)] * N; sl1[idx] = 1
        p0 = float(np.sum(np.abs(psi[tuple(sl0)])**2))
        p1 = float(np.sum(np.abs(psi[tuple(sl1)])**2))
        return p0 - p1

def _two_site_expectation(mps: qtn.MatrixProductState, A: np.ndarray, i: int,
                          B: np.ndarray, j: int, max_bond=256, cutoff=1e-10) -> float:
    """⟨A_i ⊗ B_j⟩ via 2-site reduced RDM (dense fallback)."""
    try:
        rdm2 = mps.partial_trace_to_mpo(keep=[i, j], max_bond=max_bond, cutoff=cutoff)
        rho = rdm2.to_dense() if hasattr(rdm2, "to_dense") else rdm2.to_array()  # (4,4)
        AB = np.kron(A, B)
        return float(np.trace(rho @ AB).real)
    except Exception:
        vec = mps.to_dense()
        N = mps.nsites
        psi = vec.reshape([2] * N)
        red = np.zeros((4, 4), dtype=complex)
        for a in (0, 1):
            for b in (0, 1):
                sl = [slice(None)] * N
                sl[i] = a; sl[j] = b
                block = psi[tuple(sl)]
                v = block.reshape(-1)
                idx = 2 * a + b
                for c in (0, 1):
                    for d in (0, 1):
                        slb = [slice(None)] * N
                        slb[i] = c; slb[j] = d
                        blockb = psi[tuple(slb)].reshape(-1)
                        jdx = 2 * c + d
                        red[idx, jdx] = np.vdot(blockb, v)
        AB = np.kron(A, B)
        return float(np.trace(red @ AB).real)

def energy_from_mps(mps: qtn.MatrixProductState, system_sites, fields, couplings,
                    max_bond=256, cutoff=1e-10) -> float:
    """E = <H_sys> = -Σ_field coeff Σ_i⟨P_i⟩ - Σ_coupling coeff Σ_i⟨P_i P_{i+1}⟩ (system sites).
    fields=[(coeff,P)], couplings=[(coeff,P_left,P_right)] (same spec as model_system_terms)."""
    terms = {}
    for c, P in fields:
        op = -c * P
        for site in system_sites:
            k = (site,); terms[k] = terms.get(k, np.zeros_like(op)) + op
    for c, Pl, Pr in couplings:
        op = -c * np.kron(Pl, Pr)
        for left, right in zip(system_sites[:-1], system_sites[1:]):
            k = (left, right); terms[k] = terms.get(k, np.zeros_like(op)) + op
    try:
        return float(np.real(mps.compute_local_expectation(terms, method="envs")))
    except Exception:
        e = 0.0
        for c, P in fields:
            e += -c * sum(float(np.real(mps.local_expectation_exact(P, (site,))))
                          for site in system_sites)
        for c, Pl, Pr in couplings:
            e += -c * sum(_two_site_expectation(mps, Pl, left, Pr, right,
                                                max_bond=max_bond, cutoff=cutoff)
                          for left, right in zip(system_sites[:-1], system_sites[1:]))
        return float(e)

# ---------------- System MPO (cache) -----------------------------------------------
# SpinHam1D uses Sx=σx/2, Sz=σz/2; to get H=-J σxσx - g σz, use -4J for XX and -2g for Z
def build_tfim_mpo_system(L: int, J: float, g: float, cyclic: bool = False,
                          dtype='complex128'):
    builder = SpinHam1D(S=1 / 2)
    builder += -4.0 * J, 'X', 'X'  # σxσx term
    builder += -2.0 * g, 'Z'       # σz term
    H_mpo = builder.build_mpo(L=L)
    return H_mpo

_HMPO_CACHE = {}
def get_tfim_mpo_cached(L, J, G, cyclic=False):
    key = (L, float(J), float(G), bool(cyclic))
    H = _HMPO_CACHE.get(key)
    if H is None:
        H = build_tfim_mpo_system(L, J, G, cyclic=cyclic)
        _HMPO_CACHE[key] = H
    return H

def energy_from_mps_mpo_system_only(mps_full: qtn.MatrixProductState, L_sys: int,
                                    J: float, g: float, chi_cut: int = 1,
                                    cutoff: float = 0.0, cyclic: bool = False,
                                    verbose: bool = False) -> float:
    """Compute E = ⟨ψ_S| H |ψ_S⟩ where H is TFIM MPO on the *system only*."""
    mps_c = mps_full.H
    # reindex conjugate physical legs for the *system* sites so MPO sits in between
    for idx in range(L_sys):
        mps_c = mps_c.reindex({f'k{idx}': f'b{idx}'})
    H_mpo = get_tfim_mpo_cached(L_sys, J, g, cyclic=cyclic)
    E = (mps_full & H_mpo & mps_c).contract()
    if verbose and RANK == 0:
        print(f"[debug] <H>={E}")
    return float(np.real(E))

# --------------- stochastic measure & reset of baths -------------------------------
def measure_and_reset_bath_stochastic(mps: qtn.MatrixProductState, bath_sites,
                                      rng: np.random.Generator, chi, cutoff,
                                      p_reset: float = 0.0):
    for idx in bath_sites:
        z_exp = _local_z_expectation(mps, idx, max_bond=chi, cutoff=cutoff)
        p0 = float(np.clip(0.5 * (1.0 + z_exp), 1e-12, 1.0 - 1e-12))
        if rng.random() < p0:
            mps.gate_(P0 / np.sqrt(p0), where=idx, contract='swap+split',
                      max_bond=chi, cutoff=cutoff)
        else:
            mps.gate_(P1 / np.sqrt(1.0 - p0), where=idx, contract='swap+split',
                      max_bond=chi, cutoff=cutoff)
            mps.gate_(X, where=idx, contract='swap+split',
                      max_bond=chi, cutoff=cutoff)
        if p_reset > 0.0 and rng.random() < p_reset:
            mps.gate_(X, where=idx, contract='swap+split',
                      max_bond=chi, cutoff=cutoff)
    mps.normalize()
    mps.compress(method='svd', max_bond=chi, cutoff=cutoff)

# ----------------- One RI cycle (single trajectory) --------------------------------
def ri_cycle_trajectory(mps: qtn.MatrixProductState, L: int, NB: int, J: float, g: float,
                        beta: float, delta: float, chi: int, cutoff: float,
                        rng: np.random.Generator, T_factor: float = 3.0,
                        trotter_order: int = 2, p1: float = 0.0, p2: float = 0.0,
                        p_reset: float = 0.0, system_sites=None, bath_sites=None,
                        site_ordering: str = "blocked",
                        h_override=None, theta_override=None, NT=None,
                        model="ising", mparams=None):
    """One repeated-interaction cycle with Gaussian filter, randomized geometry (one trajectory)."""
    fields, couplings = model_system_terms(model, J=J, g=g, **(mparams or {}))
    if trotter_order not in (1, 2):
        raise ValueError(f"trotter_order must be 1 or 2, got {trotter_order}.")
    validate_probability("p1", p1)
    validate_probability("p2", p2)
    validate_probability("p_reset", p_reset)
    if system_sites is None or bath_sites is None:
        site_ordering, system_sites, bath_sites = build_site_layout(
            L, NB, site_ordering
        )

    h = float(h_override) if h_override is not None else max(2 * g, 4 * J)
    a_true = math.sqrt(4.0 * h / max(beta, 1e-12))
    if NT is not None:
        # Jerome's window: a = delta*a_true (== his delta*sqrt(4h/beta)), MT = max(NT, int(NT/a))
        a_jer = delta * a_true
        MT = max(int(NT), int(int(NT) / a_jer)) if a_jer > 0 else int(NT)
    else:
        MT = max(2, int(math.ceil((T_factor / a_true) / delta)))
    f = gaussian_window(delta, MT, a_true)
    theta = float(theta_override) if theta_override is not None else math.sqrt(0.05 / math.sqrt(max(beta, 1e-12) * h))

    # With NB=L, bath labels are interchangeable, so fixed B_i-S_i pairing is
    # equivalent to a random bath-label permutation and enables local routing.
    if NB == L:
        sys_targets = np.arange(L)
    elif site_ordering == "distributed":
        # each bath couples to a RANDOM system site within its OWN region -> local gates (low
        # bond) AND full coverage (regions tile the chain, every site cooled over many cycles).
        edges = np.linspace(0, L, NB + 1).astype(int)
        sys_targets = np.array([int(rng.integers(int(edges[b]), max(int(edges[b]) + 1, int(edges[b + 1]))))
                                for b in range(NB)])
    else:
        sys_targets = rng.choice(L, size=min(NB, L), replace=False)
    peak_bond = int(mps.max_bond())

    if trotter_order == 1:
        Ub = build_bath_precession(h, delta)
        f1 = [expm(+1j * c * delta * P) for c, P in fields]
        c1 = [expm(+1j * c * delta * np.kron(Pl, Pr)) for c, Pl, Pr in couplings]
        for ft in f:
            for U in f1:
                apply_one_site_all_sys(mps, U, system_sites, chi, cutoff, rng, p1)
            for U in c1:
                apply_two_site_even_odd_sys(mps, U, system_sites, chi, cutoff, rng, p2)
            apply_bath_precession(mps, Ub, bath_sites, chi, cutoff, rng, p1)
            for mu, i_sys in enumerate(sys_targets):
                U_sb = build_sb_coupling(theta_ft=delta * theta * ft)
                apply_sys_bath_gate(
                    mps, system_sites[i_sys], bath_sites[mu],
                    U_sb, chi, cutoff, rng, p2
                )
            peak_bond = max(peak_bond, int(mps.max_bond()))
    else:
        # 2nd-order Strang: S(dt/2) [B(dt/2) C_j(dt) B(dt/2) S(dt)] ... S(dt/2).
        Ub_half = build_bath_precession(h, delta / 2.0)
        if model == "ising":
            # Ising keeps the validated leapfrog-merged step (Z/2 - XX - Z/2, 1q noise once).
            def sys_step(dur):
                Uz, _ = build_system_unis(J, g, dur / 2.0)
                _, Uxx = build_system_unis(J, g, dur)
                apply_system_strang(mps, system_sites, Uz, Uxx, chi, cutoff, rng, p1, p2)
        else:
            def sys_step(dur):
                apply_sys_step_generic(mps, system_sites, fields, couplings, dur,
                                       chi, cutoff, rng, p1, p2)
        sys_step(delta / 2.0)
        for j, ft in enumerate(f):
            apply_bath_precession(mps, Ub_half, bath_sites, chi, cutoff, rng, p1)
            for mu, i_sys in enumerate(sys_targets):
                U_sb = build_sb_coupling(theta_ft=delta * theta * ft)
                apply_sys_bath_gate(
                    mps, system_sites[i_sys], bath_sites[mu],
                    U_sb, chi, cutoff, rng, p2
                )
            apply_bath_precession(mps, Ub_half, bath_sites, chi, cutoff, rng, p1)
            sys_step(delta / 2.0 if j == len(f) - 1 else delta)
            peak_bond = max(peak_bond, int(mps.max_bond()))

    # stochastic trace+reset of baths
    measure_and_reset_bath_stochastic(
        mps, bath_sites, rng, chi, cutoff, p_reset=p_reset
    )

    return dict(MT=MT, theta=theta, peak_bond=peak_bond)

# ------------- Driver: many trajectories averaged (MPI) -----------------------------
def run_ri_tebd_energy_trajectories_mpi(L=6, J=1.0, g=1.0, beta=0.5, NB=3,
                                        ncycles=120, delta=np.pi/40, chi=256, cutoff=1e-10,
                                        sample_every=5, n_traj=32, seed=1,
                                        T_factor=3.0, trotter_order=2,
                                        p1=0.0, p2=0.0, p_reset=0.0,
                                        site_ordering="auto",
                                        h_override=None, theta_override=None, NT=None,
                                        model="ising", mparams=None,
                                        init='random'):
    """
    MPI-parallel version: each rank simulates a disjoint subset of trajectories and
    we average energies with an Allreduce at sampling checkpoints.
    Returns (sampled_cycles, E_av) on rank 0; (None, None) on others.
    """
    n_local = split_count(n_traj, SIZE, RANK)
    rng_master = np.random.default_rng(seed + 10000 * RANK)
    effective_ordering, system_sites, bath_sites = build_site_layout(
        L, NB, site_ordering
    )
    if init == 'random' and cutoff < 1e-7 and RANK == 0:
        print(f"[WARN] init='random' + cutoff={cutoff:.0e}: cooling from a random bitstring "
              f"builds volume-law entanglement transiently; a tight cutoff blows up the bond. "
              f"Use a noise-matched cutoff (~1e-6).", flush=True)
    trajectories = [make_initial_mps(L, NB, rng=rng_master, init=init) for _ in range(max(n_local, 0))]
    fields, couplings = model_system_terms(model, J=J, g=g, **(mparams or {}))

    sampled_cycles = []
    E_av = []

    for cyc in range(1, ncycles + 1):
        evolve_start = time.perf_counter()
        peak_bond_local = 1
        # advance each local trajectory one RI cycle
        for t in range(n_local):
            rng = np.random.default_rng(rng_master.integers(1, 10**9))
            cycle_info = ri_cycle_trajectory(
                trajectories[t], L, NB, J, g, beta, delta,
                chi, cutoff, rng, T_factor=T_factor,
                trotter_order=trotter_order, p1=p1, p2=p2,
                p_reset=p_reset,
                system_sites=system_sites, bath_sites=bath_sites,
                site_ordering=effective_ordering,
                h_override=h_override, theta_override=theta_override, NT=NT,
                model=model, mparams=mparams,
            )
            peak_bond_local = max(
                peak_bond_local, int(cycle_info["peak_bond"])
            )
        evolve_sec_local = time.perf_counter() - evolve_start
        final_bond_local = max(
            (int(trajectory.max_bond()) for trajectory in trajectories),
            default=1,
        )

        #if RANK == 0 and ((cyc % max(1, sample_every) == 0) or (cyc == ncycles)):
        #    print(f"cycle {cyc:03d} ...", flush=True)

        # sampling checkpoint
        if (cyc % sample_every) == 0 or cyc == ncycles:
            measure_start = time.perf_counter()
            # local sum
            E_sum_local = 0.0
            for t in range(n_local):
                Et = energy_from_mps(
                    trajectories[t], system_sites, fields, couplings,
                    max_bond=chi, cutoff=cutoff
                )
                E_sum_local += Et

            # reduce to global sum & count with explicit numpy buffers
            locE = np.array([E_sum_local], dtype='float64')
            locN = np.array([n_local], dtype='int64')
            if COMM is not None:
                globE = np.zeros_like(locE); COMM.Allreduce(locE, globE)
                globN = np.zeros_like(locN); COMM.Allreduce(locN, globN)
                E_sum_global = float(globE[0]); n_traj_global = int(globN[0])
                evolve_sec = float(COMM.allreduce(evolve_sec_local, op=MPI.MAX))
                measure_sec = float(COMM.allreduce(
                    time.perf_counter() - measure_start, op=MPI.MAX
                ))
                final_bond = int(COMM.allreduce(final_bond_local, op=MPI.MAX))
                peak_bond = int(COMM.allreduce(peak_bond_local, op=MPI.MAX))
            else:
                E_sum_global = float(locE[0]); n_traj_global = int(locN[0])
                evolve_sec = evolve_sec_local
                measure_sec = time.perf_counter() - measure_start
                final_bond = final_bond_local
                peak_bond = peak_bond_local

            if RANK == 0:
                E_mean = E_sum_global / max(n_traj_global, 1)
                sampled_cycles.append(cyc)
                E_av.append(E_mean)
                print(
                    f"[cycle {cyc:03d}]  E ≈ {E_mean:.6f}  "
                    f"(E/L ≈ {E_mean/L:.6f})  "
                    f"peak_bond={peak_bond}  final_bond={final_bond}  "
                    f"evolve_s={evolve_sec:.3f}  measure_s={measure_sec:.3f}",
                    flush=True,
                )

    if RANK == 0:
        print(np.array(sampled_cycles), np.array(E_av))
        return np.array(sampled_cycles), np.array(E_av)
    else:
        return None, None

# ------------------- Utilities: saving ---------------------------------------------
def ensure_parent_dir(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d and (not os.path.exists(d)) and (RANK == 0):
        os.makedirs(d, exist_ok=True)
    barrier()

def save_results(outbase: str, cycles, E, L, u_inf_val, params: dict):
    ensure_parent_dir(outbase)
    if RANK != 0:
        return

    np.savez_compressed(
        outbase + ".npz",
        cycles=cycles,
        energy=E,
        energy_per_site=(E / L),
        u_inf=u_inf_val,
        params_json=json.dumps(params, indent=2, sort_keys=True),
    )

    with open(outbase + ".csv", "w") as f:
        f.write("# cycles,E,E_per_site,u_inf\n")
        for c, e in zip(cycles, E):
            f.write(f"{c},{e:.16e},{(e/L):.16e},{u_inf_val:.16e}\n")

    print(f"[rank 0] saved: {outbase}.npz, {outbase}.csv", flush=True)

# ------------------- CLI & main ----------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="RI-TEBD trajectories (MPI, no plots)")
    p.add_argument("--L", type=int, default=20)
    p.add_argument("--J", type=float, default=1.0)
    p.add_argument("--g", type=float, default=1.0)
    p.add_argument("--model", choices=("ising", "xy", "heisenberg"), default="ising",
                   help="system Hamiltonian: ising (-J XX -g Z), xy (-Jxx XX -Jyy YY), "
                        "heisenberg (-Jxx XX -Jyy YY -Jzz ZZ).")
    p.add_argument("--gx", type=float, default=0.0, help="transverse-x field (ising only).")
    p.add_argument("--Jxx", type=float, default=None, help="XX coupling (default J).")
    p.add_argument("--Jyy", type=float, default=None, help="YY coupling (default J).")
    p.add_argument("--Jzz", type=float, default=None, help="ZZ coupling (default J).")
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--NB", type=int, default=3)
    p.add_argument("--ncycles", type=int, default=100)
    p.add_argument("--chi", type=int, default=256)
    p.add_argument("--cutoff", type=float, default=1e-6)
    p.add_argument("--sample_every", type=int, default=5)
    p.add_argument("--n_traj", type=int, default=64)
    p.add_argument("--delta", type=float, default=np.pi/40)
    p.add_argument("--T_factor", type=float, default=3.0)
    p.add_argument("--h", type=float, default=None,
                   help="override bath frequency h (default max(2g,4J)); Jerome uses 4.")
    p.add_argument("--theta", type=float, default=None,
                   help="override system-bath coupling theta (default formula); Jerome uses 1.")
    p.add_argument("--NT", type=int, default=None,
                   help="if set, use Jerome's window MT=max(NT,int(NT/a)) instead of T_factor.")
    p.add_argument("--trotter_order", type=int, choices=(1, 2), default=2)
    p.add_argument(
        "--p1", type=float, default=None,
        help="1q depolarizing probability per gate; default is p2/10.",
    )
    p.add_argument("--p2", type=float, default=0.0,
                   help="2q depolarizing probability per gate.")
    p.add_argument("--p_reset", type=float, default=0.0,
                   help="Bath bit-flip probability immediately after each reset.")
    p.add_argument(
        "--site_ordering", choices=("auto", "blocked", "interleaved", "distributed"), default="auto",
        help="MPS chain ordering; auto = interleaved (NB=L) / distributed (NB<L: local in-region "
             "coupling, low bond). 'blocked' is the old high-bond layout.",
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--init", choices=("cold", "random"), default="random")
    p.add_argument("--nk_inf", type=int, default=50_000)
    p.add_argument("--out", type=str, default=f"out/run_{int(time.time())}")
    p.add_argument("--smoke", action="store_true", help="run a tiny 1-rank test and exit")
    return p.parse_args()

def main():
    args = parse_args()
    p1 = args.p2 / 10.0 if args.p1 is None else args.p1
    validate_probability("p1", p1)
    validate_probability("p2", args.p2)
    validate_probability("p_reset", args.p_reset)

    params = dict(
        L=args.L, J=args.J, g=args.g, beta=args.beta, NB=args.NB,
        ncycles=args.ncycles, chi=args.chi, cutoff=args.cutoff,
        sample_every=args.sample_every, n_traj=args.n_traj,
        delta=args.delta, T_factor=args.T_factor,
        trotter_order=args.trotter_order, p1=p1, p2=args.p2,
        p_reset=args.p_reset, site_ordering=args.site_ordering, seed=args.seed,
        nk_inf=args.nk_inf, out=args.out, mpi_size=SIZE, mpi_rank=RANK,
    )
    noise_budget = gate_noise_budget(
        args.L, args.NB, args.J, args.g, args.beta, args.delta,
        args.T_factor, args.trotter_order, p1, args.p2,
        NT=args.NT, h_override=args.h,
    )
    params["noise_budget"] = noise_budget

    if args.smoke:
        if RANK == 0: print("[smoke] starting small test...")
        cycles, E = run_ri_tebd_energy_trajectories_mpi(
            L=6, J=1.0, g=1.0, beta=0.5, NB=2, ncycles=5,
            delta=np.pi/40, chi=64, cutoff=1e-8, sample_every=1,
            n_traj=SIZE, seed=123, T_factor=3.0,
            trotter_order=args.trotter_order, p1=p1, p2=args.p2,
            p_reset=args.p_reset, site_ordering=args.site_ordering,
        )
        if RANK == 0:
            u_inf_val = u_infinite(0.5, 1.0, 1.0, nk=20000)
            save_results(args.out, cycles, E, 6, u_inf_val, params)
        return

    if RANK == 0:
        print("=== Parameters ===")
        for k, v in params.items():
            if k != "mpi_rank":
                print(f"{k}: {v}")
        print("==================", flush=True)

    cycles, E = run_ri_tebd_energy_trajectories_mpi(
        L=args.L, J=args.J, g=args.g, beta=args.beta, NB=args.NB,
        ncycles=args.ncycles, delta=args.delta, chi=args.chi, cutoff=args.cutoff,
        sample_every=args.sample_every, n_traj=args.n_traj, seed=args.seed,
        T_factor=args.T_factor, trotter_order=args.trotter_order,
        p1=p1, p2=args.p2, p_reset=args.p_reset,
        site_ordering=args.site_ordering,
        h_override=args.h, theta_override=args.theta, NT=args.NT,
        model=args.model, mparams=dict(gx=args.gx, Jxx=args.Jxx, Jyy=args.Jyy, Jzz=args.Jzz),
        init=args.init,
    )

    if RANK == 0:
        u_inf_val = u_infinite(args.beta, args.J, args.g, nk=args.nk_inf)
        save_results(args.out, cycles, E, args.L, u_inf_val, params)

if __name__ == "__main__":
    main()
