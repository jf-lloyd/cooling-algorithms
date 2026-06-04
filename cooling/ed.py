"""
Exact diagonalisation routines using QuSpin.
ModelSpec returns (and saves in spectra/) the spectrum of a given Model, or returns the full Hamiltonian.
ThermalEnergy returns the thermal energy for given temperature and lowest k energies.

Created by Jerome Lloyd on 3rd June 2026
"""

import numpy as np
import scipy
import pickle
import os
import itertools

from quspin.operators import hamiltonian
from quspin.basis import spin_basis_general


PATH = "spectra/"

# Basis rotations that make the Hamiltonian compatible with QuSpin's spin-inversion
# operator prod_i X_i. Each rotation maps the offending single-site field to X
# (which commutes with spin-inversion). Two-site PP terms always commute.
#
# Hadamard (X<->Z): for models with a Z single-site field only
_ROT_HADAMARD  = {'xx': 'zz', 'zz': 'xx', 'yy': 'yy', 'z': 'x', 'x': 'z', 'y': 'y'}
# pi/2 around Z (Y -> -X, X -> Y): for models with a Y single-site field only
_ROT_Z90       = {'xx': 'yy', 'yy': 'xx', 'zz': 'zz', 'x': 'y', 'y': 'x', 'z': 'z'}
_ROT_Z90_SIGNS = {'y': -1}   # Y -> -X requires negating coupling strengths


def _apply_rotation(bonds, rot_map, sign_map):
    """Apply a basis rotation to a QuSpin bond dict, including any sign flips."""
    result = {}
    for op, bl in bonds.items():
        new_op = rot_map.get(op, op)
        sign = sign_map.get(op, 1)
        if sign == -1:
            bl = [[-b[0]] + b[1:] for b in bl]
        result[new_op] = bl
    return result


def saver(data, fname, path):
    with open(path + fname + '.pkl', 'wb') as f:
        pickle.dump(data, f, protocol=2)


def loader(fname, path):
    with open(path + fname + ".pkl", 'rb') as f:
        return pickle.load(f)


def _to_quspin_bonds(coupling_lists):
    """Convert model.coupling_lists to QuSpin bond lists (lowercase operator strings)."""
    result = {}
    for op, terms in coupling_lists.items():
        if len(op) == 2:
            result[op.lower()] = [[J, s, t] for J, s, t in terms]
        else:
            result[op.lower()] = [[g, s] for g, s in terms]
    return result


def _build_H(model, basis):
    """Build QuSpin Hamiltonian from model.coupling_lists. Returns (H, components)."""
    bonds = _to_quspin_bonds(model.coupling_lists)
    kw = dict(dtype=np.complex128, check_herm=False, check_symm=False, check_pcon=False)
    components = {op: hamiltonian([[op, bl]], [], basis=basis, **kw)
                  for op, bl in bonds.items()}
    H = hamiltonian([[op, bl] for op, bl in bonds.items()], [], basis=basis, **kw)
    return H, components


def _lattice_geometry(lattice):
    """
    Return (Lx, Ly, pbc_x, pbc_y) in a unified format.
    For ChainLattice1D: Ly=1, pbc_x=lattice.pbc, pbc_y=False.
    For SquareLattice2D or TriangularLattice2D: direct.
    """
    if hasattr(lattice, 'L'):
        return lattice.L, 1, lattice.pbc, False
    elif hasattr(lattice, 'Lx'):
        return lattice.Lx, lattice.Ly, lattice.pbc_x, lattice.pbc_y
    raise NotImplementedError(
        f"Symmetry sectors not implemented for {type(lattice).__name__}.")


def _is_uniform(coupling_lists):
    """True if all coupling strengths are uniform (required for translation/parity)."""
    return all(
        np.allclose([t[0] for t in terms], [t[0] for t in terms][0])
        for terms in coupling_lists.values()
    )


def ModelSpec(model, k=None, save=True, verbal=True, return_H=False,
              translation=True, parity=True, Z2=True, U1=True):
    """
    QuSpin ED for any Model. Symmetries are detected and applied automatically.

    Checks (in order): translation, parity, U1 (Nup), Z2 spin-inversion.
    Set any flag to False to disable that symmetry.

    Note: Z2 and U1 are mutually exclusive. If U1 is active, Z2 is skipped
    because spin-inversion maps Nup -> Ns-Nup, making it incompatible with
    fixed-Nup sectors.
    """
    fname = f"{model.name}_Spec"
    fname_path = os.path.join(PATH, fname + ".pkl")
    kw = dict(dtype=np.float64, check_herm=False, check_symm=False, check_pcon=False)
    Ns = model.Ns
    basis_full = spin_basis_general(Ns, pauli=True)

    if return_H:
        H, _ = _build_H(model, basis_full)
        return H

    if os.path.isfile(fname_path):
        if verbal: print("loading pre-computed energies", fname)
        Elist = loader(fname, path=PATH)
        return Elist if k is None else Elist[:k]
    if verbal: print("calculating new energies", fname)

    cl = model.coupling_lists
    Lx, Ly, pbc_x, pbc_y = _lattice_geometry(model.lattice)
    s = np.arange(Ns)
    x, y = s % Lx, s // Lx
    is_1d = (Ly == 1)
    uniform = _is_uniform(cl)

    sym_names, sym_perms, sym_qnums = [], [], []

    # --- check for translation symmetry ---
    any_pbc = pbc_x or pbc_y
    translation_active = False
    if translation and any_pbc:
        if uniform:
            T_x = (x + 1) % Lx + Lx * y
            T_y = x + Lx * ((y + 1) % Ly)
            if pbc_x:
                sym_names.append('kxblock'); sym_perms.append(T_x); sym_qnums.append(list(range(Lx)))
                if verbal: print("  translation symmetry (x) found")
            if pbc_y:
                sym_names.append('kyblock'); sym_perms.append(T_y); sym_qnums.append(list(range(Ly)))
                if verbal: print("  translation symmetry (y) found")
            translation_active = True
        else:
            if verbal: print("  PBC present but couplings non-uniform — skipping translation symmetry")

    # --- check for parity (reflection) symmetry ---
    # Incompatible with translation on PBC rings (parity maps k -> -k).
    # For 1D: reverse chain. For 2D: x-reflection.
    if parity and uniform and not translation_active:
        P = np.arange(Ns - 1, -1, -1) if is_1d else (Lx - 1 - x) + Lx * y
        sym_names.append('pblock'); sym_perms.append(P); sym_qnums.append([0, 1])
        if verbal: print("  parity symmetry found")
    elif parity and translation_active:
        if verbal: print("  parity skipped: incompatible with translation symmetry on PBC lattice")

    # --- check for U(1) / total Sz symmetry ---
    # Must be evaluated before Z2: if U1 is active, Z2 is skipped.
    single_site_fields = {op for op in ('X', 'Y', 'Z') if op in cl}
    has_xx, has_yy = 'XX' in cl, 'YY' in cl
    conserves_sz = U1 and (
        not single_site_fields & {'X', 'Y'}
        and (not (has_xx or has_yy)
             or (has_xx and has_yy and np.isclose(cl['XX'][0][0], cl['YY'][0][0])))
    )
    nup_list = list(range(Ns + 1)) if conserves_sz else [None]
    if conserves_sz and verbal:
        print("  U(1) / Nup symmetry found")

    # --- check for Z2 spin-inversion symmetry ---
    # Skipped if U1 is active: spin-inversion maps Nup -> Ns-Nup,
    # so it is incompatible with fixed-Nup sectors.
    rotation = None
    if Z2 and not conserves_sz:
        Z_perm = -(s + 1)
        if len(single_site_fields) <= 1:
            if single_site_fields <= {'X'}:
                sym_names.append('zblock'); sym_perms.append(Z_perm); sym_qnums.append([0, 1])
                if verbal: print("  Z2 spin-inversion symmetry found")
            elif single_site_fields == {'Z'}:
                rotation = (_ROT_HADAMARD, {})
                sym_names.append('zblock'); sym_perms.append(Z_perm); sym_qnums.append([0, 1])
                if verbal: print("  Z2 spin-inversion symmetry found (Hadamard rotation: Z->X)")
            elif single_site_fields == {'Y'}:
                rotation = (_ROT_Z90, _ROT_Z90_SIGNS)
                sym_names.append('zblock'); sym_perms.append(Z_perm); sym_qnums.append([0, 1])
                if verbal: print("  Z2 spin-inversion symmetry found (pi/2 Z-rotation: Y->-X)")
        else:
            if verbal: print("  Z2 skipped: multiple single-site field types present")
    elif Z2 and conserves_sz:
        if verbal: print("  Z2 skipped: incompatible with U(1) Nup sectors")

    # Build operator list (applying basis rotation if needed for Z2)
    bonds = _to_quspin_bonds(cl)
    if rotation is not None:
        rot_map, sign_map = rotation
        bonds = _apply_rotation(bonds, rot_map, sign_map)
    ops = list(bonds.items())

    # Translation sectors with k!=0 have complex Bloch factors even for real operators.
    # Use float32 (real) for OBC and complex64 for PBC — matches old code precision and speed.
    sector_dtype = np.float32 if not translation_active else np.complex64
    sector_kw = dict(kw, dtype=sector_dtype)

    def get_basis(sector, nup=None):
        sym = {name: (perm, q) for name, perm, q in zip(sym_names, sym_perms, sector)}
        if nup is not None:
            return spin_basis_general(Ns, Nup=nup, pauli=True, **sym)
        return spin_basis_general(Ns, pauli=True, **sym)

    def build_H_for_basis(basis):
        return hamiltonian([[op, bl] for op, bl in ops], [], basis=basis, **sector_kw)

    if k is None:
        Elist = []
        for sector in itertools.product(*sym_qnums):
            for nup in nup_list:
                H = build_H_for_basis(get_basis(sector, nup))
                E, _ = H.eigh()
                Elist.extend(E)
        assert len(Elist) == 2 ** Ns, \
            f"Sector decomposition incomplete: got {len(Elist)} eigenvalues, expected {2**Ns}"
        Elist = np.array(sorted(Elist))
        if save: saver(Elist, fname, PATH)
        return Elist
    else:
        H, _ = _build_H(model, basis_full)
        if k >= H.shape[0]:
            Elist, _ = H.eigh()
        else:
            Elist = H.eigsh(k=k, which="SA", maxiter=int(1e8), return_eigenvectors=False)
        return np.sort(np.real(Elist))[:k]


def ThermalEnergy(model, beta=None, k=10, **spec_kwargs):
    """
    Thermal energy at inverse temperature beta, or lowest k energies if beta=None.
    spec_kwargs are passed to ModelSpec (save, verbal, translation, parity, Z2, U1).
    """
    energies = ModelSpec(model, **spec_kwargs)
    if beta is None:
        return energies[:k]
    weights = np.exp(-beta * energies)
    weights /= weights.sum()
    return energies[:k], np.dot(energies, weights)
