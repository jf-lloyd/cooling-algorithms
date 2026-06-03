import sys
import numpy as np
import scipy
import pickle
import os

from quspin.operators import hamiltonian, commutator # Hamiltonians and operators
from quspin.basis import spin_basis_general 
from quspin.basis import spin_basis_1d

def saver(data, fname, path):
    with open(path+fname+'.pkl', 'wb') as f:
        pickle.dump(data, f, protocol=2)
        
def loader(fname, path):
    with open(path+fname+".pkl", 'rb') as f:
        data = pickle.load(f)
    return data

PATH = "spectra/"

def IsingModelSpec(model, k=None, save=True, verbal=True, return_H=False, return_h_operators=False, delta=0):

    J = model.params["J"]
    g = model.params["g"]
    gx = model.params["gx"]
    Ns = model.Ns
    dim = model.lattice.Dim
    
    if dim == 1:
        Lx = model.lattice.L
        Ly = 1
        pbc = model.lattice.pbc
    elif dim == 2:
        Lx = model.lattice.Lx
        Ly = model.lattice.Ly
        pbc_x = model.lattice.pbc_x
        pbc_y = model.lattice.pbc_y

    fname = f"{model.name}_d{delta:.3f}_Spec"

    fname_path = os.path.join(PATH, fname + ".pkl")
    # --- load or generate energies ---
    if (return_H is False) and (return_h_operators is False):
        if os.path.isfile(fname_path):
            if verbal:
                print("loading pre-computed energies", fname)
            Elist = loader(fname, path=PATH)
    
            if k is None:
                return Elist
            else:
                return Elist[:k]        
        else:
            if verbal:
                print("calculating new energies", fname)

        # N = Lx*Ly
        # s = np.arange(N)
        # x = s%Lx
        # y = s//Lx
        # T_x = (x+1)%Lx+y*Lx
        # T_y = x+((y+1)%Ly)*Lx

        # ## Hamiltonian in QuSpin 
        # basis_2d = spin_basis_general(N, pauli=True) # 2d - basis
        # #print("Size of 2D H-space: {Ns:d}".format(Ns=basis_2d.Ns))

        # ibc = 0
        # if pbc is False:
        #     ibc = 1

        # Jx = [[J,ix+Lx*iy,T_x[ix+Lx*iy]] for ix in range(0, Lx-ibc) for iy in range(0, Ly)]
        # Jy = [[J,ix+iy*Lx,T_y[ix+iy*Lx]] for ix in range(0, Lx) for iy in range(0, Ly-ibc)]
        # gall = [[-g,i] for i in s]

        # HJ_ops = [['xx',Jx], ['xx',Jy]]
        # Hg_ops = [['z',gall]]

        # HJ = hamiltonian(HJ_ops,[],basis=basis_2d,dtype=np.complex64, check_herm=False, check_symm=False)
        # Hg = hamiltonian(Hg_ops,[],basis=basis_2d,dtype=np.complex64, check_herm=False, check_symm=False)

    
    # Two-site XX term
    Jbonds = []
    if J != 0.0:
        for (s, t) in model.lattice.nearest_neighbour_pairs():
            Jbonds.append([J,s,t])
            
    # Transverse field
    gbonds = []
    if g != 0.0:
        for i in range(Ns):
            gbonds.append([-g,i])
    
    # X field
    gxbonds = []
    if gx != 0.0:
        for i in range(Ns):
            gxbonds.append([-gx,i])


    if return_h_operators:

        ### check this code! 
        HJ_ops = [['xx',Jbonds]] # note quspin expects ZZ+X Ising
        Hg_ops = [['z',gbonds]]
        Hgx_ops = [['x',gxbonds]]
    
        basis = spin_basis_general(Ns, pauli=True) # 2d - basis

        HJ = hamiltonian(HJ_ops,[],basis=basis,dtype=np.complex64, check_herm=False, check_symm=False)
        Hg = hamiltonian(Hg_ops,[],basis=basis,dtype=np.complex64, check_herm=False, check_symm=False)
        Hgx = hamiltonian(Hgx_ops,[],basis=basis,dtype=np.complex64, check_herm=False, check_symm=False)

        H0 = HJ+Hg+Hgx
        H1 = 0.
        if delta != 0:
            cm = commutator
            H1 = (delta)/2j*(cm(Hgx,Hg)+cm(Hgx,HJ)+cm(Hg,HJ)) # 1-order Floquet 
        
        H = H0+H1 

        hJ_ops = [hamiltonian([['zz', [bond]]],[],basis=basis,dtype=np.complex64,
                              check_herm=False, check_symm=False) for bond in Jbonds]
        hg_ops = [hamiltonian([['x', [bond]]],[],basis=basis,dtype=np.complex64,
                              check_herm=False, check_symm=False) for bond in gbonds]
        hgx_ops = [hamiltonian([['z', [bond]]],[],basis=basis,dtype=np.complex64,
                              check_herm=False, check_symm=False) for bond in gxbonds]

        Sx_ops = [hamiltonian([['z', [[1.,i]]]],[],basis=basis,dtype=np.complex64,
                              check_herm=False, check_symm=False) for i in range(Ns)]
        Sy_ops = [hamiltonian([['y', [[1.,i]]]],[],basis=basis,dtype=np.complex64,
                      check_herm=False, check_symm=False) for i in range(Ns)]
        Sz_ops = [hamiltonian([['x', [[1.,i]]]],[],basis=basis,dtype=np.complex64,
                      check_herm=False, check_symm=False) for i in range(Ns)]
        
        return H, hJ_ops, hg_ops, hgx_ops, Sx_ops, Sy_ops, Sz_ops

    if return_H:
        
        HJ_ops = [['xx',Jbonds]] 
        Hg_ops = [['z',gbonds]]
        Hgx_ops = [['x',gxbonds]]
    
        basis = spin_basis_general(Ns, pauli=True) # 2d - basis

        HJ = hamiltonian(HJ_ops,[],basis=basis,dtype=np.complex64, check_herm=False, check_symm=False)
        Hg = hamiltonian(Hg_ops,[],basis=basis,dtype=np.complex64, check_herm=False, check_symm=False)
        Hgx = hamiltonian(Hgx_ops,[],basis=basis,dtype=np.complex64, check_herm=False, check_symm=False)
        
        H0 = HJ+Hg+Hgx
        H1 = 0.
        H2 = 0.
        if delta != 0:
            cm = commutator
            H1 = (delta)/2j*(cm(Hgx,Hg)+cm(Hgx,HJ)+cm(Hg,HJ)) # 1-order Floquet 
            
            # def ncm(A, B, C):
            #     return cm(A, cm(B, C))

            # # K = [Hgx,HJ] + [Hg,HJ]
            # K = cm(Hgx, HJ) + cm(Hg, HJ)
        
            # # H2 here is the coefficient of δ^2 in Heff, so the contribution to Heff is δ^2 * H2.
            # # We return it already multiplied by δ^2 for convenience (mirroring H1).
            # H2 = -(delta**2) * (
            #     (1/12) * ( ncm(Hgx, Hgx, Hg) - ncm(Hg, Hgx, Hg) )
            #     + (1/4)  * cm(cm(Hgx, Hg), HJ)
            #     + (1/12) * ( cm(Hgx + Hg, K) - cm(HJ, K) ))
            
        H = H0+H1+H2
        return H

    elif k is None: 
        # return full spectrum
        # get basis
        HJ_ops = [['zz',Jbonds]] # note quspin expects ZZ+X Ising for symmetry
        Hg_ops = [['x',gbonds]]
        Hgx_ops = [['z',gxbonds]]

        kxlist = [None]
        kylist = [None]
        zlist = [None]
    
        if dim == 1:
            if pbc is True:
                kxlist = range(Lx)
                
        elif dim == 2:
            if pbc_x is True:
                kxlist = range(Lx)
            if pbc_y is True:
                kylist = range(Ly)
                
        if gx == 0.: # spin inversion symmetry
            zlist = [0,1]
            
        s = np.arange(Ns) # sites [0 ,1,2,....]
        x = s%Lx # x positions for sites
        y = s//Lx # y positions for sites
        T_x = (x+1)%Lx + Lx*y # translation along x-direction
        T_y = x+Lx*((y+1)%Ly) # translation along y-direction
        Z = -(s+1)  # spin inversion
    
        def get_basis(Ns, z=None,kx=None,ky=None):
    
            symmetries = {}
            if z is not None:
                symmetries['zblock'] = (Z,z)
            if kx is not None:
                symmetries['kxblock'] = (T_x,kx)
            if ky is not None:
                symmetries['kyblock'] = (T_y,ky)
            
            basis = spin_basis_general(Ns, pauli=True, **symmetries)
            return basis
            

    
        Elist = []
        for kx in kxlist:
            for ky in kylist:
                for z in zlist:
    
                    basis = get_basis(Ns,z,kx,ky)
        
                    HJ = hamiltonian(HJ_ops,[],basis=basis,dtype=np.complex64, check_herm=False, check_symm=False)
                    Hg = hamiltonian(Hg_ops,[],basis=basis,dtype=np.complex64, check_herm=False, check_symm=False)
                    Hgx = hamiltonian(Hgx_ops,[],basis=basis,dtype=np.complex64, check_herm=False, check_symm=False)
                    
                    H0 = HJ+Hg+Hgx
                    H1 = 0.
                    if delta != 0:
                        cm = commutator
                        H1 = (delta)/2j*(cm(Hgx,Hg)+cm(Hgx,HJ)+cm(Hg,HJ)) # 1-order Floquet 
                    H = H0+H1 
                    H_dense = H.toarray()
                    E, V = np.linalg.eigh(H_dense)
                    Elist.extend(E)
    
        Elist = sorted(Elist)
        assert len(Elist) == 2**(Lx*Ly)
        if save:
            saver(np.array(Elist), fname, PATH)
        return np.array(Elist)
    
    else:
        HJ_ops = [['zz',Jbonds]] # note quspin expects ZZ+X Ising for symmetry
        Hg_ops = [['x',gbonds]]
        Hgx_ops = [['z',gxbonds]]
        basis = spin_basis_general(Ns, pauli=True) # 2d - basis

        HJ = hamiltonian(HJ_ops,[],basis=basis,dtype=np.complex64, check_herm=False, check_symm=False)
        Hg = hamiltonian(Hg_ops,[],basis=basis,dtype=np.complex64, check_herm=False, check_symm=False)
        Hgx = hamiltonian(Hgx_ops,[],basis=basis,dtype=np.complex64, check_herm=False, check_symm=False)
        
        H0 = HJ+Hg+Hgx
        H1 = 0.
        if delta != 0:
            cm = commutator
            H1 = (delta)/2j*(cm(Hgx,Hg)+cm(Hgx,HJ)+cm(Hg,HJ)) # 1-order Floquet 
        H = H0+H1 

        if k >= H.shape[0]:
            Elist, V = scipy.linalg.eigh(H.toarray())
            Elist = Elist[:k]
        else:
            Elist = H.eigsh(k=k,which="SA",maxiter=1E8,return_eigenvectors=False)
    
        Elist = sorted(Elist)
        return np.array(Elist)


def IsingModelThermalEnergy(model, beta=None, delta=0., k=10):
    energies = IsingModelSpec(model)
    if beta is None: #GS
        return energies[:k]
    weights = np.exp(-beta*energies)
    weights /= np.sum(weights)
    thermal_energy = np.dot(energies,weights)
    return energies[:k], thermal_energy

def IsingModelThermalBondExpectations(model, beta=None, direct=True, delta=0.):

    try:
        assert model.lattice.Dim == 1
    except:
        print("the following code is currently implemented for d=1 only!")
        return
    
    H, HJ_ops, Hg_ops, Hgx_ops, Sx_ops, Sy_ops, Sz_ops = IsingModelSpec(model, return_h_operators=True, delta=delta)
    E, V = H.eigh()
    w = np.exp(-beta*E)
    Z = w.sum()
    p = w / Z
    
    bond_ops = []
    for i, Jop in enumerate(HJ_ops):
        bond_op = Jop + 1/2*(Hg_ops[i]+Hg_ops[i+1]+Hgx_ops[i]+Hgx_ops[i+1])
        bond_ops.append(bond_op)
    
    if model.lattice.pbc is False:
        boundary_ops = [1/2*(Hg_ops[0]+Hgx_ops[0]), 1/2*(Hg_ops[-1]+Hgx_ops[-1])]

    bond_ops_exp = []
    boundary_ops_exp = []
    Sx_ops_exp = []
    Sy_ops_exp = []
    Sz_ops_exp = []
    
    if direct:
        for O in bond_ops:
            OV = O.dot(V) 
            diag = np.einsum("ij,ij->j", V.conj(), OV)
            exp = np.real(np.dot(p, diag))
            bond_ops_exp.append(exp)
        for O in boundary_ops:
            OV = O.dot(V) 
            diag = np.einsum("ij,ij->j", V.conj(), OV)
            exp = np.real(np.dot(p, diag))
            boundary_ops_exp.append(exp)
        for O in Sx_ops:
            OV = O.dot(V) 
            diag = np.einsum("ij,ij->j", V.conj(), OV)
            exp = np.real(np.dot(p, diag))
            Sx_ops_exp.append(exp)
        for O in Sy_ops:
            OV = O.dot(V) 
            diag = np.einsum("ij,ij->j", V.conj(), OV)
            exp = np.real(np.dot(p, diag))
            Sy_ops_exp.append(exp)   
        for O in Sz_ops:
            OV = O.dot(V) 
            diag = np.einsum("ij,ij->j", V.conj(), OV)
            exp = np.real(np.dot(p, diag))
            Sz_ops_exp.append(exp)

    print("energies", np.sum(bond_ops_exp)+np.sum(boundary_ops_exp), np.dot(E,p))
    
    return bond_ops_exp, boundary_ops_exp, Sx_ops_exp, Sy_ops_exp, Sz_ops_exp


def HeisModelSpec(model, k=None, save=True, verbal=True, return_H=False, delta=0.):

    Jx = model.params["Jx"]
    Jy = model.params["Jy"]
    Jz = model.params["Jz"]
    Ns = model.Ns
    dim = model.lattice.Dim
    
    if dim == 1:
        Lx = model.lattice.L
        Ly = 1
        pbc = model.lattice.pbc
    elif dim == 2:
        Lx = model.lattice.Lx
        Ly = model.lattice.Ly
        pbc_x = model.lattice.pbc_x
        pbc_y = model.lattice.pbc_y

    fname = f"{model.name}_d{delta:.3f}_Spec"

    fname_path = os.path.join(PATH, fname + ".pkl")
    # --- load or generate energies ---
    if (return_H is False):
        if os.path.isfile(fname_path):
            if verbal:
                print("loading pre-computed energies", fname)
            Elist = loader(fname, path=PATH)
    
            if k is None:
                return Elist
            else:
                return Elist[:k]        
        else:
            if verbal:
                print("calculating new energies", fname)

    # Two-site XX term
    Jxbonds = []
    if Jx != 0.0:
        for (s, t) in model.lattice.nearest_neighbour_pairs():
            Jxbonds.append([Jx,s,t])

    # Two-site YY term
    Jybonds = []
    if Jy != 0.0:
        for (s, t) in model.lattice.nearest_neighbour_pairs():
            Jybonds.append([Jy,s,t])

    # Two-site ZZ term
    Jzbonds = []
    if Jz != 0.0:
        for (s, t) in model.lattice.nearest_neighbour_pairs():
            Jzbonds.append([Jz,s,t])

    if return_H:
        
        HJx_ops = [['xx',Jxbonds]] 
        HJy_ops = [['yy',Jybonds]] 
        HJz_ops = [['zz',Jzbonds]] 
    
        basis = spin_basis_general(Ns, pauli=True) # 2d - basis

        HJx = hamiltonian(HJx_ops,[],basis=basis,dtype=np.complex64, check_herm=False, check_symm=False)
        HJy = hamiltonian(HJy_ops,[],basis=basis,dtype=np.complex64, check_herm=False, check_symm=False)
        HJz = hamiltonian(HJz_ops,[],basis=basis,dtype=np.complex64, check_herm=False, check_symm=False)
        
        H0 = HJx+HJy+HJz
        H1 = 0.
        if delta != 0:
            cm = commutator
            H1 = (delta)/2j*(cm(HJz,HJy)+cm(HJz,HJx)+cm(HJy,HJx)) # 1-order Floquet 
        
        H = H0+H1 
        return H

    elif k is None: 
        # return full spectrum
        # get basis
        HJx_ops = [['xx',Jxbonds]] 
        HJy_ops = [['yy',Jybonds]] 
        HJz_ops = [['zz',Jzbonds]] 

        kxlist = [None]
        kylist = [None]
    
        if dim == 1:
            if pbc is True:
                kxlist = range(Lx)
                
        elif dim == 2:
            if pbc_x is True:
                kxlist = range(Lx)
            if pbc_y is True:
                kylist = range(Ly)

        s = np.arange(Ns) # sites [0 ,1,2,....]
        x = s%Lx # x positions for sites
        y = s//Lx # y positions for sites
        T_x = (x+1)%Lx + Lx*y # translation along x-direction
        T_y = x+Lx*((y+1)%Ly) # translation along y-direction
    
        def get_basis(Ns, kx=None,ky=None):
    
            symmetries = {}
            if kx is not None:
                symmetries['kxblock'] = (T_x,kx)
            if ky is not None:
                symmetries['kyblock'] = (T_y,ky)
            
            basis = spin_basis_general(Ns, pauli=True, **symmetries)
            return basis
            
        Elist = []
        for kx in kxlist:
            for ky in kylist:
    
                basis = get_basis(Ns,kx,ky)
    
                HJx = hamiltonian(HJx_ops,[],basis=basis,dtype=np.complex64, check_herm=False, check_symm=False)
                HJy = hamiltonian(HJy_ops,[],basis=basis,dtype=np.complex64, check_herm=False, check_symm=False)
                HJz = hamiltonian(HJz_ops,[],basis=basis,dtype=np.complex64, check_herm=False, check_symm=False)
   
                H0 = HJx+HJy+HJz
                H1 = 0.
                if delta != 0:
                    cm = commutator
                    H1 = (delta)/2j*(cm(HJz,HJy)+cm(HJz,HJx)+cm(HJy,HJx)) # 1-order Floquet 
                H = H0+H1 
                H_dense = H.toarray()
                E, V = np.linalg.eigh(H_dense)
                Elist.extend(E)
    
        Elist = sorted(Elist)
        assert len(Elist) == 2**(Lx*Ly)
        if save:
            saver(np.array(Elist), fname, PATH)
        return np.array(Elist)
    
    else:
        HJx_ops = [['xx',Jxbonds]] 
        HJy_ops = [['yy',Jybonds]] 
        HJz_ops = [['zz',Jzbonds]] 
        basis = spin_basis_general(Ns, pauli=True) # 2d - basis

        HJx = hamiltonian(HJx_ops,[],basis=basis,dtype=np.complex64, check_herm=False, check_symm=False)
        HJy = hamiltonian(HJy_ops,[],basis=basis,dtype=np.complex64, check_herm=False, check_symm=False)
        HJz = hamiltonian(HJz_ops,[],basis=basis,dtype=np.complex64, check_herm=False, check_symm=False)
      
        H0 = HJx+HJy+HJz
        H1 = 0.
        if delta != 0:
            cm = commutator
            H1 = (delta)/2j*(cm(HJz,HJy)+cm(HJz,HJx)+cm(HJy,HJx)) # 1-order Floquet 
        H = H0+H1 

        if k >= H.shape[0]:
            Elist, V = scipy.linalg.eigh(H.toarray())
            Elist = Elist[:k]
        else:
            Elist = H.eigsh(k=k,which="SA",maxiter=1E8,return_eigenvectors=False)
    
        Elist = sorted(Elist)
        return np.array(Elist)

def HeisModelThermalEnergy(model, k=10, beta=None, delta=0.):
    energies = HeisModelSpec(model, delta=delta)
    if beta is None: #GS
        return energies[:k]
    weights = np.exp(-beta*energies)
    weights /= np.sum(weights)
    thermal_energy = np.dot(energies,weights)
    return energies[:k], thermal_energy
