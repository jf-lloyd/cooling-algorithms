"""Heisenberg (H = sum_<ij> XX+YY+ZZ) 1st- vs 2nd-order Trotter: 2q-GATE COUNT (with boundary merge)
and ACCURACY vs exact ED.  Tests the 2nd-order gate overhead = (2C-2)/C for C non-commuting 2-body
color classes (1D C=2 -> 1x free; 2D C=4 -> 1.5x), unlike Ising (commuting -> always free)."""
import numpy as np, scipy.linalg as sla, json
np.random.seed(0)
X=np.array([[0,1],[1,0]],complex); Y=np.array([[0,-1j],[1j,0]]); Z=np.diag([1,-1]).astype(complex)
hb=np.kron(X,X)+np.kron(Y,Y)+np.kron(Z,Z)                      # XX+YY+ZZ on a bond (4x4)
def Ub(t): return sla.expm(-1j*t*hb).reshape(2,2,2,2)

def geom(kind):
    if kind=='1D':
        L=12; nq=L; bonds=[(i,i+1) for i in range(L-1)]
        classes=[[(i,i+1) for i in range(0,L-1,2)], [(i,i+1) for i in range(1,L-1,2)]]
    else:                                                      # 2D 3x4
        Lx,Ly=3,4; nq=Lx*Ly; idx=lambda x,y:x+y*Lx
        he,ho,ve,vo=[],[],[],[]
        for y in range(Ly):
            for x in range(Lx):
                if x+1<Lx: (he if x%2==0 else ho).append((idx(x,y),idx(x+1,y)))
                if y+1<Ly: (ve if y%2==0 else vo).append((idx(x,y),idx(x,y+1)))
        classes=[he,ho,ve,vo]; bonds=[b for c in classes for b in c]
    return nq,bonds,[c for c in classes if c]

def Hfull(nq,bonds):
    H=np.zeros((2**nq,2**nq),complex)
    for (a,b) in bonds:
        for P in (X,Y,Z):
            ops=[np.eye(2,dtype=complex)]*nq; ops[a]=P; ops[b]=P
            M=np.array([[1]],complex)
            for o in ops: M=np.kron(M,o)
            H+=M
    return H

def apply2(psi,U,a,b):
    psi=np.tensordot(psi,U,axes=([a,b],[2,3])); return np.moveaxis(psi,[-2,-1],[a,b])

def layers_1st(C,k): return [(ci,1.0) for _ in range(k) for ci in range(C)]
def layers_2nd(C,k):                                           # symmetric Suzuki over the C classes
    per=[(ci,0.5) for ci in range(C-1)]+[(C-1,1.0)]+[(ci,0.5) for ci in range(C-2,-1,-1)]
    raw=per*k; merged=[]                                       # merge adjacent same-class (boundary merge)
    for ci,fr in raw:
        if merged and merged[-1][0]==ci: merged[-1]=(ci,merged[-1][1]+fr)
        else: merged.append([ci,fr])
    return [tuple(m) for m in merged]
def count2q(classes,layers): return sum(len(classes[ci]) for ci,_ in layers)
def evolve(psi0,classes,layers,dt,nq):
    psi=psi0.reshape((2,)*nq).copy()
    for ci,fr in layers:
        U=Ub(fr*dt)
        for (a,b) in classes[ci]: psi=apply2(psi,U,a,b)
    return psi.reshape(-1)

out={}
for kind in ('1D','2D'):
    nq,bonds,classes=geom(kind); C=len(classes); T=1.0
    H=Hfull(nq,bonds); Uex=sla.expm(-1j*T*H)
    st=[np.random.randn(2**nq)+1j*np.random.randn(2**nq) for _ in range(4)]; st=[s/np.linalg.norm(s) for s in st]
    ex=[Uex@s for s in st]
    print(f"\n=== Heisenberg {kind}: {nq}q, {len(bonds)} bonds, C={C} non-commuting classes, T={T} ===",flush=True)
    print(f"{'k':>3} {'1st gates':>10} {'1st infid':>11} {'2nd gates':>10} {'2nd infid':>11} {'gate 2nd/1st':>13}",flush=True)
    res=[]
    for k in (1,2,3,4,6,8,12,16,20):
        dt=T/k; l1=layers_1st(C,k); l2=layers_2nd(C,k); g1=count2q(classes,l1); g2=count2q(classes,l2)
        i1=float(np.mean([1-abs(np.vdot(ex[i],evolve(st[i],classes,l1,dt,nq)))**2 for i in range(4)]))
        i2=float(np.mean([1-abs(np.vdot(ex[i],evolve(st[i],classes,l2,dt,nq)))**2 for i in range(4)]))
        res.append(dict(k=k,g1=g1,i1=i1,g2=g2,i2=i2))
        print(f"{k:>3} {g1:>10} {i1:>11.3e} {g2:>10} {i2:>11.3e} {g2/g1:>12.2f}x",flush=True)
    out[kind]=dict(nq=nq,nbonds=len(bonds),C=C,res=res)
json.dump(out,open('cc_heisenberg.json','w'),indent=2)
print("\nsaved cc_heisenberg.json",flush=True)
