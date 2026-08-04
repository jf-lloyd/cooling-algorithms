"""Fixed 2nd-order Trotter (NO optimization) channel infidelity vs ideal, vs depth k.
Same machinery as cc_cycle_adam.py (forward()), forward-only -> fast at any size.
Usage: cc_trot_sweep.py Lx Ly [beta]"""
import numpy as np, json, torch, sys
torch.set_num_threads(8); cd=torch.complex64; fd=torch.float32
Lx,Ly,beta=(int(sys.argv[1]),int(sys.argv[2]),float(sys.argv[3])) if len(sys.argv)>3 else (3,4,1.0)
J,g,theta=-1.0,1.0,0.6; delta=0.4*(0.1*np.pi/2); NT=5; h=4.0
nq=Lx*Ly; nb=1; nn=nq+nb; bq=nq; cpl=nq//2
def idx(x,y): return x+y*Lx
bonds=[(idx(x,y),idx(x+1,y)) for y in range(Ly) for x in range(Lx) if x+1<Lx]+[(idx(x,y),idx(x,y+1)) for y in range(Ly) for x in range(Lx) if y+1<Ly]
nB=len(bonds); Gg=nB+1
a=delta*np.sqrt(4*h/beta); MT=max(NT,int(NT/a)); steps=2*MT+1
def gauss_cpl(M): ts=np.linspace(-MT,MT,M); gg=np.exp(-a**2*ts**2/2); return gg/gg.sum()*theta
Zn=np.array([[1,0],[0,-1]],complex); Yn=np.array([[0,-1j],[1j,0]])
OPm=torch.tensor(np.kron((Zn+Yn)/np.sqrt(2),Yn).reshape(2,2,2,2),dtype=cd)
def rin(s):
    rng=np.random.default_rng(s); st=np.ones((1,),complex)
    for q in range(nq):
        v=rng.normal(size=2)+1j*rng.normal(size=2); v/=np.linalg.norm(v); st=np.kron(st,v)
    return np.kron(st,np.array([1,0],complex))
Ztot_np=np.zeros((2,)*nn)
for q in range(nq): Ztot_np=Ztot_np+np.array([1.,-1.]).reshape([2 if i==q else 1 for i in range(nn)])
Ztot=torch.tensor(Ztot_np,dtype=fd)
NS=6; psi0=torch.tensor(np.stack([rin(s) for s in range(NS)]).reshape((NS,)+(2,)*nn),dtype=cd)
def xx(psi,th,p,q): return torch.cos(th)*psi-1j*torch.sin(th)*torch.flip(psi,dims=(p+1,q+1))
def zf(psi,ph,q):
    sh=[1]*psi.ndim; sh[q+1]=2; return psi*torch.stack([torch.exp(1j*ph),torch.exp(-1j*ph)]).reshape(sh)
def zfall(psi,phi): return psi*torch.exp(1j*phi*Ztot)
def cpgate(psi,al):
    OPpsi=torch.movedim(torch.tensordot(psi,OPm,dims=([cpl+1,bq+1],[2,3])),(-2,-1),(cpl+1,bq+1))
    return torch.cos(al)*psi-1j*torch.sin(al)*OPpsi
def forward(th,ph,bf,al):                # 2nd-order (Strang: Z/2 - XX - Z/2)
    psi=psi0
    for l in range(th.shape[0]):
        psi=zfall(psi,ph[l]/2)
        for (i,j) in bonds: psi=xx(psi,th[l],i,j)
        psi=zfall(psi,ph[l]/2); psi=zf(psi,bf[l],bq); psi=cpgate(psi,al[l])
    return psi.reshape(NS,2**nq,2)
def forward1(th,ph,bf,al):               # 1st-order (sequential: XX - Z - bath - coupling)
    psi=psi0
    for l in range(th.shape[0]):
        for (i,j) in bonds: psi=xx(psi,th[l],i,j)
        psi=zfall(psi,ph[l]); psi=zf(psi,bf[l],bq); psi=cpgate(psi,al[l])
    return psi.reshape(NS,2**nq,2)
def trot2(M):
    th=torch.full((M,),J*delta*steps/M,dtype=fd); ph=torch.full((M,),g*delta*steps/M,dtype=fd)
    bf=torch.full((M,),h*delta/2*steps/M,dtype=fd); al=torch.tensor(gauss_cpl(M),dtype=fd)
    with torch.no_grad(): return forward(th,ph,bf,al)
Aid=trot2(160)
def native1st():
    th=torch.full((steps,),J*delta,dtype=fd); ph=torch.full((steps,),g*delta,dtype=fd)
    bf=torch.full((steps,),h*delta/2,dtype=fd); al=torch.tensor(gauss_cpl(steps),dtype=fd)
    psi=psi0
    with torch.no_grad():
        for l in range(steps):
            for (i,j) in bonds: psi=xx(psi,th[l],i,j)
            psi=zfall(psi,ph[l]); psi=zf(psi,bf[l],bq); psi=cpgate(psi,al[l])
    return psi.reshape(NS,2**nq,2)
def chan_infid(A):
    Mi=Aid.conj().transpose(-1,-2)@A; num=(Mi.abs()**2).sum((-1,-2))
    gi=(Aid.conj().transpose(-1,-2)@Aid).abs().pow(2).sum((-1,-2)).sqrt()
    ga=(A.conj().transpose(-1,-2)@A).abs().pow(2).sum((-1,-2)).sqrt()
    return float(1-(num/(gi*ga)).mean())
eps_nat=chan_infid(native1st())
print(f"{Lx}x{Ly}+1 ({nn}q) steps={steps} native={steps*Gg} 2q/cyc eps_nat={eps_nat:.4f}",flush=True)
res1={}; res2={}
for k in range(2,21,2):
    th=torch.full((k,),J*delta*steps/k,dtype=fd); ph=torch.full((k,),g*delta*steps/k,dtype=fd)
    bf=torch.full((k,),h*delta/2*steps/k,dtype=fd); al=torch.tensor(gauss_cpl(k),dtype=fd)
    with torch.no_grad():
        i1=chan_infid(forward1(th,ph,bf,al)); i2=chan_infid(forward(th,ph,bf,al))
    res1[k]=i1; res2[k]=i2
    print(f"  k={k:2d} ({k*Gg:3d} gates, {steps/k:.2f}x): 1st-Trotter={i1:.3e}  2nd-Trotter={i2:.3e}",flush=True)
json.dump(dict(Lx=Lx,Ly=Ly,beta=beta,steps=steps,native=steps*Gg,Gg=Gg,eps_nat=eps_nat,t1=res1,t2=res2),
          open(f'cc_trot_{Lx}x{Ly}.json','w'),indent=2)
print(f"saved cc_trot_{Lx}x{Ly}.json",flush=True)
