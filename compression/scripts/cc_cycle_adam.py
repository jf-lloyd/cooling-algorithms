"""Variational compression of one cooling cycle (3x4+1) with torch autodiff + Adam.
Same gate set as 2nd-order Trotter (XX on bonds, Z fields, coupling), angles free per layer.
Target = IDEAL channel (fine 2nd-order Trotter). Dense 13q (exact). Many k points."""
import numpy as np, scipy.linalg as spla, json, torch, sys
torch.set_num_threads(8)
cd=torch.complex64; fd=torch.float32
Lx,Ly,beta=(int(sys.argv[1]),int(sys.argv[2]),float(sys.argv[3])) if len(sys.argv)>3 else (3,4,1.0)
J,g,theta=-1.0,1.0,0.6; delta=0.4*(0.1*np.pi/2); NT=5; h=4.0
nq=Lx*Ly; nb=1; nn=nq+nb; bq=nq; cpl=nq//2
def idx(x,y): return x+y*Lx
bonds=[(idx(x,y),idx(x+1,y)) for y in range(Ly) for x in range(Lx) if x+1<Lx]+[(idx(x,y),idx(x,y+1)) for y in range(Ly) for x in range(Lx) if y+1<Ly]
nB=len(bonds); Gg=nB+1
a=delta*np.sqrt(4*h/beta); MT=max(NT,int(NT/a)); steps=2*MT+1
def gauss_cpl(M):
    ts=np.linspace(-MT,MT,M); gg=np.exp(-a**2*ts**2/2); return gg/gg.sum()*theta
Zn=np.array([[1,0],[0,-1]],complex); Yn=np.array([[0,-1j],[1j,0]])
OPm=torch.tensor(np.kron((Zn+Yn)/np.sqrt(2),Yn).reshape(2,2,2,2),dtype=cd)   # OP, OP^2=I
def rin(s):
    rng=np.random.default_rng(s); st=np.ones((1,),complex)
    for q in range(nq):
        v=rng.normal(size=2)+1j*rng.normal(size=2); v/=np.linalg.norm(v); st=np.kron(st,v)
    return np.kron(st,np.array([1,0],complex))
Ztot_np=np.zeros((2,)*nn)
for q in range(nq):
    Ztot_np=Ztot_np+np.array([1.,-1.]).reshape([2 if i==q else 1 for i in range(nn)])
Ztot=torch.tensor(Ztot_np,dtype=fd)
NS=6; psi0=torch.tensor(np.stack([rin(s) for s in range(NS)]).reshape((NS,)+(2,)*nn),dtype=cd)
# differentiable gates (real angle params)
def xx(psi,th,p,q): return torch.cos(th)*psi-1j*torch.sin(th)*torch.flip(psi,dims=(p+1,q+1))
def zf(psi,ph,q):
    sh=[1]*psi.ndim; sh[q+1]=2; ph2=torch.stack([torch.exp(1j*ph),torch.exp(-1j*ph)]).reshape(sh); return psi*ph2
def zfall(psi,phi): return psi*torch.exp(1j*phi*Ztot)
def cpgate(psi,al):
    OPpsi=torch.movedim(torch.tensordot(psi,OPm,dims=([cpl+1,bq+1],[2,3])),(-2,-1),(cpl+1,bq+1))
    return torch.cos(al)*psi-1j*torch.sin(al)*OPpsi
def forward(th,ph,bf,al):                 # 2nd-order (Strang: Z/2 - XX - Z/2)
    psi=psi0; k=th.shape[0]
    for l in range(k):
        psi=zfall(psi,ph[l]/2)
        for (i,j) in bonds: psi=xx(psi,th[l],i,j)
        psi=zfall(psi,ph[l]/2)
        psi=zf(psi,bf[l],bq); psi=cpgate(psi,al[l])
    return psi.reshape(NS,2**nq,2)
def forward1(th,ph,bf,al):                # 1st-order (sequential: XX - Z - bath - coupling)
    psi=psi0; k=th.shape[0]
    for l in range(k):
        for (i,j) in bonds: psi=xx(psi,th[l],i,j)
        psi=zfall(psi,ph[l])
        psi=zf(psi,bf[l],bq); psi=cpgate(psi,al[l])
    return psi.reshape(NS,2**nq,2)
# ideal (fine 2nd-order Trotter, M=160) — build once, no grad
def trot2(M):
    th=torch.full((M,),J*delta*steps/M,dtype=fd); ph=torch.full((M,),g*delta*steps/M,dtype=fd)
    bf=torch.full((M,),h*delta/2*steps/M,dtype=fd); al=torch.tensor(gauss_cpl(M),dtype=fd)
    with torch.no_grad(): return forward(th,ph,bf,al)
Aid=trot2(160)
def native1st():                                  # exact native protocol (1st-order, `steps` micro-steps)
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
    return 1-(num/(gi*ga)).mean()
eps_nat=float(chan_infid(native1st()))
print(f"{Lx}x{Ly}+1 ({nq}+1={nn}q) beta={beta} steps={steps} nB={nB} native={steps*Gg} 2q/cyc | eps_nat(1st vs ideal)={eps_nat:.4f} | torch+Adam",flush=True)
import os
KS=tuple(int(x) for x in os.environ.get("KS","6,8,10,12,14").split(","))
NIT=int(os.environ.get("NIT","2500")); NREST=int(os.environ.get("NREST","3"))
FWD={1:forward1, 2:forward}                        # VARIATIONAL ON BOTH ORDERS (same gate set/count, free angles)
print(f"VARIATIONAL both orders: KS={KS} NIT={NIT} NREST={NREST} | lr=0.02 cosine, bad>800, perturbed restarts | eps_nat={eps_nat:.4f}",flush=True)
res={1:{},2:{}}
for k in KS:
    base=(np.full(k,J*delta*steps/k), np.full(k,g*delta*steps/k), np.full(k,h*delta/2*steps/k), gauss_cpl(k))
    for order in (1,2):
        fwd=FWD[order]; kbest=1.0; bestit=0
        for r in range(NREST):
            rng=np.random.default_rng(100+r); pert=0.0 if r==0 else 0.2
            ps=[torch.tensor(b+pert*rng.standard_normal(k),dtype=fd,requires_grad=True) for b in base]
            opt=torch.optim.Adam(ps,lr=0.02); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=NIT)
            best=1.0; bad=0
            for it in range(NIT):
                opt.zero_grad(); L=chan_infid(fwd(*ps)); L.backward(); opt.step(); sch.step()
                l=float(L)
                if l<best-1e-9: best=l; bad=0
                else: bad+=1
                if bad>800: break
            if best<kbest: kbest=best; bestit=it+1
        res[order][k]=kbest
        tag="  <= native acc" if kbest<=eps_nat else ""
        print(f"k={k:2d} order={order} ({k*Gg:3d} gates): optimized = {kbest:.3e}  [conv {bestit} steps]{tag}",flush=True)
    json.dump(dict(Lx=Lx,Ly=Ly,beta=beta,steps=steps,native=steps*Gg,Gg=Gg,eps_nat=eps_nat,opt1=res[1],opt2=res[2]),
              open(f'cc_cycle_adam_{Lx}x{Ly}.json','w'),indent=2)     # incremental save after each k (survives kills)
print(f"saved cc_cycle_adam_{Lx}x{Ly}.json",flush=True)
