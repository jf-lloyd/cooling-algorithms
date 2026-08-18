"""Acceptance tests for the tilted staggered frame (phi) in GroundStateProtocol.

Run: python tests/test_ground_pc_phi.py
"""
import sys, numpy as np, cirq
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cooling
NS=NB=6
def mk(phi=None,order=1,g=0.5,Lx=None):
    if Lx is None: lat=cooling.ChainLattice1D(NS,pbc=False); nb=NB
    else:
        try: lat=cooling.SquareLattice2D(Lx,Lx,pbc_x=False,pbc_y=False)
        except TypeError: lat=cooling.SquareLattice2D(Lx,Lx,pbc=False)
        nb=Lx*Lx
    dev=cooling.CoolingDevice.from_lattice(lat,Nb=nb)
    mod=cooling.IsingModel(dev,{"J":1.0,"g":g,"gx":0.0})
    par={"T":0.5,"N":3,"h":2.0,"theta":1.1}
    if phi is not None: par["phi"]=phi
    pr=cooling.GroundStateProtocol(dev,mod,params=par,function="constant",
        noise_model=None,trotter_order=order,allow_iSWAP=True)
    ns=dev.Ns
    return pr,dev,{k:k%ns for k in range(nb)}

print("T1  向后兼容: 不传 phi vs phi=0 vs 旧行为")
for order in (1,2):
    pa,_,ga=mk(None,order); pb,_,gb=mk(0.0,order)
    ca=cirq.Circuit(pa.channel(ga,{k:'iSWAP' for k in ga}))
    cb=cirq.Circuit(pb.channel(gb,{k:'iSWAP' for k in gb}))
    ok = cirq.allclose_up_to_global_phase(
        cirq.Circuit(o for m in ca for o in m if not isinstance(o.gate,(cirq.ResetChannel,cirq.IdentityGate))).unitary(),
        cirq.Circuit(o for m in cb for o in m if not isinstance(o.gate,(cirq.ResetChannel,cirq.IdentityGate))).unitary())
    n1=sum(1 for o in ca.all_operations()); n2=sum(1 for o in cb.all_operations())
    print(f"   order={order}: unitary 相同={ok}  门数 {n1} vs {n2}  {'OK' if ok and n1==n2 else 'FAIL'}")

print("T2  两比特门数与 phi 无关 (编译后)")
for ph in (0.0,np.radians(30),np.radians(60),np.radians(90)):
    pr,dev,geo=mk(ph)
    c=cirq.Circuit(pr.channel(geo,{k:'iSWAP' for k in geo}))
    cz=sum(1 for o in c.all_operations() if len(o.qubits)==2 and isinstance(o.gate,cirq.CZPowGate))
    print(f"   phi={np.degrees(ph):5.1f}°: CZ={cz}")

print("T3  子格符号 + 非二分格子报错")
pr,dev,geo=mk(np.radians(60))
print(f"   1D chain signs: {pr.sublattice_signs}")
pr2,_,_=mk(np.radians(60),Lx=3)
print(f"   3x3 OBC signs : {pr2.sublattice_signs}")
try:
    lat=cooling.SquareLattice2D(3,3,pbc_x=True,pbc_y=True)
    dev=cooling.CoolingDevice.from_lattice(lat,Nb=9)
    mod=cooling.IsingModel(dev,{"J":1.0,"g":0.5,"gx":0.0})
    p3=cooling.GroundStateProtocol(dev,mod,params={"T":0.5,"N":3,"h":2.,"theta":1.1,"phi":1.0},
        function="constant",noise_model=None,trotter_order=1,allow_iSWAP=True)
    p3.sublattice_signs; print("   3x3 PBC: 未报错  FAIL")
except ValueError as e: print(f"   3x3 PBC: 正确报错 -> {str(e)[:70]}...")
except Exception as e: print(f"   3x3 PBC: 其它异常 {type(e).__name__}")

print("T4  物理基准: L=6 AFM, phi=50°, (T=.5,N=3,th=1.1,h=2.0) -> P_gs 0.9829 / F_brk 0.9826")
from scipy.linalg import expm
I2=np.eye(2,dtype=complex); PX=np.array([[0,1],[1,0]],complex); PZ=np.diag([1,-1]).astype(complex)
def OP(d):
    m=np.array([[1]],complex)
    for k in range(NS): m=np.kron(m,d.get(k,I2))
    return m
H0=sum(OP({i:PX,i+1:PX}) for i in range(NS-1))-0.5*sum(OP({i:PZ}) for i in range(NS))
ev,vec=np.linalg.eigh(H0); E0v,E1v=vec[:,0],vec[:,1]
PSIP=(E0v+E1v)/np.sqrt(2); PSIM=(E0v-E1v)/np.sqrt(2)
lat=cooling.ChainLattice1D(NS,pbc=False); dev=cooling.CoolingDevice.from_lattice(lat,Nb=NB)
mod=cooling.IsingModel(dev,{"J":1.0,"g":0.5,"gx":0.0})
pr=cooling.GroundStateProtocol(dev,mod,params={"T":0.5,"N":3,"h":2.0,"theta":1.1,"phi":np.radians(50)},
    function="constant",noise_model=None,trotter_order=1,allow_iSWAP=True)
geo={k:k%NS for k in range(NB)}
circ=cirq.Circuit(pr.channel(geo,{k:'iSWAP' for k in geo},compile=False))
uni=cirq.Circuit(o for m in circ for o in m if not isinstance(o.gate,(cirq.ResetChannel,cirq.IdentityGate)))
U=uni.unitary(qubit_order=list(dev.system_qubits)+list(dev.bath_qubits))
DS=DB=64
K=np.ascontiguousarray(U.reshape(DS,DB,DS,DB)[:,:,:,0].transpose(1,0,2))
tr=np.einsum('jab,jac->bc',K.conj(),K); assert abs(tr-np.eye(DS)).max()<1e-8,"not TP"
Kd=K.conj().transpose(0,2,1); rho=np.eye(DS,dtype=complex)/DS; Ep=None
for r in range(6000):
    rho=np.einsum('jab,bc,jcd->ad',K,rho,Kd,optimize=True)
    E=float(np.real(np.trace(rho@H0)))
    if Ep is not None and abs(E-Ep)<1e-12: break
    Ep=E
P=float(np.real(E0v.conj()@rho@E0v+E1v.conj()@rho@E1v))
fb=max(float(np.real(PSIP.conj()@rho@PSIP)),float(np.real(PSIM.conj()@rho@PSIM)))
print(f"   P_gs={P:.4f}  F_brk={fb:.4f}  E rel={(E-ev[0])/abs(ev[0])*100:.2f}%  "
      f"{'OK' if abs(P-0.9829)<0.002 and abs(fb-0.9826)<0.002 else 'FAIL'}")
