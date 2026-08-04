"""Regenerate the key figures from results/*.json.  Usage: python plot_results.py
Reads ../results/{cc_trot_3x4,cc_cycle_beta_var,cc_heisenberg}.json, writes ../figures/*.png.
Only needs numpy + matplotlib."""
import json, os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); R=os.path.join(HERE,'..','results'); F=os.path.join(HERE,'..','figures')
os.makedirs(F,exist_ok=True)
def L(name): return json.load(open(os.path.join(R,name)))
def ser(d): ks=sorted(int(k) for k in d); return np.array(ks,float),np.array([d[str(k)] for k in ks])

# ---------- Fig 1: 1st vs 2nd vs OPTIMIZED Trotter, one cooling cycle (3x4+1, beta=1) ----------
t3=L('cc_trot_3x4.json'); v3=L('cc_cycle_beta_var.json')['1.0']['var']
Gg=t3['Gg']; eps=t3['eps_nat']; N=t3['native']
k1,i1=ser(t3['t1']); k2,i2=ser(t3['t2']); kv,iv=ser(v3)
plt.figure(figsize=(7.5,5.6))
plt.plot(k1*Gg,i1,'^-',color='#7f7f7f',ms=5,lw=1.7,label='1st-order Trotter')
plt.plot(k2*Gg,i2,'o-',color='#1f77b4',ms=5,lw=1.9,label='2nd-order Trotter (default)')
plt.plot(kv*Gg,iv,'s-',color='#d62728',ms=9,lw=2.3,label='optimized (tuned angles)')
plt.axhline(0.01,ls='--',color='k',lw=1.6); plt.text(250,0.0108,'1% cutoff',fontsize=9.5,fontweight='bold')
plt.axhline(eps,ls=':',color='gray',lw=1.1); plt.text(250,eps*1.05,f'native acc {eps*100:.1f}%',color='gray',fontsize=8.5)
plt.yscale('log'); plt.xlabel('2-qubit gates per cooling cycle'); plt.ylabel('channel infidelity vs ideal')
plt.title(f'One cooling-cycle compression, 3x4+1 (13q), beta=1\nnative full cycle = {N} 2q-gates (39 micro-steps)')
plt.grid(True,which='both',alpha=0.25); plt.legend(); plt.ylim(2e-3,0.7); plt.tight_layout()
plt.savefig(os.path.join(F,'cc_1st_2nd_opt.png'),dpi=150); plt.close()

# ---------- Fig 2: Heisenberg 1st vs 2nd-order Trotter (per-gate accuracy) ----------
H=L('cc_heisenberg.json')
fig,axs=plt.subplots(1,2,figsize=(13,5.3),sharey=True)
for ax,kind in zip(axs,('1D','2D')):
    r=H[kind]['res']; C=H[kind]['C']
    g1=[x['g1'] for x in r]; ii1=[x['i1'] for x in r]; g2=[x['g2'] for x in r]; ii2=[x['i2'] for x in r]
    ax.plot(g1,ii1,'o-',color='#7f7f7f',ms=6,lw=2,label='1st-order')
    ax.plot(g2,ii2,'s-',color='#1f77b4',ms=7,lw=2.2,label='2nd-order')
    ax.set_yscale('log'); ax.set_xlabel('2-qubit gate count (T=1)')
    ax.set_title(f'Heisenberg {kind} ({H[kind]["nq"]}q, C={C} classes)\n2nd-order overhead {g2[-1]/g1[-1]:.2f}x/step')
    ax.grid(True,which='both',alpha=0.25); ax.legend()
axs[0].set_ylabel('Trotter infidelity vs exact'); axs[0].set_ylim(1e-5,1.2)
plt.tight_layout(); plt.savefig(os.path.join(F,'cc_heisenberg.png'),dpi=150); plt.close()

# ---------- Fig 3: compression ratio vs beta (optimized vs 2nd-order), 3x4 ----------
B=L('cc_cycle_beta_var.json')
def cross(d,y):
    ks,i=ser(d); im=np.minimum.accumulate(i)
    for j in range(len(ks)-1):
        if im[j]>=y>im[j+1]:
            f=(np.log(im[j])-np.log(y))/(np.log(im[j])-np.log(im[j+1])); return ks[j]+f*(ks[j+1]-ks[j])
    return None
bs=sorted(B,key=float); rt=[]; rv=[]
for b in bs:
    e=B[b]; st=e['steps']; eps=e['eps_nat']
    kt=cross(e['t2'],eps); kv=cross(e['var'],eps)
    rt.append(st/kt if kt else np.nan); rv.append(st/kv if kv else np.nan)
plt.figure(figsize=(7,5.2))
plt.plot([float(b) for b in bs],rt,'o-',color='#1f77b4',ms=8,lw=2,label='2nd-order Trotter')
plt.plot([float(b) for b in bs],rv,'s-',color='#d62728',ms=10,lw=2.3,label='optimized')
plt.xscale('log'); plt.xticks([float(b) for b in bs],[b for b in bs])
plt.xlabel('inverse temperature  beta'); plt.ylabel('compression ratio (native / compressed @ native acc)')
plt.title('Compression vs temperature, 3x4 (beta>=2 optimizer-limited)')
plt.grid(True,alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(F,'cc_beta_sweep.png'),dpi=150); plt.close()
print("wrote figures/cc_1st_2nd_opt.png, cc_heisenberg.png, cc_beta_sweep.png")
