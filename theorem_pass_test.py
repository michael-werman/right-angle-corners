"""Brute-force checks of three results no outside reader has checked:
Theorem 6 (closed-form interval), Proposition 5(2) (unbounded iff orthographic rule),
and the mirror theorem (only the true pair of hypotheses is consistent)."""
import numpy as np
rng=np.random.default_rng(7)
# ---- Theorem 6 and Prop 5(2): random junctions, F from a fine sign chart of the exact product
bad6=bad5=n=0
mus=np.concatenate([np.geomspace(1e-6,1e6,40001)])
for _ in range(20000):
    psi=rng.uniform(-np.pi,np.pi,3); rho=rng.uniform(1,500)
    if np.min(np.abs(np.sin(psi)))<1e-3: continue
    lam={}
    for i,j in ((0,1),(1,2),(2,0)): lam[(i,j)]=np.cos(psi[i]-psi[j])/(np.sin(psi[i])*np.sin(psi[j]))
    if min(abs(np.sin(psi[i]-psi[j])) for i,j in lam)<1e-3: continue          # distinct lines, margin
    n+=1
    prod=np.ones_like(mus)
    for (i,j),l in lam.items(): prod*=(np.cos(psi[i]-psi[j])*mus*rho**2+rho**2*np.sin(psi[i])*np.sin(psi[j]))   # A f^2 + e e
    neg=prod<0
    S=[k for k,l in lam.items() if l<0]; th=sorted(rho/np.sqrt(-lam[k]) for k in S)
    f=np.sqrt(mus)*rho
    pred=np.zeros_like(neg)
    if len(S)==1: pred=f>th[0]
    elif len(S)==2: pred=(f>th[0])&(f<th[1])
    near=np.zeros_like(neg)
    for t in th: near|=np.abs(f/t-1)<1e-3
    bad6+= np.any((neg!=pred)&~near) or len(S)>2
    A=np.prod([np.cos(psi[i]-psi[j]) for i,j in lam]); unb=neg[-1]
    bad5+= (A<0)!=unb if abs(A)>1e-6 else 0
print(f"Theorem 6: closed form against the exact sign chart: {n-bad6}/{n} junctions agree")
print(f"Prop 5(2): unbounded above <=> product of raw-angle cosines < 0: {n-bad5}/{n} agree")
# ---- mirror theorem
def H(r): r=r/np.linalg.norm(r); return np.eye(3)-2*np.outer(r,r)
def is_sign_diag(M): return np.allclose(np.abs(M),np.eye(3),atol=1e-9)
cnt={'true only':0,'other':0}; par_ok=0; npar=0
for _ in range(20000):
    Q,_r=np.linalg.qr(rng.standard_normal((3,3))); R=Q*np.sign(np.diag(_r))
    D=np.diag(rng.choice([-1,1],3)); r1=rng.standard_normal(3); r1[2]=abs(r1[2])+0.3; r2=rng.standard_normal(3); r2[2]=abs(r2[2])+0.3
    if min(abs((R[:,k]/1)@(r/np.linalg.norm(r))) for r in (r1,r2) for k in range(3))>0.999: continue
    hyps=[(G1,G2) for G1 in (np.eye(3),H(r1)) for G2 in (np.eye(3),H(r2))]
    cons=[is_sign_diag((G1@R).T@(G2@R@D)) for G1,G2 in hyps]
    cnt['true only' if cons==[True,False,False,False] else 'other']+=1
    # parallel rays: the double flip is consistent too
    rp=r1*rng.uniform(0.5,2); hp=[(G1,G2) for G1 in (np.eye(3),H(r1)) for G2 in (np.eye(3),H(rp))]
    cp=[is_sign_diag((G1@R).T@(G2@R@D)) for G1,G2 in hp]; npar+=1; par_ok+= cp==[True,False,False,True]
print(f"Theorem 19: non-parallel rays, only the true pair consistent: {cnt['true only']}/{sum(cnt.values())}")
print(f"            parallel rays, true pair and double flip consistent: {par_ok}/{npar}")
