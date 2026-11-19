"""Replicated widths of the joint admissible set for the frame seen down a cube diagonal (referee #25),
from the closed form of Proposition 14, which reproduces the intersection of single-junction intervals.
Vertices uniform over the sampled square of half-width 480 px; 1000 px image; noiseless."""
import numpy as np
rng=np.random.default_rng(2026)
def diagonal_frame():
    z=np.array([0,0,1.0]); g=np.array([1,1,1.0])/np.sqrt(3)
    v=np.cross(g,z); sn=np.linalg.norm(v); ct=g@z
    vx=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3)+vx+vx@vx*((1-ct)/sn**2)
R=diagonal_frame(); zax=R[2,:]
def width(fov,M):
    f=500.0/np.tan(np.radians(fov)/2); Gp=-np.inf; Gm=-np.inf; done=0
    while done<M:
        n=min(200000,M-done); u=rng.uniform(-480,480,n); v=rng.uniform(-480,480,n)
        r=np.stack([u/f,v/f,np.ones(n)],1); Z=r@R            # zeta_a = r . d_a
        W=zax[None,:]/Z
        for a,p,q in ((0,1,2),(1,2,0),(2,0,1)):
            g=(zax[a]-Z[:,a]*W[:,p])*(zax[a]-Z[:,a]*W[:,q])
            Gp=max(Gp,g.max()); Gm=max(Gm,(-g).max())
        done+=n
    up=Gp/(Gp-1) if Gp>1 else np.inf; lo=Gm/(1+Gm) if Gm>0 else 0.0
    return np.sqrt(up)-np.sqrt(lo)
Ms=[1000,3000,10000,30000,100000,300000]
print(f"{'FOV':>4}  "+"  ".join(f"M={m:>6}" for m in Ms)+"   slope over 3e4..3e5 [95% CI]")
for fov in (54,56,58,60,65,70):
    reps=30; W=np.array([[width(fov,m) for m in Ms] for _ in range(reps)])
    med=np.median(W,0)
    x=np.log(Ms[3:]); boot=[]
    for _ in range(2000):
        idx=rng.integers(0,reps,reps); mb=np.median(W[idx][:,3:],0)
        if np.all(np.isfinite(mb)) and np.all(mb>0): boot.append(np.polyfit(x,np.log(mb),1)[0])
    sl=np.polyfit(x,np.log(med[3:]),1)[0] if np.all(np.isfinite(med[3:])) else np.nan
    ci=np.percentile(boot,[2.5,97.5]) if boot else [np.nan,np.nan]
    print(f"{fov:>4}  "+"  ".join(f"{w:>8.2e}" for w in med)+f"   {sl:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]",flush=True)
