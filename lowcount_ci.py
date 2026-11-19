"""What does the bound give at the junction counts a real image supplies?
E5 found 1.8 full junctions per image, so the interesting range is M = 1..10,
not M = 50."""
import numpy as np
from scipy.stats import norm
from common import *
rng=np.random.default_rng(0); HALF=500.0; ell=150.0; f=866.0
def place(rng):
    c=rng.uniform(-HALF*0.95,HALF*0.95,2); z=rng.uniform(2,6)
    ray=np.linalg.inv(Kmat(f))@np.array([c[0],c[1],1.0]); return ray/ray[2]*z
def one(sigma):
    while True:
        o=observe_corner(place(rng),rand_frame(rng),f,sigma,rng)
        if o is None: continue
        c,Ls=o; co=coeffs(c,Ls); comps=admissible(co)
        if not comps: continue
        rho=np.hypot(c[0],c[1]); psi=[]
        for l in Ls:
            a,b=l[0],l[1]; d=np.array([b,-a])
            psi.append(np.arctan2(c[0]*d[1]-c[1]*d[0], d[0]*c[0]+d[1]*c[1]))
        sd=[]
        for i,j in ((0,1),(1,2),(2,0)):
            D=np.cos(psi[i]-psi[j])
            sd.append(sigma*rho**2*np.hypot(np.sin(2*psi[i]),np.sin(2*psi[j]))
                      /(np.sqrt(2)*ell*D*D) if sigma>0 else 1e-9)
        xth=np.array([(-E/A if abs(A)>1e-300 else np.nan) for A,E in co])
        return co, comps, xth, np.array(sd)
xs=np.linspace((0.3*f)**2,(3*f)**2,3000)
def ll(data):
    out=np.zeros_like(xs)
    for co,_,xth,sd in data:
        v=np.array([[A*x+E for A,E in co] for x in xs])
        S=np.sign(np.prod(v,axis=1)); p=np.zeros_like(v)
        for k in range(3):
            if np.isfinite(xth[k]) and sd[k]>0:
                p[:,k]=norm.cdf(-np.abs(xs-xth[k])/sd[k])
        out+=np.log(np.clip((1-S*np.prod(1-2*p,axis=1))/2,1e-12,1))
    return out
def consensus(ivs,k):
    ev=sorted([(a,1) for a,_ in ivs]+[(b,-1) for _,b in ivs])
    cnt=0; lo=None; hi=None
    for xv,d in ev:
        prev=cnt; cnt+=d
        if prev<k<=cnt and lo is None: lo=xv
        if prev>=k>cnt: hi=xv
    return (lo,hi) if lo is not None else None
# Table 3 with bootstrap intervals; empty strict intersections counted apart from bounded ones (referee #25)
rngb=np.random.default_rng(99)
def bci(v):
    v=np.asarray(v); b=[np.median(rngb.choice(v,len(v))) for _ in range(1000)]; return np.percentile(b,[2.5,97.5])
print(f"{'sigma':>5}{'M':>4}{'bounded,nonempty':>17}{'empty':>7}{'unbounded':>10}{'width|bnd':>10}{'maximizer err [95% CI]':>26}")
for sigma in (0.5,2.0):
    for M in (1,2,3,5,10,20,50):
        nb=ne=nu=0; w=[]; em=[]; T=200
        for _ in range(T):
            data=[one(sigma) for _ in range(M)]
            lo=max(d[1][0][0] for d in data); hi=min(d[1][-1][1] for d in data)
            if hi>=1e13: nu+=1
            elif hi<lo: ne+=1
            else: nb+=1; w.append((np.sqrt(hi)-np.sqrt(lo))/f)
            em.append(abs(np.sqrt(xs[int(np.argmax(ll(data)))])-f)/f)
        ci=bci(em); ws=f"{np.median(w):.2f}" if w else "--"
        print(f"{sigma:>5}{M:>4}{100*nb/T:>16.0f}%{100*ne/T:>6.0f}%{100*nu/T:>9.0f}%{ws:>10}{100*np.median(em):>12.1f}% [{100*ci[0]:.1f}, {100*ci[1]:.1f}]",flush=True)
