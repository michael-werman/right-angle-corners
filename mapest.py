"""A likelihood for f from many junctions, replacing the depth-k consensus.

For a candidate x = f^2, junction i has three linear forms.  By Prop. 4 the
probability that form j has the WRONG sign at x is p_j = Phi(-|x-x_ij|/sd_ij).
Admissibility is "an odd number of forms negative", so the observed verdict is
correct iff an even number of signs have flipped:

    P(even # flips) = (1 + prod_j (1 - 2 p_j)) / 2 ,

and with S = sign of the observed product,

    pi_i(x) = ( 1 - S * prod_j (1 - 2 p_j) ) / 2 .

The log-likelihood is sum_i log pi_i(x).  (The review's proposed
prod_i (1 - Phi(|f^2-x_ij|/sd)) is maximised where the margins are SMALLEST,
i.e. inverted, and ignores that admissibility is a parity condition.)
"""
import numpy as np
from scipy.stats import norm
from common import *
rng=np.random.default_rng(0); HALF=500.0; L=150.0; f_true=866.0

def place(rng):
    c=rng.uniform(-HALF*0.95,HALF*0.95,2); z=rng.uniform(2,6)
    ray=np.linalg.inv(Kmat(f_true))@np.array([c[0],c[1],1.0]); return ray/ray[2]*z

def junction(sigma):
    while True:
        o=observe_corner(place(rng),rand_frame(rng),f_true,sigma,rng)
        if o is None: continue
        c,Ls=o; co=coeffs(c,Ls)
        rho=np.hypot(c[0],c[1])
        psi=[]
        for a,b,e in [np.array([l[0],l[1],-(l[0]*c[0]+l[1]*c[1])]) for l in Ls]:
            d=np.array([b,-a])
            psi.append(np.arctan2(c[0]*d[1]-c[1]*d[0], d[0]*c[0]+d[1]*c[1]))
        sd=[]
        for i,j in ((0,1),(1,2),(2,0)):
            D=np.cos(psi[i]-psi[j])
            sd.append(sigma*rho**2*np.hypot(np.sin(2*psi[i]),np.sin(2*psi[j]))
                      /(np.sqrt(2)*L*D*D))
        x=[(-E/A if abs(A)>1e-300 else np.nan) for A,E in co]
        return co, np.array(x), np.array(sd)

def loglik(xs, data):
    out=np.zeros_like(xs)
    for co,xth,sd in data:
        v=np.array([[A*x+E for A,E in co] for x in xs])     # (nx,3)
        S=np.sign(np.prod(v,axis=1))
        p=np.zeros_like(v)
        for k in range(3):
            if not np.isfinite(xth[k]) or sd[k]<=0: continue
            p[:,k]=norm.cdf(-np.abs(xs-xth[k])/sd[k])
        pi=(1-S*np.prod(1-2*p,axis=1))/2
        out+=np.log(np.clip(pi,1e-12,1))
    return out

def consensus(ivs,k):
    ev=sorted([(a,1) for a,_ in ivs]+[(b,-1) for _,b in ivs])
    cnt=0; lo=None; hi=None
    for xv,d in ev:
        prev=cnt; cnt+=d
        if prev<k<=cnt and lo is None: lo=xv
        if prev>=k>cnt: hi=xv
    return (lo,hi) if lo is not None else None

xs=np.linspace((0.3*f_true)**2,(3*f_true)**2,4000)
print(f"{'sigma':>6}{'m':>5}{'MAP err':>12}{'consensus-mid err':>20}{'cons. covers':>14}")
for sigma in (0.5,1.0,2.0):
    for m in (5,20,50):
        emap=[]; econ=[]; cov=0; T=60
        for _ in range(T):
            data=[junction(sigma) for _ in range(m)]
            ll=loglik(xs,data)
            emap.append(abs(np.sqrt(xs[int(np.argmax(ll))])-f_true)/f_true)
            ivs=[]
            for co,xth,sd in data:
                c=admissible(co)
                if c: ivs.append((c[0][0], min(c[-1][1],1e9)))
            kk=max(1,int(np.ceil(0.9*len(ivs))))
            cc=consensus(ivs,kk)
            if cc:
                mid=0.5*(cc[0]+min(cc[1],(3*f_true)**2))
                econ.append(abs(np.sqrt(max(mid,1))-f_true)/f_true)
                cov += (cc[0]<=f_true**2<=cc[1])
        print(f"{sigma:>6}{m:>5}{100*np.median(emap):>11.2f}%"
              f"{100*np.median(econ):>19.2f}%{100*cov/T:>13.0f}%")
