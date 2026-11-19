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
        return co, comps, xth, np.array(sd), np.array(psi), rho
xs=np.linspace((0.3*f)**2,(3*f)**2,3000)
def ll(data):
    out=np.zeros_like(xs)
    for co,_,xth,sd,_psi,_rho in data:
        v=np.array([[A*x+E for A,E in co] for x in xs])
        S=np.sign(np.prod(v,axis=1)); p=np.zeros_like(v)
        for k in range(3):
            if np.isfinite(xth[k]) and sd[k]>0:
                p[:,k]=norm.cdf(-np.abs(xs-xth[k])/sd[k])
        out+=np.log(np.clip((1-S*np.prod(1-2*p,axis=1))/2,1e-12,1))
    return out

def ll_form(data, sigma):
    """the same log-plausibility with the form-based scale of eq. (18): each form is linearized in the two angles"""
    tau2=2*sigma**2/ell**2; out=np.zeros_like(xs)
    for co,_,xth,sd,psi,rho in data:
        v=np.array([[A*x+E for A,E in co] for x in xs])
        S=np.sign(np.prod(v,axis=1)); p=np.zeros_like(v)
        for k,(i,j) in enumerate(((0,1),(1,2),(2,0))):
            di=-np.sin(psi[i]-psi[j])*xs+rho**2*np.cos(psi[i])*np.sin(psi[j])
            dj= np.sin(psi[i]-psi[j])*xs+rho**2*np.sin(psi[i])*np.cos(psi[j])
            sk=np.sqrt(tau2*(di**2+dj**2))+1e-300
            p[:,k]=norm.cdf(-np.abs(v[:,k])/sk)
        out+=np.log(np.clip((1-S*np.prod(1-2*p,axis=1))/2,1e-12,1))
    return out
rngb=np.random.default_rng(5)
print(f"{'sigma':>5}{'M':>4}{'threshold scale':>17}{'form scale':>12}{'paired median difference':>27}")
for sigma in (0.5,2.0):
    for M in (1,3,10,50):
        e1=[];e2=[]
        for _ in range(100):
            data=[one(sigma) for _ in range(M)]
            e1.append(abs(np.sqrt(xs[int(np.argmax(ll(data)))])-f)/f); e2.append(abs(np.sqrt(xs[int(np.argmax(ll_form(data,sigma)))])-f)/f)
        e1=np.array(e1); e2=np.array(e2)
        print(f"{sigma:>5}{M:>4}{100*np.median(e1):>16.1f}%{100*np.median(e2):>11.1f}%{100*np.median(e2-e1):>25.2f} pts",flush=True)
