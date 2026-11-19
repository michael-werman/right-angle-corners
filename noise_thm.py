"""A first-order (delta-method) distribution for the endpoints of F.

Observation: each edge i gives two endpoints, p_i0 near the junction and
p_i1 = p_i0 + L d_i, each perturbed by iid N(0, sigma^2 I).  Everything
downstream -- the fitted lines, the estimated junction, the thresholds
f_ij -- is a smooth function g of those 12 numbers, so

    sd(f_ij) ~= sigma * ||grad g||,

and the true f leaves F when the nearest threshold crosses it, giving

    P(f not in F) ~= sum_j Phi( -|f - f_ij| / sd(f_ij) ).
"""
import numpy as np
from scipy.stats import norm
from common import Kmat, coeffs, admissible, reproject
rng = np.random.default_rng(0)
HALF, L = 500.0, 150.0

def frame(rng):
    A=rng.standard_normal((3,3)); Q,R=np.linalg.qr(A); Q=Q@np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q)<0: Q[:,0]*=-1
    return Q

def endpoints(P,D,f):
    """the 6 noiseless image endpoints, as a flat 12-vector"""
    Km=Kmat(f); c=Km@P; c=c/c[2]; c=c[:2]
    out=[]
    for i in range(3):
        X,Y,Z=P; d=D[:,i]
        di=np.array([f*d[0]-c[0]*d[2], f*d[1]-c[1]*d[2]])/Z
        nd=np.linalg.norm(di)
        if nd<1e-12: return None
        di/=nd
        out += list(c); out += list(c+L*di)
    return np.asarray(out)

def thresholds(z):
    """endpoints (12) -> the positive thresholds of the three pair factors"""
    pts=z.reshape(3,2,2); Ls=[]
    for i in range(3):
        p1,p2=pts[i]; e=p2-p1; n=np.linalg.norm(e)
        if n<1e-9: return None
        e=e/n; a,b=e[1],-e[0]
        Ls.append(np.array([a,b,-(a*p1[0]+b*p1[1])]))
    A=np.array([l[:2] for l in Ls]); rhs=-np.array([l[2] for l in Ls])
    c,*_=np.linalg.lstsq(A,rhs,rcond=None)
    co=coeffs(c,Ls)                       # re-projects through c internally
    return np.array([(-E/Aa if abs(Aa)>1e-300 else np.nan) for Aa,E in co]), c, co

def grad_thresholds(z, h=1e-4):
    base=thresholds(z)
    if base is None: return None
    x0=base[0]; G=np.zeros((3,12))
    for k in range(12):
        zp=z.copy(); zp[k]+=h; a=thresholds(zp)
        zm=z.copy(); zm[k]-=h; b=thresholds(zm)
        if a is None or b is None: return None
        G[:,k]=(a[0]-b[0])/(2*h)
    return x0,G,base[2]

def grad_forms(z, xt, h=1e-4):
    """gradient of the three forms A_ij x + e_i e_j at x = xt: linearizes the event of (17) itself"""
    base=thresholds(z)
    if base is None: return None
    F0=np.array([A*xt+E for A,E in base[2]]); G=np.zeros((3,12))
    for k in range(12):
        zp=z.copy(); zp[k]+=h; a=thresholds(zp); zm=z.copy(); zm[k]-=h; b=thresholds(zm)
        if a is None or b is None: return None
        G[:,k]=(np.array([A*xt+E for A,E in a[2]])-np.array([A*xt+E for A,E in b[2]]))/(2*h)
    return F0,G
print(f"{'sigma':>6}{'eq.(18) as published':>22}{'linearized forms':>18}{'measured P(miss)':>19}{'n':>7}")
for sigma in (0.5,1.0,2.0):
    pred=[]; pred2=[]; meas=[]; n=0
    while n<600:
        f=866.0; D=frame(rng)
        P=np.array([rng.uniform(-2.5,2.5),rng.uniform(-2.5,2.5),rng.uniform(2,6)])
        z=endpoints(P,D,f)
        if z is None or np.max(np.abs(z))>HALF: continue
        g=grad_thresholds(z)
        if g is None: continue
        x0,G,co=g
        xt=f*f
        # probability that a threshold crosses the true x = f^2
        p=0.0
        for k in range(3):
            if not np.isfinite(x0[k]) or x0[k]<=0: continue
            sd=sigma*np.linalg.norm(G[k])
            if sd<=0: continue
            p+=norm.cdf(-abs(xt-x0[k])/sd)
        gf=grad_forms(z,xt)
        if gf is None: continue
        F0,GF=gf; p2=sum(norm.cdf(-abs(F0[k])/(sigma*np.linalg.norm(GF[k]))) for k in range(3) if np.linalg.norm(GF[k])>0)
        pred.append(min(p,1.0)); pred2.append(min(p2,1.0)); n+=1
        # Monte Carlo for the same corner
        miss=0; M=200
        for _ in range(M):
            zz=z+rng.normal(0,sigma,12)
            t=thresholds(zz)
            if t is None: miss+=1; continue
            comps=admissible(t[2])
            if not comps or not any(lo<=xt<=hi for lo,hi in comps): miss+=1
        meas.append(miss/M)
    print(f"{sigma:>6}{np.mean(pred):>22.4f}{np.mean(pred2):>18.4f}{np.mean(meas):>19.4f}{n:>7}")
