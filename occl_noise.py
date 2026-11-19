"""Does Prop. 3's closed-form sd still predict coverage when the vertex is
extrapolated?  It conditions on the junction estimate, which was harmless with
the vertex visible (ratio 1.004) but the junction error grows with the gap."""
import numpy as np
from scipy.stats import norm
from common import Kmat, coeffs, admissible, rand_frame, project, edge_image_dir
rng=np.random.default_rng(8); f=866.0; HALF=500.0; ell=150.0; sigma=0.5
def place():
    c=rng.uniform(-HALF*0.9,HALF*0.9,2); z=rng.uniform(2,6)
    ray=np.linalg.inv(Kmat(f))@np.array([c[0],c[1],1.0]); return ray/ray[2]*z
def obs(P,D,gap,noise=True):
    c0=project(P,f); Ls=[]
    for i in range(3):
        d=edge_image_dir(P,D[:,i],f); d/=np.linalg.norm(d)
        p1=c0+gap*d+(rng.normal(0,sigma,2) if noise else 0)
        p2=c0+(gap+ell)*d+(rng.normal(0,sigma,2) if noise else 0)
        e=p2-p1; e/=np.linalg.norm(e); a,b=e[1],-e[0]
        if np.dot(c0-p1,np.array([b,-a]))>0: a,b=-a,-b
        Ls.append(np.array([a,b,-(a*p1[0]+b*p1[1])]))
    A=np.array([l[:2] for l in Ls]); r=-np.array([l[2] for l in Ls])
    c,*_=np.linalg.lstsq(A,r,rcond=None); return c,Ls
def sd_closed(c,Ls):
    rho=np.hypot(*c); psi=[]
    for a,b,e in Ls:
        d=np.array([b,-a]); psi.append(np.arctan2(c[0]*d[1]-c[1]*d[0],d@c))
    out=[]
    for i,j in ((0,1),(1,2),(2,0)):
        D=np.cos(psi[i]-psi[j])
        out.append(sigma*rho**2*np.hypot(np.sin(2*psi[i]),np.sin(2*psi[j]))/(np.sqrt(2)*ell*D*D))
    return np.array(out)
print(f"{'gap px':>7}{'predicted miss':>16}{'measured miss':>15}{'ratio':>8}")
for gap in (0,150,300):
    pred=[];meas=[]
    for _ in range(300):
        P=place(); D=rand_frame(rng)
        c,Ls=obs(P,D,gap,noise=False)
        co=coeffs(c,Ls); xt=f*f
        x=[(-E/A if abs(A)>1e-300 else np.nan) for A,E in co]
        sd=sd_closed(c,Ls)
        pred.append(min(1,sum(norm.cdf(-abs(xt-x[k])/sd[k]) for k in range(3)
                              if np.isfinite(x[k]) and x[k]>0 and sd[k]>0)))
        miss=0
        for _ in range(120):
            cc,LL=obs(P,D,gap); cm=admissible(coeffs(cc,LL))
            if not cm or not any(lo<=xt<=hi for lo,hi in cm): miss+=1
        meas.append(miss/120)
    p,m=np.mean(pred),np.mean(meas)
    print(f"{gap:>7}{100*p:>15.1f}%{100*m:>14.1f}%{m/p:>8.2f}")
