"""Consensus sets: how often disconnected; mirror resolution: Wilson intervals; parity of sign flips: independent
approximation against correlated forms."""
import numpy as np, math
from scipy.stats import norm
import referee3_experiments as R
rng=np.random.default_rng(4)
# ---- (1) how often the depth-kappa set is disconnected
print("(1) depth-kappa consensus, kappa = ceil(0.9M): share of trials with a disconnected set (200 trials each)")
for sigma in (0.5,1.0):
    row=[]
    for M in (2,5,10,25,50):
        k=max(1,math.ceil(0.9*M)); dis=0; n=0
        for _ in range(200):
            iv=[]
            while len(iv)<M:
                b=R.one(sigma)
                if b: iv.append(b)
            ds=R.depth_set(iv,k)
            if ds: n+=1; dis+= len(ds)>1
        row.append(f"M={M}: {100*dis/max(n,1):.1f}%")
    print(f"   sigma={sigma}: "+", ".join(row))
# ---- (4) Wilson intervals for mirror resolution
def wilson(k,n,z=1.959964):
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n); h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)); return 100*(c-h)/d,100*(c+h)/d
for lab,k,n in (("overall",5970,6000),("<5 deg",141,150),("5-15 deg",850,859),("15-45 deg",3971,3987),(">45 deg",1004,1004),("York Urban",383,387)):
    lo,hi=wilson(k,n); print(f"(4) mirror {lab:>10}: {100*k/n:5.1f}%  Wilson 95% [{lo:.1f}, {hi:.1f}]  (n={n})")
# ---- (6) parity of sign flips: independent approximation against the correlated forms
tau=np.sqrt(2)*1.0/150.0                                  # sigma = 1 px, 150 px edges
f=866.0; diffs=[]; big=0; N=600
for _ in range(N):
    rho=rng.uniform(50,480); psi=rng.uniform(-np.pi,np.pi,3); x=f*f*rng.uniform(0.6,1.6)
    pairs=((0,1),(1,2),(2,0))
    F=np.array([np.cos(psi[i]-psi[j])*x+rho**2*np.sin(psi[i])*np.sin(psi[j]) for i,j in pairs])
    G=np.zeros((3,3))                                      # gradient of each form in (psi_1,psi_2,psi_3)
    for k,(i,j) in enumerate(pairs):
        G[k,i]=-np.sin(psi[i]-psi[j])*x+rho**2*np.cos(psi[i])*np.sin(psi[j])
        G[k,j]= np.sin(psi[i]-psi[j])*x+rho**2*np.sin(psi[i])*np.cos(psi[j])
    s=tau*np.linalg.norm(G,axis=1); p=norm.cdf(-np.abs(F)/np.maximum(s,1e-300))
    p_ind=(1-np.prod(1-2*p))/2                             # independent flips
    xi=rng.normal(0,tau,(200000,3)); P=psi[None,:]+xi      # the exact model: perturb the three angles jointly
    Fm=np.stack([np.cos(P[:,i]-P[:,j])*x+rho**2*np.sin(P[:,i])*np.sin(P[:,j]) for i,j in pairs],1)
    flips=(np.sign(Fm)!=np.sign(F)[None,:]).sum(1); p_mc=np.mean(flips%2==1)
    diffs.append(abs(p_ind-p_mc)); big+= abs(p_ind-p_mc)>0.01
d=np.array(diffs)
print(f"(6) parity of sign flips, sigma = 1 px, {N} random junctions: |independent - exact| median {d.mean()*0+np.median(d):.1e}, 99th pct {np.percentile(d,99):.1e}, max {d.max():.1e}; above 0.01 in {big}")
