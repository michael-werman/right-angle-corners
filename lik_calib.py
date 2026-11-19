"""Likelihood estimator for a single focal length from junction bounds, using
the first-order noise model with PER-EDGE segment lengths:
    sd(x_ij) = rho^2/(2 cos^2 D) * sqrt( sin^2(2psi_j) 2s^2/l_i^2 + sin^2(2psi_i) 2s^2/l_j^2 )
and the parity form  pi(x) = (1 - S prod(1 - 2 eps_k))/2.
Junction tuples: (L, c, image_index, edge_lengths)."""
import numpy as np
from scipy.stats import norm
PAIRS=((0,1),(1,2),(2,0))
def prep(Js):
    out=[]
    for L,c,img,lens in Js:
        u,v=c[0],c[1]; rho=np.hypot(u,v)
        psi=[np.arctan2(u*(-l[0])-v*l[1], l[1]*u+(-l[0])*v) for l in L]   # angle from radial dir to (b,-a)
        A=[L[i][0]*L[j][0]+L[i][1]*L[j][1] for i,j in PAIRS]
        E=[L[i][2]*L[j][2] for i,j in PAIRS]
        g=[]
        for (i,j) in PAIRS:
            D=np.cos(psi[i]-psi[j])
            g.append(rho**2/(2*D*D+1e-300)*np.sqrt(2*np.sin(2*psi[j])**2/lens[i]**2+2*np.sin(2*psi[i])**2/lens[j]**2))
        out.append((np.array(A),np.array(E),np.array(g)))
    return out
def loglik(xs, P, sigma, floor=1e-3):
    ll=np.zeros_like(xs)
    for A,E,g in P:
        v=np.outer(xs,A)+E                                # forms at each x
        S=np.sign(np.prod(v,axis=1))
        with np.errstate(divide='ignore',invalid='ignore'):
            xth=-E/A
        eps=np.zeros_like(v)
        for k in range(3):
            if np.isfinite(xth[k]) and g[k]>0:
                eps[:,k]=norm.cdf(-np.abs(xs-xth[k])/(sigma*g[k]))
        ll+=np.log(np.clip((1-S*np.prod(1-2*eps,axis=1))/2,floor,1.0))
    return ll
def estimate(Js, f_lo=150.0, f_hi=4000.0, sigmas=(0.25,0.5,1.0,2.0,4.0), n=1500):
    """returns f_hat, sigma_hat, and a 1/2-log-likelihood-drop interval"""
    P=prep(Js); fs=np.exp(np.linspace(np.log(f_lo),np.log(f_hi),n)); xs=fs**2
    best=None
    for s in sigmas:
        ll=loglik(xs,P,s); k=int(np.argmax(ll))
        if best is None or ll[k]>best[0]: best=(ll[k],fs[k],s,ll)
    ll=best[3]; ok=ll>=best[0]-0.5
    return best[1], best[2], (fs[ok].min(), fs[ok].max())
