"""G4: an OCCLUDED or off-image vertex.  The theory never needs the vertex
visible -- only that the three image lines are concurrent, which they are at
the projection of the vertex whether or not it is seen.  So the junction can be
recovered by intersecting three edge segments seen AWAY from the vertex.
Test: segments start a gap g px from the vertex; does the bound still hold,
and what does extrapolation cost?"""
import numpy as np
from common import Kmat, coeffs, admissible, rand_frame, project, edge_image_dir
rng=np.random.default_rng(3); f=866.0; HALF=500.0; L=150.0
def place(rng):
    c=rng.uniform(-HALF*0.9,HALF*0.9,2); z=rng.uniform(2,6)
    ray=np.linalg.inv(Kmat(f))@np.array([c[0],c[1],1.0]); return ray/ray[2]*z
def observe_gapped(P,D,sigma,gap):
    c0=project(P,f); Ls=[]
    for i in range(3):
        d=edge_image_dir(P,D[:,i],f); d/=np.linalg.norm(d)
        p1=c0+gap*d+rng.normal(0,sigma,2)        # segment starts gap px out
        p2=c0+(gap+L)*d+rng.normal(0,sigma,2)
        e=p2-p1; e/=np.linalg.norm(e); a,b=e[1],-e[0]
        # orient so the half-edge leaves the (unseen) vertex along (b,-a)
        if np.dot(c0-p1, np.array([b,-a]))>0: a,b=-a,-b
        Ls.append(np.array([a,b,-(a*p1[0]+b*p1[1])]))
    A=np.array([l[:2] for l in Ls]); rhs=-np.array([l[2] for l in Ls])
    c,*_=np.linalg.lstsq(A,rhs,rcond=None)
    return c,Ls,np.linalg.norm(c-c0)
print(f"{'gap (px)':>9}{'covers f':>11}{'median vertex error (px)':>27}{'median fmin/f':>15}")
for gap in (0,25,75,150,300):
    cov=0;n=0;err=[];fm=[]
    while n<1500:
        o=observe_gapped(place(rng),rand_frame(rng),0.5,gap)
        c,Ls,e=o
        comps=admissible(coeffs(c,Ls))
        if not comps: continue
        n+=1; err.append(e); fm.append(np.sqrt(comps[0][0])/f)
        cov+=any(lo<=f*f<=hi for lo,hi in comps)
    print(f"{gap:>9}{100*cov/n:>10.1f}%{np.median(err):>27.2f}{np.median(fm):>15.2f}")
