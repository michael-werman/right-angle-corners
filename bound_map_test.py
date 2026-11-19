"""Where a junction of a frame bounds the focal length (camera diag(f,f,1)).
Claim: gamma_a = -(r.(zhat x d_p)) (r.(zhat x d_q)) / ((r.d_p)(r.d_q)) for cyclic (a,p,q), any orientation of the frame;
a junction with vertex c has  F = ( f sqrt(G-/(1+G-)), f sqrt(G+/(G+-1)) )  with G+ = max_a gamma_a, G- = max_a(-gamma_a),
so it is bounded above iff some gamma_a > 1 and bounded below (by a positive value) iff some gamma_a < 0."""
import numpy as np
from common import coeffs, admissible
rng=np.random.default_rng(12); zh=np.array([0,0,1.0])
def rand_rot():
    Q,R=np.linalg.qr(rng.standard_normal((3,3))); return Q*np.sign(np.diag(R))
def gam_def(r,D):
    Z=D.T@r; z=D[2,:]; W=z/Z; return np.array([(z[a]-Z[a]*W[p])*(z[a]-Z[a]*W[q]) for a,p,q in ((0,1,2),(1,2,0),(2,0,1))])
def gam_formula(r,D):
    out=[]
    for a,p,q in ((0,1,2),(1,2,0),(2,0,1)):
        dp,dq=D[:,p],D[:,q]
        out.append(-(r@np.cross(zh,dq))*(r@np.cross(zh,dp))/((r@dp)*(r@dq)))
    return np.array(out)
worst=0; agree=0; n=0
for _ in range(20000):
    f=rng.uniform(300,1500); D=rand_rot()                     # either handedness
    c=rng.uniform(-500,500,2); r=np.array([c[0]/f,c[1]/f,1.0])
    g1,g2=gam_def(r,D),gam_formula(r,D); worst=max(worst,np.max(np.abs(g1-g2)/(1+np.abs(g1))))
    # the junction's own interval, from Theorem 6, against the formula
    K=np.diag([f,f,1.0]); L=[]
    ch=np.array([c[0],c[1],1.0])
    for k in range(3):
        l=np.cross(ch,K@D[:,k]); l=l/np.linalg.norm(l[:2]); L.append((l[0],l[1],l[2]))
    comps=admissible(coeffs(c,L))
    if not comps: continue
    lo,hi=np.sqrt(comps[0][0])/f,(np.sqrt(comps[-1][1])/f if comps[-1][1]<1e13 else np.inf)
    Gp,Gm=g1.max(),(-g1).max()
    plo=np.sqrt(Gm/(1+Gm)) if Gm>0 else 0.0; phi=np.sqrt(Gp/(Gp-1)) if Gp>1 else np.inf
    n+=1; agree+= (abs(lo-plo)<1e-6*(1+lo)) and ((np.isinf(hi) and np.isinf(phi)) or (np.isfinite(hi) and np.isfinite(phi) and abs(hi-phi)<1e-6*hi))
print(f"gamma_a: definition against the factored formula, worst relative difference {worst:.1e} over 20000 cases (both handednesses)")
print(f"single-junction interval from Theorem 6 against the formula from gamma: agree in {agree}/{n}")
# York Urban: the share of the image where a junction would bound f from above, against what the 181 junctions show
import yorkurban as Y, yu_junctions
from yud_path import YUD
fT,pp,images=Y.load_packed(YUD); J,_,_,_=yu_junctions.collect(); used=sorted(set(j[2] for j in J))
fr=[]
for idx in used:
    V=np.array(images[idx][3],float); D=V/np.linalg.norm(V,axis=0)
    m=200000; u=rng.uniform(0,640,m)-pp[0]; v=rng.uniform(0,480,m)-pp[1]
    R=np.stack([u/fT,v/fT,np.ones(m)],1); Z=R@D; z=D[2,:]; W=z[None,:]/Z
    G=np.max(np.stack([(z[a]-Z[:,a]*W[:,p])*(z[a]-Z[:,a]*W[:,q]) for a,p,q in ((0,1,2),(1,2,0),(2,0,1))],1),1)
    fr.append(np.mean(G>1))
fr=np.array(fr)
print(f"York Urban, 59 images: share of the image where a junction bounds f from above: median {100*np.median(fr):.1f}%, mean {100*fr.mean():.1f}%, max {100*fr.max():.1f}%")
print("   against: 4 of 181 real junction positions (2.2%) in the noiseless replay, 2 of 181 (1.1%) measured")
