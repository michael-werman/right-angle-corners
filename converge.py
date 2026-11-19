"""Does the intersection of admissible sets keep shrinking as corners accumulate,
or stabilise?  Frame seen down a cube diagonal, 54 and 60 degree fields of view."""
import numpy as np
import common as C
rng=np.random.default_rng(1)
def diagonal_frame():
    """frame whose three axes all make the same angle with the optical axis z"""
    z=np.array([0,0,1.0]); g=np.array([1,1,1.0])/np.sqrt(3)
    # rotate g onto z
    v=np.cross(g,z); s=np.linalg.norm(v); cth=g@z
    vx=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    R=np.eye(3)+vx+vx@vx*((1-cth)/s**2)
    return R@np.eye(3)                     # columns = rotated axes
def run(fov_deg, D, N=30000):
    f=500.0/np.tan(np.radians(fov_deg)/2)  # 1000 px image
    lo,hi=0.0,np.inf; n=0
    for _ in range(N):
        c=rng.uniform(-480,480,2); z=rng.uniform(2,6)
        ray=np.linalg.inv(C.Kmat(f))@np.array([c[0],c[1],1.0]); P=ray/ray[2]*z
        o=C.observe_corner(P,D,f,0.0,rng)
        if o is None: continue
        comps=C.admissible(C.coeffs(*o))
        if not comps: continue
        n+=1; lo=max(lo,comps[0][0]); hi=min(hi,comps[-1][1])
    return f, np.sqrt(lo)/f, (np.sqrt(hi)/f if np.isfinite(hi) else np.inf), n
def edge_on_in_view(f,D,half=480):
    """exact: does r.d_i change sign over the sampled image square, for some i?"""
    g=np.linspace(-half,half,401); U,V=np.meshgrid(g,g)
    R=np.stack([U,V,np.full_like(U,f)],-1); R/=np.linalg.norm(R,axis=-1,keepdims=True)
    return any((R@D[:,i]).min()<0<(R@D[:,i]).max() for i in range(3))
D=diagonal_frame()
print("does the intersection keep shrinking with more corners, or stabilise?\n")
print(f"{'FOV':>5}{'edge-on in view':>17}" + "".join(f"{'N='+str(N):>16}" for N in (3000,30000,300000)))
for fov in (54,60):
    f=500.0/np.tan(np.radians(fov)/2)
    row=[]
    for N in (3000,30000,300000):
        _,lo,hi,n=run(fov,D,N)
        row.append((hi-lo) if np.isfinite(hi) else np.inf)
    print(f"{fov:>5}{str(edge_on_in_view(f,D)):>17}"+"".join(f"{w:>16.3e}" for w in row))
