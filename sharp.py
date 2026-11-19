"""Sharp test.  Prediction: identifiability for corners of ONE frame needs some
axis d_i with |d_i . z| < sin(FOV/2) (its edge-on plane of rays crosses the
image).  Looking down a cube diagonal, |d_i.z| = 1/sqrt3 for all i, so it
should FAIL below FOV = 2 asin(1/sqrt3) = 70.53 deg and HOLD above it."""
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
D=diagonal_frame()
print("frame: cube diagonal along the optical axis; |d_i . z| =",
      np.round(np.abs(D[2,:]),4), " (1/sqrt3 = %.4f)"%(1/np.sqrt(3)))
print(f"critical FOV = 2 asin(1/sqrt3) = {2*np.degrees(np.arcsin(1/np.sqrt(3))):.2f} deg\n")
print(f"{'FOV':>6}{'sin(FOV/2)':>12}{'edge-on plane in view?':>24}{'intersection of F  (x f)':>30}")
for fov in (40,60,68,73,80,90):
    f,lo,hi,n=run(fov,D)
    inview = np.min(np.abs(D[2,:])) < np.sin(np.radians(fov)/2)
    hs = f"{hi:.4f}" if np.isfinite(hi) else "inf"
    print(f"{fov:>6}{np.sin(np.radians(fov)/2):>12.4f}{str(inview):>24}{f'[{lo:.4f}, {hs}]':>30}")

def edge_on_in_view(f,D,half=480):
    """exact: does r.d_i change sign over the sampled image square, for some i?"""
    g=np.linspace(-half,half,401); U,V=np.meshgrid(g,g)
    R=np.stack([U,V,np.full_like(U,f)],-1); R/=np.linalg.norm(R,axis=-1,keepdims=True)
    return any((R@D[:,i]).min()<0<(R@D[:,i]).max() for i in range(3))
def min_abs_rd(f,D,half=480):
    g=np.linspace(-half,half,401); U,V=np.meshgrid(g,g)
    R=np.stack([U,V,np.full_like(U,f)],-1); R/=np.linalg.norm(R,axis=-1,keepdims=True)
    return min(np.abs(R@D[:,i]).min() for i in range(3))
print("\n--- with the EXACT criterion (sign change of r.d_i over the image square) ---")
print(f"{'FOV':>6}{'edge-on in view':>17}{'min |r.d_i|':>13}{'intersection of F (x f)':>28}{'width/f':>11}")
for fov in (30,40,46,50,52,54,60,73,90):
    f,lo,hi,n=run(fov,D)
    hs=f"{hi:.5f}" if np.isfinite(hi) else "inf"
    w=(hi-lo) if np.isfinite(hi) else np.inf
    print(f"{fov:>6}{str(edge_on_in_view(f,D)):>17}{min_abs_rd(f,D):>13.4f}{f'[{lo:.5f}, {hs}]':>28}{w:>11.2e}")
