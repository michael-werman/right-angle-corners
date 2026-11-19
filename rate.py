"""Rate of identifiability.  Prediction: width(cap F) = Theta(1/M) when some
axis's edge-on plane crosses the image transversally."""
import numpy as np
from common import Kmat, coeffs, admissible, rand_frame, observe_corner
def run(fov, D, N, rng, half=480):
    f=500.0/np.tan(np.radians(fov)/2); lo=0.0; hi=np.inf; out={}
    checks=set(int(round(x)) for x in np.geomspace(20,N,14)); n=0
    while n<N:
        c=rng.uniform(-half,half,2); z=rng.uniform(2,6)
        ray=np.linalg.inv(Kmat(f))@np.array([c[0],c[1],1.0]); P=ray/ray[2]*z
        o=observe_corner(P,D,f,0.0,rng)
        if o is None: continue
        cm=admissible(coeffs(*o))
        if not cm: continue
        n+=1; lo=max(lo,cm[0][0]); hi=min(hi,cm[-1][1])
        if n in checks: out[n]=(np.sqrt(hi)-np.sqrt(lo))/f if np.isfinite(hi) else np.nan
    return out
rng=np.random.default_rng(0)
R=24; N=20000; fov=70
allw={}
for r in range(R):
    D=rand_frame(rng)
    for M,w in run(fov,D,N,rng).items(): allw.setdefault(M,[]).append(w)
Ms=np.array(sorted(allw)); med=np.array([np.nanmedian(allw[M]) for M in Ms])
ok=np.isfinite(med)&(Ms>=200)
slope,icpt=np.polyfit(np.log(Ms[ok]),np.log(med[ok]),1)
print(f"FOV {fov} deg, {R} random frames, noiseless; median width of cap F over frames")
for M,w in zip(Ms,med): print(f"   M={M:6d}  width/f = {w:.3e}")
print(f"\nlog-log slope (M >= 200): {slope:.3f}    prediction: -1")
