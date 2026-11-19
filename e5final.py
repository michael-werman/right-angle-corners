import numpy as np
from yud_path import YUD
import yorkurban as Y
from common import coeffs, admissible, Kmat
f_true, pp, images = Y.load_packed(YUD)
class Args: radius=10.0; min_len=15.0; iters=600; no_sign=False; verbose=False; vertical_family=None
rng=np.random.default_rng(0); recs=[]
fmins=[]; unb=0; nb=0; rhos=[]; fr=[]
for name,lines,labels,vp in images:
    seg=np.stack([Y.to_centred(lines[:,0],pp),Y.to_centred(lines[:,1],pp)],axis=1)
    vert=Y.pick_vertical_family(seg,labels,vp)
    others=sorted(set(int(x) for x in labels if x>0)-{vert})
    if len(others)<2: continue
    for c,fa in Y.find_junctions(seg,labels,10.0,15.0):
        if len(fa)<3: continue
        b=Y.build_corner(seg,fa,tuple(others[:2])+(vert,))
        if b is None: continue
        cc,ls=b; comps=admissible(coeffs(cc,ls))
        if not comps: continue
        nb+=1; lo,hi=comps[0][0],comps[-1][1]
        fmins.append(np.sqrt(lo)/f_true); unb += hi>=1e13
        rhos.append(np.hypot(cc[0],cc[1]))
        fr.append(np.sqrt(lo)/max(np.hypot(cc[0],cc[1]),1e-9))
fm=np.array(fmins); rh=np.array(rhos); frr=np.array(fr)
print(f"full junctions used: {nb}")
print(f"  f_min/f    : p25 {np.percentile(fm,25):.2f}  median {np.median(fm):.2f}  p75 {np.percentile(fm,75):.2f}")
print(f"  unbounded above: {100*unb/nb:.1f}%")
print(f"  rho (px)   : median {np.median(rh):.0f}  (image half-diagonal {np.hypot(320,240):.0f})")
print(f"  f_min/rho  : p25 {np.percentile(frr,25):.2f}  median {np.median(frr):.2f}  p75 {np.percentile(frr,75):.2f}")
print(f"  similarity: f_min/f spread {np.percentile(fm,75)/np.percentile(fm,25):.2f}x  vs  "
      f"f_min/rho spread {np.percentile(frr,75)/np.percentile(frr,25):.2f}x")
# correlation check
print(f"  corr(f_min, rho) = {np.corrcoef(np.array(fmins)*f_true, rh)[0,1]:.3f}")
