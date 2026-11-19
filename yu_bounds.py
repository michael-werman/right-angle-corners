import numpy as np, sys
import yu_junctions
from common import coeffs, admissible
J,F,PP,_=yu_junctions.collect()
Jt,_,_,_=yu_junctions.collect(centre=(320.0+PP[0],240.0-PP[1]))
for label,Js in (("true principal point",Jt),("image-centre principal point",J)):
    lo=[];hi=[];empty=0
    for L,c,img,lens in Js:
        cm=admissible(coeffs(c[:2],L))
        if not cm: empty+=1; continue
        lo.append(np.sqrt(cm[0][0])/F); hi.append(np.sqrt(cm[-1][1])/F if cm[-1][1]<1e13 else np.inf)
    lo=np.array(lo); hi=np.array(hi); fin=np.isfinite(hi)
    print(f"{label}: {len(lo)} intervals ({empty} empty)")
    print(f"   lower bounds / f : top five {np.round(np.sort(lo)[-5:],3)}   95th pct {np.percentile(lo,95):.3f}   median {np.median(lo):.3f}")
    print(f"   upper bounds     : {fin.sum()} of {len(hi)} junctions have one: {np.round(np.sort(hi[fin]),3)}")
    ok=(lo<=1)&(hi>=1)
    print(f"   so the junctions jointly say:  f >= {np.sort(lo)[-2]:.3f} f_true (second-largest lower bound),  f <= {np.sort(hi[fin&(hi>=1)])[0] if (fin&(hi>=1)).any() else np.inf:.3f} f_true")
