"""Close the loop: noiseless synthetic corners at the REAL junction positions,
under each image's REAL Manhattan frame.  How many are bounded above?"""
import numpy as np, itertools
from yud_path import YUD
import yorkurban as Y, yu_junctions
from common import Kmat, coeffs, admissible
f_true,pp,images=Y.load_packed(YUD)
Jt,_,_,_=yu_junctions.collect(centre=(pp[0],pp[1]))          # centred on the true principal point
K=Kmat(f_true)
def lines_at(c,D,signs):
    P=np.linalg.inv(K)@np.array([c[0],c[1],1.0])*5.0; L=[]
    for i in range(3):
        q=K@(P+1e-5*signs[i]*D[:,i]); q=q[:2]/q[2]; e=q-c; n=np.linalg.norm(e)
        if n<1e-12: return None
        e/=n; a,b=e[1],-e[0]; L.append(np.array([a,b,-(a*c[0]+b*c[1])]))
    return L
for variant,name in ((0,"stored sign"),(1,"y flipped")):
    nb=n=0; per_tilt=[]
    for L_real,c,idx,lens in Jt:
        vp=images[idx][3]; D=Y.align_gt_frame(vp)[variant]
        for signs in itertools.product((1,-1),repeat=3):
            L=lines_at(c[:2],D,signs)
            if L is None: continue
            cm=admissible(coeffs(c[:2],L))
            if not cm: continue
            n+=1; nb+=cm[-1][1]<1e13
    print(f"real positions, real frames ({name}): bounded above {100*nb/n:.1f}% of {n} noiseless corners")
print("observed on the real junctions: 2 of 181 = 1.1%  (one of the two is wrong)")
