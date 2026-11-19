"""Prediction: one frame carries 3 constraints, so a model with <= 3 unknowns is
identifiable from ONE frame.  Aspect ratio (2 unknowns, w* = diag(X,Y,1)) is linear."""
import numpy as np
from scipy.optimize import linprog
from calib_core import predicted_pattern, rand_frame, junction
rng=np.random.default_rng(6)
FX,FY=900.0,700.0; K=np.diag([FX,FY,1.0])
def box(Js):
    A=[];b=[]
    for L,c in Js:
        for (i,j),s in zip(((0,1),(1,2),(2,0)),predicted_pattern(L,c)):
            A.append([-s*L[i][0]*L[j][0], -s*L[i][1]*L[j][1]]); b.append(s*L[i][2]*L[j][2]-1e-9)
    A=np.array(A); b=np.array(b); out=[]
    for k,sg in ((0,1),(0,-1),(1,1),(1,-1)):
        cv=np.zeros(2); cv[k]=sg
        r=linprog(cv,A_ub=A,b_ub=b,bounds=[(1e-6,None)]*2,method="highs")
        out.append(sg*r.fun if r.status==0 else np.inf)
    return out
for label,D in (("ONE frame",rand_frame(rng)),):
    print(f"{label}: aspect-ratio camera f_x={FX}, f_y={FY}")
    Js=[]
    for M in (30,200,1000,4000):
        while len(Js)<M: Js.append(junction(K,rng,D))
        Xmn,Xmx,Ymn,Ymx=box(Js)
        fmt=lambda a,b: f"[{np.sqrt(a):.1f}, {np.sqrt(b):.1f}]" if np.isfinite(b) else f"[{np.sqrt(a):.1f}, inf)"
        print(f"   {M:5d} junctions:  f_x {fmt(Xmn,Xmx):>20}   f_y {fmt(Ymn,Ymx):>20}")
