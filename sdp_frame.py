import numpy as np
from scipy.optimize import linprog
from sdp_calib import *
def junction_in(K,D):
    while True:
        P=np.array([rng.uniform(-3,3),rng.uniform(-3,3),rng.uniform(2,6)])
        c=K@P; c=c/c[2]
        if max(abs(c[:2]))>600: continue
        L=[];ok=True
        for i in range(3):
            q=K@(P+1e-6*D[:,i]); q=q/q[2]; d=(q-c)[:2]; n=np.linalg.norm(d)
            if n<1e-12: ok=False;break
            d/=n; a,b=d[1],-d[0]; L.append(np.array([a,b,-(a*c[0]+b*c[1])]))
        if ok: return L,c
def params_extent(A,b):
    """bounds on each camera parameter, by optimising over w* then reading K"""
    pts=[]
    for k in range(5):
        for sg in (1,-1):
            cv=np.zeros(5); cv[k]=sg; x=extreme(A,b,cv)
            if x is not None: pts.append(K_of(W_of(x)))
    P=np.array([[k[0,0],k[1,1],k[0,1],k[0,2],k[1,2]] for k in pts])
    return P.min(0),P.max(0)
Kt=np.array([[900.,15.,40.],[0.,700.,-30.],[0.,0.,1.]])
names=["fx","fy","skew","px","py"]; true=[900,700,15,40,-30]
for label,frames in (("independent frames",None),):   # one frame: see sdp_frame2.py
    Js=[]
    for _ in range(2000):
        Js.append(junction(Kt) if frames is None else junction_in(Kt,frames))
    A,b=build(Js); lo,hi=params_extent(A,b)
    print(f"\n{label}, 2000 junctions:")
    for n,t,l,h in zip(names,true,lo,hi):
        print(f"   {n:5s} true {t:6.1f}   feasible [{l:8.2f}, {h:8.2f}]   width {h-l:7.2f}")
