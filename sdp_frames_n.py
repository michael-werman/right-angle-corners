"""Prediction: one frame's junctions carry only its 3 vanishing points = 3
constraints on w* (5 dof) -> NOT identifiable; two frames give 6 >= 5 -> identifiable."""
import numpy as np
from calib_core import *
rng=np.random.default_rng(9)
Kt=np.array([[900.,15.,40.],[0.,700.,-30.],[0.,0.,1.]]); Wt=Kt@Kt.T
def extent(nframes, M):
    Ds=[rand_frame(rng) for _ in range(nframes)]
    Js=[junction(Kt,rng,Ds[k%nframes]) for k in range(M)]
    A,b=build(Js); worst=0.0; mine=np.inf
    for k in range(5):
        v=[]
        for sg in (1,-1):
            cv=np.zeros(5); cv[k]=sg; x=extreme(A,b,cv)
            if x is None: return np.inf, np.nan
            v.append(x[k]); mine=min(mine,np.linalg.eigvalsh(W_of(x))[0])
        worst=max(worst, abs(v[0]-v[1])/abs(Wt[IDX[k]]))
    return worst, mine
print(f"{'frames':>7}{'junctions':>11}{'worst relative extent over w* entries':>40}{'min eig':>12}")
for nf in (1,2,3):
    for M in (600,3000):
        w,me=extent(nf,M)
        print(f"{nf:>7}{M:>11}{w:>40.4f}{me:>12.2e}")
