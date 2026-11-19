import numpy as np
from calib_core import *
rng=np.random.default_rng(4)
Kt=np.array([[900.,15.,40.],[0.,700.,-30.],[0.,0.,1.]]); Wt=Kt@Kt.T
D=rand_frame(rng)
print("ONE Manhattan frame: extent of each entry of w* (ratio to true value),")
print("and the smallest eigenvalue of w* over the extreme points (0 = degenerate camera)\n")
print(f"{'junctions':>10}  "+"".join(f"{'w'+str(a)+str(b):>18}" for a,b in IDX)+f"{'min eig':>12}")
Js=[]
for M in (100,500,2000,6000):
    while len(Js)<M: Js.append(junction(Kt,rng,D))
    A,b=build(Js); cells=[]; mine=np.inf
    for k in range(5):
        vals=[]
        for sg in (1,-1):
            cv=np.zeros(5); cv[k]=sg; x=extreme(A,b,cv)
            if x is None: vals.append(np.nan); continue
            vals.append(x[k]); mine=min(mine,np.linalg.eigvalsh(W_of(x))[0])
        t=Wt[IDX[k]]; r=sorted([vals[0]/t,vals[1]/t]); cells.append(f"[{r[0]:7.3f},{r[1]:7.3f}]")
    print(f"{M:>10}  "+"".join(f"{c:>18}" for c in cells)+f"{mine:>12.2e}")
