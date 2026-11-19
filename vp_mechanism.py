"""Direct test of the mechanism: for ONE frame, do the feasible cameras all make
the frame's three vanishing points mutually orthogonal, v_i^T w v_j = 0?"""
import numpy as np
from calib_core import *
rng=np.random.default_rng(4)
Kt=np.array([[900.,15.,40.],[0.,700.,-30.],[0.,0.,1.]])
D=rand_frame(rng)
V=[Kt@D[:,i] for i in range(3)]                      # the frame's vanishing points
def vp_orth(Wstar):
    W=np.linalg.inv(Wstar)
    n=[np.sqrt(abs(V[i]@W@V[i])) for i in range(3)]
    return max(abs(V[i]@W@V[j])/(n[i]*n[j]) for i,j in ((0,1),(1,2),(2,0)))   # |cos| between VPs
Js=[junction(Kt,rng,D) for _ in range(4000)]
A,b=build(Js)
print("cosine between vanishing-point directions under each camera (0 = orthogonal)\n")
print(f"   true camera                         : {vp_orth(Kt@Kt.T):.2e}")
rnd=[]
for _ in range(2000):
    Kr=np.array([[rng.uniform(500,1500),rng.normal(0,60),rng.normal(0,200)],
                 [0,rng.uniform(500,1500),rng.normal(0,200)],[0,0,1.0]])
    rnd.append(vp_orth(Kr@Kr.T))
print(f"   2000 RANDOM cameras (median)         : {np.median(rnd):.2e}")
fe=[]
for k in range(5):
    for sg in (1,-1):
        cv=np.zeros(5); cv[k]=sg; x=extreme(A,b,cv)
        W=W_of(x)
        if np.linalg.eigvalsh(W)[0]>1e-6: fe.append(vp_orth(W))
print(f"   FEASIBLE extreme cameras ({len(fe)}, median): {np.median(fe):.2e}   max {max(fe):.2e}")
print("\nIf the feasible cameras sit near 0 while random ones do not, the junctions of one")
print("frame are constraining exactly the orthogonality of its three vanishing points.")
