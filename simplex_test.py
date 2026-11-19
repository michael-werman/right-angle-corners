"""Claim: a positive-definite w* admits the junction (c; l1,l2,l3) iff
lambda_k = (l_i^T w* l_j) / (D_ki D_kj) > 0 for k = 1,2,3,  D_ki = det(c, l_k, l_i),
i.e. iff w* restricted to c-perp is a POSITIVE combination of (c x l_k)(c x l_k)^T."""
import numpy as np
from calib_core import predicted_pattern
rng=np.random.default_rng(0)
PAIRS=((0,1,2),(1,2,0),(2,0,1))          # (i, j, k)
def junction():
    c=np.array([rng.normal(0,300),rng.normal(0,300),1.0]); L=[]
    for _ in range(3):
        th=rng.uniform(0,np.pi); a,b=np.cos(th),np.sin(th); L.append(np.array([a,b,-(a*c[0]+b*c[1])]))
    return c,L
def randK():
    return np.array([[rng.uniform(300,2000),rng.normal(0,80),rng.normal(0,300)],[0,rng.uniform(300,2000),rng.normal(0,300)],[0,0,1.0]])
agree=n=adm=0; pat_ok=0; worst_rec=0.0
for _ in range(200000):
    c,L=junction(); K=randK(); W=K@K.T
    D=lambda k,i: np.linalg.det(np.array([c,L[k],L[i]]))
    lam=[(L[i]@W@L[j])/(D(k,i)*D(k,j)) for i,j,k in PAIRS]
    prod=np.prod([L[i]@W@L[j] for i,j,k in PAIRS])
    a1=prod<0; a2=all(l>0 for l in lam)
    n+=1; agree+=(a1==a2); adm+=a1
    if n<=20000:
        # the closed formula for the cell's sign pattern, against the cyclic-order rule
        pat=tuple(int(np.sign(D(k,i)*D(k,j))) for i,j,k in PAIRS)
        pat_ok+= pat==predicted_pattern(L,c)
        # reconstruction on c-perp:  x^T W x = sum lam_k (m_k.x)^2  for x in c-perp
        m=[np.cross(c,l) for l in L]; x=np.cross(c,rng.standard_normal(3))
        lhs=x@W@x; rhs=sum(lam[k2]*(m[k2]@x)**2 for k2 in range(3))
        # lam above is indexed by pair order (k = 2,0,1); map to k
        lamk={k:lam[q] for q,(i,j,k) in enumerate(PAIRS)}
        rhs=sum(lamk[k2]*(m[k2]@x)**2 for k2 in range(3))
        worst_rec=max(worst_rec,abs(lhs-rhs)/abs(lhs))
print(f"{n} random junctions x random cameras (all five intrinsics varied)")
print(f"  admissible (product of forms < 0) in {100*adm/n:.1f}%")
print(f"  'all lambda_k > 0'  agrees with admissibility in {agree}/{n}")
print(f"  closed-form sign pattern agrees with the cyclic-order rule in {pat_ok}/20000")
print(f"  w* on c-perp equals sum lambda_k (m_k.x)^2: worst relative error {worst_rec:.1e}")
