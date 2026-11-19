"""Claim: for three concurrent lines, the admissible set of omega* is ONE cell
of the arrangement, intersected with the PD cone -- hence convex.

Reduction: l_i . c = 0, so all three l_i lie in the 2-D subspace V = c^perp.
The forms l_i^T w* l_j depend on w* only through Q = E^T w* E (E a basis of V),
a 2x2 PD form.  Write u_i for the coordinates of l_i in that basis.

Step 1. Pick A with Q = A^T A.  Then <u_i,u_j>_Q = (A u_i).(A u_j), a Euclidean
dot product.  For three lines through the origin, sort their directions in
[0,pi): theta1<theta2<theta3, gaps a=t2-t1, b=t3-t2, c=pi-(t3-t1), a+b+c=pi.
   prod cos(theta_i - theta_j) = cos a cos b cos(a+b) = -cos a cos b cos c,
so  product < 0  <=>  cos a cos b cos c > 0  <=>  ALL THREE GAPS < 90 deg.

Step 2. The six oriented rays +-A u_i keep their cyclic order under any A with
det > 0.  In an all-acute configuration the gaps between consecutive rays are
a,b,c,a,b,c, so the sign of <u_i,u_j> is fixed by how many steps separate the
rays u_i and u_j in that order: + for 1 or 5 steps, - for 2, 3 or 4.
The step counts do not depend on Q.  So on the WHOLE admissible set every form
has a fixed sign: one sign pattern, hence one cell.  Convex.
"""
import numpy as np
rng=np.random.default_rng(0)

def basis_of_V(c):
    """orthonormal basis of the plane perpendicular to the homogeneous point c"""
    c=c/np.linalg.norm(c); a=np.array([1.,0,0]) if abs(c[0])<0.9 else np.array([0,1.,0])
    e1=np.cross(c,a); e1/=np.linalg.norm(e1); e2=np.cross(c,e1)
    return np.column_stack([e1,e2])

def predicted_pattern(L,c):
    """the sign pattern on the admissible set, from the junction ALONE (no K):
    sort the six rays +-u_i by angle, read step counts."""
    E=basis_of_V(c); U=[E.T@l for l in L]
    rays=[]
    for i,u in enumerate(U):
        for sgn in (1,-1):
            v=sgn*u; rays.append((np.arctan2(v[1],v[0])%(2*np.pi), i, sgn))
    rays.sort()
    pos={(i,s):k for k,(_,i,s) in enumerate(rays)}
    pat=[]
    for i,j in ((0,1),(1,2),(2,0)):
        steps=(pos[(j,1)]-pos[(i,1)])%6
        pat.append(1 if steps in (1,5) else -1)
    return tuple(pat)

def step1_check(n=20000):
    bad=0
    for _ in range(n):
        th=np.sort(rng.uniform(0,np.pi,3))
        a,b,c=th[1]-th[0],th[2]-th[1],np.pi-(th[2]-th[0])
        prod=np.cos(th[0]-th[1])*np.cos(th[1]-th[2])*np.cos(th[2]-th[0])
        if (prod<0)!=(a<np.pi/2 and b<np.pi/2 and c<np.pi/2): bad+=1
    return bad
print("Step 1: product<0 <=> all three gaps acute, violations:", step1_check())

def frame(rng):
    Q,R=np.linalg.qr(rng.standard_normal((3,3))); Q=Q@np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q)<0: Q[:,0]*=-1
    return Q
def junction(K,rng):
    D=frame(rng); P=np.array([rng.uniform(-2,2),rng.uniform(-2,2),rng.uniform(2,6)])
    c=K@P; c=c/c[2]; L=[]
    for i in range(3):
        q=K@(P+1e-6*D[:,i]); q=q/q[2]; d=(q-c)[:2]; d/=np.linalg.norm(d)
        a,b=d[1],-d[0]; L.append(np.array([a,b,-(a*c[0]+b*c[1])]))
    return L,c
def rand_K(rng,Kt,s):
    return np.array([[Kt[0,0]*np.exp(rng.normal(0,s)), Kt[0,1]+rng.normal(0,s*200), Kt[0,2]+rng.normal(0,s*300)],
                     [0, Kt[1,1]*np.exp(rng.normal(0,s)), Kt[1,2]+rng.normal(0,s*300)],[0,0,1.0]])
forms=lambda L,W: np.array([L[i]@W@L[j] for i,j in ((0,1),(1,2),(2,0))])

Kt=np.array([[900.,12,40],[0,700.,-30],[0,0,1]])
mism_true=mism_any=n=samples=0; outside=0
for _ in range(1500):
    L,c=junction(Kt,rng); n+=1
    P=predicted_pattern(L,c)
    if tuple(np.sign(forms(L,Kt@Kt.T)).astype(int))!=P: mism_true+=1
    for _ in range(1500):
        K=rand_K(rng,Kt,0.9); W=K@K.T; F=forms(L,W)
        adm = np.prod(F)<0
        inP = tuple(np.sign(F).astype(int))==P
        samples+=1
        if adm and not inP: mism_any+=1          # admissible but in another cell
        if inP and not adm: outside+=1           # in the cell but not admissible
print(f"\nStep 2, {n} junctions, {samples} sampled cameras (all 5 entries of K):")
print(f"  predicted pattern != pattern at the TRUE camera     : {mism_true}")
print(f"  admissible camera lying in a DIFFERENT cell         : {mism_any}")
print(f"  camera in the predicted cell but NOT admissible     : {outside}")
print("  => admissible set == predicted cell  intersected with the PD cone")

# Corollary: PGL(2) is 3-transitive on lines through the origin, so ANY three
# distinct concurrent lines are the image of a right-angle corner under SOME
# camera.  Test on RANDOM concurrent triples (not images of real corners).
def find_Q(U,rng,tries=4000):
    for _ in range(tries):
        A=rng.standard_normal((2,2))
        if np.linalg.det(A)<=0: continue
        V=[A@u for u in U]
        p=np.prod([V[i]@V[j] for i,j in ((0,1),(1,2),(2,0))])
        if p<0: return True
    return False
ok=n=0
for _ in range(3000):
    c=np.array([rng.uniform(-400,400),rng.uniform(-400,400),1.0])
    E=basis_of_V(c)
    U=[E.T@np.array([np.cos(t),np.sin(t),-(np.cos(t)*c[0]+np.sin(t)*c[1])])
       for t in rng.uniform(0,np.pi,3)]
    n+=1; ok+=find_Q(U,rng)
print(f"\nCorollary: random concurrent triples admissible under SOME camera: {ok}/{n}")

def explicit_A(U):
    """send the three lines to directions 0, 60, 120 deg explicitly:
    v3 = alpha v1 + beta v2;  A v1 = s1 e0,  A v2 = e60,  and since
    e120 = e60 - e0 we need s1 = -beta/alpha."""
    v1,v2,v3=U
    M=np.column_stack([v1,v2]); alpha,beta=np.linalg.solve(M,v3)
    e0=np.array([1.,0]); e60=np.array([.5,np.sqrt(3)/2])
    s1=-beta/alpha
    A=np.column_stack([s1*e0,e60])@np.linalg.inv(M)
    return A
ok=n=0
for _ in range(3000):
    c=np.array([rng.uniform(-400,400),rng.uniform(-400,400),1.0]); E=basis_of_V(c)
    U=[E.T@np.array([np.cos(t),np.sin(t),-(np.cos(t)*c[0]+np.sin(t)*c[1])])
       for t in rng.uniform(0,np.pi,3)]
    A=explicit_A(U); V=[A@u for u in U]; n+=1
    ok+= np.prod([V[i]@V[j] for i,j in ((0,1),(1,2),(2,0))])<0
print(f"Corollary, EXPLICIT construction: admissible under some camera: {ok}/{n}")

# (i) every PD Q on V comes from a real camera: extend Q to a PD w* on R^3,
#     rescale so w*_33 = 1, and recover an upper-triangular K with K K^T = w*.
def K_from_wstar(W):
    """upper-triangular K with K K^T = W, via Cholesky of the reversed matrix"""
    P=np.eye(3)[::-1]; Lc=np.linalg.cholesky(P@W@P); return P@Lc@P
bad=0
for _ in range(2000):
    c=np.array([rng.uniform(-400,400),rng.uniform(-400,400),1.0]); E=basis_of_V(c)
    A=rng.standard_normal((2,2)); Q=A.T@A+0.1*np.eye(2)
    ch=c/np.linalg.norm(c)
    W=E@Q@E.T+rng.uniform(0.5,3)*np.outer(ch,ch); W/=W[2,2]
    K=K_from_wstar(W)
    if not (np.allclose(K@K.T,W) and np.allclose(np.tril(K,-1),0) and abs(K[2,2]-1)<1e-9
            and np.allclose(E.T@W@E/ (E.T@W@E)[0,0], Q/Q[0,0])):
        bad+=1
print(f"(i) every PD Q realised by an upper-triangular camera K: failures {bad}/2000")

# (ii) the signs from the step-count rule always multiply to -1 (odd # negative)
bad=0
for _ in range(3000):
    c=np.array([rng.uniform(-400,400),rng.uniform(-400,400),1.0])
    L=[np.array([np.cos(t),np.sin(t),-(np.cos(t)*c[0]+np.sin(t)*c[1])]) for t in rng.uniform(0,np.pi,3)]
    if np.prod(predicted_pattern(L,c))!=-1: bad+=1
print(f"(ii) step-count pattern always has an odd number of negatives: failures {bad}/3000")
