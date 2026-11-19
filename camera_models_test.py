"""Checks for Section 4.3: the unknown aspect ratio (a polygon of three half-planes) and the unknown principal point
(three hyperbolas centered at the junction, scaled with f)."""
import numpy as np, sympy as sp
from common import rand_frame
rng=np.random.default_rng(11)
# ---- the decomposition behind the principal-point result
f,px,py=sp.symbols('f p_x p_y',positive=True); K=sp.Matrix([[f,0,px],[0,f,py],[0,0,1]]); pt=sp.Matrix([px,py,1])
print("K K^T = f^2 diag(1,1,0) + p p^T:", sp.simplify(K*K.T-(f**2*sp.diag(1,1,0)+pt*pt.T))==sp.zeros(3,3))
def junction(f0=866.0):
    while True:
        D=rand_frame(rng); P=np.array([*rng.uniform(-2.5,2.5,2),rng.uniform(2,6)]); Kt=np.diag([f0,f0,1.0])
        c=np.array([*(Kt@P)[:2]/P[2],1.0])
        if np.max(np.abs(c[:2]))>480: continue
        L=[]
        for k in range(3):
            l=np.cross(c,Kt@D[:,k]); L.append(l/np.hypot(l[0],l[1]))
        return c,L
def admitted(om,c,L):                                     # Proposition 3, for a positive-definite omega*
    g=[L[i]@om@L[j] for i,j in ((0,1),(1,2),(2,0))]; return np.prod(g)<0
# ---- (1) unknown aspect ratio: admitted <=> three half-planes with the signs of Delta_ki Delta_kj
bad=0; n=0
for _ in range(400):
    c,L=junction()
    Dl=lambda k,i: np.linalg.det(np.column_stack([c,L[k],L[i]]))
    for X,Y in rng.uniform(0.05,4.0,(60,2))*866.0**2:
        om=np.diag([X,Y,1.0]); a=admitted(om,c,L)
        hp=all(np.sign(L[i]@om@L[j])==np.sign(Dl(k,i)*Dl(k,j)) for k,(i,j) in ((2,(0,1)),(0,(1,2)),(1,(2,0))))
        bad+= a!=hp; n+=1
print(f"(1) aspect ratio: admitted <=> the three half-planes, disagreements {bad} of {n}")
# ---- (2) unknown principal point, known f: the form is f^2 A_ij + d_i(p) d_j(p); scaling about c by f
bad=0; badh=0; n=0
for _ in range(400):
    c,L=junction(); f0=rng.uniform(300,1500)
    for p in rng.uniform(-900,900,(30,2)):
        Kp=np.array([[f0,0,p[0]],[0,f0,p[1]],[0,0,1.0]]); om=Kp@Kp.T; pt=np.array([p[0],p[1],1.0])
        for i,j in ((0,1),(1,2),(2,0)):
            A=L[i][:2]@L[j][:2]; lhs=L[i]@om@L[j]; rhs=f0**2*A+(L[i]@pt)*(L[j]@pt)
            bad+= abs(lhs-rhs)>1e-6*max(1,abs(lhs))
        # homothety: p = c + f q  has the same verdict for every f
        q=(p-c[:2])/f0; verdicts=set()
        for f1 in (f0*0.3,f0,f0*3):
            p1=c[:2]+f1*q; K1=np.array([[f1,0,p1[0]],[0,f1,p1[1]],[0,0,1.0]]); verdicts.add(admitted(K1@K1.T,c,L))
        badh+= len(verdicts)>1; n+=1
print(f"(2) principal point: form identity failures {bad}; verdict changing under p = c + f q: {badh} of {n}")
