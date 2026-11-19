"""Shared core: K-free sign patterns, junction generators, the cutting-plane
convex program over the full w* = K K^T (w22 = 1)."""
import numpy as np
from scipy.optimize import linprog
def basis_of_V(c):   # kept for the scripts that still use it
    c=c/np.linalg.norm(c); a=np.array([1.,0,0]) if abs(c[0])<0.9 else np.array([0,1.,0])
    e1=np.cross(c,a); e1/=np.linalg.norm(e1); e2=np.cross(c,e1); return np.column_stack([e1,e2])
def predicted_pattern(L,c):
    """Sign pattern of the admissible cell, from the junction ALONE: on that cell the
    form l_i^T w* l_j has the sign of D_ki D_kj, D_ki = det(c, l_k, l_i)  (eq. nuk)."""
    D=lambda k,i: np.linalg.det(np.array([c,L[k],L[i]]))
    return tuple(int(np.sign(D(k,i)*D(k,j))) for i,j,k in ((0,1,2),(1,2,0),(2,0,1)))
def rand_frame(rng):
    Q,R=np.linalg.qr(rng.standard_normal((3,3))); Q=Q@np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q)<0: Q[:,0]*=-1
    return Q
def junction(K,rng,D=None,lim=600):
    while True:
        DD=rand_frame(rng) if D is None else D
        P=np.array([rng.uniform(-3,3),rng.uniform(-3,3),rng.uniform(2,6)])
        c=K@P; c=c/c[2]
        if max(abs(c[:2]))>lim: continue
        L=[];ok=True
        for i in range(3):
            q=K@(P+1e-6*DD[:,i]); q=q/q[2]; d=(q-c)[:2]; n=np.linalg.norm(d)
            if n<1e-12: ok=False;break
            d/=n; a,b=d[1],-d[0]; L.append(np.array([a,b,-(a*c[0]+b*c[1])]))
        if ok: return L,c
IDX=[(0,0),(0,1),(0,2),(1,1),(1,2)]
def row(li,lj):
    co=np.zeros(5)
    for k,(a,b) in enumerate(IDX): co[k]=li[a]*lj[b]+(li[b]*lj[a] if a!=b else 0.0)
    return co, li[2]*lj[2]
def build(Js):
    A=[];b=[]
    for L,c in Js:
        for (i,j),s in zip(((0,1),(1,2),(2,0)),predicted_pattern(L,c)):
            co,cst=row(L[i],L[j]); A.append(-s*co); b.append(s*cst-1e-9)
    return np.array(A),np.array(b)
def W_of(w):
    W=np.eye(3)
    for k,(a,bb) in enumerate(IDX): W[a,bb]=W[bb,a]=w[k]
    W[2,2]=1.0; return W
def extreme(A,b,cvec,box=5e7,iters=600):
    Ac,bc=list(A),list(b); x=None
    for _ in range(iters):
        r=linprog(cvec,A_ub=np.array(Ac),b_ub=np.array(bc),bounds=[(-box,box)]*5,method="highs")
        if r.status!=0: return None
        x=r.x; ev,evec=np.linalg.eigh(W_of(x))
        if ev[0]>=-1e-9*max(1,abs(ev[-1])): return x
        co,cst=row(evec[:,0],evec[:,0]); Ac.append(-co); bc.append(cst)
    return x
