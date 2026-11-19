"""Full 5-parameter calibration: w* = K K^T with skew and principal point free,
w*_33 = 1.  Junction constraints are linear (K-free patterns); positive-
definiteness is handled by Kelley cutting planes, so only linprog is needed."""
import numpy as np
from scipy.optimize import linprog
from lp_calib import predicted_pattern, junction, frame, rng
IDX=[(0,0),(0,1),(0,2),(1,1),(1,2)]            # free entries; w22 = 1
def row(li,lj):
    """coefficients of l_i^T w l_j in the 5 free entries, plus the w22 constant"""
    co=np.zeros(5)
    for k,(a,b) in enumerate(IDX):
        co[k]= li[a]*lj[b] + (li[b]*lj[a] if a!=b else 0.0)
    return co, li[2]*lj[2]
def build(Js):
    A=[];b=[]
    for L,c in Js:
        for (i,j),s in zip(((0,1),(1,2),(2,0)),predicted_pattern(L,c)):
            co,cst=row(L[i],L[j]); A.append(-s*co); b.append(s*cst-1e-9)
    return np.array(A),np.array(b)
def W_of(w):
    W=np.eye(3); 
    for k,(a,bb) in enumerate(IDX): W[a,bb]=W[bb,a]=w[k]
    W[2,2]=1.0; return W
def extreme(A,b,cvec,box=5e7,iters=400):
    Ac,bc=list(A),list(b)
    for _ in range(iters):
        r=linprog(cvec,A_ub=np.array(Ac),b_ub=np.array(bc),bounds=[(-box,box)]*5,method="highs")
        if r.status!=0: return None
        W=W_of(r.x); ev,evec=np.linalg.eigh(W)
        if ev[0]>=-1e-9*max(1,abs(ev[-1])): return r.x
        v=evec[:,0]                                   # cut: v^T W v >= 0
        co,cst=row(v,v); Ac.append(-co); bc.append(cst)
    return r.x
def K_of(W):
    P=np.eye(3)[::-1]; return P@np.linalg.cholesky(P@W@P)@P
if __name__ == "__main__":
    Kt=np.array([[900.,15.,40.],[0.,700.,-30.],[0.,0.,1.]])
    Wt=Kt@Kt.T
    print("true K: fx=900 fy=700 skew=15 px=40 py=-30;  junctions from independent frames")
    print(f"{'junctions':>10}  extent of each entry of w* / its true value  (1.000 = pinned)")
    Js=[]
    for M in (30,120,500,2000):
        while len(Js)<M: Js.append(junction(Kt))
        A,b=build(Js)
        spans=[]
        for k in range(5):
            cv=np.zeros(5); cv[k]=1; lo=extreme(A,b,cv)
            cv[k]=-1; hi=extreme(A,b,cv)
            tv=W_of(np.zeros(5)); t=Wt[IDX[k]]
            spans.append((lo[k] if lo is not None else np.nan, hi[k] if hi is not None else np.nan, t))
        txt="  ".join(f"{('w'+str(a)+str(bb))}:[{lo/t:.3f},{hi/t:.3f}]" for (lo,hi,t),(a,bb) in zip(spans,IDX))
        print(f"{M:>10}  {txt}")
