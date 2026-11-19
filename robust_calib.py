"""Noise-tolerant convex calibration from right-angle junctions.

Every junction contributes three constraints  s_k * (l_i^T w* l_j) >= 0  whose
signs s_k are read from the junction alone (the convexity theorem).  Under noise
they cannot all hold, so each gets a slack and the program minimises the total
violation.  Violations are measured as cosines between interpretation planes,
(l_i^T w* l_j) / sqrt((l_i^T w* l_i)(l_j^T w* l_j)), so they are scale-free; the
denominators are frozen at the current estimate and refreshed (a few LPs).

Models:  'f'      w* = diag(x, x, 1)            1 parameter
         'aspect' w* = diag(X, Y, 1)            2 parameters
         'full'   w* = K K^T, w*_33 = 1         5 parameters (PSD by cutting planes)
Coordinates are divided by S0 so the entries of w* are O(1).
"""
import numpy as np
from scipy.optimize import linprog
from calib_core import predicted_pattern, IDX, row as row_full, W_of
S0=400.0
PAIRS=((0,1),(1,2),(2,0))
def normalise(Js):
    """pixel junctions -> normalised: lines (a,b,e/S0), vertex (u/S0, v/S0, 1)"""
    return [([np.array([l[0],l[1],l[2]/S0]) for l in L], np.array([c[0]/S0,c[1]/S0,1.0])) for L,c,*_ in Js]
def coef(model, li, lj):
    if model=='f':      return np.array([li[0]*lj[0]+li[1]*lj[1]]), li[2]*lj[2]
    if model=='aspect': return np.array([li[0]*lj[0], li[1]*lj[1]]), li[2]*lj[2]
    return row_full(li,lj)
def W_from(model, w):
    if model=='f': return np.diag([w[0],w[0],1.0])
    if model=='aspect': return np.diag([w[0],w[1],1.0])
    return W_of(w)
NPAR={'f':1,'aspect':2,'full':5}
def assemble(model, Jn):
    R=[];C=[];S=[];LL=[]
    for L,c in Jn:
        pat=predicted_pattern(L,c)
        for (i,j),s in zip(PAIRS,pat):
            co,cst=coef(model,L[i],L[j]); R.append(co); C.append(cst); S.append(s); LL.append((L[i],L[j]))
    return np.array(R),np.array(C),np.array(S,float),LL
def weights(LL, W):
    return np.array([1.0/np.sqrt(max(li@W@li,1e-12)*max(lj@W@lj,1e-12)) for li,lj in LL])
def violation(model, w, R, C, S, LL):
    W=W_from(model,w); n=weights(LL,W)
    return float(np.sum(np.maximum(0.0, -S*(R@w+C)*n)))
def _bounds(model):
    if model in ('f','aspect'): return [(1e-4,400.0)]*NPAR[model]
    return [(1e-4,400.0),(-50,50),(-20,20),(1e-4,400.0),(-20,20)]
def _lp(model, R, C, S, n, cw, cs, extra=None, cuts=()):
    """min cw.w + cs*sum(slack)  s.t.  S*(R w + C)*n + slack >= 0, slack >= 0, cuts, extra"""
    p=R.shape[1]; K=len(C)
    A=np.hstack([-(S*n)[:,None]*R, -np.eye(K)]); b=(S*n)*C
    A=[A]; b=[b]
    for co,cst in cuts: A.append(np.hstack([-co,np.zeros(K)])[None,:]); b.append(np.array([cst]))
    if extra is not None:                                 # sum(slack) <= extra
        A.append(np.hstack([np.zeros(p),np.ones(K)])[None,:]); b.append(np.array([extra]))
    r=linprog(np.hstack([cw,np.full(K,cs)]),A_ub=np.vstack(A),b_ub=np.concatenate(b),
              bounds=_bounds(model)+[(0,None)]*K,method="highs")
    return r
def _solve(model, R, C, S, n, cw, cs, extra=None):
    cuts=[]
    for _ in range(200):
        r=_lp(model,R,C,S,n,cw,cs,extra,cuts)
        if r.status!=0: return None
        w=r.x[:R.shape[1]]
        if model!='full': return w, r
        ev,evec=np.linalg.eigh(W_of(w))
        if ev[0]>=1e-7: return w, r
        co,cst=row_full(evec[:,0],evec[:,0]); cuts.append((co,cst-1e-6))   # v^T W v >= 1e-6
    return w, r
def fit(model, Js, reweights=4, tol=0.05, ranges=True):
    """returns dict: w (point estimate), W, total violation, per-parameter range of the
    near-optimal set {violation <= (1+tol)*min}"""
    Jn=normalise(Js); R,C,S,LL=assemble(model,Jn); p=R.shape[1]
    w=np.array({'f':[2.0],'aspect':[2.0,2.0],'full':[2.0,0,0,2.0,0]}[model])   # nominal start
    for _ in range(reweights):
        n=weights(LL,W_from(model,w))
        out=_solve(model,R,C,S,n,np.zeros(p),1.0)
        if out is None: break
        w=out[0]
    n=weights(LL,W_from(model,w))
    out=_solve(model,R,C,S,n,np.zeros(p),1.0); w0,r0=out; V=r0.fun
    res={'model':model,'V':V,'n':len(C)//3}
    lo=np.full(p,np.nan); hi=np.full(p,np.nan)
    if ranges:
        for k in range(p):
            for sg,store in ((1,lo),(-1,hi)):
                cw=np.zeros(p); cw[k]=sg
                o=_solve(model,R,C,S,n,cw,0.0,extra=V*(1+tol)+1e-9)
                if o is not None: store[k]=o[0][k]
        w=np.where(np.isfinite(lo)&np.isfinite(hi),0.5*(lo+hi),w0)
        # keep the centre only if it is itself near-optimal; else fall back to the LP vertex
        if violation(model,w,R,C,S,LL)>1.5*violation(model,w0,R,C,S,LL)+1e-9: w=w0
    else: w=w0
    res.update(w=w,lo=lo,hi=hi,W=W_from(model,w)); return res
def K_of(W):
    P=np.eye(3)[::-1]; return P@np.linalg.cholesky(P@W@P)@P
def params(model, res):
    """camera parameters in PIXELS: fx, fy, skew, px, py"""
    Kn=K_of(res['W']); return dict(fx=Kn[0,0]*S0, fy=Kn[1,1]*S0, skew=Kn[0,1]*S0, px=Kn[0,2]*S0, py=Kn[1,2]*S0)
