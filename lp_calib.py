"""Self-calibration from junctions as an exact convex program, K-free.
Each junction's sign pattern P* is read from the junction alone (Theorem S1),
so its three constraints sigma_ij * (l_i^T w* l_j) > 0 are known without K.
For w* = diag(X, Y, 1) positive-definiteness is X, Y > 0: a linear program."""
import numpy as np
from scipy.optimize import linprog
from calib_core import predicted_pattern
rng=np.random.default_rng(2)
FX,FY=900.0,700.0
def frame():
    Q,R=np.linalg.qr(rng.standard_normal((3,3))); Q=Q@np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q)<0: Q[:,0]*=-1
    return Q
def junction(K):
    while True:
        D=frame(); P=np.array([rng.uniform(-3,3),rng.uniform(-3,3),rng.uniform(2,6)])
        c=K@P; c=c/c[2]
        if max(abs(c[:2]))>600: continue
        L=[]
        for i in range(3):
            q=K@(P+1e-6*D[:,i]); q=q/q[2]; d=(q-c)[:2]; d/=np.linalg.norm(d)
            a,b=d[1],-d[0]; L.append(np.array([a,b,-(a*c[0]+b*c[1])]))
        return L,c
if __name__ == "__main__":
    K=np.diag([FX,FY,1.0])
    def box(Js, margin=1e-9):
        """exact extent of {X,Y>0 : sigma*(aaX+bbY+ee) > 0 for all junctions}, via 4 LPs"""
        A=[];b=[]
        for L,c in Js:
            P=predicted_pattern(L,c)                       # K-free
            for (i,j),s in zip(((0,1),(1,2),(2,0)),P):
                # s*(aa X + bb Y + ee) >= margin  ->  -s*aa X - s*bb Y <= s*ee - margin
                A.append([-s*L[i][0]*L[j][0], -s*L[i][1]*L[j][1]]); b.append(s*L[i][2]*L[j][2]-margin)
        A=np.array(A); b=np.array(b); bd=[(1e-6,None),(1e-6,None)]
        ext=[]
        for k,sgn in ((0,1),(0,-1),(1,1),(1,-1)):
            cvec=np.zeros(2); cvec[k]=sgn
            r=linprog(cvec,A_ub=A,b_ub=b,bounds=bd,method="highs")
            ext.append(sgn*r.fun if r.status==0 else (np.inf if sgn<0 else np.nan))
        return ext   # [Xmin, Xmax, Ymin, Ymax]
    print(f"true camera f_x={FX}, f_y={FY};   sign patterns computed WITHOUT K")
    print(f"{'junctions':>10}{'f_x range':>24}{'f_y range':>24}{'true feasible':>15}")
    Js=[]
    for M in (1,4,15,60,250,1000):
        while len(Js)<M: Js.append(junction(K))
        Xmn,Xmx,Ymn,Ymx=box(Js)
        feas=all(np.sign(L[i]@np.diag([FX**2,FY**2,1])@L[j])==s
                 for L,c in Js for (i,j),s in zip(((0,1),(1,2),(2,0)),predicted_pattern(L,c)))
        fx=f"[{np.sqrt(Xmn):.1f}, {np.sqrt(Xmx):.1f}]" if np.isfinite(Xmx) else f"[{np.sqrt(Xmn):.1f}, inf)"
        fy=f"[{np.sqrt(Ymn):.1f}, {np.sqrt(Ymx):.1f}]" if np.isfinite(Ymx) else f"[{np.sqrt(Ymn):.1f}, inf)"
        print(f"{M:>10}{fx:>24}{fy:>24}{str(feas):>15}")
