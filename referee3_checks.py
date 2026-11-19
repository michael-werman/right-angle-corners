"""Checks of the claims in the third referee report."""
import numpy as np, itertools
from math import erf, sqrt, comb
Phi=lambda z: 0.5*(1+erf(z/sqrt(2)))
rng=np.random.default_rng(5)
# ---- #6: eq (18) linearizes the threshold, not the sign of the form
rho,f,sig,ell=300.0,866.0,0.5,150.0
s2=2*sig**2/ell**2
print("#6  eq.(18) near a raw right angle: pair (i,j) with psi_i - psi_j close to 90 deg")
for d in (2.0,0.5,0.1):
    pi_,pj=np.radians(40.0),np.radians(40.0-90.0+d)
    A=np.cos(pi_-pj); ee=rho**2*np.sin(pi_)*np.sin(pj); x=-ee/A
    sdx=rho**2*np.sqrt(np.sin(2*pi_)**2+np.sin(2*pj)**2)/(np.sqrt(2)*ell/sig*A**2)
    p18=Phi(-abs(f*f-x)/sdx)
    form=A*f*f+ee; dfi=-np.sin(pi_-pj)*f*f+rho**2*np.cos(pi_)*np.sin(pj); dfj=np.sin(pi_-pj)*f*f+rho**2*np.sin(pi_)*np.cos(pj)
    pnew=Phi(-abs(form)/np.sqrt(s2*(dfi**2+dfj**2)))
    n=400000; ei=rng.normal(0,np.sqrt(s2),n); ej=rng.normal(0,np.sqrt(s2),n)
    fm=np.cos(pi_+ei-pj-ej)*f*f+rho**2*np.sin(pi_+ei)*np.sin(pj+ej)
    print(f"    90-{d:>4} deg: Monte Carlo {np.mean(np.sign(fm)!=np.sign(form)):.2e}   eq.(18) {p18:.3f}   linearized form {pnew:.2e}")
# ---- #7: at a root of (28) the candidate camera admits the corner that defined it
print("#7  the corner's own sign test at the roots of (28): by construction, v1, v2, v3 are omega-orthogonal at a root;")
print("    K^-1 v_i lies in interpretation plane i, so the triad is admissible and the test passes (algebraic, no numerics needed)")
# ---- #9: one matched axis, double flip, rays 90 deg apart and the axis in their plane
def H(r): r=r/np.linalg.norm(r); return np.eye(3)-2*np.outer(r,r)
Q,_r=np.linalg.qr(rng.standard_normal((3,3))); dk=Q[:,0]
r1=Q[:,1]+0.3*dk; r1/=np.linalg.norm(r1)            # dk lies in span(r1,r2)
r2=np.cross(np.cross(r1,dk),r1); r2=r2/np.linalg.norm(r2)   # r2 perpendicular to r1, in the plane of r1 and dk
v=H(r2)@H(r1)@dk
print(f"#9  one matched axis in the plane of two rays 90 deg apart: angle(r1,r2)={np.degrees(np.arccos(r1@r2)):.1f} deg, "
      f"|H2 H1 d x d| = {np.linalg.norm(np.cross(v,dk)):.1e}  (0 = double flip consistent on that axis)")
# ---- #10: tail of gamma near an interior vanishing point: y Pr(gamma>y) against log y
from common import Kmat
f=500.0; K=Kmat(f); Ki=np.linalg.inv(K)
th=np.radians(25); R=np.array([[1,0,0],[0,np.cos(th),-np.sin(th)],[0,np.sin(th),np.cos(th)]])@np.array([[np.cos(.4),0,np.sin(.4)],[0,1,0],[-np.sin(.4),0,np.cos(.4)]])
V=K@R; vp=[V[:2,a]/V[2,a] for a in range(3)]
inside=[bool(np.all(np.abs(p)<480)) for p in vp]; z=R[2,:]
n=4_000_000; C=np.column_stack([rng.uniform(-480,480,n),rng.uniform(-480,480,n),np.ones(n)])
Z=(R.T@Ki@C.T).T; W=z[None,:]/Z
gam=np.max(np.stack([(z[a]-Z[:,a]*W[:,p])*(z[a]-Z[:,a]*W[:,q]) for a,p,q in ((0,1,2),(1,2,0),(2,0,1))],1),1)
print(f"#10 vanishing points inside the image: {inside}; tilts z = {np.round(z,3)}")
for y in (1e2,1e3,1e4,1e5): print(f"    y={y:>8.0f}: y*Pr(gamma>y) = {y*np.mean(gam>y):.3f}   (log y = {np.log(y):.1f})")
# ---- #22: pre-fixed rules on the 181 real junctions
lo=np.array([0.38*0]); 
# ---- #26, #57
def wilson(k,n,z=1.959964):
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n); h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)); return (c-h)/d,(c+h)/d
print("#26 Wilson 95%% interval for 179/181: %.3f - %.3f"%wilson(179,181))
p=sum(comb(59,k) for k in range(41,60))/2**59
print(f"#57 sign test, line sampling better on 41 of 59 images: one-sided p = {p:.4f}, two-sided p = {2*p:.4f}")
