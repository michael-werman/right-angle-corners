"""Checks for the robustness results: corners that are not right-angled, and radial lens distortion."""
import numpy as np
from common import coeffs, admissible, rand_frame
rng=np.random.default_rng(17); f=866.0; HALF=500.0
def junction_lines(c,dirs2d):
    L=[]
    for u in dirs2d:
        a,b=u[1],-u[0]; n=np.hypot(a,b); a/=n; b/=n; e=-(a*c[0]+b*c[1]); L.append((a,b,e))   # half-edge direction (b,-a)=u
    return L
def interval(c,L):
    comps=admissible(coeffs(c,L))
    if not comps: return None
    return np.sqrt(comps[0][0])/f,(np.sqrt(comps[-1][1])/f if comps[-1][1]<1e13 else np.inf)
def image_dirs(c,D,K):
    out=[]
    for k in range(3):
        v=K@D[:,k]; P=np.array([c[0],c[1],1.0])
        # image direction of edge k at c: derivative of the projection along d_k
        X=np.linalg.inv(K)@P; X=X*3.0                       # a point on the ray
        Y=X+1e-4*D[:,k]; y=K@Y; y=y[:2]/y[2]; u=y-c; out.append(u/np.linalg.norm(u))
    return out
K=np.diag([f,f,1.0])
# ---- (2) non-right-angled corners: |c_ij| < |a_i a_j| for all pairs  =>  the true camera admits the junction
viol=0; tested=0; cov={}
for deg in (2,5,10):
    hits=[]; 
    for _ in range(4000):
        D0=rand_frame(rng); D=D0.copy()
        for k in range(3):                                   # tilt each edge by up to deg degrees
            w=rng.standard_normal(3); w-=w@D[:,k]*D[:,k]; w/=np.linalg.norm(w); t=np.radians(rng.uniform(0,deg))
            D[:,k]=np.cos(t)*D[:,k]+np.sin(t)*w
        c=rng.uniform(-HALF*0.95,HALF*0.95,2); r=np.linalg.inv(K)@np.array([c[0],c[1],1.0]); rh=r/np.linalg.norm(r)
        a=D.T@rh; C=D.T@D
        if (D.T@r<=0).any(): pass
        L=junction_lines(c,image_dirs(c,D,K)); iv=interval(c,L)
        ok=iv is not None and iv[0]<=1.0<=iv[1]
        cond=all(abs(C[i,j])<abs(a[i]*a[j]) for i,j in ((0,1),(1,2),(2,0)))
        if cond: tested+=1; viol+= not ok
        hits.append(ok)
    cov[deg]=100*np.mean(hits)
print(f"(2) corners off 90 deg: the condition |c_ij|<|a_i a_j| held in {tested} cases, and the true f was outside F in {viol} of them (must be 0)")
print(f"    coverage of the true f, 60 deg FOV, noiseless, edges tilted by up to 2/5/10 deg: {cov[2]:.1f}% / {cov[5]:.1f}% / {cov[10]:.1f}%")
# ---- (3) radial distortion x -> x(1+k|x|^2): with tangent edges, the observed junction is the true one with
#      rho -> rho(1+k rho^2) and tan psi_i -> mu tan psi_i, mu=(1+k rho^2)/(1+3k rho^2); check against direct computation
worst=0; covd={}
for kmax in (0.01,0.02,0.05):                               # k*rho_max^2: the relative displacement at the image corner
    k=kmax/(2*HALF**2); hits=[]
    for _ in range(3000):
        D=rand_frame(rng); c=rng.uniform(-HALF*0.95,HALF*0.95,2); dirs=image_dirs(c,D,K)
        rho2=c@c; J=(1+k*rho2)*np.eye(2)+2*k*np.outer(c,c); cd=c*(1+k*rho2)
        dd=[J@u/np.linalg.norm(J@u) for u in dirs]
        iv=interval(cd,junction_lines(cd,dd))
        # prediction from the transformation of (rho, psi)
        rho=np.sqrt(rho2); mu=(1+k*rho2)/(1+3*k*rho2); ur=c/rho; ut=np.array([-ur[1],ur[0]])
        pd=[]
        for u in dirs:
            ps=np.arctan2(u@ut,u@ur); ps2=np.arctan2(mu*np.sin(ps),np.cos(ps)); pd.append(np.cos(ps2)*ur+np.sin(ps2)*ut)
        ip=interval(cd,junction_lines(cd,pd))
        if iv and ip: worst=max(worst,abs(iv[0]-ip[0])/(1+iv[0]))
        hits.append(iv is not None and iv[0]<=1.0<=iv[1])
    covd[kmax]=100*np.mean(hits)
print(f"(3) distortion: interval from the (rho,psi) transformation against direct computation, worst relative difference {worst:.1e}")
print(f"    coverage of the true f, 60 deg FOV, noiseless, displacement at the image corner 1/2/5%: {covd[0.01]:.1f}% / {covd[0.02]:.1f}% / {covd[0.05]:.1f}%")
