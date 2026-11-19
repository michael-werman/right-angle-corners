"""Independent check of the referee's counterexample A: a vanishing line that only
TOUCHES the image (at a corner) forces exactly one linear condition, N_ik = t0 N_jk."""
import numpy as np
half=480.0; f=1440.0; K=np.diag([f,f,1.0])
R=np.array([[0.2805,-0.5596,0.7798],[-0.8551,0.2233,0.4679],[-0.4360,-0.7981,-0.4159]])
U_,S_,Vt=np.linalg.svd(R); R=U_@Vt                      # nearest exact rotation to the 4-digit entries
# the entries are given to four digits; make axis 3 exactly perpendicular to the corner's ray
r0=np.array([half/f,half/f,1.0]); d3=R[:,2]-(R[:,2]@r0)*r0/(r0@r0); d3/=np.linalg.norm(d3)
d1=R[:,0]-(R[:,0]@d3)*d3; d1/=np.linalg.norm(d1); R=np.column_stack([d1,np.cross(d3,d1),d3])
V=K@R; Vi=np.linalg.inv(V)
corner=np.array([half,half,1.0]); z0=Vi@corner
print("zeta at the corner (480,480):",np.round(z0/np.abs(z0).max(),5),"  -> zeta_3 = 0 there: line 3 passes through the corner")
g=np.linspace(-half,half,481); X,Y=np.meshgrid(g,g); C=np.stack([X.ravel(),Y.ravel(),np.ones(X.size)],1); Z=(Vi@C.T).T
print("zeta_3 over the image, relative to its largest magnitude: min %.2e, max %.2e (one sign: the line touches, it does not cross)"%(Z[:,2].min()/np.abs(Z[:,2]).max(),Z[:,2].max()/np.abs(Z[:,2]).max()))
print("sign changes of zeta over the closed image, axes 1,2,3:",[bool(Z[:,a].min()<-1e-9*np.abs(Z[:,a]).max() and Z[:,a].max()>1e-9*np.abs(Z[:,a]).max()) for a in range(3)],
      "| min |zeta|:",np.round(np.abs(Z).min(0)/np.abs(Z).max(0),6))
t0=z0[0]/z0[1]; print(f"t0 = zeta_1/zeta_2 at the corner = {t0:.5f}")
def nu_min(N,pts):
    Zp=(Vi@pts.T).T; m=np.inf
    for a in range(3):
        p,q=[b for b in range(3) if b!=a]
        nu=N[a,a]-Zp[:,a]/Zp[:,p]*N[a,p]-Zp[:,a]/Zp[:,q]*N[a,q]+Zp[:,a]**2/(Zp[:,p]*Zp[:,q])*N[p,q]; m=min(m,nu.min())
    return m
rng=np.random.default_rng(0)
pts=np.column_stack([rng.uniform(-half,half,400000),rng.uniform(-half,half,400000),np.ones(400000)])
near=np.column_stack([half-10.0**rng.uniform(-9,1,200000),half-10.0**rng.uniform(-9,1,200000),np.ones(200000)])   # approaching the corner
allp=np.vstack([pts,near])
def Nmat(n13,n23,n12=0.0):
    N=np.eye(3); N[0,2]=N[2,0]=n13; N[1,2]=N[2,1]=n23; N[0,1]=N[1,0]=n12; return N
N=Nmat(t0*0.2,0.2)
W=V@N@V.T; Kp=np.linalg.cholesky(np.eye(3)[::-1]@(W/W[2,2])@np.eye(3)[::-1]); Kp=np.eye(3)[::-1]@Kp@np.eye(3)[::-1]
Wi=np.linalg.inv(W); cos=lambda a,b: (V[:,a]@Wi@V[:,b])/np.sqrt((V[:,a]@Wi@V[:,a])*(V[:,b]@Wi@V[:,b]))
print(f"\nN with N23 = 0.2, N13 = t0*0.2: positive-definite {np.linalg.eigvalsh(N)[0]>0}; camera f_x={Kp[0,0]:.1f}, f_y={Kp[1,1]:.1f}, skew={Kp[0,1]:.1f}")
print(f"   cosines between its vanishing points: (1,3) {cos(0,2):+.3f}, (2,3) {cos(1,2):+.3f}  -> not orthogonal")
print(f"   smallest nu over 600,000 junctions, down to 1e-9 px from the corner: {nu_min(N,allp):+.3f}  -> every junction admitted")
for d in (1e-2,1e-4,1e-6):
    print(f"   N13 moved by {d:g}: smallest nu = {nu_min(Nmat(t0*0.2+d,0.2),allp):+.2e}")
print(f"   N23 = 0.2 with N13 = 0 (ignoring the condition): smallest nu = {nu_min(Nmat(0.0,0.2),allp):+.2e}")
