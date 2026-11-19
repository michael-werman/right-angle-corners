"""Global admitted set when one edge-on line crosses the image and the others miss it.
With N_ik = N_jk = 0 (k the crossing axis), the remaining conditions are
   N_ii > r N_ij,  N_jj > N_ij / r  for r = zeta_i/zeta_j at its two extremes over P,
   N_kk + u* N_ij > 0              with u* the extreme of zeta_k^2/(zeta_i zeta_j) over P.
Checked against brute force on a dense set of junctions, and applied to photographs that only pitch."""
import numpy as np
rng=np.random.default_rng(4)
half=480.0
def coords(V,n=1500000):
    C=np.column_stack([rng.uniform(-half,half,n),rng.uniform(-half,half,n),np.ones(n)])
    corners=np.array([[x,y,1.0] for x in (-half,half) for y in (-half,half)])
    Vi=np.linalg.inv(V); return (Vi@C.T).T,(Vi@corners.T).T
def nu_min(N,Z):
    m=np.inf
    for a in range(3):
        p,q=[b for b in range(3) if b!=a]
        m=min(m,(N[a,a]-Z[:,a]/Z[:,p]*N[a,p]-Z[:,a]/Z[:,q]*N[a,q]+Z[:,a]**2/(Z[:,p]*Z[:,q])*N[p,q]).min())
    return m
# ---- (1) the example of the paper: f=900, frame at 45 deg yaw, 4 deg tilt
f=900.0; K=np.diag([f,f,1.0]); th=np.radians(45); p=np.radians(4)
Ry=np.array([[np.cos(th),0,np.sin(th)],[0,1,0],[-np.sin(th),0,np.cos(th)]]); Rx=np.array([[1,0,0],[0,np.cos(p),-np.sin(p)],[0,np.sin(p),np.cos(p)]])
D=Rx@Ry; V=K@D
Z,Zc=coords(V)
cross=[bool(Z[:,a].min()<0<Z[:,a].max()) for a in range(3)]; k=cross.index(True); i,j=[a for a in range(3) if a!=k]
# extremes over the closed image: r = zeta_i/zeta_j is a ratio of affine functions, so its extremes are at corners;
# u = zeta_k^2/(zeta_i zeta_j) is not, so its extreme is found on the dense sample plus the corners
r=np.concatenate([Z[:,i]/Z[:,j],Zc[:,i]/Zc[:,j]]); u=np.concatenate([Z[:,k]**2/(Z[:,i]*Z[:,j]),Zc[:,k]**2/(Zc[:,i]*Zc[:,j])])
rm,rp=r.min(),r.max(); us=u.max() if u.max()>0 else u.min()
# the interval for N_ij with unit diagonal, from the five inequalities
lo,hi=-1.0,1.0
for rr in (rm,rp):
    if rr*1>0: hi=min(hi,1/rr) if rr>0 else hi; lo=max(lo,1/rr) if rr<0 else lo
    # 1 - rr*x > 0
for rr in (rm,rp):
    # 1 - x/rr > 0
    if rr>0: hi=min(hi,rr)
    else: lo=max(lo,rr)
for rr in (rm,rp):
    if rr>0: hi=min(hi,1/rr)
    else: lo=max(lo,1/rr)
if us>0: lo=max(lo,-1/us)
else: hi=min(hi,-1/us)
print(f"crossing axis {k+1}; r over the image in [{rm:.3f},{rp:.3f}], extreme of u = {us:.3f}")
print(f"(1) formula: N_{i+1}{j+1} in ({lo:.4f}, {hi:.4f})")
xs=np.linspace(-0.99,0.99,397); ok=[]
for x in xs:
    N=np.eye(3); N[i,j]=N[j,i]=x; ok.append(nu_min(N,Z)>0)
ok=np.array(ok); print(f"    brute force over 1.5 million junctions: admitted for N_ij in [{xs[ok].min():.4f}, {xs[ok].max():.4f}]  (grid step {xs[1]-xs[0]:.4f})")
# ---- (2) photographs that only pitch: the exact interval of t in K(I + t e1 e1^T)K^T
def rot(ax,deg):
    a=np.radians(deg); c,s=np.cos(a),np.sin(a)
    return {'x':np.array([[1,0,0],[0,c,-s],[0,s,c]]),'y':np.array([[c,0,s],[0,1,0],[-s,0,c]])}[ax]
Kg=np.array([[900.,15.,40.],[0.,700.,-30.],[0.,0.,1.]])
frames=[rot('x',4)@rot('y',40),rot('x',11)@rot('y',50),rot('x',-7)@rot('y',45)]
e1=np.array([1.0,0,0]); tlo,thi=-1.0,np.inf; data=[]
for R in frames:
    Vm=Kg@R; Zm,Zcm=coords(Vm,600000)
    cr=[bool(Zm[:,a].min()<0<Zm[:,a].max()) for a in range(3)]; km=cr.index(True); im,jm=[a for a in range(3) if a!=km]
    a=R.T@e1                     # N = R^T (I + t e1 e1^T) R = I + t a a^T, and a_k = 0 (the camera only pitches)
    rr=np.concatenate([Zm[:,im]/Zm[:,jm],Zcm[:,im]/Zcm[:,jm]]); uu=np.concatenate([Zm[:,km]**2/(Zm[:,im]*Zm[:,jm]),Zcm[:,km]**2/(Zcm[:,im]*Zcm[:,jm])])
    rlist=[rr.min(),rr.max()]; ustar=uu.max() if uu.max()>0 else uu.min()
    cons=[]                        # each condition is  alpha + beta t > 0
    for r0 in rlist:
        cons.append((1.0, a[im]**2 - r0*a[im]*a[jm]))      # N_ii - r N_ij
        cons.append((1.0, a[jm]**2 - a[im]*a[jm]/r0))      # N_jj - N_ij / r
    cons.append((1.0, ustar*a[im]*a[jm]))                  # N_kk + u* N_ij
    for al,be in cons:
        if be>0: tlo=max(tlo,-al/be)
        elif be<0: thi=min(thi,-al/be)
    data.append((R,Zm))
print(f"(2) three photographs that only pitch: formula gives t in ({tlo:.4f}, {thi:.4f})")
ts=np.linspace(-0.9,3.0,391); adm=[]
for t in ts:
    Xi=np.eye(3)+t*np.outer(e1,e1); adm.append(all(nu_min(R.T@Xi@R,Zm)>0 for R,Zm in data))
adm=np.array(adm); print(f"    brute force: admitted for t in [{ts[adm].min():.4f}, {ts[adm].max():.4f}]  (grid step {ts[1]-ts[0]:.4f})")
