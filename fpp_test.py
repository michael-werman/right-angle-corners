"""Focal length AND principal point unknown (square pixels, zero skew).
Prediction: determined unless (1) no visible axis is tilted, or (2) the visible tilted axes are
one axis, or two perpendicular axes, in a plane through the optical axis, every other visible
axis being normal to that plane.  In case (2): p = nu*u, s^2 = 1 + k nu - nu^2, k = c/delta - delta/c."""
import numpy as np, itertools
rng=np.random.default_rng(1)
z=np.array([0,0,1.0])
def predicted_determined(axes,tol=1e-9):
    T=[d for d in axes if abs(d[2])>tol]; L=[d for d in axes if abs(d[2])<=tol]
    if not T: return False
    bars=[d[:2]/np.linalg.norm(d[:2]) for d in T]
    if any(abs(a[0]*b[1]-a[1]*b[0])>1e-7 for a,b in itertools.combinations(bars,2)): return True
    u=bars[0]
    if any(abs(e[:2]@u)>1e-7 for e in L): return True
    distinct=[]
    for d in T:
        if not any(np.linalg.norm(np.cross(d,e))<1e-7 for e in distinct): distinct.append(d)
    if len(distinct)==1: return False
    if len(distinct)==2 and abs(distinct[0]@distinct[1])<1e-7: return False
    return True
def linear_injective(axes):
    """first-order map (sigma,alpha,beta) -> (I - d d^T) dXi d over the axes; injective?"""
    cols=[]
    for e in np.eye(3):
        sig,al,be=e; dM=np.array([[sig,0,al],[0,sig,be],[0,0,0]]); dX=dM+dM.T
        cols.append(np.concatenate([(np.eye(3)-np.outer(d,d))@dX@d for d in axes]))
    s=np.linalg.svd(np.array(cols).T,compute_uv=False); return s[-1]>1e-9*s[0]
def rand_unit(): v=rng.standard_normal(3); return v/np.linalg.norm(v)
def level(): v=rng.standard_normal(3); v[2]=0; return v/np.linalg.norm(v)
def in_plane_with_z(u,theta): return np.array([np.cos(theta)*u[0],np.cos(theta)*u[1],np.sin(theta)])
sets=[]
for _ in range(1500):
    kind=rng.integers(0,7); u=level()[:2]; nrm=np.array([-u[1],u[0],0.0]); th=rng.uniform(0.15,1.4)*rng.choice([-1,1])
    if kind==0: ax=[level() for _ in range(rng.integers(1,4))]                                   # case (1)
    elif kind==1: ax=[in_plane_with_z(u,th)]+[nrm]*rng.integers(0,2)                           # case (2), one axis
    elif kind==2: ax=[in_plane_with_z(u,th),in_plane_with_z(u,th+np.pi/2)]+[nrm]*rng.integers(0,2)  # case (2), two perpendicular
    elif kind==3: ax=[in_plane_with_z(u,th),in_plane_with_z(u,th+rng.uniform(0.2,1.2))]          # coplanar with z, not perpendicular
    elif kind==4: ax=[in_plane_with_z(u,th),level()]                                            # tilted + generic level
    elif kind==5: R=np.linalg.qr(rng.standard_normal((3,3)))[0]; ax=[R[:,0],R[:,1],R[:,2]]          # a full frame
    else: ax=[rand_unit() for _ in range(rng.integers(1,4))]
    sets.append(ax)
agree=sum(predicted_determined(a)==linear_injective(a) for a in sets)
print(f"(a) prediction vs first-order injectivity: agree in {agree}/{len(sets)} axis sets (all seven kinds of configuration)")
# (b) junction level
def rot(ax,deg):
    a=np.radians(deg); c,s=np.cos(a),np.sin(a)
    return {'x':np.array([[1,0,0],[0,c,-s],[0,s,c]]),'y':np.array([[c,0,s],[0,1,0],[-s,0,c]]),'z':np.array([[c,-s,0],[s,c,0],[0,0,1]])}[ax]
half=480.0
def junction_check(f,frames,M):
    K=np.array([[f,0,40.0],[0,f,-30.0],[0,0,1]]); Ki=np.linalg.inv(K); Xi=M@M.T
    C=np.column_stack([rng.uniform(-half,half,300000),rng.uniform(-half,half,300000),np.ones(300000)])
    m=np.inf; vis=[]
    for R in frames:
        Z=(R.T@Ki@C.T).T; vis.append([bool(Z[:,a].min()<0<Z[:,a].max()) for a in range(3)])
        N=R.T@Xi@R; Z=Z[np.abs(Z).min(1)>1e-9]
        for a in range(3):
            p,q=[b for b in range(3) if b!=a]
            m=min(m,(N[a,a]-Z[:,a]/Z[:,p]*N[a,p]-Z[:,a]/Z[:,q]*N[a,q]+Z[:,a]**2/(Z[:,p]*Z[:,q])*N[p,q]).min())
    return m,vis
Mof=lambda s,p: np.array([[s,0,p[0]],[0,s,p[1]],[0,0,1.0]])
def family(nu,k,u): return Mof(np.sqrt(1+k*nu-nu**2),nu*np.asarray(u))
print("\n(b) junction level, f and principal point unknown")
# case (2), two perpendicular tilted axes and their level normal, all three lines in view (needs a wide view)
th=np.radians(30); R=np.column_stack([[1,0,0],[0,np.cos(th),np.sin(th)],[0,-np.sin(th),np.cos(th)]]); k=1/np.tan(th)-np.tan(th)
for nu in (0.05,0.12):
    m,vis=junction_check(250.0,[R],family(nu,k,(0,1)))
    print(f"   one frame, one axis parallel to the image plane, lines in view {vis[0]}: family nu={nu:+.2f}  smallest nu_a {m:+.3f} -> {'ADMITTED' if m>0 else 'rejected'}")
m,_=junction_check(250.0,[R],Mof(1.05,(0,0))); print(f"      focal length alone changed by 5%:                                 smallest nu_a {m:+.2e} -> {'ADMITTED' if m>0 else 'rejected'}")
# horizons only
fr=lambda pitch,yaw,roll=0: rot('z',roll)@rot('x',pitch)@rot('y',yaw)
g1=fr(6,40)[:,1]; c1=np.sign(g1[1])*np.linalg.norm(g1[:2]); k1=c1/g1[2]-g1[2]/c1; u1=g1[:2]/(np.sign(g1[1])*np.linalg.norm(g1[:2]))
for nu in (0.03,0.08):
    m,vis=junction_check(900.0,[fr(6,40)],family(nu,k1,u1))
    print(f"   one photo, horizon only {vis[0]}: family of photo 1, nu={nu:+.2f}            smallest nu_a {m:+.3f} -> {'ADMITTED' if m>0 else 'rejected'}")
for nu in (0.03,0.08):
    m,vis=junction_check(900.0,[fr(6,40),fr(14,70)],family(nu,k1,u1))
    print(f"   two photos, horizons only, camera only pitches {vis}: same family nu={nu:+.2f} smallest nu_a {m:+.2e} -> {'ADMITTED' if m>0 else 'rejected'}")
# strictly horizon-only pairs, the camera only pitching
def vertical_family(R,nu):
    g=R[:,1]; c=np.sign(g[1])*np.linalg.norm(g[:2]); kk=c/g[2]-g[2]/c; uu=g[:2]/c
    return family(nu,kk,uu)
for pitches in ((6,13),(4,9)):
    frames=[fr(p,45) for p in pitches]; Mf=vertical_family(frames[0],0.03)
    m1,_=junction_check(900.0,frames[:1],Mf); m,vis=junction_check(900.0,frames,Mf)
    print(f"   pitches {pitches}, yaw 45, lines in view {vis}: family of photo 1 (nu=0.03): photo 1 alone {m1:+.3f}, both photos {m:+.2e} -> {'ADMITTED' if m>0 else 'rejected'}")
