"""Touching vanishing lines in the several-frames theorem.
(1) the touch condition of Thm 12, zeta_j N_ik = zeta_i N_jk at c0, equals (Xi d_k).(d_k x r0) = 0
(2) for the focal length alone it vanishes for all f' exactly at the exceptional point c* of Thm 16
(3) junction level: three photographs that only pitch, showing only their horizons, admit
    K(I + t n n^T)K^T; a fourth frame whose vanishing line only TOUCHES a corner removes the
    family when n.d and n.(d x r0) are nonzero, and leaves it when n.(d x r0) = 0."""
import numpy as np
rng=np.random.default_rng(3)
def rand_rot():
    Q,R=np.linalg.qr(rng.standard_normal((3,3))); Q=Q@np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q)<0: Q[:,0]*=-1
    return Q
# (1)
worst=0.0
for _ in range(3000):
    K=np.array([[rng.uniform(400,1500),rng.normal(0,30),rng.normal(0,100)],[0,rng.uniform(400,1500),rng.normal(0,100)],[0,0,1.0]])
    R=rand_rot(); A=rng.standard_normal((3,3)); Xi=A@A.T+0.1*np.eye(3)
    k=rng.integers(3); i,j=(k+1)%3,(k+2)%3                      # cyclic
    r0=np.cross(R[:,k],rng.standard_normal(3))                   # a ray perpendicular to d_k: c0 on l_k
    N=R.T@Xi@R; z=R.T@r0                                         # zeta at c0 for V = K R
    lhs=z[j]*N[i,k]-z[i]*N[j,k]; rhs=-(Xi@R[:,k])@np.cross(R[:,k],r0)
    worst=max(worst,abs(lhs-rhs)/(abs(rhs)+np.linalg.norm(Xi)*np.linalg.norm(r0)))
print(f"(1) zeta_j N_ik - zeta_i N_jk = -(Xi d_k).(d_k x r0): worst relative difference over 3000 cases {worst:.1e}")
# (2)
zh=np.array([0,0,1.0]); worst2=0.0; at_star=[]
for _ in range(2000):
    d=rng.standard_normal(3); d/=np.linalg.norm(d); s2=rng.uniform(0.3,3)
    Xi=s2*np.eye(3)+(1-s2)*np.outer(zh,zh)
    r0=np.cross(d,rng.standard_normal(3))
    val=(Xi@d)@np.cross(d,r0); pred=(1-s2)*d[2]*(zh@np.cross(d,r0))
    worst2=max(worst2,abs(val-pred)/(1+abs(pred)))
    rstar=zh-d[2]*d; at_star.append(abs((Xi@d)@np.cross(d,rstar)))
print(f"(2) focal length alone: (Xi d).(d x r0) = (1-s^2) delta zhat.(d x r0): worst difference {worst2:.1e};")
print(f"    at r0 = zhat - delta d, the ray of the point c* of Theorem 16, it vanishes for every s: max {max(at_star):.1e}")
# (3) junction level.  A frame with one vanishing line touching a corner and the other two missing
# the image needs a narrow view, so f is larger here than elsewhere.
half=480.0
K=np.array([[1800.,15.,40.],[0.,1500.,-30.],[0.,0.,1.]]); Ki=np.linalg.inv(K)
def rot(ax,deg):
    a=np.radians(deg); c,s_=np.cos(a),np.sin(a)
    return {'x':np.array([[1,0,0],[0,c,-s_],[0,s_,c]]),'y':np.array([[c,0,s_],[0,1,0],[-s_,0,c]]),'z':np.array([[c,-s_,0],[s_,c,0],[0,0,1]])}[ax]
corners=[np.array([x,y,1.0]) for x in (-half,half) for y in (-half,half)]
C=np.column_stack([rng.uniform(-half,half,300000),rng.uniform(-half,half,300000),np.ones(300000)])
nearpts=[]
for c0 in corners:
    sx,sy=np.sign(c0[0]),np.sign(c0[1])
    nearpts.append(np.column_stack([c0[0]-sx*10.0**rng.uniform(-9,1.5,60000),c0[1]-sy*10.0**rng.uniform(-9,1.5,60000),np.ones(60000)]))
C=np.vstack([C]+nearpts)
def line_state(l):
    vals=np.array([l@c for c in corners]); sc=np.abs(vals).max(); on=(np.abs(vals)<1e-9*sc).sum()
    if vals.min()<-1e-9*sc and vals.max()>1e-9*sc: return 'crosses'
    return 'contains an edge' if on>=2 else ('touches' if on==1 else 'misses')
def states(R): return [line_state(Ki.T@R[:,a]) for a in range(3)]
def complete(d3):
    """a frame with third axis d3 whose other two vanishing lines miss the image"""
    e=np.cross(d3,[1.0,0,0]); e/=np.linalg.norm(e); f2=np.cross(d3,e)
    for phi in np.linspace(0,np.pi,1441):
        d1=np.cos(phi)*e+np.sin(phi)*f2; R=np.column_stack([d1,np.cross(d3,d1),d3])
        st=states(R)
        if st[0]=='misses' and st[1]=='misses': return R
    return None
def nu_min(Xi,frames):
    m=np.inf
    for R in frames:
        N=R.T@Xi@R; Z=(R.T@Ki@C.T).T; Z=Z[np.abs(Z).min(1)>1e-12]
        for a in range(3):
            p,q=[x for x in range(3) if x!=a]
            m=min(m,(N[a,a]-Z[:,a]/Z[:,p]*N[a,p]-Z[:,a]/Z[:,q]*N[a,q]+Z[:,a]**2/(Z[:,p]*Z[:,q])*N[p,q]).min())
    return m
fr=lambda pitch,yaw: rot('x',pitch)@rot('y',yaw)
photos=[fr(4,40),fr(11,50),fr(-7,45)]; n=np.array([1.0,0,0])
print("\n(3) three photographs that only pitch, horizons only (f_x = 1800):",[states(R) for R in photos])
for t in (0.05,0.2): print(f"    K(I + {t} n n^T)K^T, n = e1: smallest nu over the three = {nu_min(np.eye(3)+t*np.outer(n,n),photos):+.3f}")
def try_axis(label,d3,c0):
    d3=d3/np.linalg.norm(d3); l=Ki.T@d3
    if line_state(l)!='touches': return False
    R4=complete(d3)
    if R4 is None: return False
    r0=Ki@c0
    print(f"    {label}: corner {c0[:2].astype(int)}, lines {states(R4)}, n.d = {n@d3:+.3f}, n.(d x r0) = {n@np.cross(d3,r0):+.2e}")
    for t in (0.05,0.2):
        m=nu_min(np.eye(3)+t*np.outer(n,n),photos+[R4])
        print(f"        all four frames, t={t}: smallest nu = {m:+.3e} -> {'ADMITTED' if m>0 else 'rejected'}")
    return True
for c0 in corners:
    sx,sy=np.sign(c0[0]),np.sign(c0[1]); l=np.array([sx,sy,-(sx*c0[0]+sy*c0[1])]); d3=K.T@l; d3/=np.linalg.norm(d3)
    if line_state(Ki.T@d3)!='touches': continue
    R4=complete(d3)
    if R4 is None: continue
    r0=Ki@c0; rh=r0/np.linalg.norm(r0)
    print(f"    touching frame at corner {c0[:2].astype(int)}: lines {states(R4)};  n.d = {n@d3:+.3f}, n.(d x r0) = {n@np.cross(d3,r0):+.2e}")
    for t in (0.05,0.2):
        m=nu_min(np.eye(3)+t*np.outer(n,n),photos+[R4])
        print(f"        with the three photographs, family t={t}: smallest nu = {m:+.3e} -> {'ADMITTED' if m>0 else 'rejected'}")
    print("    the touching frame alone, perturbations I + t a a^T:")
    for lab,a in (("a = r0 (satisfies the touch condition)",rh),("a = e1 (violates it)",n)):
        row=[]
        for t in (1e-3,1e-2,0.05,0.2):
            row.append(f"t={t:g}: {nu_min(np.eye(3)+t*np.outer(a,a),[R4]):+.2e}")
        print(f"        {lab:40s} "+"   ".join(row))
    break
