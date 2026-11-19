"""Junction-level check.  Three photographs, each with ONLY its horizon in view.
If the camera only pitches, the verticals are coplanar and K (I + t n n^T) K^T, n normal to
their plane, should admit every junction of all three.  With roll as well, it should not."""
import numpy as np
def rot(ax,deg):
    a=np.radians(deg); c,s=np.cos(a),np.sin(a)
    return {'x':np.array([[1,0,0],[0,c,-s],[0,s,c]]),'y':np.array([[c,0,s],[0,1,0],[-s,0,c]]),'z':np.array([[c,-s,0],[s,c,0],[0,0,1]])}[ax]
K=np.array([[900.,15.,40.],[0.,700.,-30.],[0.,0.,1.]]); Ki=np.linalg.inv(K); half=480.0
gr=np.linspace(-half,half,241); X,Y=np.meshgrid(gr,gr); C=np.stack([X.ravel(),Y.ravel(),np.ones(X.size)],1)
rng=np.random.default_rng(0); C=np.vstack([C,np.column_stack([rng.uniform(-half,half,200000),rng.uniform(-half,half,200000),np.ones(200000)])])
def visible(R):
    Z=(R.T@Ki@C.T).T; return [bool(Z[:,a].min()<0<Z[:,a].max()) for a in range(3)]
def nu_min(Xi,R):
    N=R.T@Xi@R; Z=(R.T@Ki@C.T).T; m=np.inf
    keep=np.abs(Z).min(1)>1e-9*np.abs(Z).max()
    Z=Z[keep]
    for a in range(3):
        p,q=[b for b in range(3) if b!=a]
        m=min(m,(N[a,a]-Z[:,a]/Z[:,p]*N[a,p]-Z[:,a]/Z[:,q]*N[a,q]+Z[:,a]**2/(Z[:,p]*Z[:,q])*N[p,q]).min())
    return m
frame=lambda pitch,yaw,roll=0: rot('z',roll)@rot('x',pitch)@rot('y',yaw)
def run(label,frames):
    print(label); print("   vanishing lines in view per photo (axes 1,2,3; axis 2 is the vertical):",[visible(R) for R in frames])
    G=np.array([R[:,1] for R in frames]); sv=np.linalg.svd(G,compute_uv=False)
    print(f"   verticals coplanar: {sv[2]<1e-9*sv[0]}")
    n=np.linalg.svd(G)[2][2]                               # normal to the best plane through the verticals
    for t in (-0.3,0.2,1.0):
        Xi=np.eye(3)+t*np.outer(n,n); m=min(nu_min(Xi,R) for R in frames)
        print(f"   K(I + {t:+.1f} n n^T)K^T : smallest nu over all three photos = {m:+.3e}  -> {'ADMITTED' if m>0 else 'rejected'}")
run("camera only pitches:",[frame(4,40),frame(11,50),frame(-7,45)])
run("\npitch and roll both vary:",[frame(4,40,6),frame(11,50,-5),frame(-7,45,9)])
