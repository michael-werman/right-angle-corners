"""General several-frames theorem.  Claim: with w* = K Xi K^T, a visible vanishing line of axis d
forces 'd is an eigenvector of Xi'.  So the camera is determined iff the visible axes do NOT all
lie in the union of a plane through the origin and its normal."""
import numpy as np, itertools
rng=np.random.default_rng(0)
def rot(ax,deg):
    a=np.radians(deg); c,s=np.cos(a),np.sin(a)
    return {'x':np.array([[1,0,0],[0,c,-s],[0,s,c]]),'y':np.array([[c,0,s],[0,1,0],[-s,0,c]]),'z':np.array([[c,-s,0],[s,c,0],[0,0,1]])}[ax]
IDX=[(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]
def eig_rows(d):
    """Xi d parallel to d  <=>  u^T Xi d = 0 for two vectors u spanning d-perp: rows on the 6 entries of Xi"""
    u1=np.cross(d,[1,0,0] if abs(d[0])<0.9 else [0,1,0]); u1/=np.linalg.norm(u1); u2=np.cross(d,u1)
    return [[u[p]*d[q]+(u[q]*d[p] if p!=q else 0) for p,q in IDX] for u in (u1,u2)]
def free_dim(axes):
    A=np.array([r for d in axes for r in eig_rows(d)]); s=np.linalg.svd(A,compute_uv=False)
    return 6-int((s>1e-9*s[0]).sum())-1          # parameters left free, after the scale of Xi
def in_plane_and_normal(axes,tol=1e-9):
    """is there a unit n with every axis either parallel or perpendicular to n?"""
    A=[d/np.linalg.norm(d) for d in axes]; cands=[d for d in A]
    cands+=[np.cross(a,b)/np.linalg.norm(np.cross(a,b)) for a,b in itertools.combinations(A,2) if np.linalg.norm(np.cross(a,b))>1e-6]
    return any(all(min(abs(d@n),np.linalg.norm(np.cross(d,n)))<tol*1e3 for d in A) for n in cands)
g=lambda pitch,roll=0: rot('z',roll)@rot('x',pitch)@np.array([0,1.0,0])      # the vertical, in camera coordinates
cases=[("one photo, horizon only",[g(4)]),
 ("two photos, horizon only",[g(4),g(11,3)]),
 ("three photos, the camera only pitches (verticals coplanar)",[g(4),g(11),g(-7)]),
 ("three photos, pitch and roll both vary",[g(4),g(11,3),g(-7,-5)]),
 ("three photos, same attitude, different yaw",[g(6),g(6),g(6)]),
 ("two full frames sharing an axis",list(rot('x',20).T)+list((rot('x',20)@rot('y',37)).T)),
 ("two full frames, no common axis",list(rot('x',20).T)+list((rot('z',40)@rot('x',-35)@rot('y',15)).T))]
print(f"{'visible axes':60s}{'free':>5}{'  in plane+normal':>18}")
for lab,ax in cases: print(f"{lab:60s}{free_dim(ax):>5}{str(in_plane_and_normal(ax)):>18}")
# the criterion against the rank, on random sets of axes, half of them built to be degenerate
ok=n=0
for _ in range(4000):
    k=rng.integers(1,6)
    if rng.random()<0.5:
        nn=rng.standard_normal(3); nn/=np.linalg.norm(nn); ax=[]
        for _ in range(k):
            v=rng.standard_normal(3); ax.append(nn.copy() if rng.random()<0.3 else v-(v@nn)*nn)
    else: ax=[rng.standard_normal(3) for _ in range(k)]
    ax=[a/np.linalg.norm(a) for a in ax]; n+=1; ok+= (free_dim(ax)>0)==in_plane_and_normal(ax)
print(f"\n'camera not determined' <=> 'axes lie in a plane and its normal': agree in {ok}/{n} random sets of 1 to 5 axes")
