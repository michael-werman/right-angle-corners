import numpy as np
from calib_core import rand_frame
rng=np.random.default_rng(1)
def Lam(M,g):
    out=[]
    for a in range(3):
        b,c=[q for q in range(3) if q!=a]
        out.append(M[a,a]-g[a]/g[b]*M[a,b]-g[a]/g[c]*M[a,c]+g[a]**2/(g[b]*g[c])*M[b,c])
    return np.array(out)
def admissible(W,c,V):
    L=[np.cross(c,V[:,k]) for k in range(3)]
    return np.prod([L[i]@W@L[j] for i,j in ((0,1),(1,2),(2,0))])<0
# (1) the formula: admissible  <=>  all Lambda_a > 0, for random PD w*, random frames, random junctions
agree=n=0
for _ in range(100000):
    K=np.array([[rng.uniform(400,1500),rng.normal(0,60),rng.normal(0,200)],[0,rng.uniform(400,1500),rng.normal(0,200)],[0,0,1.0]])
    V=K@rand_frame(rng)
    K2=np.array([[rng.uniform(400,1500),rng.normal(0,60),rng.normal(0,200)],[0,rng.uniform(400,1500),rng.normal(0,200)],[0,0,1.0]])
    W=K2@K2.T; c=np.array([rng.uniform(-500,500),rng.uniform(-500,500),1.0])
    M=np.linalg.inv(V)@W@np.linalg.inv(V).T; g=np.linalg.solve(V,c)
    n+=1; agree+= (admissible(W,c,V)==bool((Lam(M,g)>0).all()))
print(f"(1) junctions of a frame: 'all Lambda_a > 0' agrees with admissibility in {agree}/{n}")
# (2) exact joint admissible set when ONLY the horizon crosses the image (level-ish camera)
f=900.0; K=np.diag([f,f,1.0]); th=np.radians(45); p=np.radians(4)
Rx=np.array([[1,0,0],[0,np.cos(p),-np.sin(p)],[0,np.sin(p),np.cos(p)]])
D=Rx@np.column_stack([[np.cos(th),0,np.sin(th)],[-np.sin(th),0,np.cos(th)],[0,1,0]])
V=K@D; half=480; gr=np.linspace(-half,half,241); U,Vv=np.meshgrid(gr,gr)
C=np.stack([U.ravel(),Vv.ravel(),np.ones(U.size)],1); G=np.linalg.solve(V,C.T).T
cross=[bool(G[:,a].min()<0<G[:,a].max()) for a in range(3)]
print(f"\n(2) vanishing lines crossing the image (gamma_a changes sign), axes 1,2,3: {cross}")
k=[a for a in range(3) if cross[a]][0]; i,j=[a for a in range(3) if a!=k]
tau=G[:,i]/G[:,j]
print(f"    tau = gamma_i/gamma_j over the image: [{tau.min():.3f}, {tau.max():.3f}]")
def all_adm(M):
    W=V@M@V.T
    if np.linalg.eigvalsh(W)[0]<=0: return False
    L=[np.cross(C,V[:,q]) for q in range(3)]
    P=np.ones(len(C))
    for a,b in ((0,1),(1,2),(2,0)): P*=np.einsum('ij,jk,ik->i',L[a],W,L[b])
    return bool((P<0).all())
# predicted: with M_ik = M_jk = 0 and M = I + t(E_ij+E_ji): admissible for all c iff 1 > sup(tau t) and 1 > sup(t/tau)
kap=G[:,k]**2/(G[:,i]*G[:,j])
def predicted(t): return (1-(tau*t).max()>0) and (1-(t/tau).max()>0) and (1+(kap*t).min()>0)
ts=np.linspace(-1.5,1.5,301); ok=0; lo=hi=None
for t in ts:
    M=np.eye(3); M[i,j]=M[j,i]=t
    a=all_adm(M); b=predicted(t) and abs(t)<1
    ok+= a==b
    if a: lo=t if lo is None else lo; hi=t
print(f"    M = I + t(E_ij+E_ji): exact prediction matches the brute-force check for {ok}/{len(ts)} values of t")
print(f"    admissible range of the free entry: t in [{lo:.2f}, {hi:.2f}]  (0 = the vanishing-point family)")
for lab,(a,b) in (("M_ik",(i,k)),("M_jk",(j,k))):
    M=np.eye(3); M[a,b]=M[b,a]=1e-3
    print(f"    {lab} = 0.001 (an entry involving the crossing axis): admissible for all junctions? {all_adm(M)}")
