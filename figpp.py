"""Figure: the principal points admitted by one junction, for the true focal length.
Each pairwise condition f^2 A_ij + d_i(p) d_j(p) = 0 is a hyperbola centered at the junction, with the two edge lines
of the pair as asymptotes."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import figstyle
from common import rand_frame
rng=np.random.default_rng(7); f=500.0; K=np.diag([f,f,1.0]); H=500   # a 90-degree field of view
P3=((0,1),(1,2),(2,0))
def make():
    D=rand_frame(rng); P=np.array([*rng.uniform(-2.0,2.0,2),rng.uniform(2.5,5)])
    c=np.array([*(K@P)[:2]/P[2],1.0])
    if np.max(np.abs(c[:2]))>330: return None
    L=[]
    for k in range(3):
        l=np.cross(c,K@D[:,k]); L.append(l/np.hypot(l[0],l[1]))
    return c,L
R=900; xs=np.linspace(-R,R,700); PX,PY=np.meshgrid(xs,xs)
def fields(c,L):
    d=[L[i][0]*PX+L[i][1]*PY+L[i][2] for i in range(3)]
    return [f**2*(L[i][:2]@L[j][:2])+d[i]*d[j] for i,j in P3]
best=None
for _ in range(20000):
    J=make()
    if J is None: continue
    c,L=J; F=fields(c,L); adm=np.prod(F,axis=0)<0
    share=adm.mean(); inside=adm[np.argmin(np.abs(xs))][np.argmin(np.abs(xs))]
    dirs=[np.array([-l[1],l[0]]) for l in L]
    sep=min(np.degrees(np.arccos(min(1,abs(dirs[i]@dirs[j])))) for i,j in P3)
    core=(np.abs(PX)<650)&(np.abs(PY)<650)
    visible=all(Fij[core].min()<0<Fij[core].max() for Fij in F) and min(abs(L[i][:2]@L[j][:2]) for i,j in P3)>=0.3
    if inside and 0.2<share<0.6 and sep>25 and visible:
        best=(c,L,F,adm); break
c,L,F,adm=best
fig,ax=plt.subplots(figsize=(3.4,3.5))
ax.contourf(PX,PY,adm.astype(float),levels=[0.5,1.5],colors=["#cfe0f3"])
cols=["#2d4a7a","#c8521b","#4f7d43"]
for (i,j),Fij,col in zip(P3,F,cols):
    ax.contour(PX,PY,Fij,levels=[0],colors=[col],linewidths=1.1)
    ax.plot([],[],color=col,lw=1.1,label=f"pair ({i+1},{j+1})")
t=np.linspace(-2*R,2*R,2)
for l in L:                                   # edge lines through the junction: the asymptotes
    u=np.array([-l[1],l[0]]); ax.plot(c[0]+t*u[0],c[1]+t*u[1],color="0.35",lw=0.7,ls=(0,(3,3)))
ax.plot([-H,H,H,-H,-H],[-H,-H,H,H,-H],color="0.2",lw=0.8)
ax.plot(*c[:2],"o",color="0.1",ms=3.5,zorder=5); ax.annotate(r"$\mathbf{c}$",c[:2],xytext=(5,4),textcoords="offset points",fontsize=9)
ax.plot(0,0,"k*",ms=8,zorder=6,label="true principal point")
ax.set_xlim(-R,R); ax.set_ylim(-R,R); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.legend(frameon=False,fontsize=6.5,loc="upper center",bbox_to_anchor=(0.5,-0.01),ncol=2,handlelength=1.6,columnspacing=1.2)
fig.tight_layout(); fig.savefig("figures/fig_ppregion.pdf")
print(f"wrote figures/fig_ppregion.pdf; junction at ({c[0]:.0f},{c[1]:.0f}); admitted share of the plotted window {adm.mean():.2f}")
