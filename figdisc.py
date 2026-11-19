"""Theorem 4 in the projective disc model, with Theorem 6 as a slice.
Up to scale, the positive-definite forms on the plane c-perp make up a disc whose boundary circle
holds the rank-one forms.  One junction admits exactly the cameras in the triangle spanned by its
three rank-one forms (m_k . x)^2, and the cameras diag(x, x, 1) form a straight segment, which
meets the triangle in the interval F of Theorem 6.  Writes figures/fig10_disc.pdf."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle  # noqa: F401
from common import coeffs, admissible, rand_frame, observe_corner, Kmat
rng=np.random.default_rng(11); f=866.0
def disc(S):   # symmetric 2x2 form -> projective (Klein) disc coordinates
    a,b,c=S[0,0],S[0,1],S[1,1]; return np.array([(a-c)/(a+c),2*b/(a+c)])
def setup(c2,L):
    c=np.array([c2[0],c2[1],1.0]); Ls=[np.array(l,float) for l in L]
    e1=np.cross(c,[1.0,0,0]); e1/=np.linalg.norm(e1); e2=np.cross(c,e1); e2/=np.linalg.norm(e2); B=np.column_stack([e1,e2])
    # put the true camera at the center: the disc picture is defined up to its isometries, and this is one
    S0=B.T@np.diag([f*f,f*f,1.0])@B; w,V=np.linalg.eigh(S0); B=B@(V@np.diag(w**-0.5)@V.T)
    return c,Ls,B
best=None
for _ in range(3000):   # a junction with a bounded interval and a well-shaped triangle
    P=np.array([rng.uniform(-0.5,0.5),rng.uniform(-0.5,0.5),1.0])*rng.uniform(2,6)
    o=observe_corner(P,rand_frame(rng),f,0.0,rng)
    if o is None: continue
    c2,L=o; cm_=admissible(coeffs(c2,L))
    if not(len(cm_)==1 and cm_[0][1]<1e12): continue
    c_,Ls_,B_=setup(c2,L)
    qq=[disc(np.outer(B_.T@np.cross(c_,l),B_.T@np.cross(c_,l))) for l in Ls_]
    area=0.5*abs(np.cross(qq[1]-qq[0],qq[2]-qq[0]))
    fl=lambda x: disc(B_.T@np.diag([x,x,1.0])@B_)
    seglen=np.linalg.norm(fl(cm_[0][1])-fl(cm_[0][0]))
    score=area*seglen
    if best is None or score>best[0]: best=(score,c2,L,cm_)
_,c2,L,cm=best
c,Ls,B=setup(c2,L)
q=[disc(np.outer(B.T@np.cross(c,l),B.T@np.cross(c,l))) for l in Ls]
fam=lambda x: disc(B.T@np.diag([x,x,1.0])@B)
fig,ax=plt.subplots(figsize=(3.2,3.0))
th=np.linspace(0,2*np.pi,400); ax.plot(np.cos(th),np.sin(th),color="0.3",lw=0.8)
ax.fill([p[0] for p in q],[p[1] for p in q],color="#cfe0f3",zorder=1)
for i in range(3): ax.plot([q[i][0],q[(i+1)%3][0]],[q[i][1],q[(i+1)%3][1]],color="0.45",lw=0.7,ls="--",zorder=2)
for i,p in enumerate(q):
    ax.plot(*p,"o",color="0.2",ms=3.5,zorder=4); ax.annotate(f"$q_{i+1}$",p,xytext=(p*1.13),ha="center",va="center",fontsize=9)
ts=np.linspace(1e-4,1-1e-4,800); xs=(ts/(1-ts))*f**2*0.02
seg=np.array([fam(x) for x in np.concatenate([[1e-9],np.geomspace(1e-6,1e12,2000)])])
ax.plot(seg[:,0],seg[:,1],color="0.25",lw=0.9,zorder=3)
xin=np.geomspace(cm[0][0],cm[0][1],400); inn=np.array([fam(x) for x in xin])
ax.plot(inn[:,0],inn[:,1],color="#c8521b",lw=3,solid_capstyle="butt",zorder=3,label=r"$F$")
_cen=sum(np.asarray(p) for p in q)/3                   # each edge is one linear condition: the edge opposite q_k is nu_k = 0
for _k in range(3):
    _a,_b=np.asarray(q[(_k+1)%3]),np.asarray(q[(_k+2)%3])
    _ts=np.linspace(0.35,0.65,13); _pts=[_a+(t)*(_b-_a) for t in _ts]          # along the edge, away from the segment of cameras
    _mid=max(_pts,key=lambda p: np.min(np.linalg.norm(seg-p,axis=1))); _out=(_mid-_cen)/np.linalg.norm(_mid-_cen)
    ax.annotate(rf"$\nu_{_k+1}=0$",_mid+0.11*_out,ha="center",va="center",fontsize=7.5,color="0.25",zorder=6)
tc=fam(f**2); ax.plot(*tc,"k*",ms=8,zorder=5,label="true camera")
p0=fam(1e-9); ax.annotate(r"$f'\to0$",p0,xytext=(-22,-12),textcoords="offset points",fontsize=7)
pinf=fam(1e12); ax.plot(*pinf,"o",ms=2.5,color="0.25",zorder=4); ax.annotate(r"$f'\to\infty$",pinf,xytext=(-30,6),textcoords="offset points",fontsize=7)
ax.set_aspect("equal"); ax.axis("off"); ax.legend(frameon=False,fontsize=7,loc="lower left")
fig.tight_layout(); fig.savefig("figures/fig10_disc.pdf"); print("wrote figures/fig10_disc.pdf; F =",np.sqrt(cm[0][0])/f,np.sqrt(cm[0][1])/f,"x f")
