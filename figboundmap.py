"""Figure: where a junction bounds the focal length from above (Theorem: where a junction bounds the
focal length).  (a) a synthetic oblique view; (b) a simulated nearly level camera, matched to York Urban."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle  # noqa: F401
zh=np.array([0,0,1.0])
def gammas(X,Y,f,D):
    R=np.stack([X/f,Y/f,np.ones_like(X)],-1)
    G=[]
    for a,p,q in ((0,1,2),(1,2,0),(2,0,1)):
        dp,dq=D[:,p],D[:,q]
        G.append(-(R@np.cross(zh,dp))*(R@np.cross(zh,dq))/((R@dp)*(R@dq)))
    return np.stack(G,0)
def rot(ax,deg):
    a=np.radians(deg); c,s=np.cos(a),np.sin(a)
    return {'x':np.array([[1,0,0],[0,c,-s],[0,s,c]]),'y':np.array([[c,0,s],[0,1,0],[-s,0,c]]),'z':np.array([[c,-s,0],[s,c,0],[0,0,1]])}[ax]
fig,ax=plt.subplots(1,2,figsize=(7.0,3.3))
# ---------------- (a) synthetic oblique view
f=500.0; half=480.0; D=rot('z',12)@rot('x',32)@rot('y',38)
vps=[f*D[:2,k]/D[2,k] for k in range(3)]
E=1500.0; n=900; xs=np.linspace(-E,E,n); X,Y=np.meshgrid(xs,xs)
G=gammas(X,Y,f,D); up=np.max(G,0)>1
a0=ax[0]
a0.contourf(X,Y,up.astype(float),levels=[0.5,1.5],colors=["#f6c28b"],alpha=0.85)
cols=["#1b6ca8","#c8521b","#2a8f3a"]
def boundary(X,Y,f,D,a):
    p,q=[(1,2),(2,0),(0,1)][a]; R=np.stack([X/f,Y/f,np.ones_like(X)],-1)
    return (R@np.cross(zh,D[:,p]))*(R@np.cross(zh,D[:,q]))+(R@D[:,p])*(R@D[:,q])    # the circle of the theorem, without poles
for a in range(3): a0.contour(X,Y,boundary(X,Y,f,D,a),levels=[0.0],colors=[cols[a]],linewidths=1.0)
for k in range(3):        # edge-on line of axis k: points c with r . d_k = 0
    d=D[:,k]; t=np.linspace(-E*2,E*2,2)
    if abs(d[1])>abs(d[0]): xl=t; yl=-(d[0]*xl/f+d[2])*f/d[1]
    else: yl=t; xl=-(d[1]*yl/f+d[2])*f/d[0]
    a0.plot(xl,yl,"--",color="0.35",lw=0.8)
    mid=np.array([xl.mean(),yl.mean()])
for k,v in enumerate(vps):
    if np.all(np.abs(v)<E): a0.plot(*v,"o",color="k",ms=3.5); a0.annotate(f"$v_{k+1}$",v,xytext=(4,4),textcoords="offset points",fontsize=8)
for k in range(3):        # c*_k: the point of l_k nearest the principal point, ray proportional to zhat - z_k d_k
    d=D[:,k]; r=zh-d[2]*d
    if abs(r[2])>1e-9:
        c=f*r[:2]/r[2]
        if np.all(np.abs(c)<E): a0.plot(*c,"x",color="k",ms=5,mew=1.2); a0.annotate(f"$c^\\star_{k+1}$",c,xytext=(4,-10),textcoords="offset points",fontsize=8)
a0.add_patch(plt.Rectangle((-half,-half),2*half,2*half,fill=False,lw=1.1,color="k"))
# two junctions: one far from the edge-on lines, one on the edge-on line of axis 3
def junction(c,L=150,col="k"):
    for k in range(3):
        u=vps[k]-c; u=u/np.linalg.norm(u)
        a0.plot([c[0],c[0]+L*u[0]],[c[1],c[1]+L*u[1]],color=col,lw=2.2,solid_capstyle="round",zorder=6)
    a0.plot(*c,"o",color=col,ms=3.5,zorder=7)
def dist_line(c,k):         # pixel distance from c to the edge-on line of axis k
    d=D[:,k]; return abs(d[0]*c[0]/f+d[1]*c[1]/f+d[2])/(np.hypot(d[0],d[1])/f)
g=np.linspace(-half*0.8,half*0.8,121); cand=np.array([[x,y] for x in g for y in g])
far=cand[np.argmax([min(dist_line(c,k) for k in range(3)) for c in cand])]
crossing=[k for k in range(3) if (lambda v: v.min()<0<v.max())(np.array([D[0,k]*c[0]/f+D[1,k]*c[1]/f+D[2,k] for c in cand]))]
k0=crossing[0]; d=D[:,k0]
onl=[]
for x in np.linspace(-half*0.8,half*0.8,401):
    if abs(d[1])<1e-9: continue
    y=-(d[0]*x/f+d[2])*f/d[1]
    if abs(y)<half*0.8: onl.append(np.array([x,y]))
onl=np.array(onl); rs=zh-D[2,k0]*D[:,k0]; cst=f*rs[:2]/rs[2]
score=[min(min(dist_line(c,k) for k in range(3) if k!=k0),min(np.linalg.norm(c-v) for v in vps),np.linalg.norm(c-cst)) for c in onl]
on=onl[int(np.argmax(score))]
junction(far,col="#111111"); junction(on,col="#7a1f1f")
a0.annotate("on $\\ell_%d$"%(k0+1),on,xytext=(4,-16),textcoords="offset points",fontsize=8,color="#7a1f1f")
print("junctions drawn at",np.round(far),"(far from all edge-on lines) and",np.round(on),"(on the edge-on line of axis",k0+1,")")
a0.set_xlim(-E,E); a0.set_ylim(E,-E); a0.set_aspect("equal"); a0.set_xticks([]); a0.set_yticks([])
a0.set_title("(a)",fontsize=9,loc="left")
# ---------------- (b) a simulated nearly level camera, matched to York Urban: 640 x 480 px, f = 675 px,
# yawed 30 degrees about the vertical and tilted up 3 degrees; fifteen junction vertices at random positions
fT=675.0; W,H=640,480; Dy=rot('x',3)@rot('y',30)
xs=np.linspace(-W/2,W/2,641); ys=np.linspace(-H/2,H/2,481); Xb,Yb=np.meshgrid(xs,ys)
Gb=gammas(Xb,Yb,fT,Dy); upb=np.max(Gb,0)>1
a1=ax[1]
a1.contourf(Xb,Yb,upb.astype(float),levels=[0.5,1.5],colors=["#f6c28b"],alpha=0.85)
for a in range(3): a1.contour(Xb,Yb,boundary(Xb,Yb,fT,Dy,a),levels=[0.0],colors=[cols[a]],linewidths=1.0)
for k in range(3):        # the edge-on lines that pass through this image, dashed
    d=Dy[:,k]
    if abs(d[1])>=abs(d[0]): xl=np.array([-W/2,W/2]); yl=-(d[0]*xl/fT+d[2])*fT/d[1]
    else: yl=np.array([-H/2,H/2]); xl=-(d[1]*yl/fT+d[2])*fT/d[0]
    if np.any((np.abs(xl)<=W/2)&(np.abs(yl)<=H/2)): a1.plot(xl,yl,"--",color="0.25",lw=0.9)
rngb=np.random.default_rng(12); jv=np.column_stack([rngb.uniform(-W/2*0.95,W/2*0.95,15),rngb.uniform(-H/2*0.95,H/2*0.95,15)])
inside=[]
for c in jv:
    g=gammas(np.array([[c[0]]]),np.array([[c[1]]]),fT,Dy).max(); inside.append(bool(g>1))
    a1.plot(c[0],c[1],"o",mfc=("k" if inside[-1] else "none"),mec="k",ms=5,mew=1.0)
a1.plot([-W/2,W/2,W/2,-W/2,-W/2],[-H/2,-H/2,H/2,H/2,-H/2],color="k",lw=0.9)
a1.set_xlim(-W/2,W/2); a1.set_ylim(H/2,-H/2); a1.set_aspect("equal"); a1.set_xticks([]); a1.set_yticks([])
a1.set_title("(b)",fontsize=9,loc="left")
name="simulated nearly level camera"
fig.tight_layout(); fig.savefig("figures/fig11_boundmap.pdf")
print(f"wrote figures/fig11_boundmap.pdf; (a) share of the image bounded above {100*np.mean(up[(np.abs(X)<half)&(np.abs(Y)<half)]):.1f}%;"
      f" (b) {name}: region covers {100*upb.mean():.1f}% of the image, {sum(inside)} of {len(jv)} junctions inside")
