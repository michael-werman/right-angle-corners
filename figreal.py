"""Real data: the bound on the York Urban Database.
(a) one image: hand-labelled segments by Manhattan family, detected full
    junctions marked by whether their admissible set contains the true f.
(b) every detected junction's admissible interval, sorted, against the true f."""
import numpy as np, matplotlib
from yud_path import YUD
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle  # noqa: F401
import yorkurban as Y
from common import coeffs, admissible
plt.rcParams.update({"font.size":8,"axes.spines.top":False,"axes.spines.right":False})
f_true,pp,images=Y.load_packed(YUD)
COL={1:"#1b6ca8",2:"#2e7d32",3:"#c8521b"}
records=[]; best=None
for name,lines,labels,vp in images:
    seg=np.stack([Y.to_centred(lines[:,0],pp),Y.to_centred(lines[:,1],pp)],axis=1)
    vert=Y.pick_vertical_family(seg,labels,vp)
    others=sorted(set(int(x) for x in labels if x>0)-{vert})
    if len(others)<2: continue
    fam=tuple(others[:2])+(vert,); js=[]
    for c,fa in Y.find_junctions(seg,labels,10.0,15.0):
        if len(fa)<3: continue
        b=Y.build_corner(seg,fa,fam)
        if b is None: continue
        cc,ls=b; cm=admissible(coeffs(cc,ls))
        if not cm: continue
        lo,hi=np.sqrt(cm[0][0]),np.sqrt(cm[-1][1]) if cm[-1][1]<1e13 else np.inf
        ok=lo<=f_true<=hi
        records.append((lo/f_true,hi/f_true,ok)); js.append((cc,ok))
    if best is None or len(js)>len(best[3]): best=(name,lines,labels,js,vert)
name,lines,labels,js,vert_best=best
fig,ax=plt.subplots(1,2,figsize=(7.0,2.6),gridspec_kw={"width_ratios":[1.15,1]})
for k in range(len(lines)):
    lab=int(labels[k])
    if lab not in COL: continue
    (x0,y0),(x1,y1)=lines[k]
    ax[0].plot([x0,x1],[y0,y1],color=COL[lab],lw=0.8)
for cc,ok in js:
    x=cc[0]+pp[0]; y=pp[1]-cc[1]
    ax[0].plot(x,y,"o",ms=6,mfc="none" if ok else "#d62728",mec="k" if ok else "#d62728",mew=1.0)
ax[0].plot(pp[0],pp[1],"+",color="k",ms=11,mew=1.6)
ax[0].set_xlim(0,640); ax[0].set_ylim(480,0); ax[0].set_aspect("equal")
ax[0].set_xticks([]); ax[0].set_yticks([])
for sp_ in ax[0].spines.values(): sp_.set_visible(True); sp_.set_color("0.7")
hor=[l for l in (1,2,3) if l!=vert_best]
from matplotlib.lines import Line2D
hd=[Line2D([],[],color=COL[vert_best],lw=1.2,label="vertical")]+[Line2D([],[],color=COL[l],lw=1.2,label=f"horizontal {i+1}") for i,l in enumerate(hor)]
hd.append(Line2D([],[],marker="+",color="k",ls="",ms=7,mew=1.4,label="principal point"))
ax[0].legend(handles=hd,fontsize=5.5,loc="lower right",framealpha=0.92,borderpad=0.4,handlelength=1.6)
ax[0].set_title("(a)",fontsize=9,loc="left")
R=sorted(records,key=lambda r:r[0])
xs_=np.arange(len(R)); lo_=np.array([r[0] for r in R]); ok_=np.array([r[2] for r in R])
ax[1].plot(xs_[ok_],lo_[ok_],".",ms=2.2,color="#1b6ca8",label="lower bound")
ax[1].plot(xs_[~ok_],lo_[~ok_],"o",ms=4,mfc="none",mec="#d62728")
nub=0
for i,(lo,hi,ok) in enumerate(R):
    if np.isfinite(hi):
        nub+=1
        ax[1].plot([i,i],[lo,hi],color="#d62728" if not ok else "#1b6ca8",lw=1.0)
        ax[1].plot(i,hi,"v",ms=4,color="#d62728" if not ok else "#1b6ca8")
        ax[1].annotate(f"upper bound {hi:.2f}$f$",(i,hi),xytext=(6,4 if hi>1 else 8),textcoords="offset points",fontsize=6)
ax[1].axhline(1.0,color="k",lw=0.8)
ax[1].text(4,2.75,f"{len(R)-nub} of {len(R)} unbounded above",fontsize=7)
ax[1].set_ylim(0,3.05); ax[1].set_xlim(-2,len(R)+1)
ax[1].set_xlabel("junction, sorted by lower bound")
ax[1].set_ylabel(r"admissible focal length, $/f$")
cov=100*np.mean([r[2] for r in R])
ax[1].set_title("(b)",fontsize=9,loc="left")
fig.tight_layout(); fig.savefig("figures/fig8_real_junctions.pdf")
print(f"wrote figures/fig8_real_junctions.pdf  image={name}  junctions={len(js)}  "
      f"total={len(R)}  coverage={cov:.1f}%  unbounded={100*np.mean([not np.isfinite(r[1]) for r in R]):.1f}%")
