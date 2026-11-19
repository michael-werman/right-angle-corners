"""Figure 3: the admissible intervals F of Theorem 6 for ten junctions, and their running intersection.
60-degree field of view, 1000 x 1000 px image, noiseless junctions, each from its own random frame."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import figstyle
from common import coeffs, admissible, rand_frame
rng=np.random.default_rng(31); f=866.0; K=np.diag([f,f,1.0]); N=10
def junction_interval():
    while True:
        D=rand_frame(rng); P=np.array([rng.uniform(-2.5,2.5),rng.uniform(-2.5,2.5),rng.uniform(2,6)])
        c=(K@P)[:2]/P[2]
        if np.max(np.abs(c))>480: continue
        L=[]
        for k in range(3):
            v=K@D[:,k]; l=np.cross(np.array([c[0],c[1],1.0]),v); L.append(tuple(l/np.hypot(l[0],l[1])))
        comps=admissible(coeffs(c,L))
        if not comps: continue
        lo=np.sqrt(comps[0][0])/f; hi=np.sqrt(comps[-1][1])/f if comps[-1][1]<1e13 else np.inf
        if lo<=1<=hi: return lo,hi
J=[junction_interval() for _ in range(N)]
XMIN,XMAX=0.05,20
fig,ax=plt.subplots(figsize=(3.4,2.3))
run_lo,run_hi=0,np.inf
for j,(lo,hi) in enumerate(J,1):
    y=-j
    ax.barh(y,min(hi,XMAX)-lo,left=lo,height=0.62,color="#bcd3ec",edgecolor="none",zorder=1)
    if hi==np.inf: ax.plot(XMAX*0.93,y,">",color="#7fa6cf",ms=3.2,zorder=2)
    run_lo,run_hi=max(run_lo,lo),min(run_hi,hi)
    ax.barh(y,min(run_hi,XMAX)-run_lo,left=run_lo,height=0.24,color="#c8521b",edgecolor="none",zorder=3)
ax.axvline(1.0,color="0.15",ls=(0,(4,3)),lw=0.8,zorder=4)
ax.set_xscale("log"); ax.set_xlim(XMIN,XMAX); ax.set_ylim(-N-0.8,-0.2)
ax.set_yticks([-1,-5,-10]); ax.set_yticklabels(["1","5","10"])
ax.set_ylabel("junction"); ax.set_xlabel(r"admissible focal length, $f'/f$")
ax.set_xticks([0.1,0.3,1,3,10]); ax.set_xticklabels(["0.1","0.3","1","3","10"])
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#bcd3ec",label="interval $F$ of one junction"),Patch(color="#c8521b",label="intersection so far")],
          frameon=False,fontsize=6.8,loc="lower center",bbox_to_anchor=(0.5,1.0),ncol=2,handlelength=1.4,columnspacing=1.2)
for sp in ("top","right"): ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig("figures/fig3_intervals.pdf")
print(f"wrote figures/fig3_intervals.pdf; final intersection [{run_lo:.3f}, {run_hi:.3f}] f; unbounded above: {sum(h==np.inf for _,h in J)} of {N}")
