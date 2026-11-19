import numpy as np
from yud_path import YUD
import yorkurban as Y
f_true, pp, images = Y.load_packed(YUD)

class Args:
    def __init__(s,r,ml,it=600): s.radius=r; s.min_len=ml; s.iters=it; s.no_sign=False; s.verbose=False; s.vertical_family=None

def run(args, seed=0):
    rng=np.random.default_rng(seed); recs=[]
    for name,lines,labels,vp in images:
        try: recs.append(Y.eval_image(name,lines,labels,vp,f_true,pp,rng,args))
        except Exception: pass
    return recs

for r,ml in ((10,15),(20,10),(25,10)):
    recs=run(Args(r,ml))
    nf=sum(x["n_full"] for x in recs)
    # paired: images where BOTH loops returned an estimate
    both=[x for x in recs if x["f_lines"] and x["f_junction"]]
    el=np.array([x["err_lines"] for x in both]); ej=np.array([x["err_junction"] for x in both])
    allv=[x for x in recs if x["f_vp3"]]
    ev=np.array([x["err_vp3"] for x in allv])
    bn=sum(x["bound_n"] for x in recs); bc=sum(x["bound_contains"] for x in recs)
    sp=sum(x["sign_pass"] for x in recs)
    mt=sum(x["mirror_pairs"] for x in recs); mo=sum(x["mirror_ok"] for x in recs)
    fr=np.array([x["frame_err_med"] for x in recs if np.isfinite(x.get("frame_err_med",np.nan))])
    print(f"\n=== radius {r}, min_len {ml}: {nf} full junctions, "
          f"{sum(1 for x in recs if x['n_full'])} images with >=1 ===")
    print(f" bound  coverage {100*bc/max(1,bn):.1f}%  sign test {100*sp/max(1,nf):.1f}%  "
          f"mirror {100*mo/max(1,mt):.1f}% of {mt} pairs")
    print(f" closed form vs GT frame: median {np.median(fr):.2f} deg over {len(fr)} images")
    print(f" PAIRED on {len(both)} images:  line-sampled median {100*np.median(el):.1f}%  "
          f"p75 {100*np.percentile(el,75):.1f}%   |  junction-sampled median "
          f"{100*np.median(ej):.1f}%  p75 {100*np.percentile(ej,75):.1f}%")
    print(f"   junction better on {int(np.sum(ej<el))}/{len(both)} images")
    print(f" 3VP (all {len(allv)} images): median {100*np.median(ev):.1f}%  "
          f"p75 {100*np.percentile(ev,75):.1f}%")
