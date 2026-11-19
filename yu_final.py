import numpy as np, sys
import yu_junctions
from robust_calib import fit, params, S0
from robust_sim import scene, F_TRUE, PP_TRUE
from common import coeffs, admissible
J,F,PP,_=yu_junctions.collect()
print("ROBUST CONVEX PROGRAM ON YORK URBAN (principal point not given; true f = %.1f, pp = (%.1f, %.1f))"%(F,PP[0],PP[1]))
for model in ('f','aspect','full'):
    try:
        r=fit(model,J,ranges=True); p=params(model,r)
        rng_txt=""
        if model=='f': rng_txt=f"   near-optimal range f in [{np.sqrt(r['lo'][0])*S0:.0f}, {np.sqrt(r['hi'][0])*S0:.0f}]"
        if model=='aspect': rng_txt=f"   ranges fx [{np.sqrt(r['lo'][0])*S0:.0f}, {np.sqrt(r['hi'][0])*S0:.0f}]  fy [{np.sqrt(r['lo'][1])*S0:.0f}, {np.sqrt(r['hi'][1])*S0:.0f}]"
        print(f"  {model:7s}: fx={p['fx']:7.1f} fy={p['fy']:7.1f} skew={p['skew']:6.1f} pp=({p['px']:6.1f},{p['py']:6.1f})  violation {r['V']:.3f}{rng_txt}")
    except Exception as e: print(f"  {model:7s}: failed ({type(e).__name__}: {e})")
print("  (parameter box limit is f = %.0f px: a range ending there is unbounded)"%(np.sqrt(400.0)*S0))
print("\nIS A TIGHT LOWER BOUND EXPECTED?  noiseless level scenes shaped like York Urban, 181 junctions:")
rng=np.random.default_rng(5); K0=np.array([[F_TRUE,0,0],[0,F_TRUE,0],[0,0,1.0]]); mx=[];ub=[]
for _ in range(40):
    Js=scene(rng,0.0,K0,True); lo=[];nb=0
    for L,c,_,_ in Js:
        cm=admissible(coeffs(c[:2],L))
        if cm:
            lo.append(np.sqrt(cm[0][0])/F_TRUE); nb+=cm[-1][1]<1e13
    mx.append(max(lo)); ub.append(nb/len(lo))
print(f"  largest lower bound / f: median {np.median(mx):.3f}, 10th-90th pct {np.percentile(mx,10):.3f}-{np.percentile(mx,90):.3f}")
print(f"  share of junctions bounded above: {100*np.mean(ub):.1f}%  (York Urban: 1.1%)")
