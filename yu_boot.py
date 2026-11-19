import numpy as np, sys, json
import yu_junctions
from robust_calib import fit, params, S0
from common import coeffs, admissible
J,F,PP,_=yu_junctions.collect()
# which image holds the junction with the erroneous upper bound?
bad=[j[2] for j in J if (lambda cm: cm and cm[-1][1]<1e13 and np.sqrt(cm[-1][1])<F)(admissible(coeffs(j[1][:2],j[0])))]
print("image(s) holding a junction whose upper bound is BELOW the true f:", bad)
by={}
for j in J: by.setdefault(j[2],[]).append(j)
keys=list(by); rng=np.random.default_rng(0); withb=[]; without=[]
for b in range(120):
    pick=rng.choice(len(keys),len(keys)); S=[j for k in pick for j in by[keys[k]]]
    r=fit('f',S,ranges=True)
    f_est=params('f',r)['fx']; hi=np.sqrt(r['hi'][0])*S0
    (withb if any(keys[k] in bad for k in pick) else without).append((f_est,hi))
withb=np.array(withb); without=np.array(without)
json.dump(dict(withb=withb.tolist(),without=without.tolist()),open('yu_boot.json','w'))
for label,a in (("resamples that INCLUDE that image",withb),("resamples that EXCLUDE it",without)):
    print(f"  {label}: {len(a)}   median f {np.median(a[:,0]):7.0f} ({100*(np.median(a[:,0])-F)/F:+.0f}%)   "
          f"68% [{np.percentile(a[:,0],16):.0f}, {np.percentile(a[:,0],84):.0f}]   upper end of range: median {np.median(a[:,1]):.0f}")
