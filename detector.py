"""Is the sign test useful as a junction DETECTOR on real data?  The paper
says |S|=0 'rejects about half of random line triples at no cost' but never
measures precision/recall.  YorkUrban gives candidate 3-family clusters, some
genuine corners (pass geometric validation) and some not (fail it)."""
import numpy as np
from yud_path import YUD
import yorkurban as Y
from common import coeffs, admissible
f_true, pp, images = Y.load_packed(YUD)
gen_pass=gen_n=false_pass=false_n=0
gen_ne=false_ne=0
for name,lines,labels,vp in images:
    seg=np.stack([Y.to_centred(lines[:,0],pp),Y.to_centred(lines[:,1],pp)],axis=1)
    vert=Y.pick_vertical_family(seg,labels,vp)
    others=sorted(set(int(x) for x in labels if x>0)-{vert})
    if len(others)<2: continue
    fam=tuple(others[:2])+(vert,)
    for c,fa in Y.find_junctions(seg,labels,10.0,15.0):
        if len(fa)<3: continue
        built=Y.build_corner(seg,fa,fam)
        if built is not None:
            cc,ls=built; gen_n+=1
        else:
            # a false candidate: three families near a point but not a corner.
            # give it the most favourable reading -- lines forced through the
            # cluster centre -- and ask whether the sign test still accepts it
            ls=[]
            for k in fam:
                i,far,near=fa[k]; l=Y.line_of(seg[i])
                if l is None: break
                ls.append(Y.orient(l,c,far))
            if len(ls)<3: continue
            cc=c; false_n+=1
        x=f_true**2
        ok=np.prod([A*x+E for A,E in coeffs(cc,ls)])<0
        ne=bool(admissible(coeffs(cc,ls)))          # K-free test: some focal length admits it
        if built is not None: gen_pass+=ok; gen_ne+=ne
        else: false_pass+=ok; false_ne+=ne
print(f"K-free test (F nonempty): genuine {gen_ne}/{gen_n} = {100*gen_ne/gen_n:5.1f}%,  false {false_ne}/{false_n} = {100*false_ne/false_n:5.1f}%")
print(f"genuine corners : {gen_n:4d}, sign test ACCEPTS {100*gen_pass/gen_n:5.1f}%  (recall)")
print(f"false candidates: {false_n:4d}, sign test ACCEPTS {100*false_pass/false_n:5.1f}%  (false-accept rate)")
