"""Classical calibration from three vanishing points on the York Urban Database (referee #18).
Each family's vanishing point is the least-squares intersection of its labeled segments (lines weighted
by length).  (a) principal point unknown: it is the orthocenter of the three vanishing points, and
f^2 = -(v_i - x0).(v_j - x0);  (b) principal point known: f^2 = -(v_i - p).(v_j - p) for each pair with
both points finite, the median over the pairs."""
import numpy as np, yorkurban as Y, yu_junctions
from yud_path import YUD
f_true,pp,images=Y.load_packed(YUD)
J,_,_,_=yu_junctions.collect(); used=set(j[2] for j in J)
def vp_ls(seg):
    L=[]
    for (x0,y0),(x1,y1) in seg:
        l=np.cross([x0,y0,1.0],[x1,y1,1.0]); n=np.linalg.norm(l[:2])
        if n>0: L.append(l/n*np.hypot(x1-x0,y1-y0))
    if len(L)<2: return None
    _,_,Vt=np.linalg.svd(np.array(L)); return Vt[-1]
res_free=[]; res_known=[]; img_free=[]; img_known=[]; fail_free=fail_known=0; n_img=0
for idx,(name,lines,labels,vp) in enumerate(images):
    seg=np.stack([Y.to_centred(lines[:,0],pp),Y.to_centred(lines[:,1],pp)],axis=1)
    V=[vp_ls(seg[labels==lab]) for lab in (1,2,3)]
    if any(v is None for v in V): continue
    n_img+=1
    fin=[v[:2]/v[2] for v in V if abs(v[2])>1e-12*np.linalg.norm(v)]
    # (b) known principal point (the centered origin)
    fs=[]
    for a in range(3):
        for b in range(a+1,3):
            va,vb=V[a],V[b]
            if abs(va[2])<1e-12 or abs(vb[2])<1e-12: continue
            p,q=va[:2]/va[2],vb[:2]/vb[2]; f2=-(p@q)
            if f2>0: fs.append(np.sqrt(f2))
    if fs: e=abs(np.median(fs)-f_true)/f_true; res_known.append(e); img_known.append(idx)
    else: fail_known+=1
    # (a) orthocenter (needs all three finite)
    if len(fin)==3:
        A,B,C=fin
        M=np.array([B-C,A-C]); rhs=np.array([(B-C)@A,(A-C)@B])
        try:
            x0=np.linalg.solve(M,rhs); f2=-((A-x0)@(B-x0))
            if f2>0: res_free.append(abs(np.sqrt(f2)-f_true)/f_true); img_free.append(idx)
            else: fail_free+=1
        except np.linalg.LinAlgError: fail_free+=1
    else: fail_free+=1
def summ(r,imgs,sub=None):
    r=np.array(r); imgs=np.array(imgs)
    if sub is not None:
        m=np.isin(imgs,list(sub)); r=r[m]
    return f"n={len(r):3d}  median {100*np.median(r):5.1f}%  75th pct {100*np.percentile(r,75):5.1f}%"
print(f"images with all three labeled families: {n_img}")
print(f"3 VPs, principal point unknown (orthocenter): {summ(res_free,img_free)}   failures {fail_free}")
print(f"3 VPs, principal point known:                 {summ(res_known,img_known)}   failures {fail_known}")
print(f"  on the {len(used)} images with full junctions:   orthocenter {summ(res_free,img_free,used)}")
print(f"                                              known pp    {summ(res_known,img_known,used)}")
