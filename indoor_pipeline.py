"""The junction pipeline for an indoor study (Limitations section), ready to run on a dataset with intrinsics.

usage:  python indoor_pipeline.py IMAGE_DIR INTRINSICS.csv [--eps 0.5] [--out results.csv]
INTRINSICS.csv has columns: image,fx,fy,cx,cy   (fx is the ground truth for the focal length; fx ~ fy assumed)

Steps: (1) detect line segments (OpenCV LSD, with a probabilistic Hough fallback); (2) group segment endpoints
into candidate junctions of three segments whose lines concur at the cluster; (3) the camera-free test of
Theorem 6 (F nonempty for K=diag(f,f,1) with the given principal point); (4) per junction, the admissible
interval F and the certified set F u U of Proposition (certified bounds), with phi_i = arcsin(2 eps / l_i);
(5) per image, the strict intersection and the depth-kappa consensus; (6) a summary of yield, coverage and bounds."""
import sys, csv, argparse, numpy as np, cv2
from common import coeffs, admissible

def detect_segments(gray, min_len=15):
    segs=[]
    if hasattr(cv2,'createLineSegmentDetector'):
        try:
            lsd=cv2.createLineSegmentDetector(); out=lsd.detect(gray)[0]
            if out is not None: segs=[s[0] for s in out]
        except cv2.error: segs=[]
    if not segs:
        edges=cv2.Canny(gray,50,150)
        out=cv2.HoughLinesP(edges,1,np.pi/720,30,minLineLength=min_len,maxLineGap=3)
        segs=[] if out is None else [s[0].astype(float) for s in out]
    return [np.array(s,float) for s in segs if np.hypot(s[2]-s[0],s[3]-s[1])>=min_len]

def junctions(segs, radius=6.0, concur_tol=3.0, min_angle_deg=10.0):
    """Clusters of segment endpoints; triples of segments, with pairwise distinct directions, that concur."""
    ends=[]                                   # (point, segment index, which end)
    for k,s in enumerate(segs): ends+= [(s[:2],k,0),(s[2:],k,1)]
    P=np.array([e[0] for e in ends]); used=set(); out=[]
    for a in range(len(P)):
        if a in used: continue
        near=[b for b in range(len(P)) if np.linalg.norm(P[b]-P[a])<=radius and ends[b][1] not in {ends[a][1]}|set()]
        near=[a]+[b for b in near if b!=a]
        segids={}; 
        for b in near: segids.setdefault(ends[b][1],b)
        if len(segids)<3: continue
        cand=list(segids.items())
        best=None
        for i in range(len(cand)):
            for j in range(i+1,len(cand)):
                for m in range(j+1,len(cand)):
                    trip=[cand[i],cand[j],cand[m]]; lines=[]; dirs=[]
                    for k,b in trip:
                        s=segs[k]; near_pt=s[:2] if ends[b][2]==0 else s[2:]; far=s[2:] if ends[b][2]==0 else s[:2]
                        u=(far-near_pt)/np.linalg.norm(far-near_pt); dirs.append(u)
                        n=np.array([-u[1],u[0]]); lines.append((n,n@near_pt,np.linalg.norm(far-near_pt)))
                    ang=[np.degrees(np.arccos(np.clip(abs(dirs[p]@dirs[q]),0,1))) for p,q in ((0,1),(1,2),(0,2))]
                    if min(ang)<min_angle_deg: continue
                    A=np.array([l[0] for l in lines]); bvec=np.array([l[1] for l in lines])
                    c,res,_,_=np.linalg.lstsq(A,bvec,rcond=None)
                    if np.max(np.abs(A@c-bvec))>concur_tol: continue
                    score=np.max(np.abs(A@c-bvec))
                    if best is None or score<best[0]: best=(score,c,dirs,[l[2] for l in lines])
        if best:
            out.append((best[1],best[2],best[3])); used.update(near)
    return out

def junction_lines(c, dirs, pp):
    """Lines (a,b,e) in coordinates centered on the principal point, each through the vertex along its edge."""
    cc=c-pp; L=[]
    for u in dirs:
        a,b=u[1],-u[0]; e=-(a*cc[0]+b*cc[1]); L.append((a,b,e))
    return cc,L

def certified_set(cc,L,lens,eps,xs):
    rho2=cc@cc; phi=np.arcsin(np.minimum(1.0,2*eps/np.asarray(lens,float)))
    F=np.ones_like(xs); U=np.zeros_like(xs,bool)
    for i,j in ((0,1),(1,2),(2,0)):
        A=L[i][0]*L[j][0]+L[i][1]*L[j][1]; E=L[i][2]*L[j][2]; form=A*xs+E
        F=F*form; U|= np.abs(form)<=np.hypot(phi[i],phi[j])*(np.sqrt(2)*xs+rho2)
    return F<0, (F<0)|U

def depth_hull(intervals, k):
    ev=sorted([(a,1) for a,_ in intervals]+[(b,-1) for _,b in intervals]); cnt=0; lo=hi=None
    for x,d in ev:
        prev=cnt; cnt+=d
        if prev<k<=cnt and lo is None: lo=x
        if prev>=k>cnt: hi=x
    return (lo,hi if hi is not None else np.inf) if lo is not None else None

def run(rows, load_gray, eps=0.5, out=None):
    summary=[]; recs=[]
    for r in rows:
        gray=load_gray(r['image']); f_true=float(r['fx']); pp=np.array([float(r['cx']),float(r['cy'])])
        segs=detect_segments(gray); Js=junctions(segs)
        xs=np.geomspace((0.05*f_true)**2,(20*f_true)**2,20000)
        ivals=[]; ncert=0; npass=0
        for c,dirs,lens in Js:
            cc,L=junction_lines(c,dirs,pp); comps=admissible(coeffs(cc,L))
            if not comps: continue
            npass+=1; lo,hi=np.sqrt(comps[0][0]),(np.sqrt(comps[-1][1]) if comps[-1][1]<1e13 else np.inf)
            Fh,C=certified_set(cc,L,lens,eps,xs)
            inC=bool(C[np.argmin(np.abs(xs-f_true**2))]); ncert+=inC
            ivals.append((lo,hi)); recs.append(dict(image=r['image'],lo=lo,hi=hi,covered=lo<=f_true<=hi,certified=inC))
        strict=(max(a for a,_ in ivals),min(b for _,b in ivals)) if ivals else None
        cons=depth_hull(ivals,max(1,int(np.ceil(0.9*len(ivals))))) if ivals else None
        summary.append(dict(image=r['image'],segments=len(segs),junctions=len(Js),passed=npass,certified_cover=ncert,
                            strict=strict,consensus=cons,f=f_true))
    if out:
        with open(out,'w',newline='') as fh:
            w=csv.DictWriter(fh,fieldnames=list(recs[0].keys()) if recs else ['image']); w.writeheader(); [w.writerow(x) for x in recs]
    n=len(summary); nj=sum(s['junctions'] for s in summary); npass=sum(s['passed'] for s in summary)
    cov=np.mean([x['covered'] for x in recs]) if recs else float('nan'); cert=np.mean([x['certified'] for x in recs]) if recs else float('nan')
    noj=sum(1 for s in summary if s['passed']==0)
    print(f"images {n}; junctions found {nj} ({nj/max(n,1):.1f} per image), passing the camera-free test {npass}; images with none {noj}")
    print(f"true f in the measured interval: {100*cov:.1f}%; in the certified set (eps={eps} px): {100*cert:.1f}%")
    return summary,recs

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("image_dir"); ap.add_argument("intrinsics"); ap.add_argument("--eps",type=float,default=0.5); ap.add_argument("--out",default="indoor_results.csv")
    a=ap.parse_args(); rows=list(csv.DictReader(open(a.intrinsics)))
    run(rows, lambda name: cv2.imread(f"{a.image_dir}/{name}",cv2.IMREAD_GRAYSCALE), a.eps, a.out)
