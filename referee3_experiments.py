"""Experiments for the third referee report: Table 2 statistics over all trials (#24) and robustness
ablations of the single-junction bound (#23)."""
import numpy as np, math
from common import observe_corner, admissible, coeffs, rand_frame
from experiments import place
rng=np.random.default_rng(77); f=866.0
def one(sigma,frame=None,seglen=150.0,off=(0.0,0.0)):
    while True:
        D=rand_frame(rng) if frame is None else frame()
        o=observe_corner(place(f,rng),D,f,sigma,rng,seglen=seglen)
        if o is None: continue
        c,L=o
        if off!=(0.0,0.0):     # principal point assumed off by `off` pixels
            c=np.array([c[0]-off[0],c[1]-off[1]]); L=[(l[0],l[1],l[2]+l[0]*off[0]+l[1]*off[1]) for l in L]
        comps=admissible(coeffs(c,L))
        if not comps: return None
        lo,hi=comps[0][0],comps[-1][1]
        return np.sqrt(lo)/f, (np.sqrt(hi)/f if hi<1e13 else np.inf)
def depth_set(iv,k):
    """the set covered by at least k intervals, as a list of intervals (not its hull)"""
    ev=sorted([(a,1) for a,_ in iv]+[(b,-1) for _,b in iv]); cnt=0; out=[]; start=None
    for x,d in ev:
        prev=cnt; cnt+=d
        if prev<k<=cnt: start=x
        if prev>=k>cnt and start is not None: out.append((start,x)); start=None
    if start is not None: out.append((start,np.inf))
    return out
print("=== Table 2 over all trials (60 deg, 400 trials; kappa = ceil(0.9 M); depth set, not its hull)")
print(f"{'sig':>4}{'M':>4}{'cov(int)':>9}{'cov(depth)':>11}{'cov(hull)':>10}{'binom.pred':>11}{'med lo/f':>9}{'%bnd':>6}{'med hi/f':>9}")
for sigma in (0.5,1.0):
    single=[]; 
    for _ in range(3000):
        b=one(sigma)
        if b: single.append(b[0]<=1.0<=b[1])
    p=1-np.mean(single)
    for M in (2,5,10,25,50):
        k=max(1,math.ceil(0.9*M)); ci=cd=ch=0; los=[]; his=[]; bnd=0
        for _ in range(400):
            iv=[]
            while len(iv)<M:
                b=one(sigma)
                if b: iv.append(b)
            lo=max(a for a,_ in iv); hi=min(b for _,b in iv); ci+=lo<=1.0<=hi
            ds=depth_set(iv,k)
            if ds:
                cd+=any(a<=1.0<=b for a,b in ds); ch+=ds[0][0]<=1.0<=ds[-1][1]
                los.append(ds[0][0]); his.append(ds[-1][1]); bnd+=np.isfinite(ds[-1][1])
        pred=sum(math.comb(M,j)*p**j*(1-p)**(M-j) for j in range(0,M-k+1))
        mh=np.median(his) if his else np.nan
        print(f"{sigma:>4}{M:>4}{100*ci/400:>8.0f}%{100*cd/400:>10.0f}%{100*ch/400:>9.0f}%{100*pred:>10.0f}%{np.median(los):>9.2f}{100*bnd/400:>5.0f}%{(f'{mh:.2f}' if np.isfinite(mh) else 'inf'):>9}",flush=True)
print(f"   per-junction miss rate used for the prediction: sigma=0.5 and 1 from 3000 single junctions each")
# ---- #23 ablations of the single-junction bound, 60 deg, sigma = 0.5 px
def stats(**kw):
    cov=[];bnd=[];fm=[]
    for _ in range(2000):
        b=one(0.5,**kw)
        if b is None: continue
        cov.append(b[0]<=1<=b[1]); bnd.append(np.isfinite(b[1])); fm.append(b[0])
    return 100*np.mean(cov),100*np.mean(bnd),np.median(fm)
print("\n=== single-junction bound, 60 deg, sigma = 0.5 px: coverage, share bounded above, median f_min/f")
print("orientations uniform:                     cov %.1f%%  bounded above %.1f%%  f_min/f %.2f"%stats())
york_tilts=np.array([1.4,3.2,6.1,0.6,2.1,4.5,8.6,1.0,2.6,5.2])     # York Urban quartiles and spread (median 3.2 deg)
def york_frame():
    t=np.radians(rng.choice(york_tilts)*rng.choice([-1,1])); y=rng.uniform(0,2*np.pi); r=np.radians(rng.normal(0,2))
    Rz=np.array([[np.cos(r),-np.sin(r),0],[np.sin(r),np.cos(r),0],[0,0,1]])
    Rx=np.array([[1,0,0],[0,np.cos(t),-np.sin(t)],[0,np.sin(t),np.cos(t)]])
    Ry=np.array([[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]])
    return Rz@Rx@Ry
print("vertical tilted as in York Urban:          cov %.1f%%  bounded above %.1f%%  f_min/f %.2f"%stats(frame=york_frame))
for L in (20,40,80,150): print(f"edge length {L:>3} px:                       "+"cov %.1f%%  bounded above %.1f%%  f_min/f %.2f"%stats(seglen=float(L)))
for o in (5,10,20,40): print(f"principal point off by {o:>2} px:              "+"cov %.1f%%  bounded above %.1f%%  f_min/f %.2f"%stats(off=(o/np.sqrt(2),o/np.sqrt(2))))
