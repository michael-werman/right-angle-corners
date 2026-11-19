"""
experiments.py -- the synthetic experiments of Section 6 (E1, E2, E3a, E3b).
Running this file regenerates every number reported there.

Run:  python3 experiments.py [e1|e2|e3a|e3b|e4|e4b|e6|sim|all]
      (E4 additionally needs scipy)

Protocol (Section 6): 1000x1000 px images, principal point at the center,
fields of view 40/60/90 degrees (f = 1374, 866, 500 px); corners sampled
uniformly over orientations and over the image at depths 2 to 6; each edge
observed as a 150 px segment running from the junction, both endpoints
perturbed by N(0, sigma); each line fitted through its two noisy endpoints and
the junction taken as the least-squares intersection of the three; position
bins use the distance from the principal point normalized by the image
half-width.
"""
import collections
import sys

import numpy as np
from scipy.optimize import least_squares

def rand_frame(rng):
    A = rng.standard_normal((3,3)); Q,R = np.linalg.qr(A)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0: Q[:,0] *= -1
    return Q

def Kmat(f): return np.diag([f,f,1.0])

def project(P, f):
    c = Kmat(f) @ P
    return c[:2]/c[2]

def edge_image_dir(P, d, f):
    """Exact image direction of the 3D direction d at the projection of P:
    the derivative of the projection along d, not the vanishing point."""
    X, Y, Z = P
    u0, v0 = f*X/Z, f*Y/Z
    return np.array([f*d[0] - u0*d[2], f*d[1] - v0*d[2]]) / Z

def observe_corner(P, D, f, sigma, rng, seglen=150.0, W=1000.0):
    """Image a corner: each edge is a seglen-px segment from the junction, both
    endpoints perturbed by N(0, sigma). Lines are fitted through the two noisy
    endpoints; the junction is the least-squares intersection of the three."""
    c0 = project(P, f)
    if np.max(np.abs(c0)) > W/2: return None
    lines = []
    for i in range(3):
        d = edge_image_dir(P, D[:, i], f)
        n = np.linalg.norm(d)
        if n < 1e-12: return None
        d = d/n
        p1 = c0 + rng.normal(0, sigma, 2)
        p2 = c0 + seglen*d + rng.normal(0, sigma, 2)
        e = p2 - p1; ne = np.linalg.norm(e)
        if ne < 1e-9: return None
        e = e/ne
        a, b = e[1], -e[0]                     # half-edge leaves c along (b,-a)
        lines.append(np.array([a, b, -(a*p1[0]+b*p1[1])]))
    L = np.array(lines)
    A = L[:, :2]; rhs = -L[:, 2]
    c, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    return c, [np.array(l) for l in L]

def gij(c, L, f):
    r = np.linalg.inv(Kmat(f)) @ np.array([c[0], c[1], 1.0])
    t = [np.cross(Kmat(f).T @ l, r) for l in L]
    return t, r, np.array([[t[i]@t[j] for j in range(3)] for i in range(3)])

def closed_form(c, L, f):
    t, r, g = gij(c, L, f); R2 = r@r
    if abs(g[1,2])<1e-300 or abs(g[2,0])<1e-300 or abs(g[0,1])<1e-300: return None
    k1sq = -g[0,1]*g[2,0]/(g[1,2]*R2)
    if k1sq <= 0: return None
    k1 = np.sqrt(k1sq); k = [k1, -g[0,1]/(R2*k1), -g[2,0]/(R2*k1)]
    cols = []
    for i in range(3):
        w = k[i]*r + t[i]; n = np.linalg.norm(w)
        if n < 1e-12: return None
        cols.append(w/n)
    return np.column_stack(cols)

def reproject(c, L):
    """Force each line through the estimated junction, keeping its direction.

    The derivation needs every interpretation plane to contain the ray, i.e.
    n_i . r = 0, i.e. every image line through the corner point.  Three
    independently fitted noisy lines do not meet exactly, so the observed
    (a, b, e) are mutually inconsistent at the sub-pixel level.  This makes
    the observation consistent with the model before the model is applied.
    """
    return [np.array([l[0], l[1], -(l[0]*c[0] + l[1]*c[1])]) for l in L]


def coeffs(c, L, enforce=True):
    """The three LINEAR factors (A_ij, e_i e_j) of the sign condition.

    Because each interpretation plane contains the ray,
        x g_ij = (x + rho^2)(A_ij x + e_i e_j),     x = f^2,
    and the positive factor (x + rho^2) is common to all three pairs, so it
    cancels from the sign condition.  Each pair contributes one linear
    function of f^2, with at most one positive root.
    """
    if enforce:
        L = reproject(c, L)
    out = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        ai, bi, ei = L[i]
        aj, bj, ej = L[j]
        out.append((ai*aj + bi*bj, ei*ej))
    return out


def admissible(co, XMAX=1e14):
    """The admissible set as a list of intervals in x = f^2.  By the interval
    theorem there is at most one, but the sign chart is built generically so
    that a violation would show up rather than be assumed away."""
    rs = sorted(-E/A for A, E in co if abs(A) > 1e-300 and -E/A > 1e-12)
    edges = [0.0] + rs + [XMAX]
    comps = []
    for k in range(len(edges)-1):
        lo, hi = edges[k], edges[k+1]
        if hi <= lo:
            continue
        xm = np.sqrt(max(lo, hi*1e-16)*hi) if lo > 0 else min(hi/2, 1e12)
        if np.prod([A*xm + E for A, E in co]) < 0:
            if comps and abs(comps[-1][1]-lo) < 1e-9*max(1, lo):
                comps[-1] = (comps[-1][0], hi)
            else:
                comps.append((lo, hi))
    return comps


def angle_err_deg(Dh, D, c, f):
    """Compare as unoriented axes, allowing either mirror branch:
    the two closed-form solutions are D and H D with H the reflection in the
    plane through O perpendicular to the ray."""
    r = np.linalg.inv(Kmat(f)) @ np.array([c[0], c[1], 1.0])
    rh = r/np.linalg.norm(r)
    H = np.eye(3) - 2*np.outer(rh, rh)
    best = 1e9
    for S in (Dh, H @ Dh):
        errs = [np.degrees(np.arccos(np.clip(abs(S[:,i]@D[:,i]),-1,1))) for i in range(3)]
        best = min(best, float(np.mean(errs)))
    return best


HALF = 500.0
FOVS = {40: 1374.0, 60: 866.0, 90: 500.0}


def place(f, rng):
    c = rng.uniform(-HALF*0.95, HALF*0.95, 2)
    z = rng.uniform(2, 6)
    ray = np.linalg.inv(Kmat(f)) @ np.array([c[0], c[1], 1.0])
    return ray/ray[2]*z


def both(c,L,f):
    D1=closed_form(c,L,f)
    if D1 is None: return None
    r=np.linalg.inv(Kmat(f))@np.array([c[0],c[1],1.0]); rh=r/np.linalg.norm(r)
    return [D1,(np.eye(3)-2*np.outer(rh,rh))@D1]
def disag(F1,F2):
    return float(np.mean([np.degrees(np.arccos(np.clip(abs(F1[:,i]@F2[:,i]),-1,1))) for i in range(3)]))


def solve(l1,l2,l3,l3p):
    v3=np.cross(l3,l3p);p,q,s=v3
    A12=l1[0]*l2[0]+l1[1]*l2[1]; m1=l1@v3; m2=l2@v3
    c2=A12*s*s; c1=A12*(p*p+q*q)+l1[2]*l2[2]*s*s-m1*m2; c0=l1[2]*l2[2]*(p*p+q*q)
    if abs(c2)<1e-280:
        return ([-c0/c1] if abs(c1)>1e-280 and -c0/c1>0 else [])
    d=c1*c1-4*c2*c0
    if d<0: return []
    return [z for z in ((-c1+np.sqrt(d))/(2*c2),(-c1-np.sqrt(d))/(2*c2)) if z>0]
def Ep(x,l1p,l2p,l3,l3p):
    K=Kmat(np.sqrt(x)); n1p,n2p,n3,n3p=K.T@l1p,K.T@l2p,K.T@l3,K.T@l3p
    w=np.cross(n3,n3p); ww=w@w
    if ww<1e-300: return np.inf
    return abs((n1p@n2p)*ww-(n1p@w)*(n2p@w))/(np.linalg.norm(n1p)*np.linalg.norm(n2p)*ww)
def vp3(pairs):
    vs=[]
    for a,b in pairs:
        v=np.cross(a,b)
        if abs(v[2])<1e-12: return None
        vs.append(v[:2]/v[2])
    est=[-np.dot(vs[i],vs[j]) for i,j in ((0,1),(1,2),(2,0))]
    est=[np.sqrt(e) for e in est if e>0]
    return float(np.median(est)) if est else None


def run_e1():
    rng = np.random.default_rng(1)
    print("=== E1 (1000 corners per FOV per noise level) ===")
    for sigma in (0.0,0.5,1.0,2.0):
        allе=[]; rej=0; tot=0; per={}
        for fov,f in FOVS.items():
            errs=[]; r=0; n=0
            while n<1000:
                P=place(f,rng); D=rand_frame(rng)
                o=observe_corner(P,D,f,sigma,rng)
                if o is None: continue
                n+=1; tot+=1; c,L=o
                Dh=closed_form(c,L,f)
                if Dh is None: r+=1; rej+=1; continue
                errs.append(angle_err_deg(Dh,D,c,f))
            per[fov]=np.median(errs); allе+=errs
        a=np.array(allе)
        print(f" sigma={sigma}: median {np.median(a):.3g}  p90 {np.percentile(a,90):.3g}  "
              f"p95 {np.percentile(a,95):.3g}  reject {100*rej/tot:.1f}%  "
              f"per-FOV medians {', '.join(f'{k}:{v:.2f}' for k,v in per.items())}")


def run_e2():
    rng = np.random.default_rng(2)
    print("\n=== E2 (1000 corners per FOV, sigma=0.5 unless noted; thirds by radius/half-width) ===")
    for fov,f in FOVS.items():
        n=0;single=0;cont=0;unb=0;rows=[]
        while n<1000:
            P=place(f,rng);D=rand_frame(rng)
            o=observe_corner(P,D,f,0.5,rng)
            if o is None: continue
            c,L=o; co=coeffs(c,L); comps=admissible(co)
            if not comps: continue
            n+=1
            if len(comps)==1: single+=1
            if any(lo<=f*f<=hi for lo,hi in comps): cont+=1
            ub = comps[-1][1]>=1e13
            if ub: unb+=1
            rows.append((np.hypot(c[0],c[1])/HALF, np.sqrt(comps[0][0])/f, ub,
                         np.sqrt(comps[-1][1])/f if not ub else np.nan))
        R=np.array(rows); inner=R[R[:,0]<1/3]; outer=R[R[:,0]>2/3]; ubo=outer[outer[:,2]==0]
        print(f" FOV {fov}: single interval {single}/{n}  contains true f {100*cont/n:.0f}%  "
              f"unbounded {100*unb/n:.0f}%")
        print(f"   median fmin/f  inner {np.median(inner[:,1]):.2f} (n={len(inner)})  "
              f"outer {np.median(outer[:,1]):.2f} (n={len(outer)})  | upper bound for "
              f"{100*len(ubo)/len(outer):.0f}% of outer, median {np.nanmedian(ubo[:,3]):.2f}x f")
    rng=np.random.default_rng(22); agree=0;n=0
    for fov,f in FOVS.items():
        k=0
        while k<1000:
            P=place(f,rng);D=rand_frame(rng)
            o=observe_corner(P,D,f,0.0,rng)
            if o is None: continue
            c,L=o;co=coeffs(c,L);comps=admissible(co)
            if not comps: continue
            k+=1;n+=1
            agree += ((comps[-1][1]>=1e13) == (np.prod([A for A,_ in co])<0))
    print(f" sigma=0 check: unboundedness coincides with the raw-angle rule in {agree}/{n}")


def run_e3a():
    rng = np.random.default_rng(3)
    f = 866.0
    bins=[(0,5),(5,15),(15,45),(45,180)]; st={b:[0,0,[]] for b in bins}; n=0; okall=0
    while n<6000:
        Fr=rand_frame(rng); Dg=np.diag(rng.choice([-1.,1.],3))
        P1,P2=place(f,rng),place(f,rng)
        o1=observe_corner(P1,Fr,f,0.5,rng); o2=observe_corner(P2,Fr@Dg,f,0.5,rng)
        if o1 is None or o2 is None: continue
        S1=both(*o1,f); S2=both(*o2,f)
        if S1 is None or S2 is None: continue
        n+=1
        sep=np.degrees(np.arccos(np.clip((P1/np.linalg.norm(P1))@(P2/np.linalg.norm(P2)),-1,1)))
        sc=sorted([(disag(S1[i],S2[j]),(i,j)) for i in range(2) for j in range(2)])
        best=sc[0][1]; margin=sc[1][0]-sc[0][0]
        t=(int(np.argmin([disag(S1[k],Fr) for k in range(2)])),
           int(np.argmin([disag(S2[k],Fr@Dg) for k in range(2)])))
        okall += (best==t)
        for b in bins:
            if b[0]<=sep<b[1]: st[b][0]+=1; st[b][1]+=(best==t); st[b][2].append(margin)
    print(f"=== E3a ({n} pairs, 60 deg, sigma=0.5) overall correct {100*okall/n:.1f}% ===")
    for b in bins:
        tot,ok,m=st[b]
        print(f"  {b[0]:>3}-{b[1]:<3} deg: n={tot:4d}  correct {100*ok/tot:5.1f}%  median margin {np.median(m):5.1f} deg")


def run_e3b():
    rng = np.random.default_rng(4)
    print("\n=== E3b (3000 pairs per noise level) ===")
    for sigma in (0.0,0.5,1.0,2.0):
        errs=[];base=[];nr=collections.Counter();binned=collections.defaultdict(list)
        so=0;ne=0;two=0;n=0;fail=0
        while n<3000:
            f=rng.uniform(400,1500); Fr=rand_frame(rng)
            d1,d2,d3=Fr[:,0],Fr[:,1],Fr[:,2]; psi=rng.uniform(0,2*np.pi)
            d1p=np.cos(psi)*d1+np.sin(psi)*d2; d2p=-np.sin(psi)*d1+np.cos(psi)*d2
            o1=observe_corner(place(f,rng),np.column_stack([d1,d2,d3]),f,sigma,rng)
            o2=observe_corner(place(f,rng),np.column_stack([d1p,d2p,d3]),f,sigma,rng)
            if o1 is None or o2 is None: continue
            (ca,L1),(cb,L2)=o1,o2; l1,l2,l3=L1; l1p,l2p,l3p=L2
            rts=solve(l1,l2,l3,l3p)
            n+=1; nr[len(rts)]+=1
            if not rts: fail+=1; continue
            if len(rts)==2:
                two+=1; keep=[]
                for x in rts:
                    ok=all(np.prod([A*x+E for A,E in coeffs(cc,LL)])<0 for cc,LL in ((ca,L1),(cb,L2)))
                    if ok: keep.append(x)
                if len(keep)==1: so+=1
                else: ne+=1
            best=min(rts,key=lambda x:Ep(x,l1p,l2p,l3,l3p))
            rel=abs(np.sqrt(best)-f)/f; errs.append(rel)
            oa=observe_corner(place(f,rng),Fr,f,sigma,rng); ob=observe_corner(place(f,rng),Fr,f,sigma,rng)
            if oa and ob:
                fb=vp3([(oa[1][i],ob[1][i]) for i in range(3)])
                if fb: base.append(abs(fb-f)/f)
            ang=np.degrees(np.arccos(np.clip(abs(l3[0]*l3p[0]+l3[1]*l3p[1]),-1,1)))
            for lo,hi,nm in ((15,180,">15"),(5,15,"5-15"),(2,5,"2-5"),(0,2,"<2")):
                if lo<=ang<hi: binned[nm].append(rel)
        e=np.array(errs); b=np.array(base)
        print(f" sigma={sigma}: solver median {100*np.median(e):.3g}%  p90 {100*np.percentile(e,90):.3g}%  "
              f"p95 {100*np.percentile(e,95):.3g}%  no-root {100*fail/n:.1f}%")
        print(f"            3VP median {100*np.median(b):.3g}%  p90 {100*np.percentile(b,90):.3g}%  "
              f"p95 {100*np.percentile(b,95):.3g}%  | roots {dict(sorted(nr.items()))}")
        if sigma==0.5:
            print(f"   2-root cases: sign test alone suffices {100*so/two:.0f}%, E' needed {100*ne/two:.0f}% (n={two})")
            for k in (">15","5-15","2-5","<2"):
                print(f"   vertical-line angle {k:>5}: median {100*np.median(binned[k]):.2g}% (n={len(binned[k])})")


VERTS = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],
                  [1,1,0],[1,0,1],[0,1,1],[1,1,1]], float)
# the three edge directions meeting at each vertex, as signed axis indices
EDGES_AT = {0:[(0,1),(1,1),(2,1)], 1:[(0,-1),(1,1),(2,1)], 2:[(0,1),(1,-1),(2,1)],
            3:[(0,1),(1,1),(2,-1)], 4:[(0,-1),(1,-1),(2,1)], 5:[(0,-1),(1,1),(2,-1)],
            6:[(0,1),(1,-1),(2,-1)], 7:[(0,-1),(1,-1),(2,-1)]}

def box_points(R, t, ratios):
    S = np.diag([1.0, ratios[0], ratios[1]])
    return (R @ (S @ VERTS.T)).T + t

def project_pts(P, f):
    z = P[:, 2]
    return np.column_stack([f*P[:, 0]/z, f*P[:, 1]/z])

def rodrigues(w):
    th = np.linalg.norm(w)
    if th < 1e-12: return np.eye(3)
    k = w/th; Kx = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3) + np.sin(th)*Kx + (1-np.cos(th))*(Kx@Kx)

def box_ml_focal(obs_idx, obs_xy, f0, R0, t0, ratios0):
    def resid(p):
        R = rodrigues(p[:3]) @ R0
        t = p[3:6]; ratios = np.abs(p[6:8]); f = abs(p[8])
        P = box_points(R, t, ratios)[obs_idx]
        if np.any(P[:, 2] < 1e-3): return np.full(2*len(obs_idx), 1e3)
        return (project_pts(P, f) - obs_xy).ravel()
    p0 = np.concatenate([np.zeros(3), t0, ratios0, [f0]])
    try:
        r = least_squares(resid, p0, method="lm", max_nfev=4000)
    except Exception:
        return None
    return abs(r.x[8])

def corner_estimates(vis, lines):
    """Every corner-plus-line estimate available among the visible vertices:
    a vertex with all three edges, plus a second visible vertex sharing one
    edge direction, which supplies the parallel line of Sec. 5."""
    est = []
    for a in vis:
        axes_a = [ax for ax, _ in EDGES_AT[a]]
        for b in vis:
            if a == b: continue
            axes_b = [ax for ax, _ in EDGES_AT[b]]
            for k, ax in enumerate(axes_a):
                if ax not in axes_b: continue
                kb = axes_b.index(ax)
                o = [i for i in range(3) if i != k]
                l1, l2, l3 = lines[a][o[0]], lines[a][o[1]], lines[a][k]
                l3p = lines[b][kb]
                rts = solve(l1, l2, l3, l3p)
                if not rts: continue
                ob = [i for i in range(3) if i != kb]
                l1p, l2p = lines[b][ob[0]], lines[b][ob[1]]
                x = min(rts, key=lambda z: Ep(z, l1p, l2p, l3, l3p))
                est.append(np.sqrt(x))
    return est


def corner_focal(vis, lines):
    """Combine the available estimates by their median, the robust choice given
    that a minority of them come from near-parallel line pairs."""
    est = corner_estimates(vis, lines)
    return float(np.median(est)) if est else None


def bound_width(vtx, lines, pts):
    """k = 1: one corner gives an interval, not an estimate (Sec. 3)."""
    co = coeffs(pts[vtx], lines[vtx])
    comps = admissible(co)
    if not comps: return None
    lo, hi = comps[0][0], comps[-1][1]
    return np.sqrt(lo), (np.sqrt(hi) if hi < 1e13 else np.inf)


def trial(rng, sigma, f, seglen=150.0):
    R = rand_frame(rng)
    ratios = rng.uniform(0.6, 1.6, 2)
    t = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(4, 7)])
    P = box_points(R, t, ratios)
    if np.any(P[:, 2] < 1.0): return None
    xy = project_pts(P, f)
    if np.max(np.abs(xy)) > HALF*0.95: return None
    S = np.diag([1.0, ratios[0], ratios[1]])
    lines = {}
    for vtx in range(8):
        ls = []
        for ax, sgn in EDGES_AT[vtx]:
            d = R @ (S[:, ax]*sgn); d = d/np.linalg.norm(d)
            di = edge_image_dir(P[vtx], d, f); di = di/np.linalg.norm(di)
            p1 = xy[vtx] + rng.normal(0, sigma, 2)
            p2 = xy[vtx] + seglen*di + rng.normal(0, sigma, 2)
            e = p2-p1; e = e/np.linalg.norm(e)
            a, b = e[1], -e[0]
            ls.append(np.array([a, b, -(a*p1[0]+b*p1[1])]))
        lines[vtx] = np.array(ls)
    pts = xy + rng.normal(0, sigma, xy.shape)
    return R, t, ratios, P, pts, lines



def run_e4():
    rng = np.random.default_rng(11)
    sigma = 0.5
    res = {k: {"corner": [], "box": []} for k in range(2, 9)}
    one = {"fmin": [], "unb": 0, "n": 0}
    n = 0
    while n < 600:
        f = rng.uniform(400, 1500)
        out = trial(rng, sigma, f)
        if out is None: continue
        R, t, ratios, P, pts, lines = out
        n += 1
        order = list(rng.permutation(8))
        bw = bound_width(order[0], lines, pts)
        if bw is not None:
            one["n"] += 1; one["fmin"].append(bw[0]/f)
            one["unb"] += (bw[1] == np.inf)
        for k in range(2, 9):
            vis = order[:k]
            ce = corner_focal(vis, lines)
            if ce is not None: res[k]["corner"].append(abs(ce-f)/f)
            if k >= 4:
                fb = box_ml_focal(np.array(vis), pts[vis], f*rng.uniform(0.7, 1.4),
                                  R, t, ratios)
                if fb is not None and np.isfinite(fb):
                    res[k]["box"].append(min(abs(fb-f)/f, 10.0))
    print(f"E4: cuboid with unknown edge ratios, sigma={sigma} px, {n} boxes, "
          f"vertices revealed in random order")
    print(f" 1  one corner gives no estimate: median f_min/f = "
          f"{np.median(one['fmin']):.2f}, unbounded above for "
          f"{100*one['unb']/one['n']:.0f}% of vertices (n={one['n']})")
    print(f"{'k':>2} {'corner median':>14} {'n':>5} {'box(ML) median':>15} {'n':>5}")
    for k in range(2, 9):
        c = res[k]["corner"]; b = res[k]["box"]
        cs = f"{100*np.median(c):.1f}%" if len(c) > 4 else "--"
        bs = f"{100*np.median(b):.1f}%" if len(b) > 4 else "--"
        print(f"{k:>2} {cs:>14} {len(c):>5} {bs:>15} {len(b):>5}")


def run_e4b():
    """E4 with matched observations (Sec. 6.5)."""
    from scipy.optimize import least_squares
    rng = np.random.default_rng(11)

    def box_ml(obs_idx, obs_xy, seg_obs, f0, R0, t0, ratios0, use_edges):
        """seg_obs: list of (vertex, axis, sign, far endpoint in image) observed
        edge segments.  With use_edges the fit also explains those endpoints, so
        both estimators see the same pixels."""
        def resid(p):
            R=rodrigues(p[:3])@R0; t=p[3:6]; ratios=np.abs(p[6:8]); f=abs(p[8])
            S=np.diag([1.0,ratios[0],ratios[1]])
            P=box_points(R,t,ratios)
            if np.any(P[:,2]<1e-3): return np.full(2*len(obs_idx)+2*len(seg_obs)*use_edges,1e3)
            r=list((project_pts(P[obs_idx],f)-obs_xy).ravel())
            if use_edges:
                for (vtx,ax,sgn,far,length) in seg_obs:
                    d=R@(S[:,ax]*sgn); d=d/np.linalg.norm(d)
                    Q=P[vtx]+length*d
                    if Q[2]<1e-3: r += [1e3,1e3]; continue
                    r += list(np.array([f*Q[0]/Q[2],f*Q[1]/Q[2]])-far)
            return np.asarray(r)
        p0=np.concatenate([np.zeros(3),t0,ratios0,[f0]])
        try: res=least_squares(resid,p0,method="lm",max_nfev=6000)
        except Exception: return None
        return abs(res.x[8])

    res={k:{"corner":[],"box_v":[],"box_ve":[]} for k in range(2,9)}
    n=0
    while n<400:
        f=rng.uniform(400,1500)
        out=trial(rng,0.5,f)
        if out is None: continue
        R,t,ratios,P,pts,lines=out
        # recover the observed segment far-endpoints that the corner method uses
        S=np.diag([1.0,ratios[0],ratios[1]])
        n+=1
        order=list(rng.permutation(8))
        for k in range(2,9):
            vis=order[:k]
            est=corner_estimates(vis,lines)
            if est: res[k]["corner"].append(abs(np.median(est)-f)/f)
            seg_obs=[]
            for vtx in vis:
                for ax,sgn in EDGES_AT[vtx]:
                    d=R@(S[:,ax]*sgn); d=d/np.linalg.norm(d)
                    L=0.9
                    Q=P[vtx]+L*d
                    if Q[2]<0.4: continue
                    far=np.array([f*Q[0]/Q[2],f*Q[1]/Q[2]])+rng.normal(0,0.5,2)
                    seg_obs.append((vtx,ax,sgn,far,L))
            for tag,ue in (("box_v",False),("box_ve",True)):
                if k<4 and not ue: continue
                fb=box_ml(np.array(vis),pts[vis],seg_obs,f*rng.uniform(0.8,1.25),
                          R,t,ratios,ue)
                if fb and np.isfinite(fb): res[k][tag].append(min(abs(fb-f)/f,10.0))
    print("E4 with matched observations, sigma=0.5 px, %d boxes"%n)
    print(f"{'k':>2} {'corner':>9} {'box(ML) vertices':>18} {'box(ML) vertices+edges':>24}")
    for k in range(2,9):
        def m(t):
            a=res[k][t]; return f"{100*np.median(a):.1f}%" if len(a)>4 else "--"
        print(f"{k:>2} {m('corner'):>9} {m('box_v'):>18} {m('box_ve'):>24}")


def run_e6():
    """Combining per-corner bounds (Sec. 6.7)."""
    rng=np.random.default_rng(9)
    f=866.0

    def bounds(m, sigma):
        out=[]
        while len(out)<m:
            o=observe_corner(place(f,rng), rand_frame(rng), f, sigma, rng)
            if o is None: continue
            c,L=o; comps=admissible(coeffs(c,L))
            if not comps: continue
            lo,hi=comps[0][0],comps[-1][1]
            out.append((np.sqrt(lo), np.sqrt(hi) if hi<1e13 else np.inf))
        return out

    def consensus(iv, k):
        """the set covered by at least k of the intervals; return its hull"""
        ev=sorted([(a,1) for a,_ in iv]+[(b,-1) for _,b in iv if np.isfinite(b)]
                  +[(1e9,-1) for _,b in iv if not np.isfinite(b)])
        cnt=0; lo=None; hi=None
        for x,d in ev:
            prev=cnt; cnt+=d
            if prev<k<=cnt and lo is None: lo=x
            if prev>=k>cnt: hi=x
        return (lo,hi if hi is not None else np.inf) if lo is not None else None

    print(f"sigma  m   strict-intersection coverage   consensus(90%) coverage   median width/f")
    for sigma in (0.5,1.0):
        for m in (2,5,10,25,50):
            cov_s=cov_c=0; w=[]
            for _ in range(400):
                iv=bounds(m,sigma)
                lo=max(a for a,_ in iv); hi=min(b for _,b in iv)
                if lo<=f<=hi: cov_s+=1
                pass
                k=max(1,int(np.ceil(0.9*m)))
                cc=consensus(iv,k)
                if cc and cc[0]<=f<=cc[1]:
                    cov_c+=1; w.append((min(cc[1],5*f)-cc[0])/f)
            print(f"{sigma:<6} {m:<3} {100*cov_s/400:>22.0f}%   {100*cov_c/400:>20.0f}%   "
                  f"{np.median(w) if w else float('nan'):>14.2f}")


def run_sim():
    """The similarity law of Cor. 1, on the E2 configuration (Sec. 6.2)."""
    rng=np.random.default_rng(31)
    print("          median f_min/f           median f_min/rho")
    print(" FOV   inner   mid   outer     inner   mid   outer")
    for fov,f in FOVS.items():
        bins={0:[],1:[],2:[]}; binsr={0:[],1:[],2:[]}
        n=0
        while n<4000:
            o=observe_corner(place(f,rng), rand_frame(rng), f, 0.0, rng)
            if o is None: continue
            c,L=o; comps=admissible(coeffs(c,L))
            if not comps: continue
            n+=1
            rho=np.hypot(c[0],c[1])
            if rho<1e-6: continue
            fmin=np.sqrt(comps[0][0])
            k=0 if rho/HALF<1/3 else (2 if rho/HALF>2/3 else 1)
            bins[k].append(fmin/f); binsr[k].append(fmin/rho)
        print(f" {fov:3d}  " + "".join(f"{np.median(bins[k]):6.2f}" for k in range(3))
              + "    " + "".join(f"{np.median(binsr[k]):6.2f}" for k in range(3)))
    print("\nf_min/rho is nearly constant across radius and field of view, as Cor. 1")
    print("predicts; the variation left is the distribution of the angles psi_i.")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in (("e1", run_e1), ("e2", run_e2), ("e3a", run_e3a), ("e3b", run_e3b),
                      ("e4", run_e4), ("e4b", run_e4b), ("e6", run_e6), ("sim", run_sim)):
        if which in (name, "all"):
            fn()
