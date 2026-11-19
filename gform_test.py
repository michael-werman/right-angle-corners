"""Claims, for K = diag(f,f,1) and one frame D:
 (1) nu_a = s + (1-s) g_a,  g_a = (z_a - zeta_a w_p)(z_a - zeta_a w_q),  w_b = z_b/zeta_b,  s = f'^2/f^2
 (2) hence  cap F = { s : Gm/(1+Gm) < s < Gp/(Gp-1) },  Gp = sup g, Gm = sup(-g)   (Gp <= 1: unbounded)"""
import numpy as np, sympy as sp
from common import Kmat, coeffs, admissible, observe_corner, rand_frame
# (1) symbolic
s_=sp.Symbol('s'); z=sp.symbols('z1:4'); ze=sp.symbols('zeta1:4')
N=s_*sp.eye(3)+(1-s_)*sp.Matrix(z)*sp.Matrix(z).T
ok=True
for a in range(3):
    p,q=[b for b in range(3) if b!=a]
    nu=N[a,a]-ze[a]/ze[p]*N[a,p]-ze[a]/ze[q]*N[a,q]+ze[a]**2/(ze[p]*ze[q])*N[p,q]
    g=(z[a]-ze[a]*z[p]/ze[p])*(z[a]-ze[a]*z[q]/ze[q])
    ok&= sp.simplify(nu-(s_+(1-s_)*g))==0
print("(1) nu_a = s + (1-s) g_a, symbolically:",ok)
# (2) numeric, against the intersection of the single-junction intervals
rng=np.random.default_rng(0); worst=0; n_id=0
for trial in range(40):
    f=rng.uniform(400,1500); D=rand_frame(rng); zz=D[2,:]; lo,hi=0.0,np.inf; Gp=-np.inf; Gm=-np.inf
    for _ in range(3000):
        c=rng.uniform(-480,480,2); P=np.array([c[0]/f,c[1]/f,1.0])*rng.uniform(2,6)
        o=observe_corner(P,D,f,0.0,rng)
        if o is None: continue
        cm=admissible(coeffs(*o))
        if not cm: continue
        lo=max(lo,cm[0][0]); hi=min(hi,cm[-1][1])
        zeta=D.T@np.array([c[0]/f,c[1]/f,1.0]); w=zz/zeta
        for a in range(3):
            p,q=[b for b in range(3) if b!=a]
            g=(zz[a]-zeta[a]*w[p])*(zz[a]-zeta[a]*w[q]); Gp=max(Gp,g); Gm=max(Gm,-g)
    smin=Gm/(1+Gm) if Gm>0 else 0.0; smax=Gp/(Gp-1) if Gp>1 else np.inf
    e1=abs(lo/f**2-smin); e2=abs(hi/f**2-smax) if np.isfinite(hi) or np.isfinite(smax) else 0.0
    if np.isfinite(hi)!=np.isfinite(smax): e2=np.inf
    worst=max(worst,e1,e2)
print(f"(2) 40 random cameras and frames, 3000 junctions each: closed form vs brute-force intersection,")
print(f"    worst discrepancy in s = f'^2/f^2 at either end: {worst:.1e}")
