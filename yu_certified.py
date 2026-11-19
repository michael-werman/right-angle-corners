"""Certified intervals (Proposition: certified bounds) on the 181 York Urban junctions.
Each edge's angle error is bounded by phi_i = arcsin(2 eps / l_i), l_i its segment length, for an assumed
endpoint accuracy eps; with per-edge bounds the uncertain set of pair (i,j) uses (phi_i^2+phi_j^2)^(1/2)."""
import numpy as np, yorkurban as Y, yu_junctions
from yud_path import YUD
from common import coeffs, admissible
fT,pp,images=Y.load_packed(YUD); J,_,_,_=yu_junctions.collect()
xs=np.geomspace((0.05*fT)**2,(20*fT)**2,40000); xt=fT**2
def certified(Ls,c,lens,eps):
    rho2=c@c; P=((0,1),(1,2),(2,0))
    phi=np.arcsin(np.minimum(1.0,2*eps/np.asarray(lens,float)))
    F=np.ones_like(xs); U=np.zeros_like(xs,bool); Ft=1.0; Ut=False
    for i,j in P:
        A=Ls[i][0]*Ls[j][0]+Ls[i][1]*Ls[j][1]; E=Ls[i][2]*Ls[j][2]
        r=np.hypot(phi[i],phi[j])
        form=A*xs+E; F=F*form; U|= np.abs(form)<=r*(np.sqrt(2)*xs+rho2)
        ft=A*xt+E; Ft*=ft; Ut|= abs(ft)<=r*(np.sqrt(2)*xt+rho2)
    Fh=F<0
    return Fh, U, (Ft<0) or Ut
print(f"{'eps (px)':>9}{'true f in measured F':>22}{'true f in certified set':>25}{'median certified lower bound':>30}{'bounded above':>15}")
for eps in (0.0,0.25,0.5,1.0,2.0):
    inF=0; inC=0; lows=[]; bnd=0
    for (Ls,c,idx,lens) in J:
        Fh,U,ok=certified(Ls,c[:2],lens,eps)
        C=Fh|U
        inF+= any(Fh[(xs>xt*0.999)&(xs<xt*1.001)])
        inC+= ok
        if C.any():
            lows.append(np.sqrt(xs[np.argmax(C)])/fT); bnd+= not C[-1]
    print(f"{eps:>9}{inF:>17d}/181{inC:>20d}/181{np.median(lows):>29.3f}f{bnd:>12d}")
# the two junctions that miss f without certification: what eps brings them in?
for (Ls,c,idx,lens) in J:
    Fh,U,ok0=certified(Ls,c[:2],lens,0.0)
    if not ok0:
        for eps in np.arange(0.05,5.01,0.05):
            if certified(Ls,c[:2],lens,eps)[2]: print(f"   a missed junction (image {images[idx][0]}, edges {np.round(lens).astype(int)} px) is covered from eps = {eps:.2f} px"); break
        else: print(f"   a missed junction (image {images[idx][0]}) is not covered up to eps = 5 px")
