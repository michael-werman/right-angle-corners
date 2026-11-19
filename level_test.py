"""Does identifiability from one frame hold for a LEVEL camera?  There the
horizon (edge-on line of the vertical axis) crosses the image, so the theorem
as stated says yes -- but vertical edges are exactly perpendicular to the
horizon in the image, so the lemma's slopes A_ij vanish along the whole line."""
import numpy as np
from common import Kmat, coeffs, admissible, observe_corner
f=900.0; half=480
def frame(pitch_deg, yaw_deg=40.0):
    y=np.radians(yaw_deg); p=np.radians(pitch_deg)
    Ry=np.array([[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]])
    Rx=np.array([[1,0,0],[0,np.cos(p),-np.sin(p)],[0,np.sin(p),np.cos(p)]])
    return Rx@Ry
def crossing(D):
    g=np.linspace(-half,half,201); U,V=np.meshgrid(g,g)
    R=np.stack([U,V,np.full_like(U,f)],-1); R/=np.linalg.norm(R,axis=-1,keepdims=True)
    return [bool((R@D[:,i]).min()<0<(R@D[:,i]).max()) for i in range(3)]
print(f"{'pitch':>6}{'edge-on lines in view (axes 1,2,3)':>38}{'bounded above':>15}{'intersection of F (x f)':>28}{'width':>10}")
for pitch in (0.0,0.5,2.0,5.0,12.0,25.0):
    D=frame(pitch); rng=np.random.default_rng(3); lo,hi=0.0,np.inf; nb=n=0
    for _ in range(40000):
        c=rng.uniform(-half,half,2); z=rng.uniform(2,6)
        ray=np.linalg.inv(Kmat(f))@np.array([c[0],c[1],1.0]); P=ray/ray[2]*z
        o=observe_corner(P,D,f,0.0,rng)
        if o is None: continue
        cm=admissible(coeffs(*o))
        if not cm: continue
        n+=1; nb+=cm[-1][1]<1e13; lo=max(lo,cm[0][0]); hi=min(hi,cm[-1][1])
    his=f"{np.sqrt(hi)/f:.4f}" if np.isfinite(hi) else "inf"
    w=(np.sqrt(hi)-np.sqrt(lo))/f if np.isfinite(hi) else np.inf
    print(f"{pitch:>6}{str(crossing(D)):>38}{100*nb/n:>14.1f}%{f'[{np.sqrt(lo)/f:.4f}, {his}]':>28}{w:>10.1e}")
