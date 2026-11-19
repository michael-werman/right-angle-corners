"""Rate when a vanishing point lies INSIDE the image: near it g ~ 1/(zeta_i zeta_k),
whose tail is heavier by a logarithm.  Prediction: W ~ 1/(M log M), i.e. M*W*log(M) flat."""
import numpy as np
half=480.0
def rot(ax,deg):
    a=np.radians(deg); c,s=np.cos(a),np.sin(a)
    return {'x':np.array([[1,0,0],[0,c,-s],[0,s,c]]),'y':np.array([[c,0,s],[0,1,0],[-s,0,c]]),'z':np.array([[c,-s,0],[s,c,0],[0,0,1]])}[ax]
def widths(D,f,Ms,rng):
    M=Ms[-1]; c=rng.uniform(-half,half,(M,2)); r=np.column_stack([c/f,np.ones(M)])
    zeta=r@D; z=D[2,:]; w=z/zeta; G=np.full(M,-np.inf); Gn=np.full(M,-np.inf)
    for a in range(3):
        p,q=[b for b in range(3) if b!=a]
        g=(z[a]-zeta[:,a]*w[:,p])*(z[a]-zeta[:,a]*w[:,q]); G=np.maximum(G,g); Gn=np.maximum(Gn,-g)
    cp=np.maximum.accumulate(G); cn=np.maximum.accumulate(Gn)
    return [1/(cp[m-1]-1)+1/(1+cn[m-1]) if cp[m-1]>1 else np.inf for m in Ms]     # width in s
Ms=[int(x) for x in np.geomspace(1e3,2e6,12)]; rng=np.random.default_rng(1)
f=700.0
D_in=rot('z',25)@rot('x',14)@rot('y',9)            # third axis near the optical axis: its vanishing point is in the image
D_out=rot('z',25)@rot('x',52)@rot('y',38)          # all vanishing points outside
for label,D in (("vanishing point inside the image",D_in),("no vanishing point in the image",D_out)):
    vps=[f*D[:2,k]/D[2,k] for k in range(3)]
    inside=[bool(max(abs(v))<half) for v in vps]
    W=np.array([widths(D,f,Ms,rng) for _ in range(40)]); med=np.median(W,axis=0); Mv=np.array(Ms)
    sl=np.polyfit(np.log(Mv[4:]),np.log(med[4:]),1)[0]
    a=Mv*med; b=Mv*med*np.log(Mv)
    print(f"{label}: vanishing points inside = {inside}")
    print(f"   log-log slope {sl:+.3f};   M*W from {a[4]:.3f} to {a[-1]:.3f} (ratio {a[-1]/a[4]:.2f});   M*W*log M from {b[4]:.2f} to {b[-1]:.2f} (ratio {b[-1]/b[4]:.2f})")
