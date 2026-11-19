"""Online Resource S10, a misplaced principal point: coverage of the true focal length by the admissible
set F of one junction (Theorem 6) when the junction is measured relative to an assumed principal point that is
offset from the true one by 0 to 40 px in a random direction. Protocol of the article's Sect. 6.1: corners
uniform over orientations and over a 1000 x 1000 px image, 150 px edges whose endpoints carry Gaussian noise of
0.5 px (an angle error of standard deviation 0.5*sqrt(2)/150 rad), 1000 corners per field of view."""
import numpy as np
rng=np.random.default_rng(7)
P=((0,1),(1,2),(2,0))
def corner(f,W=1000):
    while True:
        q,_=np.linalg.qr(rng.standard_normal((3,3))); c=rng.uniform(-W/2,W/2,2); r=np.array([c[0]/f,c[1]/f,1.0]); th=[]
        ok=True
        for k in range(3):
            d=q[:,k]
            if rng.random()<0.5: d=-d
            p2=r+1e-3*d; u=f*p2[:2]/p2[2]-c
            if np.linalg.norm(u)<1e-12: ok=False; break
            th.append(np.arctan2(u[1],u[0]))
        if ok: return c,np.array(th)
def admits(c,th,x):
    a=-np.sin(th); b=np.cos(th); e=-(a*c[0]+b*c[1])
    return np.prod([(a[i]*a[j]+b[i]*b[j])*x+e[i]*e[j] for i,j in P])<0
OFFSETS=(0,5,10,20,40)
if __name__=="__main__":
    for fov in (40,60,90):
        f=500/np.tan(np.radians(fov/2)); row=[]
        for dpp in OFFSETS:
            hit=0; N=1000
            for _ in range(N):
                c,th=corner(f); thn=th+rng.normal(0,0.5*np.sqrt(2)/150,3)
                ang=rng.uniform(0,2*np.pi); off=dpp*np.array([np.cos(ang),np.sin(ang)])
                hit+=admits(c-off,thn,f*f)
            row.append(round(100*hit/N))
        print(f"{fov} deg (f = {f:.0f} px): coverage at offsets {OFFSETS} px = {row} %")
