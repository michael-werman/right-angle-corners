"""Checks for the certified bounds and the distortion error bounds."""
import numpy as np
rng=np.random.default_rng(23)
def forms(x,rho,psi):
    P=((0,1),(1,2),(2,0))
    return np.stack([np.cos(psi[i]-psi[j])*x+rho**2*np.sin(psi[i])*np.sin(psi[j]) for i,j in P],-1)
xs=np.geomspace(1e2,1e8,6000)
bad1=bad2=miss_f=0; n=0
for _ in range(3000):
    rho=rng.uniform(5,700); psi=rng.uniform(-np.pi,np.pi,3); phi=np.radians(rng.choice([0.2,0.5,1,2,4]))
    psih=psi+rng.uniform(-phi,phi,3)
    Ft=np.prod(forms(xs,rho,psi),-1)<0; Fh=np.prod(forms(xs,rho,psih),-1)<0
    marg=np.abs(forms(xs,rho,psih)); bound=np.sqrt(2)*phi*(np.sqrt(2)*xs+rho**2)
    U=(marg<=bound[:,None]).any(-1)
    bad1+=np.sum(Fh&~U&~Ft)          # certified inner part must lie in the true set
    bad2+=np.sum(Ft&~Fh&~U)          # the true set must lie in the measured set or the uncertain set
    n+=1
print(f"(1) certified bounds, {n} junctions x {len(xs)} focal lengths: inner-part violations {bad1}, outer-part violations {bad2} (both must be 0)")
# ---- (2a) the largest angle change from tan psi' = theta tan psi
worst=0
for _ in range(20000):
    th=rng.uniform(0.3,3.0); psi=rng.uniform(-np.pi,np.pi); ps2=np.arctan2(th*np.sin(psi),np.cos(psi))
    d=abs((ps2-psi+np.pi)%(2*np.pi)-np.pi); worst=max(worst,d/abs(np.arctan((th-1)/(2*np.sqrt(th)))))
print(f"(2a) max |psi'-psi| / arctan(|1-theta|/(2 sqrt theta)) over 20000 cases: {worst:.6f}  (must be <= 1, and attained)")
# ---- (2b) chord against tangent: angle <= arcsin(3|k|(rho+s)s/m)
worst=0; tight=0
for _ in range(20000):
    k=rng.uniform(-1,1)*rng.choice([1e-7,1e-6,3e-6]); c=rng.uniform(-500,500,2); rho=np.linalg.norm(c)
    u=rng.standard_normal(2); u/=np.linalg.norm(u); s=rng.uniform(10,200)
    D=lambda x: x*(1+k*(x@x))
    J=(1+k*rho**2)*np.eye(2)+2*k*np.outer(c,c); tang=J@u; chord=D(c+s*u)-D(c)
    ang=np.arccos(np.clip(tang@chord/np.linalg.norm(tang)/np.linalg.norm(chord),-1,1))
    m=min(abs(1+k*rho**2),abs(1+3*k*rho**2)); r=3*abs(k)*(rho+s)*s/m
    if r<1: worst=max(worst,ang/np.arcsin(r)); tight+=1
print(f"(2b) chord-tangent angle / arcsin(3|k|(rho+s)s/m) over {tight} cases: max {worst:.3f}  (must be <= 1)")
