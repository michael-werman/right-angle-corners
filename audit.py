"""Scrutinize Prop. 2 (identifiability). Its hypothesis -- that the family
contains corners whose thresholds approach f from both sides -- is doing all
the work.  Is it actually satisfied by corners of ONE Manhattan frame seen by
one camera, with only the position varying?"""
import numpy as np
from common import *
rng=np.random.default_rng(0); f=866.0; HALF=500.0
Fr=rand_frame(rng)                      # ONE fixed frame
lo_best, hi_best = 0.0, np.inf
near_lo, near_hi = [], []
n=0
while n<40000:
    c=rng.uniform(-HALF*0.95,HALF*0.95,2); z=rng.uniform(2,6)
    ray=np.linalg.inv(Kmat(f))@np.array([c[0],c[1],1.0]); P=ray/ray[2]*z
    o=observe_corner(P,Fr,f,0.0,rng)    # position varies, frame fixed
    if o is None: continue
    comps=admissible(coeffs(*o))
    if not comps: continue
    n+=1
    lo_best=max(lo_best,np.sqrt(comps[0][0]))
    if comps[-1][1]<1e13: hi_best=min(hi_best,np.sqrt(comps[-1][1]))
print("ONE fixed Manhattan frame, position varying only:")
print(f"  {n} corners -> [{lo_best/f:.6f}, {hi_best/f:.6f}] x f, "
      f"width {(hi_best-lo_best)/f:.2e}")

# and with the frame also varying
rng=np.random.default_rng(1); lo2,hi2=0.0,np.inf; n=0
while n<40000:
    c=rng.uniform(-HALF*0.95,HALF*0.95,2); z=rng.uniform(2,6)
    ray=np.linalg.inv(Kmat(f))@np.array([c[0],c[1],1.0]); P=ray/ray[2]*z
    o=observe_corner(P,rand_frame(rng),f,0.0,rng)
    if o is None: continue
    comps=admissible(coeffs(*o))
    if not comps: continue
    n+=1
    lo2=max(lo2,np.sqrt(comps[0][0]))
    if comps[-1][1]<1e13: hi2=min(hi2,np.sqrt(comps[-1][1]))
print(f"frames varying too: {n} corners -> [{lo2/f:.6f}, {hi2/f:.6f}] x f, "
      f"width {(hi2-lo2)/f:.2e}")
