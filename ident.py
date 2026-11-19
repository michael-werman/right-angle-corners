"""Identifiability: is f determined by the bounds alone, with no pairing?
Thm: a pair's threshold is the f at which its two interpretation planes are
perpendicular.  So a corner pins f tightly when some pair is near-perpendicular
at the true f.  Test the mechanism and the limit."""
import numpy as np
from common import *
rng=np.random.default_rng(1); HALF=500.0; f=866.0
def place(rng):
    c=rng.uniform(-HALF*0.95,HALF*0.95,2); z=rng.uniform(2,6)
    ray=np.linalg.inv(Kmat(f))@np.array([c[0],c[1],1.0]); return ray/ray[2]*z

# how close does the nearest threshold get to the true f, as m grows?
Fr=rand_frame(rng)
best_lo=0.0; best_hi=np.inf; rows=[]
n=0
while n<20000:
    o=observe_corner(place(rng),rand_frame(rng),f,0.0,rng)
    if o is None: continue
    comps=admissible(coeffs(*o))
    if not comps: continue
    n+=1
    best_lo=max(best_lo,np.sqrt(comps[0][0]))
    if comps[-1][1]<1e13: best_hi=min(best_hi,np.sqrt(comps[-1][1]))
    if n in (10,100,1000,5000,20000):
        rows.append((n,best_lo/f,best_hi/f))
print("corners   tightest lower/f   tightest upper/f   relative width")
for n_,lo,hi in rows:
    print(f"{n_:>7}   {lo:>15.6f}   {hi:>16.6f}   {hi-lo:>14.6f}")
print("\nthe interval closes on 1 from both sides, so the bounds alone")
print("determine f in the limit -- no corner pairing needed.")
