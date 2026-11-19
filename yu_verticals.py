"""York Urban: vertical directions in camera coordinates, whether each horizon crosses
the image, and how far apart the verticals are."""
import numpy as np, itertools
import yorkurban as Y, yu_junctions
from yud_path import YUD
f,pp,images=Y.load_packed(YUD)
J,_,_,_=yu_junctions.collect()
used=sorted(set(j[2] for j in J))
G=[];inside=0
for idx in used:
    V=np.array(images[idx][3],float); V=V/np.linalg.norm(V,axis=0,keepdims=True)
    k=int(np.argmax(np.abs(V[1,:]))); g=V[:,k]*np.sign(V[1,k]); G.append(g)
    # nearest point of the horizon {r : r.g = 0} to the principal point, in pixels
    off=-f*g[2]*g[:2]/(g[:2]@g[:2])
    ok=all((-pp[0]<o[0]<640-pp[0]) and (-pp[1]<o[1]<480-pp[1]) for o in (off,off*np.array([1,-1])))   # either y convention
    inside+=ok
G=np.array(G); tilt=np.degrees(np.arcsin(np.abs(G[:,2])))
mean=G.mean(0); mean/=np.linalg.norm(mean)
ang=np.degrees(np.arccos(np.clip(G@mean,-1,1)))
pair=max(np.degrees(np.arccos(np.clip(a@b,-1,1))) for a,b in itertools.combinations(G,2))
sv=np.linalg.svd(G.T,compute_uv=False)
print(f"{len(used)} images with junctions; tilt of the vertical out of the image plane: max {tilt.max():.1f} deg")
print(f"horizon passes through the image in {inside}/{len(used)} images (under either y convention)")
print(f"vertical directions: within {ang.max():.1f} deg of their mean (median {np.median(ang):.1f}); largest pairwise angle {pair:.1f} deg")
print(f"singular values of the 3 x {len(used)} matrix of verticals: {np.round(sv,3)}  (smallest/largest {sv[2]/sv[0]:.3f})")
