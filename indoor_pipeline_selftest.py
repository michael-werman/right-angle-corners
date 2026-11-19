"""Self-test of indoor_pipeline.py on rendered scenes: opaque boxes on a floor, seen by a known camera."""
import numpy as np, cv2
from indoor_pipeline import run
rng=np.random.default_rng(5)
def rot(ax,deg):
    a=np.radians(deg); c,s=np.cos(a),np.sin(a)
    return {'x':np.array([[1,0,0],[0,c,-s],[0,s,c]]),'y':np.array([[c,0,s],[0,1,0],[-s,0,c]])}[ax]
def render(f, W=640, H=480):
    img=np.full((H,W),235,np.uint8); K=np.array([[f,0,W/2],[0,f,H/2],[0,0,1.0]])
    Rc=rot('x',rng.uniform(-8,8))                          # a camera within a few degrees of level
    boxes=[]
    for _ in range(rng.integers(3,6)):
        size=rng.uniform(0.4,1.2,3); yaw=rng.uniform(0,90); center=np.array([rng.uniform(-1.5,1.5),rng.uniform(0.2,0.9),rng.uniform(4,8)])
        corners=np.array([[sx,sy,sz] for sx in (-.5,.5) for sy in (-.5,.5) for sz in (-.5,.5)])*size
        Xw=(rot('y',yaw)@corners.T).T+center; boxes.append(Xw)
    boxes.sort(key=lambda B:-B[:,2].mean())                 # draw far boxes first: painter's algorithm
    faces=[(0,1,3,2),(4,5,7,6),(0,1,5,4),(2,3,7,6),(0,2,6,4),(1,3,7,5)]
    for Xw in boxes:
        Xc=(Rc@Xw.T).T; x=(K@Xc.T).T; x=x[:,:2]/x[:,2:]
        cam=-Xc
        for fc in sorted(faces,key=lambda fc:-Xc[list(fc)][:,2].mean()):
            n=np.cross(Xc[fc[1]]-Xc[fc[0]],Xc[fc[2]]-Xc[fc[0]]); ctr=Xc[list(fc)].mean(0)
            if n@(-ctr)<=0: n=-n
            if (-ctr)@np.cross(Xc[fc[1]]-Xc[fc[0]],Xc[fc[3]]-Xc[fc[0]])==0: pass
            poly=np.round(x[list(fc)]).astype(np.int32)
            shade=int(90+120*abs(n[2])/np.linalg.norm(n))
            cv2.fillPoly(img,[poly],shade); cv2.polylines(img,[poly],True,30,1,cv2.LINE_AA)
    img=cv2.GaussianBlur(img,(3,3),0.6)
    noise=rng.normal(0,2.0,img.shape); return np.clip(img+noise,0,255).astype(np.uint8)
rows=[]; images={}
for n in range(12):
    f=rng.uniform(450,700); name=f"scene{n:02d}.png"; images[name]=render(f)
    rows.append(dict(image=name,fx=f,fy=f,cx=320,cy=240))
cv2.imwrite("figures/indoor_selftest_example.png",images["scene00.png"])
summary,recs=run(rows, lambda name: images[name], eps=0.5)
print("per image (segments, junctions, passing):",[(s['segments'],s['junctions'],s['passed']) for s in summary])
assert all(s['segments']>0 for s in summary), "a rendered scene has no segments"
