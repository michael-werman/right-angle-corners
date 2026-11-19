import numpy as np, sympy as sp
from yud_path import YUD
# (1) level camera: axis i = (0,1,0) parallel to the image plane; j,k horizontal at yaw t.
u,v,f,t,Z=sp.symbols('u v f t Z',positive=True)
K=sp.diag(f,f,1); c=sp.Matrix([u,v,1]); P=K.inv()*c*Z
D=[sp.Matrix([0,1,0]),sp.Matrix([sp.cos(t),0,sp.sin(t)]),sp.Matrix([-sp.sin(t),0,sp.cos(t)])]
def line(d):
    q=K*(P+d); q=q/q[2]; e=sp.Matrix([q[0]-u,q[1]-v])          # image direction (unnormalised)
    a,b=e[1],-e[0]; return sp.Matrix([a,b,-(a*u+b*v)])
L=[line(d) for d in D]
def thr(i,j):   # x_ij = -e_i e_j / A_ij ; the normalisation of the lines cancels in the ratio
    return sp.simplify(-(L[i][2]*L[j][2])/(L[i][0]*L[j][0]+L[i][1]*L[j][1]))
x01=thr(0,1); x02=thr(0,2)
print("level camera, pairs containing the axis parallel to the image plane:")
print("   x_ij =",sp.simplify(x01),"     x_ik =",sp.simplify(x02))
print("   depend on v (distance from the horizon)? ", sp.simplify(sp.diff(x01,v))!=0, sp.simplify(sp.diff(x02,v))!=0)
vjx=f*sp.cos(t)/sp.sin(t); vkx=-f*sp.sin(t)/sp.cos(t)
print("   x_ij = -u * (abscissa of v_j):", sp.simplify(x01+u*vjx)==0, "   x_ik = -u * (abscissa of v_k):", sp.simplify(x02+u*vkx)==0)
print("   so x_ij = f^2 exactly when u = abscissa of v_k:", sp.simplify(x01.subs(u,vkx)-f**2)==0)
# (2) how level are the York Urban photographs?
import yorkurban as Y
f_true,pp,images=Y.load_packed(YUD)
tilt=[]
for name,lines,labels,vp in images:
    V=np.array(vp,float); V=V/np.linalg.norm(V,axis=0,keepdims=True)
    # the axis closest to the image's vertical direction; its tilt out of the image plane
    k=int(np.argmax(np.abs(V[1,:]))); tilt.append(np.degrees(np.arcsin(min(1,abs(V[2,k])))))
tilt=np.array(tilt)
print(f"\nYork Urban, tilt of the vertical axis out of the image plane, {len(tilt)} images:")
print(f"   median {np.median(tilt):.1f} deg, quartiles {np.percentile(tilt,25):.1f}-{np.percentile(tilt,75):.1f}, 90th pct {np.percentile(tilt,90):.1f}, max {tilt.max():.1f}")
