import numpy as np, sys
from robust_sim import scene
from common import coeffs, admissible
from lik_calib import estimate
SIG=(0.1,0.2,0.35,0.5,0.75,1.0,1.5,2.0,3.0,4.0,6.0,8.0)
rng=np.random.default_rng(21)
print("same 640x480 image, same edge lengths, 181 junctions, level cameras; likelihood estimator; 10 trials")
print(f"{'horizontal FOV':>15}{'f (px)':>8}{'bounded above':>15}{'sigma':>7}{'median |f error|':>18}{'worst':>8}")
for fov in (51,75,100):
    f=320.0/np.tan(np.radians(fov)/2); K=np.array([[f,0,0],[0,f,0],[0,0,1.0]])
    for sigma in (0.25,0.5,1.0):
        errs=[];ub=[]
        for _ in range(10):
            Js=scene(rng,sigma,K,True)
            nb=sum(1 for L,c,_,_ in Js if (lambda cm: cm and cm[-1][1]<1e13)(admissible(coeffs(c[:2],L))))
            ub.append(nb/len(Js))
            fh,_,_=estimate(Js,f_lo=60.0,f_hi=4000.0,sigmas=SIG,n=700); errs.append(abs(fh-f)/f)
        print(f"{fov:>13} deg{f:>8.0f}{100*np.mean(ub):>14.1f}%{sigma:>7}{100*np.median(errs):>17.1f}%{100*max(errs):>7.0f}%")
