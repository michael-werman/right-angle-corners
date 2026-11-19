import numpy as np, sys
from robust_sim import scene, F_TRUE, PP_TRUE
from robust_calib import fit, params
from lik_calib import estimate
rng=np.random.default_rng(3)
K0=np.array([[F_TRUE,0,0],[0,F_TRUE,0],[0,0,1.0]])
Kp=np.array([[F_TRUE,0,PP_TRUE[0]],[0,F_TRUE,PP_TRUE[1]],[0,0,1.0]])
print("single focal length, 181 junctions, level cameras, York-Urban edge lengths; 16 trials\n")
print(f"{'sigma':>6} | {'hinge LP: median |err|, bias':>30} | {'likelihood: median |err|, bias':>32} | {'sigma-hat':>9} | {'lik, pp 17px off':>17}")
for sigma in (0.0,0.5,1.0,2.0):
    eh=[];el=[];sh=[];eo=[]
    for _ in range(16):
        Js=scene(rng,sigma,K0,True)
        r=fit('f',Js,ranges=True); eh.append((params('f',r)['fx']-F_TRUE)/F_TRUE)
        fh,s_hat,_=estimate(Js); el.append((fh-F_TRUE)/F_TRUE); sh.append(s_hat)
        Jo=scene(rng,sigma,Kp,True); fo,_,_=estimate(Jo); eo.append((fo-F_TRUE)/F_TRUE)
    eh,el,eo=np.array(eh),np.array(el),np.array(eo)
    print(f"{sigma:>6} | {100*np.median(abs(eh)):>16.1f}%, {100*np.median(eh):>+9.1f}% | {100*np.median(abs(el)):>17.1f}%, {100*np.median(el):>+10.1f}% | {np.median(sh):>9.2f} | {100*np.median(abs(eo)):>16.1f}%")
