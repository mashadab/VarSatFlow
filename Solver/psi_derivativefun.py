#computes the analytical derivative

import numpy as np
import matplotlib.pyplot as plt
from mobilityfun import mobility

def psi_derivative(s_w,s_wp,s_nwp,mu_w,mu_nw,k_w0,k_nw0,n_w,n_nw): 
    # author: Mohammad Afzal Shadab
    # date: 04/11/2020
    # description:
    # This function computes the psi-derivative 
    # 
    # Output:
    # (dpsi/ds_w)_{s_w}

    ds_w = 0.000001
    
    if s_w + ds_w <= 1.0:
    
        [s_wc1, kr_w1, kr_nw1, lambda_w1, lambda_nw1,lambda_t1, f_w1, f_nw1] = mobility(s_w + ds_w,s_wp,s_nwp,mu_w,mu_nw,k_w0,k_nw0,n_w,n_nw)   
        
        [s_wc2, kr_w2, kr_nw2, lambda_w2, lambda_nw2,lambda_t2, f_w2, f_nw2] = mobility(s_w,s_wp,s_nwp,mu_w,mu_nw,k_w0,k_nw0,n_w,n_nw)   
    
        psi_w_1       = ( kr_w1 * kr_nw1 ) / ( kr_w1 + mu_w/mu_nw*kr_nw1 )
        psi_w_2       = ( kr_w2 * kr_nw2 ) / ( kr_w2 + mu_w/mu_nw*kr_nw2 )
        
        psi_w         =   psi_w_2
        psi_w_der     = ( psi_w_1 - psi_w_2 ) / ds_w

    else:
        
        [s_wc1, kr_w1, kr_nw1, lambda_w1, lambda_nw1,lambda_t1, f_w1, f_nw1] = mobility(s_w - ds_w,s_wp,s_nwp,mu_w,mu_nw,k_w0,k_nw0,n_w,n_nw)   
        
        [s_wc2, kr_w2, kr_nw2, lambda_w2, lambda_nw2,lambda_t2, f_w2, f_nw2] = mobility(s_w,s_wp,s_nwp,mu_w,mu_nw,k_w0,k_nw0,n_w,n_nw)   

        psi_w_1       = ( kr_w1 * kr_nw1 ) / ( kr_w1 + mu_w/mu_nw*kr_nw1 )
        psi_w_2       = ( kr_w2 * kr_nw2 ) / ( kr_w2 + mu_w/mu_nw*kr_nw2 )

        psi_w         =   psi_w_2
        psi_w_der     = ( psi_w_1 - psi_w_2 ) / (-ds_w)
        
    return psi_w,psi_w_der;

'''
#fluids
s_wp  = 0.0  #percolation threshold: wetting phase
s_nwp = 0.0  #percolation threshold: non-wetting phase
mu_w  = 1.0  #dynamic viscosity: wetting phase    
mu_nw = 1.0  #dynamic viscosity: non-wetting phase   
k_w0  = 1.0  #relative permeability threshold: wetting phase   
k_nw0 = 1.0  #relative permeability threshold: non-wetting phase   
n_w   = 4.0  #power law coefficient: wetting phase  
n_nw  = 2.0  #power law coefficient: non-wetting phase 

s_w = np.linspace(0,1,10001)
psi_w    = np.zeros((len(s_w),1))
psi_w_der  = np.zeros((len(s_w),1))

for j in range(0, len(s_w)):
    [psi_w[j],psi_w_der[j]] = psi_derivative(s_w[j],s_wp,s_nwp,mu_w,mu_nw,k_w0,k_nw0,n_w,n_nw)

#secant method
psi_w_sec_der1 = psi_w/np.transpose([s_w-s_wp])  
psi_w_sec_der2 =-psi_w/np.transpose([s_w-s_wp])  

kk1 = 1000 + np.argmin(np.abs(psi_w_der[1000:9000,:]-psi_w_sec_der1[1000:9000,:])) #to find the location of shock

kk2 = 1000 + np.argmin(np.abs(psi_w_der[1000:9000,:]-psi_w_sec_der2[1000:9000,:])) #to find the location of shock

psi_wc1 = psi_w[kk1]
s_wc1 = s_w[kk1]

psi_wc2 = psi_w[kk2]
s_wc2 = s_w[kk2]

plt.figure ()
plt.plot(s_w,psi_w,'r-',label='f(s_w)')
#plt.plot(s_w,psi_w_sec_der1,'g-',label='(d(\Psi)/ds_{w})_{s_w}_sec')
#plt.plot(s_w,psi_w_sec_der2,'g-',label='(d(\Psi)/ds_{w})_{s_w}_sec')
plt.plot(s_wc1,psi_wc1,'kX')
plt.plot(s_wc2,psi_wc2,'rX')
plt.plot(s_w,-psi_w_der,'k-',label='(d(\Psi)/ds_{w})_{s_w}') 
#plt.plot(s_w,lambda_w_der,'b-',label='(d \u03BB/ds_{w})_{s_w}') 
plt.xlabel('s_w')
plt.legend()
'''
