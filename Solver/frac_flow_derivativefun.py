#computes the fractional flow derivative

import numpy as np
import matplotlib.pyplot as plt
from mobilityfun import mobility

def frac_flow_derivative(s_w,s_wp,s_nwp,mu_w,mu_nw,k_w0,k_nw0,n_w,n_nw): 
    # author: Mohammad Afzal Shadab
    # date: 04/11/2020
    # description:
    # This function computes the fractional flow derivative 
    # 
    # Output:
    # (df/ds_w)_{s_w}

    ds_w = 0.0001
    
    if s_w + ds_w <= 1.0:
    
        [s_wc1, kr_w1, kr_nw1, lambda_w1, lambda_nw1,lambda_t1, f_w1, f_nw1] = mobility(s_w + ds_w,s_wp,s_nwp,mu_w,mu_nw,k_w0,k_nw0,n_w,n_nw)   
        
        [s_wc2, kr_w2, kr_nw2, lambda_w2, lambda_nw2,lambda_t2, f_w2, f_nw2] = mobility(s_w,s_wp,s_nwp,mu_w,mu_nw,k_w0,k_nw0,n_w,n_nw)   
    
        frac_flow1 = lambda_w1 / (lambda_w1 + lambda_nw1) #actual fractional flow
        frac_flow2 = lambda_w2 / (lambda_w2 + lambda_nw2) #fractional flow at s_w+ds_w
    
        frac_flow_der = ( frac_flow1 - frac_flow2 ) / ds_w
        lambda_w_der  = ( lambda_w1 - lambda_w2 ) / ds_w
        lambda_nw_der = ( lambda_nw1 - lambda_nw2 ) / ds_w        
        
    else:
        
        [s_wc1, kr_w1, kr_nw1, lambda_w1, lambda_nw1,lambda_t1, f_w1, f_nw1] = mobility(s_w - ds_w,s_wp,s_nwp,mu_w,mu_nw,k_w0,k_nw0,n_w,n_nw)   
        
        [s_wc2, kr_w2, kr_nw2, lambda_w2, lambda_nw2,lambda_t2, f_w2, f_nw2] = mobility(s_w,s_wp,s_nwp,mu_w,mu_nw,k_w0,k_nw0,n_w,n_nw)   
    
        frac_flow1 = lambda_w1 / (lambda_w1 + lambda_nw1) #actual fractional flow
        frac_flow2 = lambda_w2 / (lambda_w2 + lambda_nw2) #fractional flow at s_w+ds_w
    
        frac_flow_der = ( frac_flow1 - frac_flow2 ) / (-ds_w)        
        lambda_w_der  = ( lambda_w1 - lambda_w2 ) / (-ds_w) 
        lambda_nw_der = ( lambda_nw1 - lambda_nw2 ) / (-ds_w) 
        
    return frac_flow_der, frac_flow2,lambda_w_der,lambda_nw_der;
'''
#fluids
s_wp  = 0.0  #percolation threshold: wetting phase
s_nwp = 0.0  #percolation threshold: non-wetting phase
mu_w  = 1.0  #dynamic viscosity: wetting phase    
mu_nw = 1.0  #dynamic viscosity: non-wetting phase   
k_w0  = 1.0  #relative permeability threshold: wetting phase   
k_nw0 = 1.0  #relative permeability threshold: non-wetting phase   
n_w   = 3.0  #power law coefficient: wetting phase  
n_nw  = 3.0  #power law coefficient: non-wetting phase 

s_w = np.linspace(0,1,10001)
frac_flow_der = np.zeros((len(s_w),1))
frac_flow_w    = np.zeros((len(s_w),1))
lambda_w_der  = np.zeros((len(s_w),1))
lambda_nw_der  = np.zeros((len(s_w),1))

for j in range(0, len(s_w)):
    [frac_flow_der[j], frac_flow_w[j], lambda_w_der[j], lambda_nw_der[j]] = frac_flow_derivative(s_w[j],s_wp,s_nwp,mu_w,mu_nw,k_w0,k_nw0,n_w,n_nw)


#secant method
frac_flow_w_sec_der = frac_flow_w/np.transpose([s_w-s_wp])  

kk = 1000 + np.argmin(np.abs(frac_flow_der[1000:,:]-frac_flow_w_sec_der[1000:,:])) #to find the location of shock

f_wf = frac_flow_w[kk]
s_wf = frac_flow_w[kk]

plt.figure ()
plt.plot(s_w,frac_flow_w,'r-',label='f(s_w)')
plt.plot(s_w,frac_flow_w_sec_der,'g-',label='(df/ds_{w})_{s_w}_sec')
plt.plot(s_w,frac_flow_der,'k-',label='(df/ds_{w})_{s_w}') 
plt.plot(s_w,lambda_w_der,'b-',label='(d \u03BB/ds_{w})_{s_w}') 
plt.xlabel('s_w')
plt.legend()
'''