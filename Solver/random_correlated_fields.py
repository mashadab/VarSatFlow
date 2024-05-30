#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 20:34:15 2022

@author: afzal-admin
"""

import numpy as np
import matplotlib.pyplot as plt 
import sys
from matplotlib import cm

sys.path.insert(1, '../../solver')
from build_gridfun2D import build_grid 



def GeneratePermField(Grid,corr_length,amp,Kmean,type_,rng_state):
    
    [Xc,Yc] = np.meshgrid(Grid.xc,Grid.yc)
    
    if type_ =='exp':
        Xc_col = np.reshape(np.transpose(Xc), (Grid.N,-1))
        Yc_col = np.reshape(np.transpose(Yc), (Grid.N,-1))           

        sig = -np.log(0.1)/corr_length
    else:
        print('Wrong covariance model!')
    Cov = np.zeros((Grid.N,Grid.N))
    
    
    for i in range(0,Grid.N):
        dist = np.sqrt((Xc_col[i,0]-Xc_col)**2 + (Yc_col[i-0]-Yc_col)**2)
        
        if type_== 'exp':
            Cov[i,:] = (np.exp(-(sig * dist)))[:,0]
        else:
            print('Wrong covariance model!')           
    np.random.seed(rng_state)
     
    #Cov = L*L' <=> L ~ sqrt(Cov) 
    #Cholesky factorization is equivalent to square root of a matrix
     
    L     = np.linalg.cholesky(Cov)
    print(L,Cov, np.shape(L), np.shape(Cov))
    print(np.random.randn(Grid.N,1),(L @ np.random.randn(Grid.N,1)))
    Kpert = np.transpose((L @ np.random.randn(Grid.N,1)).reshape(Grid.Nx,Grid.Ny))
    K     = Kmean + Kpert
    
    return K, Xc, Yc



def GeneratePermField2D(Grid,corr_length1,corr_length2,amp,Kmean,type_,rng_state):
    
    [Xc,Yc] = np.meshgrid(Grid.xc,Grid.yc)
    
    if type_ =='exp':
        Xc_col = np.reshape(np.transpose(Xc), (Grid.N,-1))
        Yc_col = np.reshape(np.transpose(Yc), (Grid.N,-1))           

        sig1 = -np.log(0.1)/corr_length1
        sig2 = -np.log(0.1)/corr_length2
    else:
        print('Wrong covariance model!')
    Cov = np.zeros((Grid.N,Grid.N))
    
    for i in range(0,Grid.N):
        dist = np.sqrt(sig1**2 * (Xc_col[i,0]-Xc_col)**2 + sig2**2 *(Yc_col[i,0]-Yc_col)**2)
        
        if type_== 'exp':
            Cov[i,:] = (np.exp(-(dist)))[:,0]
        else:
            print('Wrong covariance model!')           
    np.random.seed(rng_state)
     
    #Cov = L*L' <=> L ~ sqrt(Cov) 
    #Cholesky factorization is equivalent to square root of a matrix
     
    L     = np.linalg.cholesky(Cov)
    #print(L,Cov, np.shape(L), np.shape(Cov))
    #print(np.random.randn(Grid.N,1),(L @ np.random.randn(Grid.N,1)))
    Kpert = np.transpose((L @ np.random.randn(Grid.N,1)).reshape(Grid.Nx,Grid.Ny))
    K     = Kmean + Kpert
    
    return K, Xc, Yc

def GeneratePermField2D_papers(Grid,corr_fluc1,corr_fluc2,amp,Kmean,type_,rng_state):
    
    [Xc,Yc] = np.meshgrid(Grid.xc,Grid.yc)
    
    if type_ =='exp':
        Xc_col = np.reshape(np.transpose(Xc), (Grid.N,-1))
        Yc_col = np.reshape(np.transpose(Yc), (Grid.N,-1))           

    else:
        print('Wrong covariance model!')
    Cov = np.zeros((Grid.N,Grid.N))
    
    for i in range(0,Grid.N):
        dist = np.sqrt((Xc_col[i,0]-Xc_col)**2/corr_fluc1**2 + (Yc_col[i,0]-Yc_col)**2/corr_fluc2**2)
        
        if type_== 'exp':
            Cov[i,:] = (np.exp(-2*dist))[:,0]
        else:
            print('Wrong covariance model!')           
    np.random.seed(rng_state)
     
    #Cov = L*L' <=> L ~ sqrt(Cov) 
    #Cholesky factorization is equivalent to square root of a matrix
     
    L     = np.linalg.cholesky(Cov)
    Kpert = np.transpose((L @ np.random.randn(Grid.N,1)).reshape(Grid.Nx,Grid.Ny))
    K     = Kmean + Kpert
    
    return K, Xc, Yc

'''

class Grid:
    def __init__(self):
        self.xmin = 0

#Marc's permeability field ###########
Grid.xmin = 0; Grid.xmax= 2; Grid.Nx = 150
Grid.ymin = 0; Grid.ymax= 1; Grid.Ny = 75

Grid = build_grid(Grid)

corr_length = 0.01; phi_mean = 0; amplitude = 1;s = 25061977

[phi_array,Xc,Yc] = GeneratePermField2D(Grid,corr_length,corr_length,amplitude,phi_mean,'exp',s)

phi_array = 10**(phi_array);
phi = np.reshape(np.transpose(phi_array), (Grid.N,-1))
#phi = phi_L*np.ones((Grid.N,1))

fig = plt.figure(figsize=(15,7.5) , dpi=100)
plot = [plt.contourf(Xc, Yc, np.log10(phi_array),cmap="coolwarm")]
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.xlabel(r'$x$')
plt.ylabel(r'$z$')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.axis('scaled')
mm = plt.cm.ScalarMappable(cmap=cm.coolwarm)
mm.set_array(np.log10(phi))
clb = plt.colorbar(mm, pad=0.1)
clb.set_label(r'$\log_{10}(K) $', labelpad=1, y=1.075, rotation=0)
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
plt.savefig(f'../Figures/correlated_fields_{Grid.Nx}by{Grid.Ny}_Marcs-logK_corr{corr_length}.pdf',bbox_inches='tight', dpi = 600)


#Combined
#Multiple_time_plots
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True,figsize=(20,20), constrained_layout=True)



plt.subplot(2,2,1)
corr_length = 0.01; phi_mean = 0; amplitude = 1;s = 25061977

[phi_array,Xc,Yc] = GeneratePermField2D(Grid,corr_length,corr_length,amplitude,phi_mean,'exp',s)

phi_array = 10**(phi_array);
phi = np.reshape(np.transpose(phi_array), (Grid.N,-1))
#phi = phi_L*np.ones((Grid.N,1))

mm = plt.cm.ScalarMappable(cmap=cm.coolwarm)
mm.set_array(np.linspace(-4,4,100))
cbar = fig.colorbar(mm, ax=axes[:, 1])
cbar.set_label(r'$\log_{10}(\phi) $', labelpad=1, y=1.075, rotation=0)

plot = [plt.contourf(Xc, Yc, np.log10(phi_array),cmap="coolwarm",vmin = -4, vmax = 4)]
plt.title(f'C.L.={corr_length}')
plt.ylabel(r'$z$')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.axis('scaled')


plt.subplot(2,2,2)
corr_length = 0.1; phi_mean = 0; amplitude = 1;s = 25061977

[phi_array,Xc,Yc] = GeneratePermField2D(Grid,corr_length,corr_length,amplitude,phi_mean,'exp',s)

phi_array = 10**(phi_array);
phi = np.reshape(np.transpose(phi_array), (Grid.N,-1))
#phi = phi_L*np.ones((Grid.N,1))

plot = [plt.contourf(Xc, Yc, np.log10(phi_array),cmap="coolwarm",vmin = -4, vmax = 4)]
plt.title(f'C.L.={corr_length}')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.axis('scaled')

plt.subplot(2,2,3)
corr_length = 1; phi_mean = 0; amplitude = 1;s = 25061977

[phi_array,Xc,Yc] = GeneratePermField2D(Grid,corr_length,corr_length,amplitude,phi_mean,'exp',s)

phi_array = 10**(phi_array);
phi = np.reshape(np.transpose(phi_array), (Grid.N,-1))
#phi = phi_L*np.ones((Grid.N,1))

plot = [plt.contourf(Xc, Yc, np.log10(phi_array),cmap="coolwarm",vmin = -4, vmax = 4)]
plt.title(f'C.L.={corr_length}')
plt.ylabel(r'$z$')
plt.xlabel(r'$x$')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.axis('scaled')

plt.subplot(2,2,4)
corr_length = 10; phi_mean = 0; amplitude = 1;s = 25061977

[phi_array,Xc,Yc] = GeneratePermField2D(Grid,corr_length,corr_length,amplitude,phi_mean,'exp',s)

phi_array = 10**(phi_array);
phi = np.reshape(np.transpose(phi_array), (Grid.N,-1))
#phi = phi_L*np.ones((Grid.N,1))

plot = [plt.contourf(Xc, Yc, np.log10(phi_array),cmap="coolwarm",vmin = -4, vmax = 4)]
plt.title(f'C.L.={corr_length}')
plt.xlabel(r'$x$')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.savefig(f'../Figures/correlation-length_combined.pdf',bbox_inches='tight', dpi = 600)

'''