#Coding the non-linear double component solution : analytical solution and aux
#Mohammad Afzal Shadab
#Date modified: 06/05/21

# clear all 
from IPython import get_ipython
#get_ipython().magic('reset -sf') #for clearing everything
#get_ipython().run_line_magic('matplotlib', 'qt') #for plotting in separate window

# import python libraries
import numpy as np
import scipy.sparse as sp
import scipy.special as sp_special
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
import matplotlib.animation as animation
from matplotlib.ticker import LinearLocator, FormatStrFormatter
plt.rcParams.update({'font.size': 20})
plt.rcParams.update({'font.family': 'Serif'})
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fsolve

import sys
sys.path.insert(1, '../../solver')

from build_gridfun import build_grid
from build_opsfun2D_optimized import build_ops
from build_bndfun_optimized import build_bnd
from flux_upwindfun2D_optimized import flux_upwind
from solve_lbvpfun_optimized import solve_lbvp
from comp_algebraicmean_optimized import comp_algebraic_mean
from comp_harmonicmean import comp_harmonicmean
import sys
from matplotlib import colors

plt.rcParams.update({'font.family': "Serif"})

brown  = [181/255 , 101/255, 29/255]
red    = [255/255 ,255/255 ,255/255 ]
blue   = [ 30/255 ,144/255 , 255/255 ]
green  = [  0/255 , 166/255 ,  81/255]
orange = [247/255 , 148/255 ,  30/255]
purple = [102/255 ,  45/255 , 145/255]
brown  = [155/255 ,  118/255 ,  83/255]
tan    = [199/255 , 178/255 , 153/255]
gray   = [100/255 , 100/255 , 100/255]

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


class Grid:
    def __init__(self):
        self.xmin = []
        self.xmax = []
        self.Nx   = []

class BC:
    def __init__(self):
        self.dof_dir = []
        self.dof_f_dir = []
        self.g = []
        self.dof_neu = []
        self.dof_f_neu = []
        self.qb = []

def h_w(H,C):
    hw = np.zeros_like(H)  #Region 1: ice + gas
    hw[H>0] = 1.0         #Region 2: ice + water + gas
    hw[H>C] = H[H>C]/C[H>C]#Region 3: water + gas
    return hw

def f_H(H,C,m,n):
    fH = np.zeros_like(H)         #Region 1: ice + gas
    fH[H>0] = (1-C[H>0]+H[H>0])**m * (H[H>0]/(1-C[H>0]+H[H>0]))**n  #Region 2: ice + water + gas
    fH[H>C] = H[H>C]*C[H>C]**(n-1)#Region 3: water + gas
    
    fH[C>=1] = H[C>=1]**m     #Region 4: single component region
    #fH[C>1]  = 0.0       #Region 4: outer region             
    return fH

def f_Hm(H,C,m,n):
    fH = np.zeros_like(H)         #Region 1: ice + gas
    fH[H>0] = (1-C[H>0]+H[H>0])**m  #Region 2: ice + water + gas
    fH[H>C] = 1.0#Region 3: water + gas
    
    fH[C>=1] = 1.0#H[C>=1]**m     #Region 4: single component region
    #fH[C>1]  = 0.0       #Region 4: outer region             
    return fH

def f_Hn(H,C,m,n):
    fH = np.zeros_like(H)         #Region 1: ice + gas
    fH[H>0] = (H[H>0]/(1-C[H>0]+H[H>0]))**n  #Region 2: ice + water + gas
    fH[H>C] = H[H>C]*C[H>C]**(n-1)#Region 3: water + gas
    
    fH[C>=1] = H[C>=1]**m     #Region 4: single component region
    #Sw #if Sw= 1, then  flux goes to zero.
    #fH[C>1]  = 0.0       #Region 4: outer region             
    return fH

def f_C(H,C,m,n):
    fC = np.zeros_like(H)          #Region 1: ice + gas
    fC[H>0] = (1-C[H>0]+H[H>0])**m * (H[H>0]/(1-C[H>0]+H[H>0]))**n        #Region 2: ice + water
    fC[H>C] = C[H>C]**(n)#Region 3: water + gas

    fC[C>=1] = 0.0#H[C==1]**m            #Region 4: single component region
    #fC[C>1]  = 0.0       #Region 4: outer region   
    return fC

def f_Cm(H,C,m,n):
    fC = np.zeros_like(H)          #Region 1: ice + gas
    fC[H>0] = (1-C[H>0]+H[H>0])**m        #Region 2: ice + water
    fC[H>C] = 1.0#Region 3: water + gas

    fC[C==1] = 0.0#H[C==1]**m            #Region 4: single component region
    fC[C>1]  = 0.0       #Region 4: outer region   
    return fC

def f_Cn(H,C,m,n):
    fC = np.zeros_like(H)          #Region 1: ice + gas
    fC[H>0] = (H[H>0]/(1-C[H>0]+H[H>0]))**n        #Region 2: ice + water
    fC[H>C] = C[H>C]**(n)#Region 3: water + gas

    fC[C==1] = 0.0            #Region 4: single component region
    fC[C>1]  = 0.0       #Region 4: outer region   
    return fC


def phi_w(H,C):
    phiw = np.zeros_like(H)#Region 1: all ice
    phiw[H>0] = H[H>0]     #Region 2: ice + water
    phiw[H>=C]= C[H>=C]    #Region 3: all water
    return phiw

def phi_i(H,C):
    phii = np.zeros_like(H)#Region 3: water + gas
    phii[H<=C]= C[H<=C] - H[H<=C] #Region 2: water + ice + gas
    phii[H<=0]= C[H<=0]     #Region 1: ice + water
    return phii

def phi_g(H,C):
    phig = 1 - C
    return phig

def porosity(H,C):
    por = phi_w(H,C) + phi_g(H,C)
    return por

def saturation(H,C):
    Sw = phi_w(H,C)/(1 - phi_i(H,C))
    Sg = phi_g(H,C)/(1 - phi_i(H,C))    
    return Sw, Sg

def T(H, C, Ste, Cpr):
    T = np.zeros_like(H)                
    T[H<0] = H[H<0] / (C[H<0] *Ste * Cpr)        #Region 1: all ice
    T[H>C] = (H[H>C]/C[H>C]-1)/Ste  #Region 2 & 3: ice + water or all water
    return T

def lambda_1(H, C, m, n):
    lambda1 = np.zeros_like(H)                
    lambda1[H>C] = C[H>C]**(n-1)  #Region 3: gas+ water
    return lambda1

def lambda_2(H, C, m, n):
    lambda2 = np.zeros_like(H)                
    lambda2[H>0] = n*H[H>0]**(n-1) * (1-C[H>0]+H[H>0])**(m-n) #Region 2: water + gas +ice
    lambda2[H>C] = n*C[H>C]**(n-1)  #Region 3: gas+ water
    return lambda2

def int_region3_curves_lambda1(u0,H, m, n): # water + gas
    C = u0[0,:]            #calculating the integrating constant
    C = C*np.ones_like(H)  #calculating the integral curves
    return C

def int_region3_curves_lambda2(u0,H, m, n):
    C = u0[0,:]/u0[1,:]  #calculating the integrating constant
    C = C*H              #calculating the integral curves
    return C

def int_curves_lambda1(u0,H, m, n):
    C = u0[1,:]**(n/(m-n))*(u0[0,:]-(1+u0[1,:]))  #calculating the integrating constant
    C = 1 + H + C*H**(-n/(m-n))   #calculating the integral curves
    return C

def int_curves_lambda2(u0,H, m, n):
    C = u0[0,:]-u0[1,:]  #calculating the integrating constant
    C = H + C            #calculating the integral curves
    return C

def plotting(simulation_name,H_plot,C_plot,m,n):
    H = np.linspace(0,1,10000)
    
    norm = colors.Normalize(vmin=0, vmax=1)
    fig = plt.figure(figsize=(10,10), dpi=100)
    
    
    #fancy plots
    light_red  = [1.0,0.5,0.5]
    light_blue = [0.5,0.5,1.0]
    light_black= [0.5,0.5,0.5]

    #Region 2 Three phase region
    C1 = int_curves_lambda2(np.array([[0.05],[0.0]]),H, m, n)   #u0 = [C,H]^T
    C2 = int_curves_lambda2(np.array([[0.2],[0]]),H, m, n) 
    C3 = int_curves_lambda2(np.array([[0.4],[0]]),H, m, n) 
    C4 = int_curves_lambda2(np.array([[0.6],[0]]),H, m, n) 
    C5 = int_curves_lambda2(np.array([[0.8],[0]]),H, m, n) 
    
    C1_2 = int_curves_lambda1(np.array([[0.05],[0.05]]),H, m, n)   #u0 = [C,H]^T
    C2_2 = int_curves_lambda1(np.array([[0.2],[0.2]]),H, m, n)
    C3_2 = int_curves_lambda1(np.array([[0.4],[0.4]]),H, m, n) 
    C4_2 = int_curves_lambda1(np.array([[0.6],[0.6]]),H, m, n) 
    C5_2 = int_curves_lambda1(np.array([[0.8],[0.8]]),H, m, n) 
    
    #removing the curves outside the region
    C1_2[C1_2<H] = np.nan
    C2_2[C2_2<H] = np.nan
    C3_2[C3_2<H] = np.nan
    C4_2[C4_2<H] = np.nan
    C5_2[C5_2<H] = np.nan
    

    plt.plot(C1_2,H,'-',c=light_blue,label=r'Slow path')
    plt.plot(C2_2,H,'-',c=light_blue)
    plt.plot(C3_2,H,'-',c=light_blue)
    plt.plot(C4_2,H,'-',c=light_blue)
    plt.plot(C5_2,H,'-',c=light_blue)
    plt.plot(C1,H,'-',c=light_red,label=r'Fast path')
    plt.plot(C2,H,'-',c=light_red)
    plt.plot(C3,H,'-',c=light_red)
    plt.plot(C4,H,'-',c=light_red)
    plt.plot(C5,H,'-',c=light_red)
    
    '''
    #Region 3 Water and gas
    C1_region3 = int_region3_curves_lambda2(np.array([[0.05],[1.0]]),H, m, n)   #u0 = [C,H]^T
    C2_region3 = int_region3_curves_lambda2(np.array([[0.2],[1.0]]),H, m, n) 
    C3_region3 = int_region3_curves_lambda2(np.array([[0.4],[1.0]]),H, m, n) 
    C4_region3 = int_region3_curves_lambda2(np.array([[0.6],[1.0]]),H, m, n) 
    C5_region3 = int_region3_curves_lambda2(np.array([[0.8],[1.0]]),H, m, n) 
    
    C1_2_region3 = int_region3_curves_lambda1(np.array([[0.05],[1.0]]),H, m, n)   #u0 = [C,H]^T
    C2_2_region3 = int_region3_curves_lambda1(np.array([[0.2],[1.0]]),H, m, n)
    C3_2_region3 = int_region3_curves_lambda1(np.array([[0.4],[1.0]]),H, m, n) 
    C4_2_region3 = int_region3_curves_lambda1(np.array([[0.6],[1.0]]),H, m, n) 
    C5_2_region3 = int_region3_curves_lambda1(np.array([[0.8],[1.0]]),H, m, n) 
    
    #removing the curves outside the region
    C1_region3[C1_region3>H] = np.nan
    C2_region3[C2_region3>H] = np.nan
    C3_region3[C3_region3>H] = np.nan
    C4_region3[C4_region3>H] = np.nan
    C5_region3[C5_region3>H] = np.nan
    C1_2_region3[C1_2_region3>H] = np.nan
    C2_2_region3[C2_2_region3>H] = np.nan
    C3_2_region3[C3_2_region3>H] = np.nan
    C4_2_region3[C4_2_region3>H] = np.nan
    C5_2_region3[C5_2_region3>H] = np.nan
    
    
    C1_region3[(0.366*3.428+1)*C1_region3<H] = np.nan
    C2_region3[(0.366*3.428+1)*C2_region3<H] = np.nan
    C3_region3[(0.366*3.428+1)*C3_region3<H] = np.nan
    C4_region3[(0.366*3.428+1)*C4_region3<H] = np.nan
    C5_region3[(0.366*3.428+1)*C5_region3<H] = np.nan
    C1_2_region3[(0.366*3.428+1)*C1_2_region3<H] = np.nan
    C2_2_region3[(0.366*3.428+1)*C2_2_region3<H] = np.nan
    C3_2_region3[(0.366*3.428+1)*C3_2_region3<H] = np.nan
    C4_2_region3[(0.366*3.428+1)*C4_2_region3<H] = np.nan
    C5_2_region3[(0.366*3.428+1)*C5_2_region3<H] = np.nan    
    

    plt.plot(C1_2_region3,H,c=light_blue)
    plt.plot(C2_2_region3,H,c=light_blue)
    plt.plot(C3_2_region3,H,c=light_blue)
    plt.plot(C4_2_region3,H,c=light_blue)
    plt.plot(C5_2_region3,H,c=light_blue)
    plt.plot(C1_region3,H,c=light_red)
    plt.plot(C2_region3,H,c=light_red)
    plt.plot(C3_region3,H,c=light_red)
    plt.plot(C4_region3,H,c=light_red)
    plt.plot(C5_region3,H,c=light_red)

    plt.plot([1,0],[(0.366*3.428+1),0],'k--')
    
    '''
    
    
    plt.plot([0,1],[0,1],'k--')
    plt.plot([0,1],[0,0],'k--')
    
    
    plt.ylim([-0.2,1])
    plt.xlim([0,1])
    plt.plot(C_plot[0],H_plot[0],'ko', markersize=12)
    plt.plot(C_plot[len(C_plot)-1],H_plot[len(C_plot)-1],'ko', markersize=12)
    plt.plot(C_plot,H_plot,'k-',linewidth=2,label='Path')
    plt.ylabel(r'Dim-less Enthalpy $\mathcal{H}$')
    plt.xlabel(r'Dim-less Composition $\mathcal{C}$')
    #plt.ylabel(r'$\mathcal{H}$')
    #plt.xlabel(r'$\mathcal{C}$')
    plt.legend(loc='upper left',borderaxespad=0.)

    #adding arrow
    x_pos1 = C_plot[int(len(H_plot)/4)-1]
    y_pos1 = H_plot[int(len(H_plot)/4)-1]
    x_direct1 = C_plot[int(len(H_plot)/4)]-C_plot[int(len(H_plot)/4)-1]
    y_direct1 = H_plot[int(len(H_plot)/4)]-H_plot[int(len(H_plot)/4)-1]
    plt.arrow(x_pos1, y_pos1, x_direct1, y_direct1,head_width=0.04, head_length=0.04, fc='k', ec='k')
    
    x_pos2 = C_plot[int(len(H_plot)*2/3)-1]
    y_pos2 = H_plot[int(len(H_plot)*2/3)-1]
    x_direct2 = C_plot[int(len(H_plot)*2/3)+1]-C_plot[int(len(H_plot)*2/3)-1]
    y_direct2 = H_plot[int(len(H_plot)*2/3)+1]-H_plot[int(len(H_plot)*2/3)-1]
    plt.arrow(x_pos2, y_pos2, x_direct2, y_direct2,head_width=0.04, head_length=0.04, fc='k', ec='k')
        

    plt.savefig(f'{simulation_name}_integral_curves.pdf')


def analytical(case_no,etaL,etaR,C_L,H_L,C_R,H_R, m, n):
    
    eta  = np.linspace(etaL,etaR,20000)
    
    if case_no==1:
        print('Case 1: Contact Discontinuity only')
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta)       
        C[eta <0.0] = C_R
        H[eta <0.0] = H_R   
        
        H_plot = np.linspace(H_L,H_R,20000)
        C_plot = int_curves_lambda1(np.array([[C_L],[H_L]]),H_plot, m, n)        
    
    elif case_no==2:
        print('Case 2: Rarefaction only')
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta)             
        
        lambda_2L = lambda_2(np.array([H_L]),np.array([C_L]), m, n)[0] #left state
        lambda_2R = lambda_2(np.array([H_R]),np.array([C_R]), m, n)[0] #right state
        
        #H[eta<lambda_2L] = (H_L**(n-1) + (eta[eta<lambda_2L]-lambda_2L)/(n*(1-C_L+H_L)**(m-n)))**(1/(n-1))
        H[eta>=lambda_2L] = (eta[eta>=lambda_2L]/(n*(1-C_L+H_L)**(m-n)))**(1/(n-1))
        C[eta>=lambda_2L] = C_L - H_L + H[eta>=lambda_2L]
        
        H[eta>=lambda_2R] = H_R
        C[eta>=lambda_2R] = C_R  

        H_plot = np.linspace(H_L,H_R,20000)
        C_plot = int_curves_lambda2(np.array([[C_L],[H_L]]),H_plot, m, n) 

    elif case_no==3:
        print('Case 3: Shock only')
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta)             
        
        s = (f_C(np.array([H_L]),np.array([C_L]),m,n)[0]-f_C(np.array([H_R]),np.array([C_R]),m,n)[0])/(C_L - C_R)#Shock speed
        
        H[eta>=s] = H_R
        C[eta>=s] = C_R
        
        H_plot = np.linspace(H_L,H_R,20000)
        C_plot = int_curves_lambda2(np.array([[C_L],[H_L]]),H_plot, m, n) 
        
    elif case_no==4:
        print('Case 4: 1-Contact Discontinuity, 2-Rarefaction')
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta)             
        
        phi_L = porosity(np.array([H_L]),np.array([C_L]))[0]
        phi_R = porosity(np.array([H_R]),np.array([C_R]))[0]
        
        func = lambda H_I: phi_R - H_I**(-1/(n-1))*H_L**(n/(n-1))*(phi_L/phi_R)**((m-n)/(n-1))+H_I-phi_L*(H_I/H_L)**(-n/(m-n))
        H_I  = fsolve(func,(H_L+H_R)/2)
        C_I  = 1 + H_I - (1 + H_L - C_L)*(H_I/H_L)**(-n/(m-n))

        H[eta>=0] = H_I
        C[eta>=0] = C_I
        
        lambda_2I = lambda_2(np.array([H_I]),np.array([C_I]), m, n)[0] #intermediate state
        lambda_2R = lambda_2(np.array([H_R]),np.array([C_R]), m, n)[0] #right state
        
        H[eta>=lambda_2I] = (eta[eta>=lambda_2I]/(n*(1-C_R+H_R)**(m-n)))**(1/(n-1))
        C[eta>=lambda_2I] = C_R - H_R + H[eta>=lambda_2I]
        
        H[eta>=lambda_2R] = H_R
        C[eta>=lambda_2R] = C_R        

        H_plot1 = np.linspace(H_L,H_I,10000)
        C_plot1 = int_curves_lambda1(np.array([[C_L],[H_L]]),H_plot1, m, n) 
        H_plot2 = np.linspace(H_I,H_R,10000)
        C_plot2 = int_curves_lambda2(np.array([[C_R],[H_R]]),H_plot2, m, n) 
        C_plot  = np.append(C_plot1,C_plot2)
        H_plot  = np.append(H_plot1,H_plot2)

    elif case_no==5:
        print('Case 5: 1-Contact Discontinuity, 2-Shock')
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta)             
        
        H_I = H_L*((1+H_L-C_L)/(1+H_R-C_R))**((m-n)/n)
        C_I = H_I + (C_R-H_R)
        
        H[eta>=0] = H_I
        C[eta>=0] = C_I
        
        s = (f_C(np.array([H_I]),np.array([C_I]),m,n)[0]-f_C(np.array([H_R]),np.array([C_R]),m,n)[0])/(C_I - C_R)#Shock speed
        
        H[eta>=s] = H_R
        C[eta>=s] = C_R
        
        H_plot1 = np.linspace(H_L,H_I,10000)
        C_plot1 = int_curves_lambda1(np.array([[C_L],[H_L]]),H_plot1, m, n) 
        H_plot2 = np.linspace(H_I,H_R,10000)
        C_plot2 = int_curves_lambda2(np.array([[C_R],[H_R]]),H_plot2, m, n) 
        C_plot  = np.append(C_plot1,C_plot2)
        H_plot  = np.append(H_plot1,H_plot2)


    elif case_no==6:
        print('Case 6: Contact Discontinuity only (Region 1: ice + gas)')
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta)       
        C[eta >0.0] = C_R
        H[eta >0.0] = H_R   
        
        H_plot = np.linspace(H_L,H_R,20000)
        C_plot = C_L+ (C_L-C_R)/(H_L-H_R)*(H_plot - H_L)  
        
        
    elif case_no==7:
        print('Case 7: Region 3 (Water + gas) - Contact Discontinuity only')
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta) 
        
        lambda1 = lambda_1(np.array([H_L]),np.array([C_L]), m, n)[0]
        
        C[eta >lambda1] = C_R
        H[eta >lambda1] = H_R   
        
        H_plot = np.linspace(H_L,H_R,20000)
        C_plot = int_region3_curves_lambda1(np.array([[C_L],[H_L]]),H_plot, m, n)        
    
    elif case_no==8:
        print('Case 8: Region 3 (Water + gas) - Rarefaction only')
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta)             
        
        
        lambda_2L = lambda_2(np.array([H_L]),np.array([C_L]), m, n)[0] #left state
        lambda_2R = lambda_2(np.array([H_R]),np.array([C_R]), m, n)[0] #right state
        
        C[eta>=lambda_2L] = (eta[eta>=lambda_2L]/n)**(1/(n-1))
        H[eta>=lambda_2L] = H_R/C_R*C[eta>=lambda_2L]
        
        H[eta>=lambda_2R] = H_R
        C[eta>=lambda_2R] = C_R  

        H_plot = np.linspace(H_L,H_R,20000)
        C_plot = int_region3_curves_lambda2(np.array([[C_L],[H_L]]),H_plot, m, n) 
        
    elif case_no==9:
        print('Case 9: Region 3 (Water + gas) - Shock only')
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta)             
        
        s = (f_C(np.array([H_L]),np.array([C_L]),m,n)[0]-f_C(np.array([H_R]),np.array([C_R]),m,n)[0])/(C_L - C_R)#Shock speed
        
        H[eta>=s] = H_R
        C[eta>=s] = C_R
        
        H_plot = np.linspace(H_L,H_R,20000)
        C_plot = int_region3_curves_lambda2(np.array([[C_L],[H_L]]),H_plot, m, n) 
        
        
    elif case_no==10:
        print('Case 10: Region 3 (Water + gas) - 1-Contact Discontinuity, 2-Rarefaction')
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta)             
        
        C_I  = C_L
        H_I  = H_R*C_I/C_R

        lambda_2L = lambda_1(np.array([H_L]),np.array([C_L]), m, n)[0] #left state
        lambda_2I = lambda_2(np.array([H_I]),np.array([C_I]), m, n)[0] #intermediate state
        lambda_2R = lambda_2(np.array([H_R]),np.array([C_R]), m, n)[0] #right state

        H[eta>=lambda_2L] = H_I
        C[eta>=lambda_2L] = C_I
    
        
        C[eta>=lambda_2I] = (eta[eta>=lambda_2I]/n)**(1/(n-1))
        H[eta>=lambda_2I] = H_R/C_R*C[eta>=lambda_2I]
        
        H[eta>=lambda_2R] = H_R
        C[eta>=lambda_2R] = C_R        

        H_plot1 = np.linspace(H_L,H_I,10000)
        C_plot1 = int_region3_curves_lambda1(np.array([[C_L],[H_L]]),H_plot1, m, n) 
        H_plot2 = np.linspace(H_I,H_R,10000)
        C_plot2 = int_region3_curves_lambda2(np.array([[C_R],[H_R]]),H_plot2, m, n) 
        C_plot  = np.append(C_plot1,C_plot2)
        H_plot  = np.append(H_plot1,H_plot2)

    elif case_no==11:
        print('Case 11: Region 3 (Water + gas) - 1-Contact Discontinuity, 2-Shock')
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta)   

        lambda_2L = lambda_1(np.array([H_L]),np.array([C_L]), m, n)[0] #left state

        s = (f_C(np.array([H_L]),np.array([C_L]),m,n)[0]-f_C(np.array([H_R]),np.array([C_R]),m,n)[0])/(C_L - C_R)#Shock speed
        
        C_I  = C_L
        H_I  = H_R*(C_R**(n-1)-s)/(C_I**(n-1)-s)

        H[eta>=lambda_2L] = H_I
        C[eta>=lambda_2L] = C_I

        H[eta>=s] = H_R
        C[eta>=s] = C_R
        
        H_plot1 = np.linspace(H_L,H_I,10000)
        C_plot1 = int_region3_curves_lambda1(np.array([[C_L],[H_L]]),H_plot1, m, n) 
        H_plot2 = np.linspace(H_I,H_R,10000)
        C_plot2 = int_region3_curves_lambda2(np.array([[C_R],[H_R]]),H_plot2, m, n) 
        C_plot  = np.append(C_plot1,C_plot2)
        H_plot  = np.append(H_plot1,H_plot2)
        
    elif case_no==12:
        print('Case 12: Contact Discontinuity + Rarefaction (Region 1 to 2)')
        
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta)             
        
        H_I  = 0.0
        C_I  = H_I + C_R - H_R

        H[eta>=0] = H_I
        C[eta>=0] = C_I
        
        lambda_2I = lambda_2(np.array([H_I]),np.array([C_I]), m, n)[0] #intermediate state
        lambda_2R = lambda_2(np.array([H_R]),np.array([C_R]), m, n)[0] #right state
        
        H[eta>=lambda_2I] = (eta[eta>=lambda_2I]/(n*(1-C_R+H_R)**(m-n)))**(1/(n-1))
        C[eta>=lambda_2I] = C_R - H_R + H[eta>=lambda_2I]
        
        H[eta>=lambda_2R] = H_R
        C[eta>=lambda_2R] = C_R        

        H_plot1 = np.linspace(H_L,H_I,10000)
        C_plot1 = C_L+ (C_L-C_I)/(H_L-H_I)*(H_plot1 - H_L) 
        H_plot2 = np.linspace(H_I,H_R,10000)
        C_plot2 = int_curves_lambda2(np.array([[C_R],[H_R]]),H_plot2, m, n) 
        C_plot  = np.append(C_plot1,C_plot2)
        H_plot  = np.append(H_plot1,H_plot2)

    elif case_no==13:
        print('Case 13: Mixed R1 to R2: Single Shock only')
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta)             
        
        s = (f_C(np.array([H_L]),np.array([C_L]),m,n)[0]-f_C(np.array([H_R]),np.array([C_R]),m,n)[0])/(C_L - C_R)#Shock speed
        
        H[eta>=s] = H_R
        C[eta>=s] = C_R
        
        H_I  = 0.0
        C_I  = H_I + C_L - H_L

        H_plot1 = np.linspace(H_L,H_I,10000)
        C_plot1 = int_curves_lambda2(np.array([[C_L],[H_L]]),H_plot1, m, n) 
        H_plot2 = np.linspace(H_I,H_R,10000)
        C_plot2 = C_R+ (C_R-C_I)/(H_R-H_I)*(H_plot2 - H_R) 
        C_plot  = np.append(C_plot1,C_plot2)
        H_plot  = np.append(H_plot1,H_plot2)
        
    elif case_no==14:
        print('Case 14: Contact Discontinuity + Shock (Region 2 to 1)')
        
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta)             

        H_I = H_L*((1+H_L-C_L)/(1+H_R-C_R))**((m-n)/n)
        C_I = H_I + (C_R-H_R)

        H[eta>=0] = H_I
        C[eta>=0] = C_I
        
        s = (f_C(np.array([H_I]),np.array([C_I]),m,n)[0]-f_C(np.array([H_R]),np.array([C_R]),m,n)[0])/(C_I - C_R)#Shock speed
        
        H[eta>=s] = H_R
        C[eta>=s] = C_R        

        H_plot1 = np.linspace(H_L,H_I,10000)
        C_plot1 = int_curves_lambda1(np.array([[C_L],[H_L]]),H_plot1, m, n) 
        H_plot2 = np.linspace(H_I,H_R,10000)
        C_plot2 = C_R+ (C_R-C_I)/(H_R-H_I)*(H_plot2 - H_R) 
        C_plot  = np.append(C_plot1,C_plot2)
        H_plot  = np.append(H_plot1,H_plot2)
    
    elif case_no==15:
        print('Case 15: Contact Discontinuity + Shock (Region 1 to 2)')
        
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta)             

        C_I = 1.0
        H_I = H_L**(n/m)*(1-C_L+H_L)**((m-n)/m)

        H[eta>=0] = H_I
        C[eta>=0] = C_I
        
        s = (f_H(np.array([H_I]),np.array([C_I]),m,n)[0]-f_H(np.array([H_R]),np.array([C_R]),m,n)[0])/(H_I - H_R)#Shock speed
        
        H[eta>=s] = H_R
        C[eta>=s] = C_R        

        H_plot1 = np.linspace(H_L,H_I,10000)
        C_plot1 = int_curves_lambda1(np.array([[C_L],[H_L]]),H_plot1, m, n) 
        H_plot2 = np.linspace(H_I,H_R,10000)
        C_plot2 = C_R+ (C_R-C_I)/(H_R-H_I)*(H_plot2 - H_R) 
        C_plot  = np.append(C_plot1,C_plot2)
        H_plot  = np.append(H_plot1,H_plot2)
        
        ''' 
    elif case_no==16:
        print('Case 16: Shock (Region 1 to 2) Single component case')
    
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta)             
        
        s = (f_H(np.array([H_L]),np.array([C_L]),m,n)[0]-f_H(np.array([H_R]),np.array([C_R]),m,n)[0])/(H_L - H_R)#Shock speed
        
        H[eta>=s] = H_R
        C[eta>=s] = C_R
        
        H_plot = np.linspace(H_L,H_R,20000)
        C_plot = C_R + (C_R-C_L)/(H_R-H_L)*(H_plot - H_R) 
        '''
    
    elif case_no==18:
        print('Case 18: Mixed Region 2 to 3 - 1,2-Contact Discontinuity, 3-Rarefaction')
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta)   

        lambda_2L = lambda_1(np.array([H_L]),np.array([C_L]), m, n)[0] #left state
        lambda_2R = lambda_2(np.array([H_R]),np.array([C_R]), m, n)[0] #right state
        
        H_I  = H_L*(1-C_L+H_L)**((m-n)/n)
        C_I  = H_I

        H[eta>=lambda_2L] = H_I
        C[eta>=lambda_2L] = C_I   
   
        C_II  = C_I
        H_II  = H_R/C_R*C_II

        lambda_1I = lambda_1(np.array([H_II]),np.array([C_II]), m, n)[0] #intermediate state     

        H[eta>=lambda_1I] = H_II
        C[eta>=lambda_1I] = C_II
        
        lambda_2II = lambda_2(np.array([H_II]),np.array([C_II]), m, n)[0] #intermediate state        
   
        C[eta>=lambda_2II] = (eta[eta>=lambda_2II]/(n))**(1/(n-1))
        H[eta>=lambda_2II] = H_R/C_R*C[eta>=lambda_2II]
    
    
        H[eta>=lambda_2R] = H_R
        C[eta>=lambda_2R] = C_R
        
        H_plot1 = np.linspace(H_L,H_I,8000)
        C_plot1 = int_curves_lambda1(np.array([[C_L],[H_L]]),H_plot1, m, n) 
        H_plot2 = np.linspace(H_I,H_II,8000)
        C_plot2 = int_region3_curves_lambda1(np.array([[C_I],[H_I]]),H_plot2, m, n) 
        H_plot3 = np.linspace(H_II,H_R,8000)
        C_plot3 = int_region3_curves_lambda2(np.array([[C_R],[H_R]]),H_plot3, m, n) 
        C_plot  = np.append(C_plot1,C_plot2)
        C_plot  = np.append(C_plot,C_plot3)
        H_plot  = np.append(H_plot1,H_plot2)
        H_plot  = np.append(H_plot,H_plot3)
        
    elif case_no==19:
        print('Case 19: Mixed Region 2 to 3 - 1,2-Contact Discontinuity, 3-Shock')
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta)   

        lambda_2L = lambda_1(np.array([H_L]),np.array([C_L]), m, n)[0] #left state
        lambda_2R = lambda_2(np.array([H_R]),np.array([C_R]), m, n)[0] #right state
        
        H_I  = H_L*(1-C_L+H_L)**((m-n)/n)
        C_I  = H_I

        H[eta>=lambda_2L] = H_I
        C[eta>=lambda_2L] = C_I   
   
        C_II  = C_I
        H_II  = H_R/C_R*C_II

        lambda_1I = lambda_1(np.array([H_II]),np.array([C_II]), m, n)[0] #intermediate state     

        H[eta>=lambda_1I] = H_II
        C[eta>=lambda_1I] = C_II
        
        s   =   (C_R**n - C_II**n) / (C_R - C_II)    
    
    
        H[eta>=s] = H_R
        C[eta>=s] = C_R
        
        H_plot1 = np.linspace(H_L,H_I,8000)
        C_plot1 = int_curves_lambda1(np.array([[C_L],[H_L]]),H_plot1, m, n) 
        H_plot2 = np.linspace(H_I,H_II,8000)
        C_plot2 = int_region3_curves_lambda1(np.array([[C_I],[H_I]]),H_plot2, m, n) 
        H_plot3 = np.linspace(H_II,H_R,8000)
        C_plot3 = int_region3_curves_lambda2(np.array([[C_R],[H_R]]),H_plot3, m, n) 
        C_plot  = np.append(C_plot1,C_plot2)
        C_plot  = np.append(C_plot,C_plot3)
        H_plot  = np.append(H_plot1,H_plot2)
        H_plot  = np.append(H_plot,H_plot3)
        
        
    elif case_no==22:
        print('Case 22: Mixed Region 3 to 2 - 1-Shock, 2-Rarefaction')
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta)   

        C_I  = lambda H_I: C_R + (H_I-H_R)
        func = lambda H_I: (H_L*C_L**(n-1)-(1-C_I(H_I)+H_I)**(m-n)*H_I**n)*(C_L - C_I(H_I)) - (H_L - H_I)*(C_L**n -(1-C_I(H_I)+H_I)**(m-n)*H_I**n)
        #func = lambda H_I: (H_L*C_L**(n-1)-(1-C_R+H_R)**(m-n)*H_I**n)*(C_L - C_R + (H_I-H_R)) - (H_L - H_I)*(C_L**n -(1-C_R+H_R)**(m-n)*H_I**n)
        
        H_I = fsolve(func,(H_L+H_R)/2)[0]
        C_I = C_I(H_I)
        
        s = (H_L*C_L**(n-1)-(1-C_I+H_I)**(m-n)*H_I**n)/(H_L - H_I)

        H[eta>=s] = H_I
        C[eta>=s] = C_I   
   
        lambda_2I = lambda_2(np.array([H_I]),np.array([C_I]), m, n)[0] #intermediate state     
        lambda_2R = lambda_2(np.array([H_R]),np.array([C_R]), m, n)[0] #right state
        

        H[eta>=lambda_2I] = (eta[eta>=lambda_2I]/(n*(1-C_R+H_R)**(m-n)))**(1/(n-1))
        C[eta>=lambda_2I] = C_R - H_R + H[eta>=lambda_2I]
        
        H[eta>=lambda_2R] = H_R
        C[eta>=lambda_2R] = C_R        

        H_plot1 = np.linspace(H_L,H_I,10000)
        C_plot1 = C_L+ (C_L-C_I)/(H_L-H_I)*(H_plot1 - H_L) 
        H_plot2 = np.linspace(H_I,H_R,10000)
        C_plot2 = int_curves_lambda2(np.array([[C_R],[H_R]]),H_plot2, m, n) 
        C_plot  = np.append(C_plot1,C_plot2)
        H_plot  = np.append(H_plot1,H_plot2)
        
    elif case_no==23:
        print('Case 23: Mixed Region 3 to 2 - 1-Shock, 2-Shock')
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta)   

        C_I  = lambda H_I: C_R + (H_I-H_R)
        func = lambda H_I: (H_L*C_L**(n-1)-(1-C_I(H_I)+H_I)**(m-n)*H_I**n)*(C_L - C_I(H_I)) - (H_L - H_I)*(C_L**n -(1-C_I(H_I)+H_I)**(m-n)*H_I**n)
        #func = lambda H_I: (H_L*C_L**(n-1)-(1-C_R+H_R)**(m-n)*H_I**n)*(C_L - C_R + (H_I-H_R)) - (H_L - H_I)*(C_L**n -(1-C_R+H_R)**(m-n)*H_I**n)
        
        H_I = fsolve(func,(H_L+H_R)/2)[0]
        C_I = C_I(H_I)
        
        s = (H_L*C_L**(n-1)-(1-C_I+H_I)**(m-n)*H_I**n)/(H_L - H_I)

        H[eta>=s] = H_I
        C[eta>=s] = C_I   
   
        lambda_2I = lambda_2(np.array([H_I]),np.array([C_I]), m, n)[0] #intermediate state     
        lambda_2R = lambda_2(np.array([H_R]),np.array([C_R]), m, n)[0] #right state
        

        s2 = (f_H(np.array([H_I]),np.array([C_I]),m,n)[0]-f_H(np.array([H_R]),np.array([C_R]),m,n)[0])/(H_I - H_R)#Shock speed
        

        H[eta>=s2] = H_R
        C[eta>=s2] = C_R        

        H_plot1 = np.linspace(H_L,H_I,10000)
        C_plot1 = C_L+ (C_L-C_I)/(H_L-H_I)*(H_plot1 - H_L) 
        H_plot2 = np.linspace(H_I,H_R,10000)
        C_plot2 = int_curves_lambda2(np.array([[C_R],[H_R]]),H_plot2, m, n) 
        C_plot  = np.append(C_plot1,C_plot2)
        H_plot  = np.append(H_plot1,H_plot2)
        
        
    elif case_no==24:
        print('Case 24: Mixed Region 1 to 3 - 1-Contact, 2-Rarefaction')
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta)   


        C_I  = 0.0
        H_I  = 0.0

        lambda_2I = lambda_2(np.array([H_I]),np.array([C_I]), m, n)[0] #intermediate state
        lambda_2R = lambda_2(np.array([H_R]),np.array([C_R]), m, n)[0] #right state

        H[eta>=0] = H_I
        C[eta>=0] = C_I
    
        
        C[eta>=lambda_2I] = (eta[eta>=lambda_2I]/n)**(1/(n-1))
        H[eta>=lambda_2I] = H_R/C_R*C[eta>=lambda_2I]
        
        H[eta>=lambda_2R] = H_R
        C[eta>=lambda_2R] = C_R        

        H_plot1 = np.linspace(H_L,H_I,10000)
        C_plot1 = C_L+ (C_L-C_I)/(H_L-H_I)*(H_plot1 - H_L) 
        H_plot2 = np.linspace(H_I,H_R,10000)
        C_plot2 = int_region3_curves_lambda2(np.array([[C_R],[H_R]]),H_plot2, m, n) 
        C_plot  = np.append(C_plot1,C_plot2)
        H_plot  = np.append(H_plot1,H_plot2)
 
    elif case_no==29:
        print('Case 29: Mixed Region 3 to 1 - 1-Shock, 2-Shock')
        C = C_L*np.ones_like(eta)  
        H = H_L*np.ones_like(eta)   

        C_I  = lambda H_I: C_R + (H_I-H_R)
        func = lambda H_I: (H_L*C_L**(n-1)-(1-C_I(H_I)+H_I)**(m-n)*H_I**n)*(C_L - C_I(H_I)) - (H_L - H_I)*(C_L**n -(1-C_I(H_I)+H_I)**(m-n)*H_I**n)
        #func = lambda H_I: (H_L*C_L**(n-1)-(1-C_R+H_R)**(m-n)*H_I**n)*(C_L - C_R + (H_I-H_R)) - (H_L - H_I)*(C_L**n -(1-C_R+H_R)**(m-n)*H_I**n)
        
        H_I = fsolve(func,(H_L+H_R)/2)[0]
        C_I = C_I(H_I)
        
        s = (H_L*C_L**(n-1)-(1-C_I+H_I)**(m-n)*H_I**n)/(H_L - H_I)

        H[eta>=s] = H_I
        C[eta>=s] = C_I   
   
        lambda_2I = lambda_2(np.array([H_I]),np.array([C_I]), m, n)[0] #intermediate state     
        lambda_2R = lambda_2(np.array([H_R]),np.array([C_R]), m, n)[0] #right state
        

        s2 = (f_H(np.array([H_I]),np.array([C_I]),m,n)[0]-f_H(np.array([H_R]),np.array([C_R]),m,n)[0])/(H_I - H_R)#Shock speed
        

        H[eta>=s2] = H_R
        C[eta>=s2] = C_R        

        H_plot1 = np.linspace(H_L,H_I,10000)
        C_plot1 = C_L+ (C_L-C_I)/(H_L-H_I)*(H_plot1 - H_L) 
        H_plot2 = np.linspace(H_I,H_R,10000)
        C_plot2 = C_R+ (C_R-C_I)/(H_R-H_I)*(H_plot2 - H_R) 
        C_plot  = np.append(C_plot1,C_plot2)
        H_plot  = np.append(H_plot1,H_plot2)
        
        

    else: #Not plotting
        C = np.nan*np.ones_like(eta)
        H = np.nan*np.ones_like(eta)
        C_plot = np.nan*np.ones_like(eta)
        H_plot = np.nan*np.ones_like(eta)
                
    return eta,C,H,C_plot,H_plot
