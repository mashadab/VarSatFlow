# import python libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse

# import personal libraries
import build_gridfun
import build_opsfun
from build_bndfun import build_bnd
from mobilityfun import mobility

# author: Mohammad Afzal Shadab
# date: 2/25/2020
# Description
# Computes the solution $u$ to the linear differential problem given by
#
# $$\mathcal{L}(u)=f \quad x\in \Omega $$
#
# with boundary conditions
#
# $$\mathcal{B}(u)=g \quad x\in\partial\Omega$$.
#
# Input:
# L = matrix representing the discretized linear operator of size N by N, 
#     where N is the number of degrees of fredom
# f = column vector representing the discretized r.h.s. and contributions
#     due non-homogeneous Neumann BC's of size N by 1
# B = matrix representing the constraints arising from Dirichlet BC's of
#     size Nc by N
# g = column vector representing the non-homogeneous Dirichlet BC's of size
#     Nc by 1.
# N = matrix representing a orthonormal basis for the null-space of B and
#     of size N by (N-Nc).
# Output:
# u = column vector of the solution of size N by 1

class Grid:
    def __init__(self):
        self.xmin = 0.0
        self.xmax = 1.0
        self.Nx = 100

class Param:
    def __init__(self):
        self.dof_dir = []       # identify cells on Dirichlet bnd
        self.dof_f_dir = []     # identify faces on Dirichlet bnd
        self.dof_neu = []       # identify cells on Neumann bnd
        self.dof_f_neu = []     # identify faces on Neumann bnd
        self.g = []             # column vector of non-homogeneous Dirichlet BCs (Nc X 1)
        self.qb = []            

grid = Grid()

build_gridfun.build_grid(grid)

[D,G,I]=build_opsfun.build_ops(grid)

#applying boundary condition
Param.dof_dir   = np.array([grid.dof_xmin])     # identify cells on Dirichlet bnd
Param.dof_f_dir = np.array([grid.dof_f_xmin])   # identify faces on Dirichlet bnd
Param.dof_neu   = np.array([grid.dof_xmax])     # identify cells on Neumann bnd
Param.dof_f_neu = np.array([grid.dof_f_xmax])   # identify faces on Neumann bnd
Param.qb = np.array([0.0])                      # set flux at Neumann bnd
Param.g  = np.array([0.0])                      # set head at Dirichlet bnd
[B,N,fn] = build_bnd(Param,grid,I)              # Build constraint matrix and basis for its nullspace
#fs = np.zeros([grid.N,1])                      # r.h.s. (zero)
#L = -np.mat(D)*np.mat(G)                       # Laplacian

#parameters
#problem
rho_w  = 1.0  #density of non-wetting phase
rho_nw = 0.980 #density of the fluid at botttom
grav = 9.801 
K = 1.0 

#fluids
s_wp  = 0.20 #percolation threshold: wetting phase
s_nwp = 0.30 #percolation threshold: non-wetting phase
mu_w  = 1.0  #dynamic viscosity: wetting phase    
mu_nw = 1.0  #dynamic viscosity: non-wetting phase   
k_w0  = 1.0  #relative permeability threshold: wetting phase   
k_nw0 = 0.60 #relative permeability threshold: non-wetting phase   
n_w   = 2.0  #power law coefficient: wetting phase  
n_nw  = 2.0  #power law coefficient: non-wetting phase 

#initial condition
if rho_w > rho_nw: #upper half is the wetting phase
    s_w = np.ones((grid.N,1))
    s_w[:(int)((grid.N)/2),:] = 0.0 #non-wetting phase
else: #lower half is the wetting phase
    s_w = np.zeros((grid.N,1))
    s_w[:(int)((grid.N)/2),:] = 1.0 #wetting phase    

#initializing arrays for fluid parameters
mobility_data = np.array([], dtype=np.float64)
mobility_data = mobility_data.reshape(0,8)
s_wc = np.zeros((grid.N,1))
kr_w = np.zeros((grid.N,1))
kr_nw =  np.zeros((grid.N,1))
lambda_w =  np.zeros((grid.N,1)) 
lambda_nw = np.zeros((grid.N,1))
lambda_t =  np.zeros((grid.N,1))
f_w =  np.zeros((grid.N,1))
f_nw =   np.zeros((grid.N,1))

#calculating fluid parameters
for i in range(0, grid.N):
    [s_wc[i], kr_w[i], kr_nw[i], lambda_w[i], lambda_nw[i],lambda_t[i], f_w[i], f_nw[i]] = mobility(s_w[i],s_wp,s_nwp,mu_w,mu_nw,k_w0,k_nw0,n_w,n_nw)

#building operators

#advective matrix for wetting and non-wetting phases
if rho_w > rho_nw: #wetting at the top, non-wetting at the bottom
    A_nw = np.diagflat(lambda_nw)
    A_nw = np.concatenate((np.zeros((1,grid.N)), A_nw), axis=0)
    A_nw = np.concatenate((A_nw, np.zeros((grid.Nfx,1))), axis=1)
    
    A_w  = np.diagflat(lambda_w)
    A_w = np.concatenate((A_w,np.zeros((1,grid.N))), axis=0)
    A_w = np.concatenate((A_w,np.zeros((grid.Nfx,1))), axis=1)
    
    A_t  = np.diagflat(lambda_t)
    A_t = np.concatenate((A_t,np.zeros((1,grid.N))), axis=0)
    A_t = np.concatenate((A_t,np.zeros((grid.Nfx,1))), axis=1)
    
else:  #non-wetting at the top, wetting at the bottom
    A_nw = np.diagflat(lambda_nw)
    A_nw = np.concatenate((A_nw, np.zeros((1,grid.N))), axis=0)
    A_nw = np.concatenate((A_nw, np.zeros((grid.Nfx,1))), axis=1)
    
    A_w  = np.diagflat(lambda_w)
    A_w = np.concatenate((np.zeros((1,grid.N)),A_w), axis=0)
    A_w = np.concatenate((A_w,np.zeros((grid.Nfx,1))), axis=1)
    
    A_t = np.diagflat(lambda_t)
    A_t = np.concatenate((np.zeros((1,grid.N)),A_t), axis=0)
    A_t = np.concatenate((A_t,np.zeros((grid.Nfx,1))), axis=1)

# left hand side operator
L  =  -np.mat(D)*(np.mat(A_t)*K*np.mat(G)) 
fs =   np.mat(D)*np.mat(A_nw)*K*(rho_nw-rho_w)*grav*np.transpose(np.ones((1,grid.Nfx)))

#plot
fig, ax= plt.subplots()
ax.plot(s_w,grid.xc,'r-',label='s_w')
ax.plot(lambda_w,grid.xc,'b--',label='lambda_w')
ax.plot(lambda_nw,grid.xc,'g--',label='lambda_nw')
ax.plot(lambda_t,grid.xc,'k--',label='lambda_t')
legend = ax.legend(loc='upper right', shadow=False, fontsize='x-large')
ax.set_xlabel('')
ax.set_ylabel('y')

'''
#evolution

#for i in np.arange(ti, tf, dt):
    #f_w vector
    #f_w = np.array(lambda_w*lambda_nw)
    #s_w = s_old + dt*K*D*fw
    #s_w = s_old
'''

f = fs+fn
if len(B)==0:
    u = np.linalg.solve(L, f)
    
else:
    up = np.empty([len(f),1])
    
    up = np.mat(np.transpose(B))*np.mat(np.linalg.solve(np.mat(np.mat(B))*np.mat(np.transpose(B)),Param.g))
    
    u0 = np.mat(N)*np.mat(np.linalg.solve(np.transpose(N)*np.mat(L)*np.mat(N),np.transpose(N)*(f-np.mat(L)*np.mat(up))))
    u = u0 + up

u_nw=u+(rho_w-rho_nw)*grav*np.transpose(np.mat(grid.xc))

#plot
fig, ax= plt.subplots()
ax.plot(u,grid.xc,'r-',label='Potential_w')
ax.plot(u_nw*grav,grid.xc,'g-',label='Potential_nw')
ax.plot(s_w,grid.xc,'k-',label='s_w')
legend = ax.legend(loc='lower right', shadow=False, fontsize='x-large')
ax.set_ylabel('y')
ax.set_xlabel('')