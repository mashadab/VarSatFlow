# import python libraries
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.sparse as sp

# import personal libraries
import build_gridfun
import build_opsfun
from build_bndfun import build_bnd
from mobilityfun import mobility

class grid:
    def __init__(self):
        self.xmin = 0.0
        self.xmax = 1.0
        self.Nx = 10

class Param:
    def __init__(self):
        self.dof_dir = []       # identify cells on Dirichlet bnd
        self.dof_f_dir = []     # identify faces on Dirichlet bnd
        self.dof_neu = []       # identify cells on Neumann bnd
        self.dof_f_neu = []     # identify faces on Neumann bnd
        self.g = []             # column vector of non-homogeneous Dirichlet BCs (Nc X 1)
        self.qb = []  


def comp_flux_ade(D,flux_op,h,fs,Grid,Param):
    # author: Mohammad Afzal Shadab
    # date: 15 September 2020
    # Description:
    # Computes the mass conservative fluxes across all boundaries from the 
    # residual of the compatability condition over the boundary cells for the advection-diffusion equation
    # Note: Current implmentation works for all cases where one face 
    #       is assigned to each bnd cell. So corner cells must have
    #       natural BC's on all but one face.
    #
    # Input:
    # D = N by Nf discrete divergence matrix.
    # Kd = Nf by Nf conductivity matrix.
    # G = Nf by N discrete gradient matrix.
    # h = N by 1 vector of flow potential in cell centers.
    # fs = N by 1 right hand side vector containing only source terms.
    # Grid = structure containing grid information.
    # Param = structure contaning problem paramters and information about BC's
    #
    # Output:
    # q = Nf by 1 vector of fluxes across all cell faces

    
    ## Compute interior fluxes
    q = flux_op @ h
    
    ## Compute boundary fluxes
    # note: check if o.k. for homogeneous Neumann problem

    #dof_cell = Param.dof_dir
    #dof_face = Param.dof_f_dir
    if not Param.dof_neu.any(): 
        dof_cell = (np.squeeze(Param.dof_dir, axis=0))
        dof_face = (np.squeeze(Param.dof_f_dir, axis=0))
    
    #For non-empt Dirichlet and Neumann BC
    else:
        dof_cell = np.concatenate((np.squeeze(Param.dof_dir, axis=0),np.squeeze(Param.dof_neu, axis=0)),axis=0)
        dof_face = np.concatenate((np.squeeze(Param.dof_f_dir, axis=0),np.squeeze(Param.dof_f_neu, axis=0)),axis=0)
        
    dof_cell = list(filter(None, dof_cell))
    dof_face = list(filter(None, dof_face))
    
    #print(f'The dof of BC cell is {dof_cell}')
    #print(f'The dof of BC face is {dof_face}')
    sign = np.multiply(np.isin(dof_face,np.concatenate((Grid.dof_f_xmin,Grid.dof_f_ymin),axis=0)), 1) - \
           np.multiply(np.isin(dof_face,np.concatenate((Grid.dof_f_xmax,Grid.dof_f_ymax),axis=0)), 1)
     
    #Because of Python indexing
    dof_cell = np.subtract(dof_cell,1)
    dof_face = np.subtract(dof_face,1)
    
    q[dof_face,:] =  np.transpose([sign]) * ((D[dof_cell,:]@ q)- fs[dof_cell,:]) *Grid.V[dof_cell,:]/Grid.A[dof_face,:]
    #print(dof_face,q[dof_face,:])  
    return q;

'''
#grid and operators
grid.xmin = 0.0
grid.xmax = 1.0
grid.Nx   = 10
build_gridfun.build_grid(grid)
[D,G,I]=build_opsfun.build_ops(grid)
#applying boundary condition
Param.dof_dir   = np.array([grid.dof_xmin])     # identify cells on Dirichlet bnd
Param.dof_f_dir = np.array([grid.dof_f_xmin])   # identify faces on Dirichlet bnd
Param.dof_neu   = np.array([])     # identify cells on Neumann bnd
Param.dof_f_neu = np.array([])     # identify faces on Neumann bnd
Param.g  = np.array([0.0])                      # set head at Dirichlet bnd
[B,N,fn] = build_bnd(Param,grid,I)              # Build constraint matrix and basis for its nullspace
fs = np.ones([grid.N,1])                       # r.h.s. (zero)
L = -np.mat(D)*np.mat(G)                        # Laplacian
f = fs+fn
if not B.any():
    u = np.linalg.solve(L, f)
else:
    up = np.empty([len(f),1])
    
    up = np.mat(np.transpose(B))*np.mat(np.linalg.solve(np.mat(np.mat(B))*np.mat(np.transpose(B)),Param.g))
    
    u0 = np.mat(N)*np.mat(np.linalg.solve(np.transpose(N)*np.mat(L)*np.mat(N),np.transpose(N)*(f-np.mat(L)*np.mat(up))))
    u = u0 + up

q = comp_flux(D,np.diagflat(np.ones(grid.Nx+1)),G,u,fs,grid,Param)

#plot
fig, ax= plt.subplots()
ax.plot(grid.xc,u,'k-',label='u')
ax.plot(grid.xf,q,'r-',label='q')
legend = ax.legend(loc='upper left', shadow=False, fontsize='x-large')
ax.set_xlabel('Position')
ax.set_ylabel('Head')
'''