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

def comp_flux(D,Kd,G,h,fs,Grid,Param):
    # author: Mohammad Afzal Shadab
    # date: 22 April 2020
    # Description:
    # Computes the mass conservative fluxes across all boundaries from the 
    # residual of the compatability condition over the boundary cells.
    # Note: Current implmentation works for all cases where one face 
    #       is assigend to each bnd cell. So corner cells must have
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
    #q = -np.mat(Kd)*np.mat(G)*np.mat(h)
    q = -Kd @ G @ h
    #q = -(Kd)*np.mat(G)*np.mat(h)
    
    ## Compute boundary fluxes
    # note: check if o.k. for homogeneous Neumann problem

    #dof_cell = Param.dof_dir
    #dof_face = Param.dof_f_dir

    dof_cell = (Param.dof_neu).copy()
    dof_face = (Param.dof_f_neu).copy()

    #dof_cell = np.concatenate([Param.dof_dir,Param.dof_neu]).astype(int)
    #dof_face = np.concatenate([Param.dof_f_dir,Param.dof_f_neu]).astype(int)
    
    #For non-empt Dirichle and Neumann BC
    #dof_cell = np.concatenate((np.squeeze(Param.dof_dir, axis=0),np.squeeze(Param.dof_neu, axis=0)),axis=0)
    #dof_face = np.concatenate((np.squeeze(Param.dof_f_dir, axis=0),np.squeeze(Param.dof_f_neu, axis=0)),axis=0)
    
    #dof_cell = list(filter(None, dof_cell))
    #dof_face = list(filter(None, dof_face))
    
    #print(f'The number of dof of BC cell is', np.shape(dof_cell))
    #print(f'The number of dof of BC face is', np.shape(dof_face))
    
    '''
    if np.any(Param.dof_neu)==False:
        dof_cell = Param.dof_dir
        dof_face = Param.dof_f_dir

    elif np.any(Param.dof_dir)==False:
        dof_cell = (Param.dof_neu).copy()
        dof_face = (Param.dof_f_neu).copy()

    else:
        #check next time
        #dof_cell = np.concatenate([Param.dof_dir,Param.dof_neu]).astype(int)
        #dof_face = np.concatenate([Param.dof_f_dir,Param.dof_f_neu]).astype(int)
        
        #For non-empt Dirichle and Neumann BC
        dof_cell = np.concatenate((np.squeeze(Param.dof_dir, axis=0),np.squeeze(Param.dof_neu, axis=0)),axis=0)
        dof_face = np.concatenate((np.squeeze(Param.dof_f_dir, axis=0),np.squeeze(Param.dof_f_neu, axis=0)),axis=0)
    
        dof_cell = list(filter(None, dof_cell))
        dof_face = list(filter(None, dof_face))
    '''

    if Grid.Nx>1 and Grid.Ny>1:  
        sign = np.multiply(np.isin(dof_face,np.concatenate((Grid.dof_f_xmin,Grid.dof_f_ymin),axis=0)), 1) - \
               np.multiply(np.isin(dof_face,np.concatenate((Grid.dof_f_xmax,Grid.dof_f_ymax),axis=0)), 1)

    else:  
        sign = np.multiply(np.isin(dof_face,Grid.dof_f_xmin), 1) - \
               np.multiply(np.isin(dof_face,Grid.dof_f_xmax), 1)
               
    #Because of Python indexing
    dof_cell = np.subtract(dof_cell,1)
    dof_face = np.subtract(dof_face,1)
    
    #print(np.shape(np.transpose([sign])),np.shape((D[dof_cell,:]@ q)- fs[dof_cell,:]), np.shape(Grid.V[dof_cell,:]/Grid.A[dof_face,:]))
    q[dof_face] =  np.transpose([np.ravel(sign) * (np.ravel(D[dof_cell,:]@ q - fs[dof_cell,:]))*np.ravel(Grid.V[dof_cell,:]/Grid.A[dof_face,:])])
    
    #q[dof_face] =  np.transpose([sign]) * ((D[dof_cell,:]@ q)- sp.csr_matrix.toarray(fs[dof_cell,:]))*Grid.V[dof_cell,:]/Grid.A[dof_face,:]
      
    return q;

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