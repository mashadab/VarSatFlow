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

def comp_flux_ade_gen(flux,res,u,Grid,Param,uOld):
      # author: Mohammad Afzal Shadab
      # date: 11 February 2021
      # Description:
      # Computes the fuxes on the interior from flux(u) and reconstructs the
      # fluxes on the boundary faces from the residuals in the adjacent boundary
      # cells, res(u). 
      #
      # Note to self: This is an attempt to make the function more general
      #               the real test will be if this works with variable
      #               densities.
    
      # Input:
      # flux = anonymous function computing the flux (correct in the interior)
      # res = anonymous function computing the residual 
      # u = vector of 'flux potential' (head, temperature,electric field,...)
      # Grid = structure containing pertinent information about the grid
      # Param = structure containing pertinent information about BC's
      # 
      # Output:
      # q = correct flux everywhere
    
    ## Compute interior fluxes
    q = flux(u,uOld)
    
    ## Compute boundary fluxes
    #1) Identify the faces and cells on the boundary
    # note: check if o.k. for homogeneous Neumann problem
    if not Param.dof_neu.any(): 
        dof_cell = (np.squeeze(Param.dof_dir, axis=0))
        dof_face = (np.squeeze(Param.dof_f_dir, axis=0))
    
    #For non-empty Dirichlet and Neumann BC
    else:
        dof_cell = np.concatenate((np.squeeze(Param.dof_dir, axis=0),np.squeeze(Param.dof_neu, axis=0)),axis=0)
        dof_face = np.concatenate((np.squeeze(Param.dof_f_dir, axis=0),np.squeeze(Param.dof_f_neu, axis=0)),axis=0)
        
    dof_cell = list(filter(None, dof_cell))
    dof_face = list(filter(None, dof_face))
    
    # 2) Determine sign of flux: Convention is that flux is positive in
    #    coordinate direction. So the boundary flux, qb is not equal to q*n,
    #    were n is the outward normal!
    sign = np.multiply(np.isin(dof_face,np.concatenate((Grid.dof_f_xmin,Grid.dof_f_ymin),axis=0)), 1) - \
           np.multiply(np.isin(dof_face,np.concatenate((Grid.dof_f_xmax,Grid.dof_f_ymax),axis=0)), 1)
     
    #Because of Python indexing
    dof_cell = np.subtract(dof_cell,1)
    dof_face = np.subtract(dof_face,1)

    # 3) Compute residuals and convert them to bnd fluxes    
    resCalc  = res(u,uOld)
    q[dof_face,:] =  np.transpose([sign]) * resCalc[dof_cell,:] *Grid.V[dof_cell,:]/Grid.A[dof_face,:]

    return q;



def comp_flux_ade_gen_enthalpy(flux,res,u,Grid,Param,uOld,flux_op,L):
      # author: Mohammad Afzal Shadab
      # date: 12 February 2021
      # Description:
      # Computes the fuxes on the interior from flux(u) and reconstructs the
      # fluxes on the boundary faces from the residuals in the adjacent boundary
      # cells, res(u). 
      #
      # Note to self: This is an attempt to make the function more general
      #               the real test will be if this works with variable
      #               densities.
    
      # Input:
      # flux = anonymous function computing the flux (correct in the interior)
      # res = anonymous function computing the residual 
      # u = vector of 'flux potential' (head, temperature,electric field,...)
      # Grid = structure containing pertinent information about the grid
      # Param = structure containing pertinent information about BC's
      # 
      # Output:
      # q = correct flux everywhere
    
    ## Compute interior fluxes
    q = flux(u,uOld,flux_op)
    
    ## Compute boundary fluxes
    #1) Identify the faces and cells on the boundary
    # note: check if o.k. for homogeneous Neumann problem
    if not Param.dof_neu.any(): 
        dof_cell = (np.squeeze(Param.dof_dir, axis=0))
        dof_face = (np.squeeze(Param.dof_f_dir, axis=0))
    
    #For non-empty Dirichlet and Neumann BC
    else:
        dof_cell = np.concatenate((np.squeeze(Param.dof_dir, axis=0),np.squeeze(Param.dof_neu, axis=0)),axis=0)
        dof_face = np.concatenate((np.squeeze(Param.dof_f_dir, axis=0),np.squeeze(Param.dof_f_neu, axis=0)),axis=0)
        
    dof_cell = list(filter(None, dof_cell))
    dof_face = list(filter(None, dof_face))
    
    # 2) Determine sign of flux: Convention is that flux is positive in
    #    coordinate direction. So the boundary flux, qb is not equal to q*n,
    #    were n is the outward normal!
    sign = np.multiply(np.isin(dof_face,np.concatenate((Grid.dof_f_xmin,Grid.dof_f_ymin),axis=0)), 1) - \
           np.multiply(np.isin(dof_face,np.concatenate((Grid.dof_f_xmax,Grid.dof_f_ymax),axis=0)), 1)

    #Because of Python indexing
    dof_cell = np.subtract(dof_cell,1)
    dof_face = np.subtract(dof_face,1)

    # 3) Compute residuals and convert them to bnd fluxes    
    resCalc  = res(u,uOld,L)
    q[dof_face,:] =  np.transpose([sign]) * resCalc[dof_cell,:] *Grid.V[dof_cell,:]/Grid.A[dof_face,:]

    return q;