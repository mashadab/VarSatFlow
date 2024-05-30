#algebraic mean operators
import numpy as np
import scipy.sparse as sp

def comp_mean_matrix(Grid): 
    # author: Mohammad Afzal Shadab
    # date: 04/15/2022
    # description:
    # This function computes the mean on interfaces using algebraic mean
    # Interface values are just the cell center values
    # Grid = structure containing all pertinent information about the grid.
    # lmbda + mobility at cell centers
    # Output:
    # Averaging matrix
    
    if Grid.Nx>1 and Grid.Ny==1:    
        M_x1 = sp.csr_matrix(sp.spdiags(([np.array(0.5*np.ones((Grid.Nx),'float64')),0.5*np.array(np.ones((Grid.Nx),'float64'))]),np.array([-1,0]),Grid.Nx+1,Grid.Nx))
        M_x1[0,0] = 1.0
        M_x1[Grid.Nx,Grid.Nx-1] = 1.0
        M = M_x1.copy()        

    elif Grid.Nx==1 and Grid.Ny>1:       
        M_y1 = sp.csr_matrix(sp.spdiags(([np.array(0.5*np.ones((Grid.Ny),'float64')),0.5*np.array(np.ones((Grid.Ny),'float64'))]),np.array([-1,0]),Grid.Ny+1,Grid.Ny))
        #M_y1 = sp.vstack([np.zeros((1,Grid.Ny)),M_y1,np.zeros((1,Grid.Ny))]) 
        M_y1[0,0] = 1.0
        M_y1[Grid.Ny,Grid.Ny-1] = 1.0
        M = M_y1.copy()
    
    if Grid.Nx>1 and Grid.Ny>1:
        #Averaging in y-direction considering the zero-flux at boundary
        M_y1 = sp.csr_matrix(sp.spdiags(([np.array(0.5*np.ones((Grid.Ny),'float64')),0.5*np.array(np.ones((Grid.Ny),'float64'))]),np.array([-1,0]),Grid.Ny+1,Grid.Ny))
        #M_y1 = sp.vstack([np.zeros((1,Grid.Ny)),M_y1,np.zeros((1,Grid.Ny))]) 
        M_y1[0,0] = 1.0
        M_y1[Grid.Ny,Grid.Ny-1] = 1.0
        M_y2 = sp.kron(sp.eye(Grid.Nx),M_y1)
    
        #Averaging in x-direction considering the zero-flux at boundary
        M_x1 = sp.csr_matrix(sp.spdiags(([np.array(0.5*np.ones((Grid.Nx),'float64')),0.5*np.array(np.ones((Grid.Nx),'float64'))]),np.array([-1,0]),Grid.Nx+1,Grid.Nx))
        M_x1[0,0] = 1.0
        M_x1[Grid.Nx,Grid.Nx-1] = 1.0
        M_x2 = sp.kron(M_x1,sp.eye(Grid.Ny))    
        
        M    = sp.vstack([M_x2, M_y2])

    return M;

'''
from classes import *
from build_gridfun2D import build_grid

Grid.xmin = 0; Grid.xmax = 1; Grid.Nx = 2
Grid.ymin = 0; Grid.ymax = 1; Grid.Ny = 2

Grid = build_grid(Grid)
M = comp_mean_matrix(Grid)
'''