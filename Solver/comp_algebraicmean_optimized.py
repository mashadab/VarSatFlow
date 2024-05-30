#algebraic mean operators
import numpy as np
import scipy.sparse as sp

def comp_algebraic_mean(Grid): 
    # author: Mohammad Afzal Shadab
    # date: 03/09/2020
    # description:
    # This function computes the mean on interfaces using algebraic mean
    # Grid = structure containing all pertinent information about the grid.
    # lmbda + mobility at cell centers
    # Output:
    # lmbda: mobility at the all the interfaces
    
    #Averaging in y-direction considering the zero-flux at boundary
    Avg_y1 = sp.spdiags(([np.array(0.5*np.ones((Grid.Ny),'float64')),0.5*np.array(np.ones((Grid.Ny),'float64'))]),np.array([0,1]),Grid.Ny-1,Grid.Ny)
    Avg_y1 = sp.vstack([np.zeros((1,Grid.Ny)),Avg_y1,np.zeros((1,Grid.Ny))]) 
    Avg_y2 = sp.kron(sp.eye(Grid.Nx),Avg_y1)

    #Averaging in x-direction considering the zero-flux at boundary
    Avg_x1 = sp.spdiags(([np.array(0.5*np.ones((Grid.Nx),'float64')),0.5*np.array(np.ones((Grid.Nx),'float64'))]),np.array([0,1]),Grid.Nx-1,Grid.Nx)
    Avg_x1 = sp.vstack([np.zeros((1,Grid.Nx)),Avg_x1, np.zeros((1,Grid.Nx))]) 
    Avg_x2 = sp.kron(Avg_x1,sp.eye(Grid.Ny))    
    
    Avg    = sp.vstack([Avg_x2, Avg_y2])

    return Avg;