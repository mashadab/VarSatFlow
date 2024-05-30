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
    Avg_y1 = (sp.spdiags(([np.array(0.5*np.ones((Grid.Ny),'float64')),0.5*np.array(np.ones((Grid.Ny),'float64'))]),np.array([0,1]),Grid.Ny-1,Grid.Ny).toarray())
    Avg_y1 = np.concatenate((Avg_y1, np.zeros((1,Grid.Ny))), axis=0)
    Avg_y1 = np.concatenate((np.zeros((1,Grid.Ny)),Avg_y1), axis=0)  
    Avg_y2 = np.kron(np.eye(Grid.Nx),Avg_y1)

    #Averaging in x-direction considering the zero-flux at boundary
    Avg_x1 = (sp.spdiags(([np.array(0.5*np.ones((Grid.Nx),'float64')),0.5*np.array(np.ones((Grid.Nx),'float64'))]),np.array([0,1]),Grid.Nx-1,Grid.Nx).toarray())
    Avg_x1 = np.concatenate((Avg_x1, np.zeros((1,Grid.Nx))), axis=0)
    Avg_x1 = np.concatenate((np.zeros((1,Grid.Nx)),Avg_x1), axis=0)  
    Avg_x2 = np.kron(Avg_x1,np.eye(Grid.Ny))    

    Avg   = np.concatenate((Avg_x2 , Avg_y2), axis=0) #harmonic mean on all faces
    Avg = sp.csr_matrix(Avg)    
    
    return Avg;