#computes the mean across an interface

import numpy as np
import scipy.sparse

def comp_harmonic_mean(lmbda,Grid): 
    # author: Mohammad Afzal Shadab
    # date: 03/09/2020
    # description:
    # This function computes the mean on interfaces using harmonic mean
    # Grid = structure containing all pertinent information about the grid.
    # lmbda + mobility at cell centers
    # Output:
    # lmbda: mobility at the all the interfaces  
    lmbda = (np.array(1.0/lmbda)) #inverse for harmonic mean
    
    #lmbda[np.isinf(lmbda)] = 0.0
    lmbda = np.nan_to_num(lmbda)
    
    #Averaging in y-direction considering the zero-flux at boundary
    Avg_y1 = (scipy.sparse.spdiags(([np.array(0.5*np.ones((Grid.Ny),'float64')),0.5*np.array(np.ones((Grid.Ny),'float64'))]),np.array([0,1]),Grid.Ny-1,Grid.Ny).toarray())
    Avg_y1 = np.concatenate((Avg_y1, np.zeros((1,Grid.Ny))), axis=0)
    Avg_y1 = np.concatenate((np.zeros((1,Grid.Ny)),Avg_y1), axis=0)  
    Avg_y2 = np.kron(np.eye(Grid.Nx),Avg_y1)
    lmbda_y = np.mat(Avg_y2)*np.mat(lmbda)   #harmonic mean lambda on the y-faces  
    lmbda_y = (np.array(1.0/lmbda_y))
    lmbda_y[np.isinf(lmbda_y)] = 0.0
    
    
    #Averaging in x-direction considering the zero-flux at boundary
    Avg_x1 = (scipy.sparse.spdiags(([np.array(0.5*np.ones((Grid.Nx),'float64')),0.5*np.array(np.ones((Grid.Nx),'float64'))]),np.array([0,1]),Grid.Nx-1,Grid.Nx).toarray())
    Avg_x1 = np.concatenate((Avg_x1, np.zeros((1,Grid.Nx))), axis=0)
    Avg_x1 = np.concatenate((np.zeros((1,Grid.Nx)),Avg_x1), axis=0)  
    Avg_x2 = np.kron(Avg_x1,np.eye(Grid.Ny))    
    lmbda_x = np.mat(Avg_x2)*np.mat(lmbda)  #harmonic mean lambda on the x-faces
    lmbda_x = (np.array(1.0/lmbda_x))    
    lmbda_x[np.isinf(lmbda_x)] = 0.0    

    #print('lambda_x',lmbda_y)
    
    lmbda   = np.diagflat(np.concatenate((lmbda_x , lmbda_y), axis=0)) #harmonic mean on all faces

    return lmbda;

def comp_algebraic_mean(lmbda,Grid): 
    # author: Mohammad Afzal Shadab
    # date: 03/09/2020
    # description:
    # This function computes the mean on interfaces using algebraic mean
    # Grid = structure containing all pertinent information about the grid.
    # lmbda + mobility at cell centers
    # Output:
    # lmbda: mobility at the all the interfaces
    
    #Averaging in y-direction considering the zero-flux at boundary
    Avg_y1 = (scipy.sparse.spdiags(([np.array(0.5*np.ones((Grid.Ny),'float64')),0.5*np.array(np.ones((Grid.Ny),'float64'))]),np.array([0,1]),Grid.Ny-1,Grid.Ny).toarray())
    Avg_y1 = np.concatenate((Avg_y1, np.zeros((1,Grid.Ny))), axis=0)
    Avg_y1 = np.concatenate((np.zeros((1,Grid.Ny)),Avg_y1), axis=0)  
    Avg_y2 = np.kron(np.eye(Grid.Nx),Avg_y1)
    lmbda_y = np.mat(Avg_y2)*np.mat(lmbda)   #harmonic mean lambda on the y-faces

    #Averaging in x-direction considering the zero-flux at boundary
    Avg_x1 = (scipy.sparse.spdiags(([np.array(0.5*np.ones((Grid.Nx),'float64')),0.5*np.array(np.ones((Grid.Nx),'float64'))]),np.array([0,1]),Grid.Nx-1,Grid.Nx).toarray())
    Avg_x1 = np.concatenate((Avg_x1, np.zeros((1,Grid.Nx))), axis=0)
    Avg_x1 = np.concatenate((np.zeros((1,Grid.Nx)),Avg_x1), axis=0)  
    Avg_x2 = np.kron(Avg_x1,np.eye(Grid.Ny))    
    lmbda_x = np.mat(Avg_x2)*np.mat(lmbda)  #harmonic mean lambda on the x-faces  
    
    lmbda   = np.diagflat(np.concatenate((lmbda_x , lmbda_y), axis=0)) #harmonic mean on all faces
   
    return lmbda;