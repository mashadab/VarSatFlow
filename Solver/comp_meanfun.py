#computes the mean across an interface

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse

def comp_mean(lmbda,Grid): 
    # author: Mohammad Afzal Shadab
    # date: 03/09/2020
    # description:
    # This function computes the mean on interfaces using harmonic mean
    # Grid = structure containing all pertinent information about the grid.
    # lmbda + mobility at cell centers
    # Output:
    # lmbda: mobility at the all the interfaces
    
    lmbda = np.nan_to_num(np.array(1.0/lmbda))
    
    A = (scipy.sparse.spdiags(([np.array(0.5*np.ones((Grid.N),'float64')),0.5*np.array(np.ones((Grid.N),'float64'))]),np.array([0,1]),Grid.N-1,Grid.N).toarray())
    
    lmbda = np.mat(A)*np.mat(lmbda)
    
    lmbda = np.nan_to_num(np.array(1.0/lmbda))
    
    lmbda = np.diagflat(lmbda)
    
    lmbda = np.concatenate((lmbda, np.zeros((1,Grid.N-1))), axis=0)
    lmbda = np.concatenate((np.zeros((1,Grid.N-1)),lmbda), axis=0)
    lmbda = np.concatenate((lmbda, np.zeros((Grid.Nfx,1))), axis=1)
    lmbda = (np.concatenate((np.zeros((Grid.Nfx,1)),lmbda), axis=1))
    
    return lmbda;