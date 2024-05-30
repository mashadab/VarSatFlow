#computes the advection operator

import numpy as np
import scipy.sparse

def adv_op(lmbda,q,Grid):
    # author: Mohammad Afzal Shadab
    # date: 03/24/2020
    # description:
    # This function computes the mean on interfaces using harmonic mean
    # Grid = structure containing all pertinent information about the Grid.
    # lmbda + mobility at cell centers
    # Output:
    # A_w: mobility at the all the interfaces

    A_w_pos = np.diagflat(lmbda)
    A_w_pos = np.concatenate((A_w_pos,np.zeros((1,Grid.N))), axis=0)
    A_w_pos = np.concatenate((A_w_pos,np.zeros((Grid.Nfx,1))), axis=1) 
    q_pos   = (np.maximum(q,0.0))
    
    A_w_neg = np.diagflat(lmbda)
    A_w_neg = np.concatenate((np.zeros((1,Grid.N)),A_w_neg), axis=0)
    A_w_neg = np.concatenate((np.zeros((Grid.Nfx,1)),A_w_neg), axis=1) 
    q_neg   = (np.minimum(q,0.0))
   
    A_w     = np.mat(A_w_pos)*np.mat(q_pos) + np.mat(A_w_neg)*np.mat(q_neg)
    
    
    return A_w;