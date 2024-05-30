#computes the advection operator

import numpy as np
import scipy.sparse
from flux_upwindfun2D import flux_upwind

def adv_op(lmbda,q,Grid):
    # author: Mohammad Afzal Shadab
    # date: 03/24/2020
    # description:
    # This function computes the mean on interfaces using harmonic mean
    # Grid = structure containing all pertinent information about the Grid.
    # lmbda + mobility at cell centers
    # Output:
    # A_w: mobility at the all the interfaces
       
    # 1st0order Upwind/Godonov fluxes (required for all discretizations)
    # Lines 19 to 45 are identical to flux_upwind.m
    Nx = Grid.Nx
    Ny = Grid.Ny 
    N = Grid.N
    Nfx = Grid.Nfx # of x faces
    Nfy = Grid.Nfy # of y faces
    Nf = length(q)

    # First-order upwind (Godunov) fluxes
    A_w = flux_upwind(q,Grid)
    
    return A_w;