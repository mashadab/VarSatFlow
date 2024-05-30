import scipy.sparse
import numpy as np

def build_ops(Grid):
    # author: Mohammad Afzal Shadab
    # date: 1/27/2020
    # description:
    # This function computes the discrete divergence and gradient matrices on a
    # regular staggered grid using central difference approximations. The
    # discrete gradient assumes homogeneous boundary conditions.
    # Input:
    # Grid = structure containing all pertinent information about the grid.
    # Output:
    # D = discrete divergence matrix
    # G = discrete gradient matrix
    # I = identity matrix

    Nx = Grid.Nx

    # One dimensional divergence
    
    #     Readable implementation
    #     # 1D divergence matrices

    Dx = scipy.sparse.spdiags(([-np.array(np.ones((Nx+1),'float64')),np.array(np.ones((Nx+1),'float64'))])/np.asarray(Grid.dx),np.array([0,1]),Nx,Nx+1).toarray() # 1D div-matrix in x-dir

    #Dx = scipy.sparse.spdiags(([-np.array(np.ones((Nx+1),'float64')),np.array(np.ones((Nx+1),'float64'))]),np.array([0,1]),Nx,Nx+1).toarray() # 1D div-matrix in x-dir   #####PROBLEM###

    Ix = (scipy.sparse.eye(Nx)).toarray()  # 1D identities in x
    #     # Complete 1D divergence
    D = Dx.copy()

    #     Implementation avoiding intermediate matrices
    #D = [np.kron(scipy.spdiags([-np.ones(Nx,1) np.ones(Nx,1)]/Grid.dx,[0,1],Nx,Nx+1))]

    dof_f_bnd = [Grid.dof_f_xmin-1, Grid.dof_f_xmax-1] # boundary faces
    dof_f_bnd = np.transpose(dof_f_bnd)

    # Gradient
    # Note this is only true in cartesian coordinates!
    # For more general coordinate systems it is worth
    # assembling G and D seperately.
    G = -np.transpose(D)
    G[dof_f_bnd,:] = 0.0

    #Identity
    I = (scipy.sparse.eye(Grid.N)).toarray()

    return D,G,I; 