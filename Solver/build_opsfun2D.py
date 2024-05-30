import scipy.sparse as sp
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
    Ny = Grid.Ny
    N  = Grid.N

    # Two dimensional divergence    
    #     Readable implementation
    #     # 2D divergence matrices
    
    if (Nx>1) and (Ny>1): #2D case
        #One diamentional divergence
        Dy = sp.spdiags(([-np.array(np.ones((Ny+1),'float64')) , np.array(np.ones((Ny+1),'float64'))])/np.asarray(Grid.dy),np.array([0,1]),Ny,Ny+1).tocsr()#.toarray() # Dy^1
        
        #Two dimensional divergence
        Dy = sp.kron(np.eye(Nx), Dy) #y component Dy^2
        
        e  = np.array(np.ones(Ny*(Nx+1),'float64'))
        Dx = sp.spdiags(([-e , e])/np.asarray(Grid.dx),np.array([0,Ny]),N,(Nx+1)*Ny).tocsr()#.toarray() # 2D div-matrix in x-dir

        #D  = np.concatenate((Dx , Dy), axis=1)
        D  = sp.hstack([Dx , Dy])    
        dof_f_bnd = np.concatenate(np.array([Grid.dof_f_xmin-1, Grid.dof_f_xmax-1, Grid.dof_f_ymin-1, Grid.dof_f_ymax-1]), axis=0 )       # boundary faces
        dof_f_bnd = np.transpose(dof_f_bnd)
        
    elif (Nx > 1) and (Ny == 1): #one dimensional in x direction
        D = sp.spdiags(([-np.array(np.ones((Nx+1),'float64')),np.array(np.ones((Nx+1),'float64'))])/np.asarray(Grid.dx),np.array([0,1]),Nx,Nx+1).tocsr()#.toarray() # 1D div-matrix in x-dir
        dof_f_bnd = [Grid.dof_f_xmin-1, Grid.dof_f_xmax-1] # boundary faces
        dof_f_bnd = np.transpose(dof_f_bnd)  

    elif (Nx == 1) and (Ny > 1): #one dimensional in y direction
        D = sp.spdiags(([-np.array(np.ones((Ny+1),'float64')),np.array(np.ones((Ny+1),'float64'))])/np.asarray(Grid.dy),np.array([0,1]),Ny,Ny+1).tocsr()#.toarray() # 1D div-matrix in y-dir
        dof_f_bnd = [Grid.dof_f_ymin-1, Grid.dof_f_ymax-1] # boundary faces
        dof_f_bnd = np.transpose(dof_f_bnd)  

    # Gradient
    # Note this is only true in cartesian coordinates!
    # For more general coordinate systems it is worth
    # assembling G and D seperately.
    G = -sp.csr_matrix.transpose(D)
    G =  zero_rows(G,dof_f_bnd)

    #Identity
    I = (sp.eye(Grid.N)).tocsr()

    return D,G,I;

def zero_rows(M, rows_to_zero):

    ixs = np.ones(M.shape[0], int)
    ixs[rows_to_zero] = 0
    D = sp.diags(ixs)
    res = D * M
    return res