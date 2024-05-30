import numpy as np

def build_bnd(Param,Grid,I):
    
    # author: Mohammad Afzal Shadab
    # date: 01/03/2020
    # Description:
    # This function computes the operators and r.h.s vectors for both Dirichlet
    # and Neumann boundary conditions. 
    # Note:
    # N is not created from I the same way B is created from I, because 
    # the vector dof_dir contains the columns that must be eliminated rather
    # then the columns that are retained in N. If you wanted to do it this way
    # you would have to create a new vector
    # dof_non_dir = setdiff(dof,dof_dir)
    # I suspect that the set-operators are expensive on large vectors, hence
    # we simply eliminate the rows.
    
    # Input:
    # Grid = structure containing all pertinent information about the grid.
    # Param = structure containing all information about the physical problem
    #         in particular this function needs the fields
    #         Param.dof_dir = Nc by 1 column vector containing 
    #                         the dof's of the Dirichlet boundary.
    #         Param.dof_neu = N by 1 column vector containing 
    #                         the dof's of the Neumann boundary.
    #         Param.qb      = column vector of prescribed fluxes on Neuman bnd.
    # I = identity matrix in the full space
    #
    # Output:
    # B = Nc by N matrix of the Dirichlet constraints
    # N = (N-Nc) by (N-Nc) matrix of the nullspace of B
    # fn = N by 1 r.h.s. vector of Neuman contributions
    
    # Dirichlet boundary conditions
    if not Param.dof_dir.any():
        B = []
        N = []    
    else:
        I = np.array(I)
        B = I[Param.dof_dir-1, :]
        N = I.copy()
        #N[:,Param.dof_dir] = []
        N = np.delete(N,Param.dof_dir-1, 1)
        
    # Neumann boundary conditions
    if not Param.dof_neu.any():
        fn = np.zeros([Grid.N,1]);                     # allocate sparse zero vector

    else:

        fn = np.zeros([Grid.N,1])  # allocate sparse vector
        

        fn[Param.dof_neu-1,:] = Param.qb*Grid.A[Param.dof_f_neu-1]/Grid.V[Param.dof_neu-1]

    return B,N, fn;