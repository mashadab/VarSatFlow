#computes the upwind flux using Godunov scheme

import numpy as np
import scipy.sparse
from build_gridfun2D import build_grid

class grid:
    def __init__(self):
        self.xmin = []
        self.xmax = []
        self.Nx = []

def flux_upwind(q,Grid):
    # author: Mohammad Afzal Shadab
    # date: 03/24/2020
    # Description:
    # This function computes the upwind flux matrix from the flux vector.
    #
    # Input:
    # q = Nf by 1 flux vector from the flow problem.
    # Grid = structure containing all pertinent information about the grid.
    #
    # Output:
    # A = Nf by Nf matrix contining the upwinded fluxes
    #
    # Example call:
    # >> Grid.xmin = 0; Grid.xmax = 1; Grid.Nx = 10;
    # >> Grid = build_grid(Grid);
    # >> q = ones(Grid.Nf,1);
    # >> [A] = flux_upwind(q,Grid);
      
    # 1st0order Upwind/Godonov fluxes (required for all discretizations)
    Nx = Grid.Nx
    Ny = Grid.Ny
    N = Grid.N
    Nfx = Grid.Nfx # of x faces
    Nfy = Grid.Nfy # of y faces
    Nf  = len(q)

    if (Nx>1) and (Ny>1): #2D case
       
        # x -fluxes
        Qdp = np.diagflat(np.maximum(q,0)) #Nf by Nf matrix
        Qdn = np.diagflat(np.minimum(q,0)) #Nf by Nf matrix  
        
        Iy = (scipy.sparse.eye(Ny)).toarray()
        
        Axp1 = scipy.sparse.spdiags(([np.array(np.ones((Nx+1),'float64'))]),np.array([-1]),Nx+1,Nx).toarray() # Axp^1
        Axn1 = scipy.sparse.spdiags(([np.array(np.ones((Nx+1),'float64'))]),np.array([0]),Nx+1,Nx).toarray() # Axn^1
        
        #Two dimensional divergence
        Axp = np.kron(Axp1,Iy) #x component Ax+
        Axn = np.kron(Axn1,Iy) #x component Ax-

        # y -fluxes       
        Ix = (scipy.sparse.eye(Nx)).toarray()
        
        Ayp1 = scipy.sparse.spdiags(([np.array(np.ones((Ny+1),'float64'))]),np.array([-1]),Ny+1,Ny).toarray() # Ayp^1
        Ayn1 = scipy.sparse.spdiags(([np.array(np.ones((Ny+1),'float64'))]),np.array([0]),Ny+1,Ny).toarray() # Ayn^1
        
        #Two dimensional divergence
        Ayp = np.kron(Ix,Ayp1) #x component Ay+
        Ayn = np.kron(Ix,Ayn1) #x component Ay-
        
        Ap   = np.concatenate((Axp , Ayp), axis=0) #A+ matrix 
        An   = np.concatenate((Axn , Ayn), axis=0) #A- matrix         
        
        A    = np.mat(Qdp)*np.mat(Ap) + np.mat(Qdn)*np.mat(An)  
        
    elif (Nx>1) and (Ny==1): #2D case    
        qn   = (np.minimum(q[0:Nx,0],0))
        qp   = (np.maximum(q[1:Nx+1,0],0))
        A    = scipy.sparse.spdiags(([qp,qn]),np.array([-1,0]),Nx+1,Nx).toarray()

    elif (Nx==1) and (Ny>1): #2D case    
        qn   = np.minimum(q[0:N,0],0)
        qp   = np.maximum(q[1:N+1,0],0)
        A    = scipy.sparse.spdiags(([qp,qn]),np.array([-1,0]),N+1,N).toarray()
    
    return A;


grid.xmin = 0 
grid.xmax = 1 
grid.Nx = 2

grid.ymin = 0 
grid.ymax = 1 
grid.Ny = 2

grid = build_grid(grid)
q = np.ones((grid.Nf,1))
A = flux_upwind(q,grid)
