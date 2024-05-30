#computes the upwind flux using Godunov scheme

import numpy as np
import scipy.sparse as sp
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
    N  = Grid.N

    if (Nx>1) and (Ny>1): #2D case
    
        Nfx = Grid.Nfx # of x faces
        Nfy = Grid.Nfy # of y faces
        Nf  = Grid.Nf
       
        # x -fluxes      
        Qdp = sp.dia_matrix((np.transpose(np.maximum(q,0)),  np.array([0])), shape=(Nf, Nf))        
        Qdn = sp.dia_matrix((np.transpose(np.minimum(q,0)),  np.array([0])), shape=(Nf, Nf))        
        
        #Iy = (sp.eye(Ny)).toarray()
        Iy  = sp.eye(Ny)
        
        #print(np.linalg.norm(Iy-Iy1))        
        
        Axp1 = sp.spdiags(([np.array(np.ones((Nx+1),'float64'))]),np.array([-1]),Nx+1,Nx) # Axp^1
        Axn1 = sp.spdiags(([np.array(np.ones((Nx+1),'float64'))]),np.array([0]),Nx+1,Nx) # Axn^1
              
        #Two dimensional divergence
        Axp = sp.kron(Axp1,Iy) #x component Ax+
        Axn = sp.kron(Axn1,Iy) #x component Ax-

        # y -fluxes       
        Ix   = sp.eye(Nx)
        
        Ayp1 = sp.spdiags(([np.array(np.ones((Ny+1),'float64'))]),np.array([-1]),Ny+1,Ny) # Ayp^1
        Ayn1 = sp.spdiags(([np.array(np.ones((Ny+1),'float64'))]),np.array([0]),Ny+1,Ny) # Ayn^1
        
        #Two dimensional divergence
        Ayp = sp.kron(Ix,Ayp1) #x component Ay+
        Ayn = sp.kron(Ix,Ayn1) #x component Ay-
        
        Ap    = sp.vstack([Axp , Ayp]) #A+ matrix 
        An    = sp.vstack([Axn , Ayn]) #A- matrix         
       
        A    = sp.csr_matrix.dot(Qdp,Ap) + sp.csr_matrix.dot(Qdn,An)  
        
    elif (Nx>1) and (Ny==1): #1D case    
        qn   = np.array(np.minimum(q[0:Nx,0],0))
        qp   = np.array(np.maximum(q[1:Nx+1,0],0))
        #qp   = np.append(qp,0)
        
        A    = sp.spdiags(([qp,qn]),np.array([-1,0]),Nx+1,Nx)


    elif (Nx==1) and (Ny>1): #1D case    
        qn   = np.minimum(q[0:N,0],0)
        qp   = np.maximum(q[1:N+1,0],0)
        qp   = np.append(qp,0)
        
        A    = sp.spdiags(([qp,qn]),np.array([-1,0]),N+1,N)

    return A;

'''
grid.xmin = 0 
grid.xmax = 1 
grid.Nx = 2

grid.ymin = 0 
grid.ymax = 1 
grid.Ny = 2

grid = build_grid(grid)
q = np.ones((grid.Nf,1))
A = flux_upwind(q,grid)
print(sp.issparse(A))
'''