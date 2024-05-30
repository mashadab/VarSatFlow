# import python libraries
import numpy as np
#import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as linalg

# import personal libraries
#import build_gridfun
#import build_opsfun
#from build_bndfun import build_bnd
#from mobilityfun import mobility

def solve_lbvp_SPD(L,f,B,g,N):

    # author: Mohammad Afzal Shadab
    # date: 2/25/2020
    # Description
    # Computes the solution $u$ to the linear differential problem given by
    #
    # $$\mathcal{L}(u)=f \quad x\in \Omega $$
    #
    # with boundary conditions
    #
    # $$\mathcal{B}(u)=g \quad x\in\partial\Omega$$.
    #
    # Input:
    # L = matrix representing the discretized linear operator of size N by N, 
    #     where N is the number of degrees of freedom
    # f = column vector representing the discretized r.h.s. and contributions
    #     due non-homogeneous Neumann BC's of size N by 1
    # B = matrix representing the constraints arising from Dirichlet BC's of
    #     size Nc by N
    # g = column vector representing the non-homogeneous Dirichlet BC's of size
    #     Nc by 1.
    # N = matrix representing a orthonormal basis for the null-space of B and
    #     of size N by (N-Nc).
    # Output:
    # u = column vector of the solution of size N by 1
    if B.nnz == 0:
        u  = np.transpose([linalg.cg(L, f,tol=1e-10,maxiter=100)[0]])
    else:

        up = sp.csr_matrix.transpose(B) @ np.transpose([linalg.cg((B @ sp.csr_matrix.transpose(B)),g,tol=1e-10,maxiter=100)[0]])
        u0 = N @ np.transpose([linalg.cg(sp.csr_matrix.transpose(N) @ L @ N,sp.csr_matrix.transpose(N) @ (f-L @ up),tol=1e-10,maxiter=100)[0]])

        u = u0 + up

    return u;
'''
class grid:
    def __init__(self):
        self.xmin = []
        self.xmax = []
        self.Nx = []

class Param:
    def __init__(self):
        self.dof_dir = []       # identify cells on Dirichlet bnd
        self.dof_f_dir = []     # identify faces on Dirichlet bnd
        self.dof_neu = []       # identify cells on Neumann bnd
        self.dof_f_neu = []     # identify faces on Neumann bnd
        self.g = []             # column vector of non-homogeneous Dirichlet BCs (Nc X 1)
        self.qb = []            

#grid and operators
grid.xmin = 0.0
grid.xmax = 1.0
grid.Nx   = 10
build_gridfun.build_grid(grid)
[D,G,I]=build_opsfun.build_ops(grid)
#applying boundary condition
Param.dof_dir   = np.array([grid.dof_xmin])     # identify cells on Dirichlet bnd
Param.dof_f_dir = np.array([grid.dof_f_xmin])   # identify faces on Dirichlet bnd
Param.dof_neu   = np.array([grid.dof_xmax])     # identify cells on Neumann bnd
Param.dof_f_neu = np.array([grid.dof_f_xmax])   # identify faces on Neumann bnd
Param.qb = np.array([1.0])                      # set flux at Neumann bnd
Param.g  = np.array([0.0])                      # set head at Dirichlet bnd
[B,N,fn] = build_bnd(Param,grid,I)              # Build constraint matrix and basis for its nullspace
fs = np.zeros([grid.N,1])                       # r.h.s. (zero)
L = -np.mat(D)*np.mat(G)                        # Laplacian

f = fs+fn

u = solve_lbvp(L,f,B,Param.g,N)                 # Solve linear boundary value problem

#plot
fig, ax= plt.subplots()
ax.plot(grid.xc,u,'r-',label='u')
legend = ax.legend(loc='upper left', shadow=False, fontsize='x-large')
ax.set_xlabel('Position')
ax.set_ylabel('Head')
'''
