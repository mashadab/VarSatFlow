#Classes for Gravity separation 2D uniform core adding thermodynamics
# author: Mohammad Afzal Shadab
# date: 2/24/2020
import warnings
warnings.filterwarnings('ignore')
from IPython import get_ipython
#get_ipython().magic('reset -sf') #for clearing everything
#get_ipython().run_line_magic('matplotlib', 'qt') #for plotting in separate window

# import python libraries
import numpy as np
import scipy.sparse as sp
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
from matplotlib.ticker import LinearLocator, FormatStrFormatter
plt.rcParams.update({'font.size': 20})
plt.rcParams.update({'font.family': 'Serif'})
from scipy.sparse import spdiags


from build_gridfun2D import build_grid
from build_opsfun2D_optimized import build_ops, zero_rows
#import comp_meanfun2D
from comp_algebraicmean_optimized import comp_algebraic_mean
from comp_harmonicmean import comp_harmonicmean
from mobilityfun import mobility, mobilityvec
from build_bndfun_optimized import build_bnd
from solve_lbvpfun_optimized import solve_lbvp
from flux_upwindfun2D_optimized import flux_upwind  #instead of from adv_opfun2D import adv_opfun
from frac_flow_derivativefun import frac_flow_derivative
from psi_derivativefun import psi_derivative
from time import perf_counter 
from eval_phase_behavior import eval_phase_behavior, enthalpyfromT, eval_h


#Conversion constants
day2s = 24*60*60
hr2s  = 60*60
yr2s  = 365.25*24*60*60
km2m  = 1e3

class grid:
    def __init__(self):
        self.xmin = np.array([])
        self.xmax = np.array([])
        self.Nx = np.array([])
        self.N = np.array([])

class BC:
    def __init__(self):
        self.dof_dir = np.array([])
        self.dof_f_dir = np.array([])
        self.g = np.array([])     

class BC_P:
    def __init__(self):
        self.dof_dir = np.array([])
        self.dof_f_dir = np.array([])
        self.g = np.array([])
        self.dof_neu = np.array([])
        self.dof_f_neu = np.array([])
        self.qb = np.array([])


class Grid:
    def __init__(self):
        self.xmin = np.array([])
        self.xmax = np.array([])
        self.Nx = np.array([])
        self.N = np.array([])

class Param:
    class P:
        def __init__(self):
            self.dof_dir = np.array([])       # identify cells on Dirichlet bnd
            self.dof_f_dir = np.array([])     # identify faces on Dirichlet bnd
            self.dof_neu = np.array([])       # identify cells on Neumann bnd
            self.dof_f_neu = np.array([])     # identify faces on Neumann bnd
            self.g = np.array([])             # column vector of non-homogeneous Dirichlet BCs (Nc X 1)
            self.qb = np.array([])         

    class S:
        def __init__(self):
            self.dof_dir = np.array([])       # identify cells on Dirichlet bnd
            self.dof_f_dir = np.array([])     # identify faces on Dirichlet bnd
            self.dof_neu = np.array([])       # identify cells on Neumann bnd
            self.dof_f_neu = np.array([])     # identify faces on Neumann bnd
            self.g = np.array([])             # column vector of non-homogeneous Dirichlet BCs (Nc X 1)
            self.qb = np.array([])

    class H:
        def __init__(self):
            self.dof_dir = np.array([])       # identify cells on Dirichlet bnd
            self.dof_f_dir = np.array([])     # identify faces on Dirichlet bnd
            self.dof_neu = np.array([])       # identify cells on Neumann bnd
            self.dof_f_neu = np.array([])     # identify faces on Neumann bnd
            self.g = np.array([])             # column vector of non-homogeneous Dirichlet BCs (Nc X 1)
            self.qb = np.array([])

#Colors
brown  = [181/255 , 101/255, 29/255]
red    = [255/255 ,255/255 ,255/255 ]
blue   = [ 30/255 ,144/255 , 255/255 ]
green  = [  0/255 , 166/255 ,  81/255]
orange = [247/255 , 148/255 ,  30/255]
purple = [102/255 ,  45/255 , 145/255]
brown  = [155/255 ,  118/255 ,  83/255]
tan    = [199/255 , 178/255 , 153/255]
gray   = [100/255 , 100/255 , 100/255]

