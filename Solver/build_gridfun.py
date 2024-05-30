import numpy as np
    
def build_grid(Grid):
    # Author: Mohammad Afzal Shadab
    # Date: 01/27/2020
    
    # This function computes takes in minimal definition of the computational
    # domain and grid and computes all containing all pertinent information 
    # about the grid. 
    
    # Input:
    # Grid.xmin = left boundary of the domain
    # Grid.xmax = right boundary of the domain
    # Grid.Nx   = number of grid cells
    # Output: (suggestions)
    # Grid.Lx = length of the domain
    # Grid.dx = cell width
    # Grid.xc = vector of cell center locations
    # Grid.xf = vector of cell face locations
    # Grid.Nfx = number of fluxes in x-direction
    # Grid.dof_xmin = degrees of freedom corrsponding to the cells along the x-min boundary
    # Grid.dof_xmax = degrees of freedom corrsponding to the cells along the x-max boundary

    # Example call: 
    # >> Grid.xmin = 0; Grid.xmax = 1; Grid.Nx = 10; 
    # >> Grid = build_grid(Grid);
    
    # Set up the geometry
    if not hasattr(Grid,'xmin'):
        Grid.xmin = 0
        print("Grid.xmin is not defined and has been set to zero.\n")
    if not hasattr(Grid,'xmax'):
        Grid.xmax = 1 
        print("Grid.xmax is not defined and has been set to 10.\n")
    if not hasattr(Grid,'Nx'): 
        Grid.Nx   = 10
        print("Grid.Nx is not defined and has been set to 10.\n")
        
    Grid.Lx = Grid.xmax-Grid.xmin    # domain length in x
    Grid.dx = Grid.Lx/Grid.Nx        # dx of the gridblocks
    
    # Number for fluxes
    Grid.Nfx = (Grid.Nx+1)
    
    # x coords of the corners of the domain
    Grid.xdom = [Grid.xmin, Grid.xmax]
    
    #Set up mesh for plotting
    #xcoords of the cell centers    
    #Grid.xc = [Grid.xmin+Grid.dx/2:Grid.dx:Grid.xmax-Grid.dx/2] # x-coords of gridblock centers
    #temp = [xmin+xi*Grid.dx + Grid.dx/2 for xi in range(Grid.Nx)]
    Grid.xc = np.transpose(np.linspace(Grid.xmin+Grid.dx/2, Grid.xmax-Grid.dx/2, Grid.Nx))
    Grid.xf = np.transpose(np.linspace(Grid.xmin, Grid.xmax, Grid.Nx+1)) # x-coords of gridblock faces
    
    # Set up dof vectors
    Grid.N = Grid.Nx                  # total number of gridblocks
    Grid.dof   = np.transpose([xi+1 for xi in range(Grid.Nx)])            # cell centered degree of freedom/gridblock number
    Grid.dof_f = np.transpose([xi+1 for xi in range(Grid.Nfx)])           # face degree of freedom/face number
    
    # Boundary dof's
    # Boundary cells
    # make more efficient by avoidng DOF
    #DOF = np.reshape(Grid.dof,Grid.Ny,Grid.Nx)
    Grid.dof_xmin = 1
    Grid.dof_xmax = Grid.Nx
    
    # Boundary faces
    # DOFx = reshape([1:Grid.Nfx],Grid.Ny,Grid.Nx+1
    Grid.dof_f_xmin = 1
    Grid.dof_f_xmax = Grid.Nfx
    
    
    Grid.A = np.ones((Grid.Nfx,1))
    Grid.V  = np.ones((Grid.N,1))*Grid.dx
    
    return Grid;