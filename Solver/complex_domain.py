#Flow in complex domains

import os


from classes import *

# Part (a) Crater in a square domain
x0 = 0;
y0 = 0;
Rc = 0.1

Grid.xmin = -1/2; Grid.xmax = 1/2; Grid.Nx = 25
Grid.ymin = -1/2; Grid.ymax = 1/2; Grid.Ny = 25

build_grid(Grid)

[D,G,I] = build_ops(Grid)
[Xc,Yc] = np.meshgrid(Grid.xc,Grid.yc)
Xc_col  = np.reshape(np.transpose(Xc), (Grid.N,-1))    #building the single X vector
Yc_col  = np.reshape(np.transpose(Yc), (Grid.N,-1))    #building the single Y vector
Avg     = comp_algebraic_mean(Grid)

'''
plt.figure(figsize=(8,8) , dpi=100)
plt.plot([Grid.xf,Grid.xf],[Grid.ymin*np.ones(Grid.Nx+1),Grid.ymax*np.ones(Grid.Nx+1)],'k')
plt.plot([Grid.xmin*np.ones(Grid.Ny+1),Grid.xmax*np.ones(Grid.Ny+1)],[Grid.yf,Grid.yf],'k')
circle = plt.Circle((x0,y0),Rc, color='r', fill=False,linewidth=2)
plt.gca().add_patch(circle)
plt.axis('equal')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.xlim([-0.6,0.6])
plt.ylim([-0.6,0.6])
plt.tight_layout() 
plt.savefig(f'../All figures/step1.pdf',bbox_inches='tight', dpi = 600)
'''

# Part (b) Modify domain to cut out crater

# Step 1: Find cells in the crater

def find_crater_dofs(x0,y0,r,Grid,Xc_col,Yc_col):
    dof_in = [];
    dist_center = (np.sqrt((x0-Xc_col)**2+(y0-Yc_col)**2))[:,0]
    dof_in =Grid.dof[dist_center <= r]
    dof_out = np.setdiff1d(Grid.dof,dof_in)
    return dof_in, dof_out

[dof_inact,dof_act] = find_crater_dofs(x0,y0,Rc,Grid,Xc_col,Yc_col)

'''
plt.figure(figsize=(8,8) , dpi=100)
plt.plot([Grid.xf,Grid.xf],[Grid.ymin*np.ones(Grid.Nx+1),Grid.ymax*np.ones(Grid.Nx+1)],'k')
plt.plot([Grid.xmin*np.ones(Grid.Ny+1),Grid.xmax*np.ones(Grid.Ny+1)],[Grid.yf,Grid.yf],'k')
circle = plt.Circle((x0,y0),Rc, color='r', fill=False,linewidth=2,label='Crater')
plt.gca().add_patch(circle)
plt.plot(Xc_col[dof_inact-1,0],Yc_col[dof_inact-1,0],'ro',label='Inactive', markerfacecolor="None")
plt.plot(Xc_col[dof_act-1,0],Yc_col[dof_act-1,0],'bo',label='Active', markerfacecolor="None")
plt.legend()
plt.axis('equal')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.xlim([-0.6,0.6])
plt.ylim([-0.6,0.6])
plt.legend(loc = 'upper right')
plt.tight_layout() 
plt.savefig(f'../All figures/step2.pdf',bbox_inches='tight', dpi = 600)
'''

# Step 2: Finding the boundary of the crater

def find_faces(cell_dofs,D,Grid):
    
    DD = D[cell_dofs-1,:]  #find the faces corresponding to inactive cells
    ## Exterior faces
    # Sum columns of DD. In the columns corresponding to internal faces the
    # entries will cancel. Therefore, only the sum of the columns corresponding
    # to exterior faces will be non-zero.
    
    dof_f_ext = Grid.dof_f[np.ravel(np.abs(DD.sum(axis=0))>1e-10)]  
    return dof_f_ext

def find_all_faces(cell_dofs,D,Grid):
    
    DD = (D[cell_dofs-1,:])  #find the faces corresponding to inactive cells
    ## Exterior faces
    # Sum columns of DD. In the columns corresponding to internal faces the
    # entries will cancel. Therefore, only the sum of the columns corresponding
    # to exterior faces will be non-zero.
    
    dof_f_ext     = Grid.dof_f[np.ravel((np.abs(DD)).sum(axis=0))>1e-10]  
    
    return dof_f_ext

def find_all_y_faces(cell_dofs,D,Grid):
    
    DD = (D[cell_dofs-1,:]).toarray()  #find the faces corresponding to inactive cells
    ## Exterior faces
    # Sum columns of DD. In the columns corresponding to internal faces the
    # entries will cancel. Therefore, only the sum of the columns corresponding
    # to exterior faces will be non-zero.
    
    dof_f_ext     = Grid.dof_f[np.ravel((np.abs(DD)).sum(axis=0))>1e-10]   
    
    dof_f_y = [i for i in range(Grid.Nfx,Grid.Nf+1)]
    
    dof_f_y = np.intersect1d(dof_f_ext,dof_f_y)
    
    return dof_f_y

def find_all_x_faces(cell_dofs,D,Grid):
    
    DD = (D[cell_dofs-1,:]).toarray()  #find the faces corresponding to inactive cells
    ## Exterior faces
    # Sum columns of DD. In the columns corresponding to internal faces the
    # entries will cancel. Therefore, only the sum of the columns corresponding
    # to exterior faces will be non-zero.
    
    dof_f_ext     = Grid.dof_f[np.ravel((np.abs(DD)).sum(axis=0))>1e-10]   
    
    dof_f_x = [i for i in range(1,Grid.Nfx+1)]
    
    dof_f_x = np.intersect1d(dof_f_ext,dof_f_x)
    
    return dof_f_x


'''
dof_f_crater = find_faces(dof_inact,D,Grid)
#all_inact_faces = find_all_faces(dof_inact,D,Grid)


[Xx,Yx] = np.meshgrid(Grid.xf,Grid.yc)
[Xy,Yy] = np.meshgrid(Grid.xc,Grid.yf)
Xx  = np.reshape(np.transpose(Xx), ((Grid.Nx+1)*Grid.Ny,-1))    #building the single X vector
Xy  = np.reshape(np.transpose(Xy), ((Grid.Nx+1)*Grid.Ny,-1))    #building the single Y vector
Yx  = np.reshape(np.transpose(Yx), (Grid.Nx*(Grid.Ny+1),-1))    #building the single X vector
Yy  = np.reshape(np.transpose(Yy), (Grid.Nx*(Grid.Ny+1),-1))    #building the single Y vector
Xf_col = np.vstack((Xx,Xy))
Yf_col = np.vstack((Yx,Yy))


plt.figure(figsize=(8,8) , dpi=100)
plt.plot([Grid.xf,Grid.xf],[Grid.ymin*np.ones(Grid.Nx+1),Grid.ymax*np.ones(Grid.Nx+1)],'k')
plt.plot([Grid.xmin*np.ones(Grid.Ny+1),Grid.xmax*np.ones(Grid.Ny+1)],[Grid.yf,Grid.yf],'k')
circle = plt.Circle((x0,y0),Rc, color='r', fill=False,linewidth=2,label='Crater')
plt.gca().add_patch(circle)
plt.plot(Xf_col[dof_f_crater-1,0],Yf_col[dof_f_crater-1,0],'bX',label='Crater face')
plt.axis('equal')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.xlim([-0.6,0.6])
plt.ylim([-0.6,0.6])
plt.legend(loc = 'upper right')
plt.tight_layout() 
plt.savefig(f'../All figures/step3.pdf',bbox_inches='tight', dpi = 600)
'''

# Step 3: Find the cells along the boundary

def find_bnd_cells(dof_act,dof_inact,dof_f_bnd,D,Grid):
    
    DD = (D[:,dof_f_bnd-1]).toarray()
    #dof_bnd = Grid.dof[(np.abs(np.sum(DD,axis=1)))>=1e-10]
    dof_bnd = np.unique((np.argwhere(D[:,dof_f_bnd-1].toarray()!=0)+1)[:,0])
    dof_bnd_inact = np.intersect1d(dof_bnd,dof_inact)  
    dof_bnd_act   = np.setdiff1d(dof_bnd,dof_inact)  
    
    return dof_bnd_inact,dof_bnd_act,dof_bnd

'''
[dof_bnd_inact,dof_bnd_act,dof_bnd] = find_bnd_cells(dof_act,dof_inact,dof_f_crater,D,Grid)


plt.figure(figsize=(8,8) , dpi=100)
plt.plot([Grid.xf,Grid.xf],[Grid.ymin*np.ones(Grid.Nx+1),Grid.ymax*np.ones(Grid.Nx+1)],'k')
plt.plot([Grid.xmin*np.ones(Grid.Ny+1),Grid.xmax*np.ones(Grid.Ny+1)],[Grid.yf,Grid.yf],'k')
circle = plt.Circle((x0,y0),Rc, color='r', fill=False,linewidth=2,label='Crater')
plt.gca().add_patch(circle)
plt.plot(Xc_col[dof_bnd_inact-1,0],Yc_col[dof_bnd_inact-1,0],'ro',label='Inactive', markerfacecolor="None")
plt.plot(Xc_col[dof_bnd_act-1,0],Yc_col[dof_bnd_act-1,0],'bo',label='Active', markerfacecolor="None")
plt.plot(Xf_col[dof_f_crater-1,0],Yf_col[dof_f_crater-1,0],'bX',label='Crater face')
plt.axis('equal')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.xlim([-0.6,0.6])
plt.ylim([-0.6,0.6])
plt.legend(loc = 'upper right')
plt.tight_layout() 
plt.savefig(f'../All figures/step4.pdf',bbox_inches='tight', dpi = 600)


#Solving a constrained problem
x0 = 0
y0 = 0
Rc = 0.1
Pi_out = 0.3
Pi_crater = 0.2

Grid.xmin = -1/2; Grid.xmax = 1/2; Grid.Nx = 250
Grid.ymin = -1/2; Grid.ymax = 1/2; Grid.Ny = 250

build_grid(Grid)

[D,G,I] = build_ops(Grid)
[Xc,Yc] = np.meshgrid(Grid.xc,Grid.yc)
Xc_col  = np.reshape(np.transpose(Xc), (Grid.N,-1))    #building the single X vector
Yc_col  = np.reshape(np.transpose(Yc), (Grid.N,-1))    #building the single Y vector
Avg     = comp_algebraic_mean(Grid)

[dof_inact,dof_act] = find_crater_dofs(x0,y0,Rc,Grid,Xc_col,Yc_col)
dof_f_crater        = find_faces(dof_inact,D,Grid)
[dof_bnd_inact,dof_bnd_act,dof_bnd] = find_bnd_cells(dof_act,dof_inact,dof_f_crater,D,Grid)
N_crater            = len(dof_bnd_act)

#Step 1: Modify the gradient to import natural BC at the crater
G_small = zero_rows(G,dof_f_crater-1)

#Step 2 Eliminate inactive cells by putting them to constraint matrix
BC.dof_dir = np.concatenate((Grid.dof_xmax, \
                             dof_bnd_act, \
                             dof_inact),axis=0)
               
BC.dof_f_dir= np.concatenate((Grid.dof_f_xmax, \
                             dof_f_crater),axis=0)

Nan_matrix = np.empty((len(dof_inact),1))
Nan_matrix[:] = np.nan
    
BC.g       =  np.concatenate((Pi_out*np.ones((Grid.Ny,1)), \
              Pi_crater*np.ones((N_crater,1)),\
              -Nan_matrix),axis=0)   
    
BC.dof_neu   = np.array([])
BC.dof_f_neu = np.array([])
BC.qb = np.array([])

[B,N,fn]   = build_bnd(BC, Grid, I)

L  = -D @ G_small
fs =  np.ones((Grid.N,1))

hD_conf = solve_lbvp(L , fs + fn, B, BC.g, N)

fig = plt.figure(figsize=(15,7.5) , dpi=100)
plot = [plt.contourf(Xc, Yc, np.transpose(hD_conf.reshape(Grid.Nx,Grid.Ny)),50,cmap="coolwarm",levels=20,vmin = 0.2, vmax = 0.4)]
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.xlabel(r'$x''$')
plt.ylabel(r'$y''$')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymin,Grid.ymax])
plt.axis('scaled')
mm = plt.cm.ScalarMappable(cmap=cm.coolwarm)
mm.set_array(hD_conf)
mm.set_clim(0.2,0.4)
clb = plt.colorbar(mm, pad=0.1)
clb.set_label(r'$h_D$', labelpad=1, y=1.075, rotation=0)
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
plt.savefig(f'../All figures/crater.pdf',bbox_inches='tight', dpi = 600)
'''
