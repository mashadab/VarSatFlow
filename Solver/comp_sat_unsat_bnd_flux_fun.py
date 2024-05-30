#A function to find the mass conservative flux at the saturated
#unsaturated boundary
'''
Input:
    dof_y_faces: y faces on the saturated-unsaturated boundary
    qs_yfaces: saturated flux at y faces
    fs_yfaces: unsaturated flux at y faces
    ytop: cell above the boundary
    ybot: cell below the boundary
    C: volume fraction of water
    phi: porosity
    s_wt: saturation threshold
'''
import numpy as np

def find_top_bot_cells(dof_f_bnd,D,Grid):
    
    DD = (D[:,dof_f_bnd-1]).toarray()
    
    dof_bnd = (np.argwhere(D[:,dof_f_bnd-1].toarray()!=0)+1)[:,0]
    dof_bnd = np.sort(np.ndarray.flatten(dof_bnd))
    
    ybot = dof_bnd[::2]
    ytop = dof_bnd[1::2]
    return ytop,ybot

def find_left_right_cells_old(dof_f_bnd,D,Grid):
    
    DD = (D[:,dof_f_bnd-1]).toarray()
    
    dof_bnd       = (np.argwhere(D[:,dof_f_bnd-1].toarray()!=0)+1)[:,0]
    dof_bnd_index = (np.argwhere(D[:,dof_f_bnd-1].toarray()!=0)+1)[:,1]
    
    dof_bnd = (np.ndarray.flatten(dof_bnd))

    print(np.shape(dof_bnd))
    print(dof_bnd)
    xleft  = np.intersect1d(dof_bnd,dof_bnd - Grid.Ny)
    xright = np.intersect1d(dof_bnd,dof_bnd + Grid.Ny) 
    #xleft  = np.setdiff1d(xleft,xright)
    #xright = np.setdiff1d(xright,xleft)
    print(xleft) 
    print(xright) 

    return xleft,xright

def find_left_right_cells(dof_f_bnd,D,Grid):
    
    xleft  = dof_f_bnd  - Grid.Ny
    xright = dof_f_bnd 
    return xleft,xright


def comp_sat_unsat_bnd_flux(qs_yfaces,fs_yfaces,ytop,ybot,C,phi,s_wt):
    
    #calculating the shock speed using Rankine-Hugoniot condition
    
    lambda_ = (qs_yfaces - fs_yfaces)/(np.maximum(C[ytop-1],C[ybot-1]) - np.minimum(C[ytop-1],C[ybot-1]))
    #print(lambda_)
    #setting the flux using RH condition
    flux = (lambda_ >= 0) * (C[ytop-1]< (s_wt*phi[ytop-1])) * qs_yfaces + \
           (lambda_ >  0) * (C[ybot-1]< (s_wt*phi[ybot-1])) * fs_yfaces + \
           (lambda_ <  0) * (C[ytop-1]< (s_wt*phi[ytop-1])) * fs_yfaces + \
           (lambda_ <= 0) * (C[ybot-1]< (s_wt*phi[ybot-1])) * qs_yfaces 
    #print(flux[:,0])           
    return flux[:,0]

'''
def comp_sat_unsat_bnd_flux_new(qs_yfaces,fs_yfaces,ytop,ybot,C,phi,s_wt):
    
    #calculating the shock speed using Rankine-Hugoniot condition
    
    #lambda_ = (qs_yfaces - fs_yfaces)/(np.maximum(C[ytop-1],C[ybot-1]) - np.minimum(C[ytop-1],C[ybot-1]))
    
    lambda_ = 1/(C[ytop-1] - C[ybot-1])* (qs_yfaces*(C[ytop-1]<=(s_wt*phi[ytop-1])) - fs_yfaces*)
        
    
    #print(lambda_)
    #setting the flux using RH condition
    flux = (lambda_ >= 0) * (C[ytop-1]<=(s_wt*phi[ytop-1])) * qs_yfaces + \
           (lambda_ >  0) * (C[ytop-1]> (s_wt*phi[ytop-1])) * fs_yfaces + \
           (lambda_ <  0) * (C[ytop-1]< (s_wt*phi[ytop-1])) * fs_yfaces + \
           (lambda_ <= 0) * (C[ytop-1]> (s_wt*phi[ytop-1])) * qs_yfaces 
    #print(flux[:,0])           
    return flux[:,0]
'''
