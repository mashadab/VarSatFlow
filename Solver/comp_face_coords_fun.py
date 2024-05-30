#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 22:45:31 2022

@author: afzal-admin
"""
import numpy as np

def comp_face_coords(dof_faces,Grid):
    
    # Coordinates of the face centers
    Xx,Yx = np.meshgrid(Grid.xf,Grid.yc) #x-face
    Xy,Yy = np.meshgrid(Grid.xc,Grid.yf) #y-face
    
    Xx_col = np.reshape(np.transpose(Xx),((Grid.Nx+1)*Grid.Ny,-1))
    Yx_col = np.reshape(np.transpose(Yx),((Grid.Nx+1)*Grid.Ny,-1))
    Xy_col = np.reshape(np.transpose(Xy),( Grid.Nx   *(Grid.Ny+1),-1))
    Yy_col = np.reshape(np.transpose(Yy),( Grid.Nx   *(Grid.Ny+1),-1))
    
    Xf = np.vstack([Xx_col,Xy_col])
    Yf = np.vstack([Yx_col,Yy_col])
    
    # Coordinates of x-face centers
    Xfx = Xf[0:Grid.Nfx] 
    Yfx = Yf[0:Grid.Nfx] 
    
    # Coordinates of y-face centers
    Xfy = Xf[Grid.Nfx:Grid.Nf] 
    Yfy = Yf[Grid.Nfx:Grid.Nf]
    
    # Center coordinates of bounding faces
    dof_x_faces = dof_faces[dof_faces-1 < Grid.Nfx]
    dof_y_faces = dof_faces[dof_faces-1 >=Grid.Nfx]-Grid.Nfx;
    
    
    Xbx = np.transpose(Xfx[dof_x_faces-1]); Ybx = np.transpose(Yfx[dof_x_faces-1])
    Xby = np.transpose(Xfy[dof_y_faces-1]); Yby = np.transpose(Yfy[dof_y_faces-1])
    
    
    # Create two matrices size 2 by Nb containing the 
    # x and y coords of the endpoints of the faces
    X_faces = np.vstack([np.hstack([Xbx,Xby+Grid.dx/2]),np.hstack([Xbx,Xby-Grid.dx/2])])
    Y_faces = np.vstack([np.hstack([Ybx+Grid.dy/2,Yby]),np.hstack([Ybx-Grid.dy/2,Yby])])    

    return X_faces,Y_faces