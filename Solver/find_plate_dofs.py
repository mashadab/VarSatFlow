#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(1, '../../solver')
from complex_domain import find_faces

import numpy as np

def find_plate_dofs(xleft,xright,ytop,ybottom,Grid,Xc_col,Yc_col,D):
    
    dof_x = Grid.dof[np.intersect1d(np.argwhere(Xc_col>= xleft),np.argwhere(Xc_col<= xright))][1:-1]
    dof_y = Grid.dof[np.intersect1d(np.argwhere(Yc_col>= ytop) ,np.argwhere(Yc_col<= ybottom))][1:-1]   

    dof_plate       = np.intersect1d(dof_x,dof_y)
    dof_out         = np.setdiff1d(Grid.dof,dof_plate)
    dof_f_plate_bnd = find_faces(dof_plate,D,Grid)
    
    return dof_plate, dof_out, dof_f_plate_bnd


def multiple_plate_dofs(xleft,xright,ytop,ybottom,Grid,Xc_col,Yc_col,D):
    
    length = xright - xleft
    height = ybottom - ytop
    
    #five plates: 3 on bottom
    dof_plate1, dof_out1, dof_f_plate_bnd1 = find_plate_dofs(xleft+length/15,xleft+length/3,ytop+2*height/5,ytop+3*height/5,Grid,Xc_col,Yc_col,D)
    dof_plate2, dof_out2, dof_f_plate_bnd2 = find_plate_dofs(xleft+length/2-length/8,xleft+length/2+length/8,ytop+2*height/5,ytop+3*height/5,Grid,Xc_col,Yc_col,D)    
    dof_plate3, dof_out3, dof_f_plate_bnd3 = find_plate_dofs(xright-length/3,xright-length/15,ytop+2*height/5,ytop+3*height/5,Grid,Xc_col,Yc_col,D)  


    #five plates: 5 on top
    dof_plate4, dof_out4, dof_f_plate_bnd4 = find_plate_dofs(xleft,xleft + length/10,ytop,ytop+height/5,Grid,Xc_col,Yc_col,D)
    dof_plate5, dof_out5, dof_f_plate_bnd5 = find_plate_dofs(xleft +length/4 - length/10,xleft +length/4+ length/10,ytop,ytop+height/5,Grid,Xc_col,Yc_col,D)
    dof_plate6, dof_out6, dof_f_plate_bnd6 = find_plate_dofs(xright-length/2 - length/10,xright-length/2+ length/10,ytop,ytop+height/5,Grid,Xc_col,Yc_col,D)   
    dof_plate7, dof_out7, dof_f_plate_bnd7 = find_plate_dofs(xright-length/4 - length/10,xright-length/4+ length/10,ytop,ytop+height/5,Grid,Xc_col,Yc_col,D)   
    dof_plate8, dof_out8, dof_f_plate_bnd8 = find_plate_dofs(xright- length/10,xright,ytop,ytop+height/5,Grid,Xc_col,Yc_col,D)   
          
    dof_plates = np.hstack([dof_plate1,dof_plate2,dof_plate3,dof_plate4,dof_plate5,dof_plate6,dof_plate7,dof_plate8])
    dof_outs   = np.hstack([dof_out1,dof_out2,dof_out3,dof_out4,dof_out5,dof_out6,dof_out7,dof_out8])
    dof_f_plate_bnds = np.hstack([dof_f_plate_bnd1,dof_f_plate_bnd2,dof_f_plate_bnd3,dof_f_plate_bnd4,dof_f_plate_bnd5,dof_f_plate_bnd6,dof_f_plate_bnd7,dof_f_plate_bnd8])
    
    return dof_plates, dof_outs, dof_f_plate_bnds
