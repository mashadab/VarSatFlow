#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 17:19:30 2022

@author: afzal-admin
"""
import numpy as np

#n:number of points
#filename
#tf_new: new time

def spin_up(file_name,Grid,tf_new,n,t_interest):
    print('Spinup initiated')
    data = np.load(file_name)
    t=data['t']
    s_w_sol =data['s_w']
    phi=data['phi']
    
    #geometry
    '''
    Xc=data['Xc']
    Yc=data['Yc']
    Xc_col=data['Xc_col']
    Yc_col=data['Yc_col']
    Grid.Nx=data['Grid_Nx']
    Grid.Ny=data['Grid_Ny']
    Grid.xc=data['Grid_xc']
    Grid.yc=data['Grid_yc']
    Grid.xf=data['Grid_xf']
    Grid.yf=data['Grid_yf']
    '''
    s_w = s_w_sol[:,-1]
    time = t[-1]
    t_new = np.linspace(t[-1],tf_new,n)
    t_interest = np.append(t_interest,t_new[1:])
    i = 1
    s_w = np.transpose([s_w])
    tf =tf_new
    s_w_sol = s_w_sol
    t = t.tolist()
    print('Spinup finished')
    return Grid,s_w,time,tf_new,time,t_interest,i,tf,t,s_w_sol


def spin_up_C(file_name,Grid,tf_new,n,t_interest):
    print('Spinup initiated')
    data = np.load(file_name)    
    t=data['t']
    C_sol =data['C_sol']
    phi=data['phi']
    flux_sol = data['flux_sol']
    [dummy,endstop] = np.shape(C_sol)
    C = C_sol[:,-1]
    flux_vert =  flux_sol[:,-1]
    
    
    #geometry
    '''
    Xc=data['Xc']
    Yc=data['Yc']
    Xc_col=data['Xc_col']
    Yc_col=data['Yc_col']
    Grid.Nx=data['Grid_Nx']
    Grid.Ny=data['Grid_Ny']
    Grid.xc=data['Grid_xc']
    Grid.yc=data['Grid_yc']
    Grid.xf=data['Grid_xf']
    Grid.yf=data['Grid_yf']
    '''

    time = t[-1]
    t_new = np.linspace(t[-1],tf_new,n)
    t_interest = np.append(t_interest,t_new[1:])
    i = 1
    C = np.transpose([C])
    tf =tf_new
    t = t.tolist()
    print('Spinup finished')
    return Grid,C,time,time,t_interest,i,tf,t,C_sol