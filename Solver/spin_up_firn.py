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

def spin_up_firn(file_name,Grid,tf_new,n,t_interest):
    print('Spinup initiated')
    data = np.load(file_name)
    t=data['t']
    s_w_sol =data['s_w_sol']
    #phi=data['phi']
    phi_w_sol =data['phi_w_sol']
    phi_i_sol =data['phi_i_sol']
    H_sol =data['H_sol']
    T_sol =data['T_sol']
    s_w_sol =data['s_w_sol']
    
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
    phi_i = phi_i_sol[:,-1]
    phi_w = phi_w_sol[:,-1]
    phi_nw = 1- phi_i - phi_w
    phi = 1 - phi_i
    H   = H_sol[:,-1]
    T   = T_sol[:,-1]
    time = t[-1]
    t_new = np.linspace(t[-1],tf_new,n)
    t_interest = np.append(t_interest,t_new[1:])
    i = 1
    s_w = np.transpose([s_w])
    tf =tf_new
    s_w_sol = s_w_sol
    t = t.tolist()
    print('Spinup finished')
    return Grid,s_w,time,tf_new,time,t_interest,i,tf,t,s_w_sol,phi_w_sol,phi_w,phi_i_sol,phi_i,H_sol,H,T_sol,T,phi