#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 17:02:49 2022

@author: afzal-admin
"""

import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, './Drainage-analytic')


def reservoir_drainage():
    from scipy.io import loadmat
    data = loadmat('./Drainage-analytic/Analytic-solution-drainage.mat')
    h = data['U'][:,0]
    x = data['xc'][:,0] 
    return x,h




def reservoir_drainage_old():
    def fun(x, y):
        return np.vstack((y[1], -1 - y[1]**2/y[0]))
    
    def bc(ya, yb):
        return np.array([ya[1], yb[0]])
    
    
    x = np.linspace(0, 1, 100)
    y = 0.3*np.ones((2, x.size))
    y[1,:] = -0.3
    
    sol = solve_bvp(fun, bc, x, y,tol=1e-10)
    if sol.status != 0:
        print("WARNING: sol.status is %d" % sol.status)
    print(sol.message)
    
    return sol.x,sol.y[0]


x,h = reservoir_drainage()
x_old,h_old = reservoir_drainage_old()

plt.figure(figsize=(10,5))
plt.plot(x,h, 'r-', label='MATLAB')
plt.plot(x_old,h_old, 'b-', label='Python')
plt.legend()
plt.savefig('../Figures/Nonlinear_BVP_solution.pdf',dpi=100)

