#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 19:18:56 2022

@author: afzal-admin
"""
import numpy as np
import matplotlib.pyplot as plt

#The analytic result for Gravity Current on Horizontal aquifer


#Parameters
g     = 9.808  #Acceleration due to gravity, m^2/s
theta = 0      #Angle of slope
rho_w = 999.9  #Density of water, kg/m^3 
rho_g = 1.0    #Density of gas, kg/m^3 
Delta_rho = rho_w - rho_g #Density difference
phi_0 = 0.72   #Porosity of the medium
mu    = 1e-3   #Viscosity of water, Pa.s
k0    = 5.6e-11#absolute permeability m^2 in pore space Meyer and Hewitt 2017
n     = 2.0    #Power law exponent for porosity-permeability relation
Q_0   = 100e3*10#Total volume of water m^3 L = 100km* H = 10m
S     = phi_0**(n-1) * Delta_rho * g * k0 / mu
gamma = 0      #Coefficient of water flux Q(t) = Q_0*t^gamma
yr2s  = 365.25*24*60*60 #Year to second conversion

#Calculating parameters
H_gamma  = (Q_0/S**gamma)**(1/(2-gamma))
D_gamma  = (S**2/Q_0)**(1/(2-gamma))


#Analytic solution
xi_0 =  lambda phi_0: (9/phi_0)**(1/3)
f0   =  lambda xi,xi_0: (xi_0**2 - xi**2)/6  #Only for gamma = 0 
h    =  lambda t,xi,phi_0 : H_gamma*(D_gamma*t)**((2*gamma - 1)/3) * f0(xi,xi_0(phi_0))
x    =  lambda t,xi: xi*H_gamma*(D_gamma*t)**((gamma+1)/3)


#Plotting
xi   = np.linspace(0,xi_0(phi_0),1000)
plt.figure(figsize=(10,10),dpi=100)
t = 10000*yr2s #time in seconds
plt.plot(x(t,xi)/(1e3),h(t,xi,phi_0),'r-',label='t=%.2f years' %(t/yr2s))

t = 100000*yr2s #time in seconds
plt.plot(x(t,xi)/(1e3),h(t,xi,phi_0),'b-',label='t=%.2f years' %(t/yr2s))

t = 1000000*yr2s #time in seconds
plt.plot(x(t,xi)/(1e3),h(t,xi,phi_0),'k-',label='t=%.2f years' %(t/yr2s))
plt.xlabel(r'$x$ [km]')
plt.ylabel(r'$h$ [m]')
plt.legend()
plt.xlim([x(t,xi)[0]/(1e3),x(t,xi)[-1]/(1e3)])
plt.tight_layout()
plt.savefig(f'../Figures/gravity-current-solution.pdf',bbox_inches='tight', dpi = 600)









