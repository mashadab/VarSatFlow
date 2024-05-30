# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

def mesh_grid(x,y): 
    # author: Mohammad Afzal Shadab
    # date: 06/06/2020
    # description:
    #Making the meshgrid() function 
    #Input: two vectors x,y containing the grid points
    #Output: X and Y matrices used by all 2D matplotlib.pyplot functions
    
    X = np.ones((len(y),1))
    X = np.kron(X,x)
    
    Y = np.ones((len(x)))
    Y = np.kron(Y,np.transpose([y]))    
    
    return X,Y;

Nx = 11; Ny =11; N = Nx*Ny

x = np.linspace(0,10,Nx)
y = np.linspace(0,20,Ny)

X =  np.zeros((Ny,Nx))
Y =  np.zeros((Ny,Nx))

[X,Y] = np.meshgrid(x,y)                 #mesh_grid(x,y)

X1 = np.reshape(np.transpose(X), (N,-1)) #Flattened changing in y-direction first so x is constant for Ny column entries

Y1 = np.reshape(np.transpose(Y), (N,-1)) #Flattened column vector in y-direction

f = lambda x,y: x*y
g = lambda x,y: y

fig = plt.figure()
ax1 = fig.gca(projection='3d')
ax1.plot_surface(X, Y,f(X,Y), cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$y$')
ax1.set_zlabel(r'$f(x,y)$')

fig = plt.figure()
ax1 = fig.gca(projection='3d')
ax1.plot_surface(X, Y,g(X,Y), cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$y$')
ax1.set_zlabel(r'$g(x,y)$')
