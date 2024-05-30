#Coding the Richard's equation for water drainage
#Case 1: Quasi-1D test
#Mohammad Afzal Shadab
#Date modified: 27/03/22

import sys
sys.path.insert(1, '../../solver')

# import personal libraries and class
from classes import *    

from two_components_aux import *
from solve_lbvpfun_SPD import solve_lbvp_SPD
from complex_domain import find_faces, find_bnd_cells, find_all_faces, find_all_x_faces,find_all_y_faces
from comp_fluxfun import comp_flux
from scipy.integrate import solve_ivp
from comp_sat_unsat_bnd_flux_fun import comp_sat_unsat_bnd_flux, find_top_bot_cells

#for 2D
from build_gridfun2D import build_grid

class Grid_P:
    def __init__(self):
        self.xmin = []
        self.xmax = []
        self.Nx = []
        self.N = []

class BC_P:
    def __init__(self):
        self.dof_dir = []
        self.dof_f_dir = []
        self.g = []
        self.dof_neu = []
        self.dof_f_neu = []
        self.qb = []

def case(i):
        switcher={
                #Separate regions
                #Region 2: Three phase region
                3: [[0.0,0.5],[0.5,0.5]]   
             }
        return switcher.get(i,"Invalid Case")

#parameters
case_no =  3              #case number
simulation_name = f'1D-drainage'
m = 3 #Cozeny-Karman coefficient for numerator K = K0 (1-phi_i)^m
n = 2 #Corey-Brooks coefficient krw = krw0 * sw^n
s_wr = 0.0 #Residual water saturation
s_gr = 0.0 #Residual gas saturation

#test details
[u_L,u_R] = case(case_no) #left and right states
[C_L,phi_L] = u_L #water sat, porosity
[C_R,phi_R] = u_R #water sat, porosity
C_R = (1-s_gr)*phi_R #resetting the saturation
xm = 0.0 #the location of jump
sat_threshold = (1-1e-5)*(1-s_gr) #threshold for saturated region formation

#temporal
tmax = 0.25  #time scaling with respect to fc
t_interest = [0,0.05,0.1,0.15,0.2,0.25]   #swr,sgr=0

#tmax = tmax / phi_L**m   #time scaling with respect to K_0 where K_0 = f_c/phi**m
Nt   = 1000
CFL  = 1
dt = tmax / (Nt)

#Non-dimensional permeability: Harmonic mean
def f_Cm(phi,m):
    fC = np.zeros_like(phi)        
    fC = phi**m / phi_L**m           #Power law porosity
    return fC

#Rel perm of water: Upwinded
def f_Cn(C,phi,n):
    fC = np.zeros_like(phi)
    fC = ((C/phi-s_wr)/(1-s_gr-s_wr))**n    #Power law rel perm
    fC[C<=0]  = 0.0      
    return fC

#spatial
Grid.xmin =  0; Grid.xmax =1; Grid.Nx = 2; 
Grid.ymin =  0; Grid.ymax =1; Grid.Ny = 400;

Grid = build_grid(Grid)
[D,G,I] = build_ops(Grid)
Avg     = comp_algebraic_mean(Grid)

#boundary condition
BC.dof_dir   = Grid.dof_ymax
BC.dof_f_dir = Grid.dof_f_ymax
BC.C_g       = 0.99*sat_threshold*C_R*np.ones((len(Grid.dof_ymin),1))

BC.dof_neu   = np.array([])
BC.dof_f_neu = np.array([])
BC.qb = np.array([])

[B,N,fn] = build_bnd(BC, Grid, I)

[Xc,Yc] = np.meshgrid(Grid.xc,Grid.yc)                 #building the (x,y) matrix
Xc_col  = np.reshape(np.transpose(Xc), (Grid.N,-1))    #building the single X vector
Yc_col  = np.reshape(np.transpose(Yc), (Grid.N,-1))    #building the single Y vector

phi = phi_L*np.ones((Grid.N,1))
phi[Yc_col>xm] = phi_R
  

C = phi*(1-s_gr)#C_L*np.zeros((Grid.N,1))
C[Grid.dof_ymin-1,:] = 0
C[BC.dof_dir-1,:] = BC.C_g 
#C = C_L*np.ones((Grid.N,1))
#C[Grid.xc>xm,:] = C_R      #Initializing

C_sol = C.copy()

t = [0.0]
v = np.ones((Grid.Nf,1))

i = 0

while t[i]<tmax:

    C_old = C.copy() 
    flux      = (comp_harmonicmean(Avg,f_Cm(phi,m))*(flux_upwind(v, Grid)@f_Cn(C_old,phi,n)))
    flux_vert = flux.copy()
    flux_vert[Grid.dof_f<Grid.Nfx,0] = 0  #making gravity based flux 0 in x direction
    
    res = D@flux_vert  #since the gradients are only zero and 1    
    res_vert = res.copy()
    ######
    #Taking out the domain to cut off single phase region
    dof_act  = Grid.dof[C_old[:,0] / (phi[:,0]*(1-s_gr)) < sat_threshold]
    dof_inact= np.setdiff1d(Grid.dof,dof_act) #saturated cells
    if len(dof_act)<Grid.N: #when atleast one layer is present
        #############################################
        dof_f_saturated = find_faces(dof_inact,D,Grid)       

        #Eliminate inactive cells by putting them to constraint matrix
        BC_P.dof_dir = (dof_act)           
        BC_P.dof_f_dir= np.array([])
        BC_P.g       =  -Yc_col[dof_act-1]   
        BC_P.dof_neu = np.array([])
        BC_P.dof_f_neu = np.array([])
        BC_P.qb = np.array([])

        [B_P,N_P,fn_P] = build_bnd(BC_P,Grid,I)
        Kd  = comp_harmonicmean(Avg,f_Cm(phi,m)) * (Avg @ f_Cn(C_old,phi,n))
        
        
        Kd  = sp.dia_matrix((Kd[:,0],  np.array([0])), shape=(Grid.Nf, Grid.Nf))
        L = - D @ Kd @ G
        u = solve_lbvp(L,fn_P,B_P,BC_P.g,N_P)   # Non dimensional water potential
        q_w = - Kd @ G @ u
        
        BC_P.dof_dir = (dof_act)           
        BC_P.dof_f_dir= np.array([dof_f_saturated])
        BC_P.g       =  -Yc_col[dof_act-1]  
        BC_P.dof_neu = np.array([])
        BC_P.dof_f_neu = np.array([])
        BC_P.qb = np.array([])
        fs_P = np.zeros((len(u),1))
        
        print(i,t[i],'Saturated cells',Grid.N-len(dof_act))

        
        #upwinding boundary y-directional flux
        #finding boundary faces
        dof_ysat_faces = dof_f_saturated[dof_f_saturated>=Grid.Nfx]



        #new try
        ytop,ybot               = find_top_bot_cells(dof_ysat_faces,D,Grid)
        q_w[dof_ysat_faces-1,0] = comp_sat_unsat_bnd_flux(q_w[dof_ysat_faces-1],flux_vert[dof_ysat_faces-1],ytop,ybot,C,phi,sat_threshold)
        
            
        #find all saturated faces
        dof_sat_faces = find_all_faces(dof_inact,D,Grid)  
        
        flux_vert[dof_sat_faces-1] = q_w[dof_sat_faces-1]
        
        res = D @ flux_vert
        

    dt   = CFL*np.abs((phi - phi*s_gr - C_old)/(res)) #Calculating the time step from the filling of volume
    
    dt[dt<1e-10] = 0.0
    dt  =  np.min(dt[dt>0])
    if i<10: 
        dt = tmax/Nt
    elif t[i]+dt >= t_interest[np.max(np.argwhere(t[i]+dt >= t_interest))] and t[i] < t_interest[np.max(np.argwhere(t[i]+dt >= t_interest))]:
        dt = t_interest[np.max(np.argwhere(t[i]+dt >= t_interest))] - t[i]   #To have the results at a specific time
      
    RHS = C_old - dt*res  #since the gradients are only zero and 1  
    
    C = solve_lbvp(I,RHS,B,BC.C_g,N)

    C_sol = np.concatenate((C_sol,C),axis=1)
    t.append(t[i] + dt)
    print(i,t[i])
    i = i+1

t = np.array(t)

#saving the tensors
np.savez(f'{simulation_name}_C{C_L}_{Grid.Nx}by{Grid.Ny}_t{tmax}.npz', t=t,C_sol =C_sol,phi=phi,Xc=Xc,Yc=Yc,Xc_col=Xc_col,Yc_col=Yc_col,Grid_Nx=Grid.Nx,Grid_Ny=Grid.Ny,Grid_xc=Grid.xc,Grid_yc=Grid.yc,Grid_xf=Grid.xf,Grid_yf=Grid.yf)

'''
#for loading data
data = np.load('NewupwindedTestingagainMarcstest_Quasi1D_hyperbolic_Richards-drainage-new_C0.0_2by400_t0.25.npz')
t=data['t']
C_sol =data['C_sol']
phi=data['phi']
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
[dummy,endstop] = np.shape(C_sol)
'''


###############################################
light_red  = [1.0,0.5,0.5]
light_blue = [0.5,0.5,1.0]
light_black= [0.5,0.5,0.5]

low_phi_dof = np.union1d(np.intersect1d(np.argwhere(Xc_col >=0.5)[:,0],np.argwhere(Xc_col < 0.75 - (1-Yc_col)/3)), \
                         np.intersect1d(np.argwhere(Xc_col < 0.5)[:,0],np.argwhere(Xc_col > 0.25 + (1-Yc_col)/3)))

fig = plt.figure(figsize=(15,7.5) , dpi=100)
plot = [plt.contourf(Xc, Yc, np.transpose(phi.reshape(Grid.Nx,Grid.Ny)),20,cmap="coolwarm",levels=20,vmin = 0, vmax = 1)]
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.xlabel(r'$x$')
plt.ylabel(r'$z$')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
#plt.axis('scaled')
mm = plt.cm.ScalarMappable(cmap=cm.coolwarm)
mm.set_array(C)
mm.set_clim(0., 1.)
clb = plt.colorbar(mm, pad=0.1)
clb.set_label(r'$\phi$', labelpad=1, y=1.075, rotation=0)
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
plt.savefig(f'../Figures/{simulation_name}_{Grid.Nx}by{Grid.Ny}_porosity.pdf',bbox_inches='tight', dpi = 600)


'''
#New Contour plot
fps = 100000 # frame per sec
#frn = endstop # frame number of the animation
[N,frn] = np.shape(C_sol) # frame number of the animation from the saved file

def update_plot(frame_number, zarray, plot,t):
    plt.clf()
    zarray  = zarray/phi
    plot[0] = plt.contourf(Xc, Yc, np.transpose(zarray[:,frame_number].reshape(Grid.Nx,Grid.Ny)), cmap="Blues",vmin = 0, vmax = 1)
    plt.title("t= %0.4f" % t[frame_number],loc = 'center', fontsize=18)
    #plt.axis('scaled')
    #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    #clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=-0.1)
    plt.clim(0,1)
    
    
    plt.xlim([Grid.xmin, Grid.xmax])
    plt.ylim([Grid.ymax,Grid.ymin])
    mm = plt.cm.ScalarMappable(cmap=cm.Blues)
    mm.set_array(C)
    mm.set_clim(0., 1.)
    clb = plt.colorbar(mm, pad=0.1)
    clb.set_label(r'$s_w$', labelpad=-3,x=-3, y=1.13, rotation=0)
    

fig = plt.figure(figsize=(10,10) , dpi=100)
plot = [plt.contourf(Xc, Yc, np.transpose((C_sol[:,0]/phi[:,0]).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = 0, vmax = 1)]
#manager = plt.get_current_fig_manager()
#manager.window.showMaximized()
#clb = fig.colorbar(plot[0], orientation='horizontal',aspect=50, pad=-0.1)
#clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=-0.1)
plt.ylabel(r'$z$')
plt.xlabel(r'$x$')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
#plt.axis('scaled')

mm = plt.cm.ScalarMappable(cmap=cm.Blues)
mm.set_array(C)
mm.set_clim(0., 1.)
clb = plt.colorbar(mm, pad=0.1)
clb.set_label(r'$s_w$', labelpad=-3,x=-3, y=1.13, rotation=0)
plt.title("t= %0.4f" % t[i],loc = 'center', fontsize=18)
#plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(C_sol[:,::25], plot[::25],t[::25]), interval=1/fps)
ani.save(f"../Figures/{simulation_name}_{C_L/phi_L}_{Grid.Nx}by{Grid.Ny}_tf{t[frn-1]}_S_wsingle.mov", writer='ffmpeg', fps=30)


#ani.save(fn+'.gif',writer='imagemagick',fps=fps)
#cmd = 'magick convert %s.gif -fuzz 5%% -layers Optimize %s_r.gif'%(fn,fn)
#subprocess.check_output(cmd)
'''

'''
#New Contour plot
fps = 100000 # frame per sec
#frn = endstop # frame number of the animation
[N,frn] = np.shape(C_sol) # frame number of the animation from the saved file

def update_plot(frame_number, zarray, plot,t):
    zarray  = zarray/phi   
    plot[0] = plt.contourf(Xc, Yc, np.transpose(zarray[:,frame_number].reshape(Grid.Nx,Grid.Ny)), cmap="Blues",vmin = -0.05, vmax = 1.05)
    plt.title("t= %0.4f" % t[frame_number],loc = 'center')
    plt.axis('scaled')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    #plt.clim(0,1)

fig = plt.figure(figsize=(10,7.5) , dpi=100)
Ind = C/phi
plot = [plt.contourf(Xc, Yc, np.transpose(Ind.reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = -0.05, vmax = 1.05)]
plt.ylabel(r'$z$')
plt.xlabel(r'$x$')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.axis('scaled')
mm = plt.cm.ScalarMappable(cmap=cm.Blues)
mm.set_array(Ind)
mm.set_clim(0., 1.)
clb = plt.colorbar(mm, pad=0.0)
#clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=0.0)
clb.set_label(r'$s_w$')
plt.title("t= %0.2f s" % t[i],loc = 'center', fontsize=18)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(C_sol[:,::50], plot[::50],t[::50]), interval=1/fps)
ani.save(f"../Figures/{simulation_name}_{Grid.Nx}by{Grid.Ny}_tf{t[frn-1]}_S_w.mov", writer='ffmpeg', fps=30)

#ani.save(fn+'.gif',writer='imagemagick',fps=fps)
#cmd = 'magick convert %s.gif -fuzz 5%% -layers Optimize %s_r.gif'%(fn,fn)
#subprocess.check_output(cmd)
'''


'''
#New Contour plot combined
fps = 100000 # frame per sec
#frn = endstop # frame number of the animation
[N,frn] = np.shape(C_sol) # frame number of the animation from the saved file

def update_plot(frame_number, zarray, plot,t):
    fig.suptitle("t= %0.4f" % t[frame_number], fontsize=22)
    #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    zarray  = zarray/phi 
    ax1.set_label(r'$x$')
    plt.subplot(1,2,1)
    plot[0] = plt.contourf(Xc, Yc, np.transpose(zarray[:,frame_number].reshape(Grid.Nx,Grid.Ny)), cmap="Blues",vmin = -0.0005, vmax = 1.0005)
    plt.ylim([Grid.ymax,Grid.ymin])
    plt.subplot(1,2,2)
    zarray[zarray<sat_threshold] = 0
    zarray[zarray>=sat_threshold] = 1
    plot1[0] = plt.contourf(Xc, Yc, np.transpose(zarray[:,frame_number].reshape(Grid.Nx,Grid.Ny)), cmap="Greys",vmin = -0.0005, vmax = 1.0005)
    ax2.set_label(r'$x$')
    plt.ylim([Grid.ymax,Grid.ymin])
    mm = plt.cm.ScalarMappable(cmap=cm.Blues)
    mm.set_array(Ind)
    mm.set_clim(0., 1.)
    #clb = plt.colorbar(mm, pad=0.05,orientation='horizontal',ax=[ax1,ax2],aspect=50)
    #plt.clim(0,1)
    ax1.set_aspect('auto')
    ax2.set_aspect('auto')
    ax1.axis('scaled')
    ax2.axis('scaled')
    ax1.set_xlabel(r'$x$')
    ax2.set_xlabel(r'$x$')
    ax2.set_ylabel(r'$z$')
    bnd = np.ones_like(Grid.xc)
    bnd[Grid.xc < 0.75] = 1 + 3*( Grid.xc[Grid.xc < 0.75] - 0.75 )
    bnd[Grid.xc < 0.5]  = 1 + 3*(-Grid.xc[Grid.xc < 0.5 ] + 0.25 )
    bnd[Grid.xc < 0.25] = 1
    #ax1.plot(Grid.xc,bnd,'-',color=brown, linewidth=0.5)
    #ax1.plot(Grid.xc,bnd,'-',color=brown, linewidth=0.5)
    #ax2.plot(Grid.xc,bnd,'-',color=brown, linewidth=0.5)
    ax1.plot(Xc_col[np.argwhere(zarray[:,frame_number]<0)][:,0],Yc_col[np.argwhere(zarray[:,frame_number]<0)][:,0],'ro-')
    #print(np.shape(zarray[:,frame_number]), np.shape(phi[:,0]))
    ax1.plot(Xc_col[np.argwhere(zarray[:,frame_number]>1)][:,0],Yc_col[np.argwhere(zarray[:,frame_number]>1)][:,0],'gX')    
    

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,20))
Ind = C_sol[:,0]/phi[:,0]
plt.subplot(1,2,1)
plot = [plt.contourf(Xc, Yc, np.transpose(Ind.reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = -0.0005, vmax = 1.0005)]
ax1.set_ylabel(r'$z$')
ax1.set_label(r'$x$')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
ax1.axis('scaled')

mm = plt.cm.ScalarMappable(cmap=cm.Blues)
mm.set_array(Ind)
mm.set_clim(0., 1.)
#clb = fig.colorbar(plot[0],orientation='horizontal', orientation='horizontal',aspect=50,pad=0.05)
#clb.set_label(r'$s_w$')
#plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
ax1.set_label(r'$x$')
ax2.set_ylabel(r'$z$')
plt.subplot(1,2,2)
Ind = C_sol[:,0]/phi[:,0]
Ind[Ind<sat_threshold] = 0
Ind[Ind>=sat_threshold] = 1.0
plot1 = [plt.contourf(Xc, Yc, np.transpose(Ind.reshape(Grid.Nx,Grid.Ny)),cmap="Greys",vmin = -0.0005, vmax = 1.0005)]
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
ax1.axis('scaled')
ax2.axis('scaled')
ax1.set_xlabel(r'$x$')
ax2.set_xlabel(r'$x$')
clb = plt.colorbar(mm, orientation='horizontal',ax=[ax1,ax2],aspect=50,pad=0.13)
clb.set_label(r'$s_w$')
fig.suptitle("t= %0.2f s" % t[0], fontsize=22)
#plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
ax1.set_aspect('auto')
ax2.set_aspect('auto')
ax1.axis('scaled')
ax2.axis('scaled')
bnd = np.ones_like(Grid.xc)
bnd[Grid.xc < 0.75] = 1 + 3*( Grid.xc[Grid.xc < 0.75] - 0.75 )
bnd[Grid.xc < 0.5]  = 1 + 3*(-Grid.xc[Grid.xc < 0.5 ] + 0.25 )
bnd[Grid.xc < 0.25] = 1
#ax1.plot(Grid.xc,bnd,'-',color=brown, linewidth=0.5)
#ax2.plot(Grid.xc,bnd,'-',color=brown, linewidth=0.5)
#ax1.plot(Xc_col[np.argwhere(C_sol[:,0]<0)[:,0]],Yc_col[np.argwhere(C_sol[:,0]<0)[:,0]],'ro-')
#ax1.plot(Xc_col[np.argwhere(C_sol[:,]>phi[:,0]),0][:,0],Yc_col[np.argwhere(C_sol[:,0]>phi[:,0]),0][:,0],'gX')    
ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(C_sol[:,::20], plot[::20],t[::20]), interval=1/fps)

ani.save(f"../Figures/combined_{simulation_name}_{C_L/phi_L}_{Grid.Nx}by{Grid.Ny}_tf{t[frn-1]}_S_w.mov", writer='ffmpeg', fps=30)

#ani.save(fn+'.gif',writer='imagemagick',fps=fps)
#cmd = 'magick convert %s.gif -fuzz 5%% -layers Optimize %s_r.gif'%(fn,fn)
#subprocess.check_output(cmd)
'''


fig = plt.figure(figsize=(15,7.5) , dpi=100)
plot = [plt.contourf(Xc, Yc, np.transpose((C/phi).reshape(Grid.Nx,Grid.Ny)),cmap="Blues")]
plt.colorbar()
plt.ylabel(r'$z$')
plt.xlabel(r'$x$')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.axis('scaled')
plt.plot(Xc_col[np.argwhere(C_sol[:,-1]<0)[:,0]],Yc_col[np.argwhere(C_sol[:,-1]<0)[:,0]],'ro',label='Sw>1')
plt.plot(Xc_col[np.argwhere(C_sol[:,-1]>phi[:,0])[:,0]],Yc_col[np.argwhere(C_sol[:,-1]>phi[:,0])[:,0]],'gX',label='Sw>1')
plt.savefig(f'../Figures/{simulation_name}_Marcs_way_Sw.pdf',bbox_inches='tight', dpi = 600)



[Xc_flux,Yf_flux] = np.meshgrid(Grid.xc,Grid.yf)     #building the (x,y) matrix

fig = plt.figure(figsize=(15,7.5) , dpi=100)
plot = [plt.contourf(Xc_flux, Yf_flux, np.transpose(flux_vert[Grid.Nfx:Grid.Nf,-1].reshape(Grid.Nx,Grid.Ny+1)),cmap="Blues",vmin = -0.05, vmax = 1.05)]
plt.ylabel(r'$z$')
plt.xlabel(r'$x$')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.colorbar()
plt.axis('scaled')
plt.plot(Xc_col[np.argwhere(C_sol[:,-1]<0)[:,0]],Yc_col[np.argwhere(C_sol[:,-1]<0)[:,0]],'ro')
plt.savefig(f'../Figures/{simulation_name}_Marcs_way_verticalflux.pdf',bbox_inches='tight', dpi = 600)

[Xf_flux,Yc_flux] = np.meshgrid(Grid.xf,Grid.yc)     #building the (x,y) matrix

fig = plt.figure(figsize=(15,7.5) , dpi=100)
plot = [plt.contourf(Xf_flux, Yc_flux, np.transpose(flux_vert[0:Grid.Nfx,-1].reshape(Grid.Nx+1,Grid.Ny)),cmap="Blues",vmin = -1, vmax = 1)]
plt.ylabel(r'$z$')
plt.xlabel(r'$x$')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.colorbar()
plt.axis('scaled')
plt.plot(Xc_col[np.argwhere(C_sol[:,-1]<0)[:,0]],Yc_col[np.argwhere(C_sol[:,-1]<0)[:,0]],'ro')
plt.savefig(f'../Figures/{simulation_name}_Marcs_way_horizontalflux.pdf',bbox_inches='tight', dpi = 600)


def analytic_solution(phi_L,C_L,phi_R,C_R,Grid,t):
    
    C_analy = C_R*np.ones((Grid.Ny+1,1))
    
    if t>0:
        eta = Grid.yf/t
    
        C_analy = (eta/(n*phi_L**(-n)*(1-s_wr-s_gr)))**(1/(n-1))
        
        lambda2_L = n*(C_L-phi_L*s_wr)**(n-1) * (phi_L*(1-s_gr-s_wr))**(-n)  
        lambda2_R = n*(C_R-phi_R*s_wr)**(n-1) * (phi_R*(1-s_gr-s_wr))**(-n)  
        
        C_analy[eta<lambda2_L] = C_L
        C_analy[eta>lambda2_R] = C_R   

    return C_analy


# First set up the figure, the axis

fig = plt.figure(figsize=(15,7.5) , dpi=100)
ax1 = fig.add_subplot(1, 6, 1)
ax2 = fig.add_subplot(1, 6, 2)
ax3 = fig.add_subplot(1, 6, 3)
ax4 = fig.add_subplot(1, 6, 4)
ax5 = fig.add_subplot(1, 6, 5)
ax6 = fig.add_subplot(1, 6, 6)

ax1.set_ylabel(r'Dimensionless depth $z/z_0$')
ax1.set_ylim([Grid.ymax-0.01,Grid.ymin-0.01])
ax1.set_xlim([-0.05,1.05])

ax2.set_xlim([-0.05,1.05])
ax2.set_ylim([Grid.ymax-0.01,Grid.ymin-0.01])
ax2.axes.yaxis.set_visible(False)
ax4.set_xlabel(r'Water saturation $s_w$')

ax3.set_xlim([-0.05,1.05])
ax3.axes.yaxis.set_visible(False)
ax3.set_ylim([Grid.ymax-0.01,Grid.ymin-0.01])
#ax3.set_ylim([grid.ymin,grid.ymax])

ax4.set_xlim([-0.05,1.05])
ax4.axes.yaxis.set_visible(False)
ax4.set_ylim([Grid.ymax-0.01,Grid.ymin-0.01])

ax5.set_xlim([-0.05,1.05])
ax5.axes.yaxis.set_visible(False)
ax5.set_ylim([Grid.ymax-0.01,Grid.ymin-0.01])

ax6.set_xlim([-0.05,1.05])
ax6.axes.yaxis.set_visible(False)
ax6.set_ylim([Grid.ymax-0.01,Grid.ymin-0.01])

manager = plt.get_current_fig_manager()
manager.window.showMaximized()

ax1.plot(C_sol[0:1,0]/phi[0:1,0],Grid.yc[0:1],'b--',label=r'S',linewidth=3)
ax1.plot(analytic_solution(phi_L,C_L,phi_R,C_R,Grid,t_interest[0])/phi_L,Grid.yf , c = 'r',linestyle='-',label=r'A',linewidth=3)
ax1.plot(C_sol[:Grid.Ny,(np.argwhere(t==t_interest[0]))[0,0]]/phi[:Grid.Ny,0],Grid.yc,'b--',linewidth=3)
ax1.legend(loc='lower left', shadow=False, fontsize='medium', framealpha=0.0)

ax2.plot(analytic_solution(phi_L,C_L,phi_R,C_R,Grid,t_interest[1])/phi_L,Grid.yf , c = 'r',linestyle='-',label=r'Numerical',linewidth=3)
ax2.plot(C_sol[:Grid.Ny,(np.argwhere(t==t_interest[1]))[0,0]]/phi[:Grid.Ny,0],Grid.yc, 'b--',label=r'Analytical',linewidth=3)

ax3.plot(analytic_solution(phi_L,C_L,phi_R,C_R,Grid,t_interest[2])/phi_L,Grid.yf , c = 'r',linestyle='-',label=r'Numerical',linewidth=3)
ax3.plot(C_sol[:Grid.Ny,(np.argwhere(t==t_interest[2]))[0,0]]/phi[:Grid.Ny,0],Grid.yc, 'b--',label=r'Analytical',linewidth=3)

ax4.plot(analytic_solution(phi_L,C_L,phi_R,C_R,Grid,t_interest[3])/phi_L,Grid.yf , c = 'r',linestyle='-',label=r'Numerical',linewidth=3)
ax4.plot(C_sol[:Grid.Ny,(np.argwhere(t==t_interest[3]))[0,0]]/phi[:Grid.Ny,0],Grid.yc, 'b--',label=r'Analytical',linewidth=3)

ax5.plot(analytic_solution(phi_L,C_L,phi_R,C_R,Grid,t_interest[4])/phi_L,Grid.yf , c = 'r',linestyle='-',label=r'Numerical',linewidth=3)
ax5.plot(C_sol[:Grid.Ny,(np.argwhere(t==t_interest[4]))[0,0]]/phi[:Grid.Ny,0],Grid.yc, 'b--',label=r'Analytical',linewidth=3)

ax6.plot(analytic_solution(phi_L,C_L,phi_R,C_R,Grid,t_interest[5])/phi_L,Grid.yf , c = 'r',linestyle='-',label=r'Numerical',linewidth=3)
ax6.plot(C_sol[:Grid.Ny,(np.argwhere(t==t_interest[5]))[0,0]]/phi[:Grid.Ny,0],Grid.yc,'b--',label=r'Analytical',linewidth=3)


ax1.set_title(r'$t^*$=%.2f'%t_interest[0])
ax2.set_title(r'%.2f'%t_interest[1])
ax3.set_title(r'%.2f'%t_interest[2])
ax4.set_title(r'%.2f'%t_interest[3])
ax5.set_title(r'%.2f'%t_interest[4])
ax6.set_title(r'%.2f'%t_interest[5])
plt.subplots_adjust(wspace=0.25, hspace=0)
plt.savefig(f"../Figures/swvsZpanelshock_Nx{Grid.Nx}_CL{C_L}CR{C_R}_phiL{phi_L}phiR{phi_R}m{m}_n{n}_drainage.pdf")



