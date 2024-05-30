#Coding the Richard's equation Reservoir drainage
#Woods and Hesse
#Mohammad Afzal Shadab
#Date modified: 03/29/2022

import sys
sys.path.insert(1, '../../solver')
sys.path.insert(1, './Drainage-analytic')

# import personal libraries and class
from classes import *    

from two_components_aux import *
from solve_lbvpfun_SPD import solve_lbvp_SPD
from complex_domain import find_faces, find_bnd_cells, find_all_faces, find_all_x_faces,find_all_y_faces
from comp_fluxfun import comp_flux
from scipy.integrate import solve_ivp
from comp_sat_unsat_bnd_flux_fun import comp_sat_unsat_bnd_flux, find_top_bot_cells, find_left_right_cells#,comp_sat_unsat_bnd_flux_xface
from comp_face_coords_fun import comp_face_coords
from find_plate_dofs import find_plate_dofs
from spin_up import spin_up_C
from comp_mean_matrix import comp_mean_matrix

from scipy.interpolate import interp1d
from Paper_drainage_analytic import reservoir_drainage
x_analytic,h_analytic = reservoir_drainage()
h_analytic_func = interp1d(x_analytic,h_analytic, kind='cubic')

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
        
class Param:
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
                3: [[0.3164247822051133,0.5],[0.0,0.2]]#[[0.4,0.5],[0.0,0.2]], #Contact discontinuity C1  (working)   
             }
        return switcher.get(i,"Invalid Case")

deg2rad = np.pi/180

#parameters
direction = 'z' #x or z

if direction == 'x':
    tilt_angle = 90#0.13/deg2rad  #angle of the slope in degrees
else:
    tilt_angle = 0 #0.13/deg2rad  #angle of the slope in degrees
    
    
case_no =  3               #case number
simulation_name = f'Reservoir_drainage'
m = 3 #Cozeny-Karman coefficient for numerator K = K0 (1-phi_i)^m
n = 2 #Corey-Brooks coefficient krw = krw0 * sw^n
s_wr = 0.0 #Residual water saturation
s_gr = 0.0 #Residual gas saturation

#test details
[u_L,u_R] = case(case_no) #left and right states
[C_L,phi_L] = u_L #water sat, porosity
[C_R,phi_R] = u_R #water sat, porosity
C_R = s_gr*phi_R #resetting the saturation
xm = 0.0 #the location of jump
sat_threshold = 1-1e-3 #threshold for saturated region formation


#temporal
tmax = 20#0.07#0.0621#2 #5.7#6.98  #time scaling with respect to fc
t_interest = np.linspace(0,tmax,401)   #swr,sgr=0

#tmax = tmax / phi_L**m   #time scaling with respect to K_0 where K_0 = f_c/phi**m
Nt   = 1000
CFL  = 0.5
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
    #fC[C<=0]  = 0.0
    #fC[C>phi] = 0.0      
    return fC

#spatial
Grid.xmin =  0; Grid.xmax =3; Grid.Nx = 300; 
Grid.ymin =  0; Grid.ymax =1; Grid.Ny = 200;
Grid = build_grid(Grid)
[D,G,I] = build_ops(Grid)
Avg     = comp_algebraic_mean(Grid)#comp_mean_matrix(Grid)
Avg_open= comp_mean_matrix(Grid)

[Xc,Yc] = np.meshgrid(Grid.xc,Grid.yc)                 #building the (x,y) matrix
Xc_col  = np.reshape(np.transpose(Xc), (Grid.N,-1))    #building the single X vector
Yc_col  = np.reshape(np.transpose(Yc), (Grid.N,-1))    #building the single Y vector


##########
#boundary condition
common_cell = np.intersect1d(Grid.dof_ymax,Grid.dof_xmax)
common_face = np.intersect1d(Grid.dof_f_ymax,Grid.dof_f_xmax)

hor_bnd = np.setdiff1d(Grid.dof_xmax,common_cell)


BC.dof_dir = np.array([])
BC.dof_f_dir = np.array([])
BC.C_g  = np.array([])


height = 1
dof_inj   = Grid.dof_xmin[np.argwhere(Grid.ymax-Grid.yc <= height)]
dof_f_inj = Grid.dof_f_xmin[np.argwhere(Grid.ymax-Grid.yc <= height)]


BC.dof_neu   = np.ravel(dof_inj)
BC.dof_f_neu = np.ravel(dof_f_inj)
BC.qb        = np.zeros_like(dof_inj)
#BC.qb[0:int(len(Grid.dof_xmin)/2),0]      = 0

[B,N,fn] = build_bnd(BC, Grid, I)


phi = phi_L*np.ones((Grid.N,1))



C = phi*sat_threshold*0.9

C_sol = C.copy()
flux_sol = np.zeros((Grid.Nf,1))
fs = np.zeros((Grid.N,1)) #RHS of elliptic equation
t    =[0.0]
time = 0
v = np.ones((Grid.Nf,1))
i = 0

#Spinup
#Grid,C,time,time,t_interest,i,tmax,t,C_sol = spin_up_C('paper_randomly_correlated_fields__C0.3164247822051133_100by50_t20.npz',Grid,tmax+10,201,t_interest)

while time<tmax:

    C_old = C.copy() 
    flux      = (comp_harmonicmean(Avg,f_Cm(phi,m))*(flux_upwind(v, Grid) @ f_Cn(C_old,phi,n)))*np.cos(tilt_angle*deg2rad)
    flux_vert = flux.copy()
    flux_vert[Grid.dof_f<Grid.Nfx,0] = flux_vert[Grid.dof_f<Grid.Nfx,0]*np.tan(tilt_angle*deg2rad)  #making gravity based flux in x direction
    
    res = D@flux_vert  #since the gradients are only zero and 1    
    res_vert = res.copy()
    ######
    #Taking out the domain to cut off single phase region
    dof_act  = Grid.dof[C_old[:,0] / (phi[:,0]*(1-s_gr)) < sat_threshold]
    dof_inact= np.setdiff1d(Grid.dof,dof_act) #saturated cells
    if len(dof_act)< Grid.N: #when atleast one layer is present
        #############################################
        dof_f_saturated = find_faces(dof_inact,D,Grid)       
        
        ##### 
        #For seepage face
        dof_act_without_inlet= np.setdiff1d(dof_act,BC.dof_neu)
        dof_seepage= np.intersect1d(dof_inact,Grid.dof_xmax)


        #Eliminate inactive cells by putting them to constraint matrix
        BC_P.dof_dir = np.hstack([dof_act_without_inlet,dof_seepage])           
        BC_P.dof_f_dir= np.array([])
        BC_P.g       =  -Yc_col[BC_P.dof_dir-1]*np.cos(tilt_angle*deg2rad) \
                        -Xc_col[BC_P.dof_dir-1]*np.sin(tilt_angle*deg2rad)  


        ######
                        
        BC_P.dof_neu   = BC.dof_neu.copy()
        BC_P.dof_f_neu = BC.dof_f_neu.copy()
        BC_P.qb        = BC.qb.copy()
        
        [B_P,N_P,fn_P] = build_bnd(BC_P,Grid,I)
        Kd  = comp_harmonicmean(Avg,f_Cm(phi,m)) * (Avg @ f_Cn(C_old,phi,n))
        
                                                                                                                                                                                                                                                                                              
        Kd  = sp.dia_matrix((Kd[:,0],  np.array([0])), shape=(Grid.Nf, Grid.Nf))
        L = - D @ Kd @ G
        u = solve_lbvp(L,fn_P,B_P,BC_P.g,N_P)   # Non dimensional water potential

        ####### New boundary condition for outflow
        Param.dof_neu   = np.hstack([Grid.dof_xmin, Grid.dof_xmax, Grid.dof_ymin, Grid.dof_ymax])
        Param.dof_f_neu = np.hstack([Grid.dof_f_xmin, Grid.dof_f_xmax, Grid.dof_f_ymin, Grid.dof_f_ymax])        
        Param.dof_dir   = np.array([]) 
        Param.dof_f_dir = np.array([]) 
        q_w = comp_flux(D, Kd, G, u, fs, Grid, Param)
        
        ####### New boundary condition for outflow
         
        #upwinding boundary y-directional flux
        #finding boundary faces
        dof_ysat_faces = dof_f_saturated[dof_f_saturated>Grid.Nfx]
        
        #removing boundary faces
        dof_ysat_faces = np.setdiff1d(dof_ysat_faces,np.append(Grid.dof_f_ymin,Grid.dof_f_ymax))
        
        ytop,ybot               = find_top_bot_cells(dof_ysat_faces,D,Grid)
        q_w[dof_ysat_faces-1,0] = comp_sat_unsat_bnd_flux(q_w[dof_ysat_faces-1],flux_vert[dof_ysat_faces-1],ytop,ybot,C,phi,sat_threshold)
        
         #upwinding boundary x-directional flux   ####new line
        #finding boundary faces
        dof_xsat_faces = dof_f_saturated[dof_f_saturated<=Grid.Nfx]
        
        #removing boundary faces
        dof_xsat_faces = np.setdiff1d(dof_xsat_faces,np.append(Grid.dof_f_xmin,Grid.dof_f_xmax))
        
        xleft,xright            = find_left_right_cells(dof_xsat_faces,D,Grid)
        q_w[dof_xsat_faces-1,0] = comp_sat_unsat_bnd_flux(q_w[dof_xsat_faces-1],flux_vert[dof_xsat_faces-1],xright,xleft,C,phi,sat_threshold)
         
            
        #find all saturated faces
        dof_sat_faces = find_all_faces(dof_inact,D,Grid)  
        dof_sat_faces = np.setdiff1d(dof_sat_faces,BC.dof_f_neu) #since Neumann boundary condition is taken care of
        
        
        flux_vert[dof_sat_faces-1] = q_w[dof_sat_faces-1]
        
        res = D @ flux_vert
    

    dt   = CFL*np.abs((phi - phi*s_gr - C_old)/(res - fn)) #Calculating the time step from the filling of volume
    
    dt[dt<1e-10] = 0.0
    dt  =  np.min(dt[dt>0])
    
    if dt > tmax/Nt: dt = tmax/Nt
    if i<100: 
        dt = tmax/(Nt*1000)
    elif time+dt >= t_interest[np.max(np.argwhere(time+dt >= t_interest))] and time < t_interest[np.max(np.argwhere(time+dt >= t_interest))]:
        dt = t_interest[np.max(np.argwhere(time+dt >= t_interest))] - time   #To have the results at a specific time
    
    RHS = C_old - dt*(res - fn)  #since the gradients are only zero and 1  
    
    C = solve_lbvp(I,RHS,B,BC.C_g,N)
    time = time + dt    
    if np.isin(time,t_interest):
        C_sol = np.concatenate((C_sol,C),axis=1)
        flux_sol = np.concatenate((flux_sol,flux_vert),axis=1)
        t.append(time)
        if len(dof_act)< Grid.N:
            print(i,time,'Saturated cells',Grid.N-len(dof_act))        
        else:    
            print(i,time)
    i = i+1
    
    if (i-1)%50:
        print(i,time)
        np.savez(f'{simulation_name}_C{C_L}_{Grid.Nx}by{Grid.Ny}_t{tmax}.npz', t=np.array(t),C_sol =C_sol,phi=phi,Xc=Xc,Yc=Yc,Xc_col=Xc_col,Yc_col=Yc_col,Grid_Nx=Grid.Nx,Grid_Ny=Grid.Ny,Grid_xc=Grid.xc,Grid_yc=Grid.yc,Grid_xf=Grid.xf,Grid_yf=Grid.yf,flux_sol = flux_sol)


t = np.array(t)


#saving the tensors
#np.savez(f'{simulation_name}_porosity.npz', phi = phi, K = K, m = m, Grid_Nx=Grid.Nx,Grid_Ny=Grid.Ny)


'''
#for loading data
data = np.load('paper_reservoir_drainage_C0.3164247822051133_150by100_t40.npz')
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
flux_sol = data['flux_sol']
[dummy,endstop] = np.shape(C_sol)
C[:,0] = C_sol[:,-1]
flux_vert =  flux_sol[:,-1]
'''



###############################################
light_red  = [1.0,0.5,0.5]
light_blue = [0.5,0.5,1.0]
light_black= [0.5,0.5,0.5]

fig = plt.figure(figsize=(10,10) , dpi=100)
plot = [plt.contourf(Xc, Yc, np.transpose(phi.reshape(Grid.Nx,Grid.Ny)),cmap="coolwarm",levels=100,vmin=np.min((phi)),vmax=np.max((phi)))]
plt.xlabel(r'$x$')
plt.ylabel(r'$z$')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.axis('scaled')
mm = plt.cm.ScalarMappable(cmap=cm.coolwarm)
mm.set_array(phi)
mm.set_clim(np.min((phi)),np.max((phi)))
clb = plt.colorbar(mm,aspect=10)
clb.ax.set_title(r'$\phi$',y=1.03)
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
plt.title("t= %0.4f" % t[0],loc = 'center', fontsize=18)
#plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(C_sol[:,:], plot[:],t[:]), interval=1/fps)
ani.save(f"../Figures/{simulation_name}_{C_L/phi_L}_{Grid.Nx}by{Grid.Ny}_tf{t[frn-1]}_S_wsingle.mov", writer='ffmpeg', fps=30)

'''


'''
#New Contour plot water content
fps = 100000 # frame per sec
#frn = endstop # frame number of the animation
[N,frn] = np.shape(C_sol) # frame number of the animation from the saved file

def update_plot(frame_number, zarray, plot,t):
    zarray[zarray>0]   = 1
    plot[0] = plt.contourf(Xc, Yc, np.transpose(zarray[:,frame_number].reshape(Grid.Nx,Grid.Ny)), cmap="Blues",vmin = -0.05, vmax = 1.05)
    plt.title("t= %0.2f" % t[frame_number],loc = 'center')
    plt.axis('scaled')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    #plt.clim(0,1)

fig = plt.figure(figsize=(20,10) , dpi=100)
Ind = C_sol[:,0]/phi[:,0]
Ind = np.ones(1)*[Ind>0]
plot = [plt.contourf(Xc, Yc, np.transpose(Ind.reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = -0.05, vmax = 1.05)]
plt.ylabel(r'$z$')
plt.xlabel(r'$x$')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.axis('scaled')
mm = plt.cm.ScalarMappable(cmap=cm.Blues)
mm.set_array(Ind)
mm.set_clim(0., 1.)
clb = plt.colorbar(mm, pad=0.1)
#clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=0.0)
clb.set_label(r'Water presence')
plt.title("t= %0.2f s" % t[0],loc = 'center', fontsize=18)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(C_sol[:,:], plot[:],t[:]), interval=1/fps)
ani.save(f"../Figures/{simulation_name}_{Grid.Nx}by{Grid.Ny}_tf{t[frn-1]}_water_presence.mov", writer='ffmpeg', fps=30)

#ani.save(fn+'.gif',writer='imagemagick',fps=fps)
#cmd = 'magick convert %s.gif -fuzz 5%% -layers Optimize %s_r.gif'%(fn,fn)
#subprocess.check_output(cmd)
'''


'''
#New Contour plot S_w
fps = 100000 # frame per sec
#frn = endstop # frame number of the animation
[N,frn] = np.shape(C_sol) # frame number of the animation from the saved file

def update_plot(frame_number, zarray, plot,t):
    plt.clf()
    zarray  = zarray/phi   
    plot[0] = plt.contourf(Xc, Yc, np.transpose(zarray[:,frame_number].reshape(Grid.Nx,Grid.Ny)), cmap="Blues",vmin = 0, vmax = 1.0,levels=100)
    plot = plt.plot(Grid.xmax*x_analytic,Grid.ymax-Grid.xmax*h_analytic/t[frame_number],'r--')
    plt.title("t= %0.2f" % t[frame_number],loc = 'center')
    plt.axis('scaled')
    #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.ylabel(r'$z$')
    plt.xlabel(r'$x$')
    plt.xlim([Grid.xmin, Grid.xmax])
    plt.ylim([Grid.ymax,Grid.ymin])
    #plt.clim(0,1)
    mm = plt.cm.ScalarMappable(cmap=cm.Blues)
    mm.set_array(Ind)
    mm.set_clim(0., 1.)
    clb = plt.colorbar(mm, pad=0.1)
    #clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=0.0)
    clb.set_label(r'$s_w$')
    

fig = plt.figure(figsize=(20,10) , dpi=100)
Ind = C/phi
plot = [plt.contourf(Xc, Yc, np.transpose(Ind.reshape(Grid.Nx,Grid.Ny)),vmin = 0, vmax = 1.0,cmap="Blues",levels=100)]
plt.ylabel(r'$z$')
plt.xlabel(r'$x$')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.axis('scaled')
mm = plt.cm.ScalarMappable(cmap=cm.Blues)
mm.set_array(Ind)
mm.set_clim(0., 1.)
clb = plt.colorbar(mm, pad=0.1)
#clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=0.0)
clb.set_label(r'$s_w$')
plt.title("t= %0.2f s" % t[0],loc = 'center', fontsize=18)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(C_sol[:,::4], plot[::4],t[::4]), interval=1/fps)
ani.save(f"../Figures/{simulation_name}_{Grid.Nx}by{Grid.Ny}_tf{t[frn-1]}_S_w.mov", writer='ffmpeg', fps=30)

#ani.save(fn+'.gif',writer='imagemagick',fps=fps)
#cmd = 'magick convert %s.gif -fuzz 5%% -layers Optimize %s_r.gif'%(fn,fn)
#subprocess.check_output(cmd)
'''

'''
#New Contour plot C
fps = 100000 # frame per sec
#frn = endstop # frame number of the animation
[N,frn] = np.shape(C_sol) # frame number of the animation from the saved file

def update_plot(frame_number, zarray, plot,t):
    zarray  = zarray  
    line_2 = plt.plot(Grid.xmax*x_analytic,Grid.ymax-Grid.xmax*h_analytic*phi_L/t[frame_number],'r--')
    plot[0] = plt.contourf(Xc, Yc, np.transpose(zarray[:,frame_number].reshape(Grid.Nx,Grid.Ny)), cmap="Blues",vmin = np.min(C_sol), vmax = np.max(C_sol),levels=100)
    plt.title("t= %0.2f" % t[frame_number],loc = 'center')
    plt.axis('scaled')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    #plt.clim(0,1)
    plt.xlim([Grid.xmin, Grid.xmax])
    plt.ylim([Grid.ymax,Grid.ymin])

fig = plt.figure(figsize=(20,10) , dpi=100)
Ind = C
plot = [plt.contourf(Xc, Yc, np.transpose(Ind.reshape(Grid.Nx,Grid.Ny)),cmap="Blues", vmin = np.min(C_sol), vmax = np.max(C_sol),levels=100)]
plt.ylabel(r'$z$')
plt.xlabel(r'$x$')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.axis('scaled')
mm = plt.cm.ScalarMappable(cmap=cm.Blues)
mm.set_array(Ind)
mm.set_clim(np.min(C_sol),np.max(C_sol))
clb = plt.colorbar(mm, pad=0.1)
#clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=0.0)
clb.set_label(r'$\theta$')
plt.title("t= %0.2f s" % t[0],loc = 'center', fontsize=18)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(C_sol[:,::1], plot[::1],t[::1]), interval=1/fps)
ani.save(f"../Figures/{simulation_name}_{Grid.Nx}by{Grid.Ny}_tf{t[frn-1]}_theta.mov", writer='ffmpeg', fps=30)

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
ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(C_sol[:,::1], plot[::1],t[::1]), interval=1/fps)

ani.save(f"../Figures/{simulation_name}_{C_L/phi_L}_{Grid.Nx}by{Grid.Ny}_tf{t[frn-1]}_S_w.mov", writer='ffmpeg', fps=30)

#ani.save(fn+'.gif',writer='imagemagick',fps=fps)
#cmd = 'magick convert %s.gif -fuzz 5%% -layers Optimize %s_r.gif'%(fn,fn)
#subprocess.check_output(cmd)
'''

'''
#New Contour plot combined with new mesh
print('New Contour plot combined with new mesh')
fps = 100000 # frame per sec
#frn = endstop # frame number of the animation
[N,frn] = np.shape(C_sol) # frame number of the animation from the saved file

[X_all,Y_all] = comp_face_coords(Grid.dof_f,Grid)
[X_plate,Y_plate] = comp_face_coords(dof_f_plate_bnd,Grid)

def update_plot(frame_number, zarray, plot,t):
    plt.cla()
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
    #plot1[0] = plt.contourf(Xc, Yc, np.transpose(zarray[:,frame_number].reshape(Grid.Nx,Grid.Ny)), cmap="Greys",vmin = -0.0005, vmax = 1.0005)
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
    ax1.set_xlim([Grid.xmin, Grid.xmax])
    ax1.set_ylim([Grid.ymax,Grid.ymin])
    ax2.set_xlim([Grid.xmin, Grid.xmax])
    ax2.set_ylim([Grid.ymax,Grid.ymin])
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
    ax2.plot(X_all,Y_all,'k-',linewidth=0.4)
    
    dof_inact= Grid.dof[zarray[:,frame_number] / (phi[:,0]*(1-s_gr)) >= sat_threshold] #saturated cells
    if np.any(dof_inact):
        dof_f_saturated = find_faces(dof_inact,D,Grid) 
        dof_sat_faces = find_all_faces(dof_inact,D,Grid) 
        [X_sat,Y_sat] = comp_face_coords(dof_sat_faces,Grid)
        [X_bnd,Y_bnd] = comp_face_coords(dof_f_saturated,Grid)
        
        ax2.plot(X_sat,Y_sat,'r-',linewidth=0.4)
        ax2.plot(X_bnd,Y_bnd,'r-',linewidth=2)
        
        ax1.plot(X_plate,Y_plate,'k-',linewidth=2)
        ax2.plot(X_plate,Y_plate,'k-',linewidth=2)
    

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
ax2.plot(X_all,Y_all,'k-',linewidth=0.4)
ax1.plot(X_plate,Y_plate,'k-',linewidth=2)
ax2.plot(X_plate,Y_plate,'k-',linewidth=2)
ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(C_sol[:,::25], plot[::25],t[::25]), interval=1/fps)

ani.save(f"../Figures/{simulation_name}_{C_L/phi_L}_{Grid.Nx}by{Grid.Ny}_tf{t[frn-1]}_S_w_mesh.mov", writer='ffmpeg', fps=30)

#ani.save(fn+'.gif',writer='imagemagick',fps=fps)
#cmd = 'magick convert %s.gif -fuzz 5%% -layers Optimize %s_r.gif'%(fn,fn)
#subprocess.check_output(cmd)
'''


dof_inact= Grid.dof[C_sol[:,-1] / (phi[:,0]*(1-s_gr)) >= sat_threshold] #saturated cells
dof_f_saturated = find_faces(dof_inact,D,Grid) 
dof_sat_faces = find_all_faces(dof_inact,D,Grid) 
[X_sat,Y_sat] = comp_face_coords(dof_sat_faces,Grid)
[X_bnd,Y_bnd] = comp_face_coords(dof_f_saturated,Grid)
[X_all,Y_all] = comp_face_coords(Grid.dof_f,Grid)

fig = plt.figure(figsize=(12,10) , dpi=100)
plot = [plt.contourf(Xc, Yc, np.transpose((C_sol[:,-1]/phi[:,0]).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",levels=100)]
plt.plot(X_sat,Y_sat,'r-',linewidth=0.4)
plt.plot(X_bnd,Y_bnd,'r-',linewidth=2)
plt.plot(X_all,Y_all,'k-',linewidth=0.4)
plt.colorbar()
plt.ylabel(r'$z$')
plt.xlabel(r'$x$')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.axis('scaled')
plt.plot(Xc_col[np.argwhere(C_sol[:,-1]<0)[:,0]],Yc_col[np.argwhere(C_sol[:,-1]<0)[:,0]],'ro',label='Sw>1')
plt.plot(Xc_col[np.argwhere(C_sol[:,-1]>(phi[:,0]+1e-5))[:,0]],Yc_col[np.argwhere(C_sol[:,-1]>(phi[:,0]+1e-5))[:,0]],'gX',label='Sw>1')
plt.savefig(f'../Figures/{simulation_name}_Marcs_way_Sw.pdf',bbox_inches='tight', dpi = 600)

fig = plt.figure(figsize=(12,10) , dpi=100)
plot = [plt.contourf(Xc, Yc, np.transpose((C).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin =np.min(C), vmax = np.max(C),levels=100)]
plt.plot(X_sat,Y_sat,'r-',linewidth=0.4)
plt.plot(X_bnd,Y_bnd,'r-',linewidth=2)
plt.plot(X_all,Y_all,'k-',linewidth=0.4)
plt.colorbar()
plt.ylabel(r'$z$')
plt.xlabel(r'$x$')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.axis('scaled')
plt.plot(Xc_col[np.argwhere(C_sol[:,-1]<0)[:,0]],Yc_col[np.argwhere(C_sol[:,-1]<0)[:,0]],'ro',label='Sw>1')
plt.plot(Xc_col[np.argwhere(C_sol[:,-1]>(phi[:,0]+1e-9))[:,0]],Yc_col[np.argwhere(C_sol[:,-1]>(phi[:,0]+1e-9))[:,0]],'gX',label='Sw>1')
plt.savefig(f'../Figures/{simulation_name}_Marcs_way_theta.pdf',bbox_inches='tight', dpi = 600)


fig = plt.figure(figsize=(10,7.5) , dpi=100)
plot = [plt.contourf(Xc, Yc, np.transpose((C/phi).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = -0.00001, vmax = 1.00001,levels=1000)]
plt.plot(Grid.xmax*x_analytic,Grid.ymax-Grid.xmax*h_analytic/t[-1],'r--',linewidth=2,label='Analytic')
plt.ylabel(r'$z$')
plt.xlabel(r'$x$')
mm = plt.cm.ScalarMappable(cmap='Blues')
mm.set_array(C/phi)
mm.set_clim(np.min((C/phi)),np.max((C/phi)))
clb = plt.colorbar(mm,aspect=15)
clb.ax.set_title(r'$s_w$',y=1.03)
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax, Grid.ymin])
plt.axis('scaled')
plt.savefig(f'../Figures/{simulation_name}_Marcs_way_Sw_withoutmesh.pdf',bbox_inches='tight', dpi = 600)


fig = plt.figure(figsize=(10,7.5) , dpi=100)
plot = [plt.contourf(Xc, Yc, np.transpose((u).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",levels=1000)]
plt.ylabel(r'$z$')
plt.xlabel(r'$x$')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax, Grid.ymin])
plt.axis('scaled')
mm = plt.cm.ScalarMappable(cmap='Blues')
mm.set_array(u)
mm.set_clim(np.min(u),np.max(u))
clb = plt.colorbar(mm,aspect=15)
clb.ax.set_title(r'$h$',y=1.03)
plt.savefig(f'../Figures/{simulation_name}_Marcs_way_head.pdf',bbox_inches='tight', dpi = 600)

fig = plt.figure(figsize=(10,7.5) , dpi=100)
plot = [plt.contourf(Xc, Yc, np.transpose((u-(Grid.ymax-Yc_col)).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",levels=1000)]
plt.ylabel(r'$z$')
plt.xlabel(r'$x$')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax, Grid.ymin])
plt.axis('scaled')
mm = plt.cm.ScalarMappable(cmap='Blues')
mm.set_array(u)
mm.set_clim(np.min(u),np.max(u))
clb = plt.colorbar(mm,aspect=15)
clb.ax.set_title(r'$p_w/\rho g$',y=1.03)
plt.savefig(f'../Figures/{simulation_name}_Marcs_way_pw.pdf',bbox_inches='tight', dpi = 600)

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


'''
#Multiple_time_plots
fig, axes = plt.subplots(nrows=5, ncols=1, sharex=True,figsize=(20,20))


plt.subplot(5,1,1)
plot = [plt.contourf(Xc, Yc, np.transpose((C_sol[:,0]/phi[:,0]).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = 0, vmax = 1,levels=100)]
#manager = plt.get_current_fig_manager()
#manager.window.showMaximized()
#clb = fig.colorbar(plot[0], orientation='horizontal',aspect=50, pad=-0.1)
#clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=-0.1)
plt.plot(Xc_col[np.argwhere(C_sol[:,0]<0)[:,0]],Yc_col[np.argwhere(C_sol[:,-1]<0)[:,0]],'ro',label='Sw>1')
plt.plot(Xc_col[np.argwhere(C_sol[:,0]>(phi[:,0]+1e-9))[:,0]],Yc_col[np.argwhere(C_sol[:,-1]>(phi[:,0]+1e-9))[:,0]],'gX',label='Sw>1')
plt.ylabel(r'$z$')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.axis('scaled')

plt.subplot(5,1,2)
plot = [plt.contourf(Xc, Yc, np.transpose((C_sol[:,np.argwhere(t>49.9999)[0,0]]/phi[:,0]).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = 0, vmax = 1,levels=100)]
#manager = plt.get_current_fig_manager()
#manager.window.showMaximized()
#clb = fig.colorbar(plot[0], orientation='horizontal',aspect=50, pad=-0.1)
#clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=-0.1)
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.plot(Xc_col[np.argwhere(C_sol[:,np.argwhere(t>49.9999)[0,0]]<0)[:,0]],Yc_col[np.argwhere(C_sol[:,np.argwhere(t>49.9999)[0,0]]<0)[:,0]],'ro',label='Sw>1')
plt.plot(Xc_col[np.argwhere(C_sol[:,np.argwhere(t>49.9999)[0,0]]>(phi[:,0]+1e-9))[:,0]],Yc_col[np.argwhere(C_sol[:,np.argwhere(t>49.9999)[0,0]]>(phi[:,0]+1e-9))[:,0]],'gX',label='Sw>1')
plt.axis('scaled')
plt.ylabel(r'$z$')

plt.subplot(5,1,3)
plot = [plt.contourf(Xc, Yc, np.transpose((C_sol[:,np.argwhere(t>99.999)[0,0]]/phi[:,0]).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = 0, vmax = 1,levels=100)]
#manager = plt.get_current_fig_manager()
#manager.window.showMaximized()
#clb = fig.colorbar(plot[0], orientation='horizontal',aspect=50, pad=-0.1)
#clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=-0.1)
plt.ylabel(r'$z$')
plt.plot(Xc_col[np.argwhere(C_sol[:,np.argwhere(t>99.9999)[0,0]]<0)[:,0]],Yc_col[np.argwhere(C_sol[:,np.argwhere(t>99.9999)[0,0]]<0)[:,0]],'ro',label='Sw>1')
plt.plot(Xc_col[np.argwhere(C_sol[:,np.argwhere(t>99.9999)[0,0]]>(phi[:,0]+1e-9))[:,0]],Yc_col[np.argwhere(C_sol[:,np.argwhere(t>99.9999)[0,0]]>(phi[:,0]+1e-9))[:,0]],'gX',label='Sw>1')

plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.axis('scaled')

plt.subplot(5,1,4)
plot = [plt.contourf(Xc, Yc, np.transpose((C_sol[:,np.argwhere(t>149.999)[0,0]]/phi[:,0]).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = 0, vmax = 1,levels=100)]
plt.plot(Xc_col[np.argwhere(C_sol[:,np.argwhere(t>149.9999)[0,0]]<0)[:,0]],Yc_col[np.argwhere(C_sol[:,np.argwhere(t>149.9999)[0,0]]<0)[:,0]],'ro',label='Sw>1')
plt.plot(Xc_col[np.argwhere(C_sol[:,np.argwhere(t>149.9999)[0,0]]>(phi[:,0]+1e-9))[:,0]],Yc_col[np.argwhere(C_sol[:,np.argwhere(t>149.9999)[0,0]]>(phi[:,0]+1e-9))[:,0]],'gX',label='Sw>1')

plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.axis('scaled')
plt.ylabel(r'$z$')
#manager = plt.get_current_fig_manager()
#manager.window.showMaximized()
#clb = fig.colorbar(plot[0], orientation='horizontal',aspect=50, pad=-0.1)
#clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=-0.1)


plt.subplot(5,1,5)
plot = [plt.contourf(Xc, Yc, np.transpose((C_sol[:,np.argwhere(t>199.999)[0,0]]/phi[:,0]).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = 0, vmax = 1,levels=100)]
plt.plot(Xc_col[np.argwhere(C_sol[:,np.argwhere(t>199.9999)[0,0]]<0)[:,0]],Yc_col[np.argwhere(C_sol[:,np.argwhere(t>199.9999)[0,0]]<0)[:,0]],'ro',label='Sw>1')
plt.plot(Xc_col[np.argwhere(C_sol[:,np.argwhere(t>199.9999)[0,0]]>(phi[:,0]+1e-9))[:,0]],Yc_col[np.argwhere(C_sol[:,np.argwhere(t>199.9999)[0,0]]>(phi[:,0]+1e-9))[:,0]],'gX',label='Sw>1')

plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.axis('scaled')
#manager = plt.get_current_fig_manager()
#manager.window.showMaximized()
#clb = fig.colorbar(plot[0], orientation='horizontal',aspect=50, pad=-0.1)
#clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=-0.1)
plt.ylabel(r'$z$')
plt.xlabel(r'$x$')
plt.subplots_adjust(wspace=0, hspace=0)
mm = plt.cm.ScalarMappable(cmap=cm.Blues)
mm.set_array(np.linspace(0,1,100))
cbar = fig.colorbar(mm, ax=axes)
cbar.ax.set_title(r'$s_w $',y=1.03)
plt.savefig(f'../Figures/{simulation_name}_Sw_combined.pdf',bbox_inches='tight', dpi = 600)



#Multiple_time_plots theta
fig, axes = plt.subplots(nrows=5, ncols=1, sharex=True,figsize=(20,20))


plt.subplot(5,1,1)
plot = [plt.contourf(Xc, Yc, np.transpose((C_sol[:,0]).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = np.min(C_sol), vmax = np.max(C_sol),levels=100)]
#manager = plt.get_current_fig_manager()
#manager.window.showMaximized()
#clb = fig.colorbar(plot[0], orientation='horizontal',aspect=50, pad=-0.1)
#clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=-0.1)
plt.plot(Xc_col[np.argwhere(C_sol[:,0]<0)[:,0]],Yc_col[np.argwhere(C_sol[:,-1]<0)[:,0]],'ro',label='Sw>1')
plt.plot(Xc_col[np.argwhere(C_sol[:,0]>(phi[:,0]+1e-9))[:,0]],Yc_col[np.argwhere(C_sol[:,-1]>(phi[:,0]+1e-9))[:,0]],'gX',label='Sw>1')
plt.ylabel(r'$z$')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.axis('scaled')

plt.subplot(5,1,2)
plot = [plt.contourf(Xc, Yc, np.transpose((C_sol[:,np.argwhere(t>49.9999)[0,0]]).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = np.min(C_sol), vmax = np.max(C_sol),levels=100)]
#manager = plt.get_current_fig_manager()
#manager.window.showMaximized()
#clb = fig.colorbar(plot[0], orientation='horizontal',aspect=50, pad=-0.1)
#clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=-0.1)
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.plot(Xc_col[np.argwhere(C_sol[:,np.argwhere(t>49.9999)[0,0]]<0)[:,0]],Yc_col[np.argwhere(C_sol[:,np.argwhere(t>49.9999)[0,0]]<0)[:,0]],'ro',label='Sw>1')
plt.plot(Xc_col[np.argwhere(C_sol[:,np.argwhere(t>49.9999)[0,0]]>(phi[:,0]+1e-9))[:,0]],Yc_col[np.argwhere(C_sol[:,np.argwhere(t>49.9999)[0,0]]>(phi[:,0]+1e-9))[:,0]],'gX',label='Sw>1')
plt.axis('scaled')
plt.ylabel(r'$z$')

plt.subplot(5,1,3)
plot = [plt.contourf(Xc, Yc, np.transpose((C_sol[:,np.argwhere(t>99.999)[0,0]]).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = np.min(C_sol), vmax = np.max(C_sol),levels=100)]
#manager = plt.get_current_fig_manager()
#manager.window.showMaximized()
#clb = fig.colorbar(plot[0], orientation='horizontal',aspect=50, pad=-0.1)
#clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=-0.1)
plt.ylabel(r'$z$')
plt.plot(Xc_col[np.argwhere(C_sol[:,np.argwhere(t>99.9999)[0,0]]<0)[:,0]],Yc_col[np.argwhere(C_sol[:,np.argwhere(t>99.9999)[0,0]]<0)[:,0]],'ro',label='Sw>1')
plt.plot(Xc_col[np.argwhere(C_sol[:,np.argwhere(t>99.9999)[0,0]]>(phi[:,0]+1e-9))[:,0]],Yc_col[np.argwhere(C_sol[:,np.argwhere(t>99.9999)[0,0]]>(phi[:,0]+1e-9))[:,0]],'gX',label='Sw>1')

plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.axis('scaled')

plt.subplot(5,1,4)
plot = [plt.contourf(Xc, Yc, np.transpose((C_sol[:,np.argwhere(t>149.999)[0,0]]).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = np.min(C_sol), vmax = np.max(C_sol),levels=100)]
plt.plot(Xc_col[np.argwhere(C_sol[:,np.argwhere(t>149.9999)[0,0]]<0)[:,0]],Yc_col[np.argwhere(C_sol[:,np.argwhere(t>149.9999)[0,0]]<0)[:,0]],'ro',label='Sw>1')
plt.plot(Xc_col[np.argwhere(C_sol[:,np.argwhere(t>149.9999)[0,0]]>(phi[:,0]+1e-9))[:,0]],Yc_col[np.argwhere(C_sol[:,np.argwhere(t>149.9999)[0,0]]>(phi[:,0]+1e-9))[:,0]],'gX',label='Sw>1')

plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.axis('scaled')
plt.ylabel(r'$z$')
#manager = plt.get_current_fig_manager()
#manager.window.showMaximized()
#clb = fig.colorbar(plot[0], orientation='horizontal',aspect=50, pad=-0.1)
#clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=-0.1)


plt.subplot(5,1,5)
plot = [plt.contourf(Xc, Yc, np.transpose((C_sol[:,np.argwhere(t>199.999)[0,0]]).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = np.min(C_sol), vmax = np.max(C_sol),levels=100)]
plt.plot(Xc_col[np.argwhere(C_sol[:,np.argwhere(t>199.9999)[0,0]]<0)[:,0]],Yc_col[np.argwhere(C_sol[:,np.argwhere(t>199.9999)[0,0]]<0)[:,0]],'ro',label='Sw>1')
plt.plot(Xc_col[np.argwhere(C_sol[:,np.argwhere(t>199.9999)[0,0]]>(phi[:,0]+1e-9))[:,0]],Yc_col[np.argwhere(C_sol[:,np.argwhere(t>199.9999)[0,0]]>(phi[:,0]+1e-9))[:,0]],'gX',label='Sw>1')

plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.axis('scaled')
#manager = plt.get_current_fig_manager()
#manager.window.showMaximized()
#clb = fig.colorbar(plot[0], orientation='horizontal',aspect=50, pad=-0.1)
#clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=-0.1)
plt.ylabel(r'$z$')
plt.xlabel(r'$x$')
plt.subplots_adjust(wspace=0, hspace=0)
mm = plt.cm.ScalarMappable(cmap=cm.Blues)
mm.set_array(np.linspace(np.min(C_sol),np.max(C_sol),100))
cbar = fig.colorbar(mm, ax=axes)
cbar.ax.set_title(r'$\theta$',y=1.03)
plt.savefig(f'../Figures/{simulation_name}_theta_combined.pdf',bbox_inches='tight', dpi = 600)

'''

'''
#Multiple_time_plots
fig, axes = plt.subplots(nrows=5, ncols=1, sharex=True,figsize=(20,20))


plt.subplot(5,1,1)
plot = [plt.contourf(Xc, Yc, np.transpose((C_sol[:,np.argwhere(t>5.9999)[0,0]]/phi[:,0]).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = 0, vmax = 1,levels=100)]
plt.plot(Grid.xmax*x_analytic,Grid.ymax-Grid.xmax*h_analytic/6,'r-',linewidth=3)

#manager = plt.get_current_fig_manager()
#manager.window.showMaximized()
#clb = fig.colorbar(plot[0], orientation='horizontal',aspect=50, pad=-0.1)
#clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=-0.1)
plt.ylabel(r'$z$')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymax/2])
plt.yticks([0.6,0.8,1.0])
#plt.axis('scaled')

plt.subplot(5,1,2)
plot = [plt.contourf(Xc, Yc, np.transpose((C_sol[:,np.argwhere(t>9.9999)[0,0]]/phi[:,0]).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = 0, vmax = 1,levels=100)]
#manager = plt.get_current_fig_manager()
#manager.window.showMaximized()
plt.plot(Grid.xmax*x_analytic,Grid.ymax-Grid.xmax*h_analytic/10,'r-',linewidth=3)
#clb = fig.colorbar(plot[0], orientation='horizontal',aspect=50, pad=-0.1)
#clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=-0.1)
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymax/2])
#plt.axis('scaled')
plt.yticks([0.6,0.8,1.0])
plt.ylabel(r'$z$')

plt.subplot(5,1,3)
plot = [plt.contourf(Xc, Yc, np.transpose((C_sol[:,np.argwhere(t>19.999)[0,0]]/phi[:,0]).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = 0, vmax = 1,levels=100)]
#manager = plt.get_current_fig_manager()
#manager.window.showMaximized()
plt.plot(Grid.xmax*x_analytic,Grid.ymax-Grid.xmax*h_analytic/19,'r-',linewidth=3)
#clb = fig.colorbar(plot[0], orientation='horizontal',aspect=50, pad=-0.1)
#clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=-0.1)
plt.ylabel(r'$z$')
plt.yticks([0.6,0.8,1.0])
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymax/2])
#plt.axis('scaled')

plt.subplot(5,1,4)
plot = [plt.contourf(Xc, Yc, np.transpose((C_sol[:,np.argwhere(t>29.999)[0,0]]/phi[:,0]).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = 0, vmax = 1,levels=100)]
plt.plot(Grid.xmax*x_analytic,Grid.ymax-Grid.xmax*h_analytic/29,'r-',linewidth=3)
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymax/2])
#plt.axis('scaled')
plt.ylabel(r'$z$')
#manager = plt.get_current_fig_manager()
#manager.window.showMaximized()
plt.yticks([0.6,0.8,1.0])
#clb = fig.colorbar(plot[0], orientation='horizontal',aspect=50, pad=-0.1)
#clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=-0.1)


plt.subplot(5,1,5)
plot = [plt.contourf(Xc, Yc, np.transpose((C_sol[:,np.argwhere(t>39.99999)[0,0]]/phi[:,0]).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = 0, vmax = 1,levels=100)]
plt.plot(Grid.xmax*x_analytic,Grid.ymax-Grid.xmax*h_analytic/40,'r-',linewidth=3)
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymax/2])
#plt.axis('scaled')
#manager = plt.get_current_fig_manager()
#manager.window.showMaximized()
#clb = fig.colorbar(plot[0], orientation='horizontal',aspect=50, pad=-0.1)
#clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=-0.1)
plt.ylabel(r'$z$')
plt.xlabel(r'$x$')
plt.subplots_adjust(wspace=0, hspace=0)
mm = plt.cm.ScalarMappable(cmap=cm.Blues)
mm.set_array(np.linspace(0,1,100))
cbar = fig.colorbar(mm, ax=axes)
cbar.ax.set_title(r'$s_w $',y=1.03)
plt.yticks([0.6,0.8,1])
plt.savefig(f'../Figures/{simulation_name}_Sw_combined.pdf',bbox_inches='tight', dpi = 600)

'''

#Total volume of water inside the reservoir
from mpltools import annotation
cell_volume =  np.kron(np.ones_like(t),Grid.V) #multiplying with time
total_volume = np.sum(cell_volume*C_sol,axis=0)

dx_analytic = x_analytic[2]-x_analytic[1]
w_analytic  = np.sum(Grid.xmax**2*h_analytic * dx_analytic)  #Integrating h and dx_analytic
plt.figure(figsize=(10,8))
plt.loglog(t,w_analytic*phi_L/(t),'r-',linewidth=3)
plt.loglog(t,total_volume,'b-',linewidth=3,label='Simulations, Present')
plt.loglog(t[0:1],w_analytic*phi_L/(t[0:1]),'r-',linewidth=3,label='Analytic')
annotation.slope_marker((1.71,2), (-1, 1))
plt.ylim([np.min(total_volume), np.max(total_volume)])
plt.legend(loc='best',framealpha=0.0)
plt.axis('scaled')
plt.ylabel(r'$\mathcal{V}/(wL^2)$')
plt.xlabel(r'$t^\prime$')
plt.tight_layout()
plt.savefig(f'../Figures/{simulation_name}_volume.pdf',bbox_inches='tight', dpi = 600)



