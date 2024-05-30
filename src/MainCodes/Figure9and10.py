#Coding the Richard's equation
#Woods and Hesse
#Mohammad Afzal Shadab
#Date modified: 03/29/2022

import sys
sys.path.insert(1, '../../solver')


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
from random_correlated_fields import GeneratePermField,GeneratePermField2D,GeneratePermField2D_papers
from comp_mean_matrix import comp_mean_matrix


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
                3: [[0.3164247822051133,0.85],[0.0,0.2]]#[[0.4,0.5],[0.0,0.2]], #Contact discontinuity C1  (working)   
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
simulation_name = f'paper_randomly_correlated_fields_final'
m = 8 #Cozeny-Karman coefficient for numerator K = K0 (1-phi_i)^m
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

jet_width = 6#0.0469343437 #
jet_intended_location = 2.5#5#0.72995

#injection
Param.xleft_inj = jet_intended_location-jet_width/2  
Param.xright_inj= jet_intended_location+jet_width/2


#temporal
tmax = 300  #time scaling with respect to fc
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
    fC[C<=0]  = 0.0
    #fC[C>phi] = 0.0      
    return fC

#spatial
Grid.xmin =  0; Grid.xmax =5; Grid.Nx = 250; 
Grid.ymin =  0; Grid.ymax =1; Grid.Ny = 100;
Grid = build_grid(Grid)
[D,G,I] = build_ops(Grid)
Avg     = comp_mean_matrix(Grid)

[Xc,Yc] = np.meshgrid(Grid.xc,Grid.yc)                 #building the (x,y) matrix
Xc_col  = np.reshape(np.transpose(Xc), (Grid.N,-1))    #building the single X vector
Yc_col  = np.reshape(np.transpose(Yc), (Grid.N,-1))    #building the single Y vector

if direction == 'x':
    #injection - X
    dof_inj   = Grid.dof_xmin[  np.intersect1d(np.argwhere(Grid.yc>= Param.xleft_inj),np.argwhere(Grid.yc <= Param.xright_inj))]
    dof_f_inj = Grid.dof_f_xmin[np.intersect1d(np.argwhere(Grid.yc>= Param.xleft_inj),np.argwhere(Grid.yc <= Param.xright_inj))]

else:
    #injection - Y
    dof_inj   = Grid.dof_ymin[  np.intersect1d(np.argwhere(Grid.xc>= Param.xleft_inj),np.argwhere(Grid.xc <= Param.xright_inj))]
    dof_f_inj = Grid.dof_f_ymin[np.intersect1d(np.argwhere(Grid.xc>= Param.xleft_inj),np.argwhere(Grid.xc <= Param.xright_inj))]


##########
#boundary condition
common_cell = np.intersect1d(Grid.dof_ymax,Grid.dof_xmax)
common_face = np.intersect1d(Grid.dof_f_ymax,Grid.dof_f_xmax)

hor_bnd = np.setdiff1d(Grid.dof_xmax,common_cell)


BC.dof_dir   = np.array([dof_inj])
BC.dof_f_dir = np.array([dof_f_inj])
BC.C_g       = np.transpose([np.ones_like(dof_inj)]) #placeholder

BC.dof_neu = np.array([])
BC.dof_f_neu = np.array([])
BC.qb  = np.array([])

[B,N,fn] = build_bnd(BC, Grid, I)

############################################

#Permeability field
corr_length = 1; K_mean = 0 ; amplitude = 1;s = 25061977

[K,Xc,Yc] = GeneratePermField2D_papers(Grid,10*corr_length,corr_length,amplitude,K_mean,'exp',s)

K = 10**(-K)

K = np.reshape(np.transpose(K), (Grid.N,-1))/np.max(K)
phi = phi_L*K**(1/m)

#Plot porosity
phi_array = np.transpose(phi.reshape(Grid.Nx,Grid.Ny))
fig = plt.figure(figsize=(15,7.5) , dpi=100)
plot = [plt.contourf(Xc, Yc, phi_array,cmap="coolwarm",levels=100)]
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.xlabel(r'$x$')
plt.ylabel(r'$z$')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.axis('scaled')
mm = plt.cm.ScalarMappable(cmap=cm.coolwarm)
mm.set_array(phi)
clb = plt.colorbar(mm, pad=0.1)
clb.set_label(r'$\phi $', labelpad=1, y=1.075, rotation=0)
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
plt.savefig(f'../Figures/correlated_fields_{Grid.Nx}by{Grid.Ny}_Marcs-logK_corr{corr_length}.pdf',bbox_inches='tight', dpi = 600)

####################################################################################
print('Correlated random field generated!')
BC.C_g         = phi[dof_inj-1,:]*sat_threshold*(1-1e-10)  #saturated boundary condition

#Initialize
C = phi*s_wr

C_sol = C.copy()
flux_sol = np.zeros((Grid.Nf,1))

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
    
    #flux_vert_mat = sp.dia_matrix((flux_vert[:,0],  np.array([0])), shape=(Grid.Nf, Grid.Nf))
    res = D@flux_vert  #since the gradients are only zero and 1    
    res_vert = res.copy()
    ######
    #Taking out the domain to cut off single phase region
    dof_act  = Grid.dof[C_old[:,0] / (phi[:,0]*(1-s_gr)) < sat_threshold]
    dof_inact= np.setdiff1d(Grid.dof,dof_act) #saturated cells
    if len(dof_act)< Grid.N: #when atleast one layer is present
        #############################################
        dof_f_saturated = find_faces(dof_inact,D,Grid)   
        
        # Eliminate inactive cells by putting them to constraint matrix
        BC_P.dof_dir = (dof_act)           
        BC_P.dof_f_dir= np.array([])
        BC_P.g       =  -Yc_col[dof_act-1]*np.cos(tilt_angle*deg2rad) \
                        -Xc_col[dof_act-1]*np.sin(tilt_angle*deg2rad)    
        BC_P.dof_neu = np.array([])
        BC_P.dof_f_neu = np.array([])
        BC_P.qb = np.array([])

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
        q_w = comp_flux(D, Kd, G, u, fn, Grid, Param)

        #upwinding boundary y-directional flux
        #finding boundary faces
        dof_ysat_faces = dof_f_saturated[dof_f_saturated>=Grid.Nfx]
        
        #removing boundary faces
        dof_ysat_faces = np.setdiff1d(dof_ysat_faces,np.append(Grid.dof_f_ymin,Grid.dof_f_ymax))
        
        ytop,ybot               = find_top_bot_cells(dof_ysat_faces,D,Grid)
        q_w[dof_ysat_faces-1,0] = comp_sat_unsat_bnd_flux(q_w[dof_ysat_faces-1],flux_vert[dof_ysat_faces-1],ytop,ybot,C,phi,sat_threshold)
        
         #upwinding boundary x-directional flux   ####new line
        #finding boundary faces
        dof_xsat_faces = dof_f_saturated[dof_f_saturated<Grid.Nfx]
        
        #removing boundary faces
        dof_xsat_faces = np.setdiff1d(dof_xsat_faces,np.append(Grid.dof_f_xmin,Grid.dof_f_xmax))
        
        xleft,xright            = find_left_right_cells(dof_xsat_faces,D,Grid)
        q_w[dof_xsat_faces-1,0] = comp_sat_unsat_bnd_flux(q_w[dof_xsat_faces-1],flux_vert[dof_xsat_faces-1],xright,xleft,C,phi,sat_threshold)
         
            
        #find all saturated faces
        dof_sat_faces = find_all_faces(dof_inact,D,Grid)  
        
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

t = np.array(t)


#saving the tensors
#np.savez(f'{simulation_name}_porosity.npz', phi = phi, K = K, m = m, Grid_Nx=Grid.Nx,Grid_Ny=Grid.Ny)

np.savez(f'{simulation_name}_C{C_L}_{Grid.Nx}by{Grid.Ny}_t{tmax}.npz', t=t,C_sol =C_sol,phi=phi,Xc=Xc,Yc=Yc,Xc_col=Xc_col,Yc_col=Yc_col,Grid_Nx=Grid.Nx,Grid_Ny=Grid.Ny,Grid_xc=Grid.xc,Grid_yc=Grid.yc,Grid_xf=Grid.xf,Grid_yf=Grid.yf,flux_sol = flux_sol)

'''
#for loading data
data = np.load('paper_randomly_correlated_fields__C0.3164247822051133_250by100_t300.npz')
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




#Multiple_time_plots
fig, axes = plt.subplots(nrows=5, ncols=1, sharex=True,figsize=(20,20))

plt.subplot(5,1,1)
plot = [plt.contourf(Xc, Yc, np.transpose((C_sol[:,np.argwhere(t>9.9999)[0,0]]/phi[:,0]).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = 0, vmax = 1,levels=100)]
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.plot(Xc_col[np.argwhere(C_sol[:,np.argwhere(t>9.9999)[0,0]]<0)[:,0]],Yc_col[np.argwhere(C_sol[:,np.argwhere(t>9.9999)[0,0]]<0)[:,0]],'ro',label='Sw>1')
plt.plot(Xc_col[np.argwhere(C_sol[:,np.argwhere(t>9.9999)[0,0]]>(phi[:,0]+1e-9))[:,0]],Yc_col[np.argwhere(C_sol[:,np.argwhere(t>9.9999)[0,0]]>(phi[:,0]+1e-9))[:,0]],'gX',label='Sw>1')
plt.axis('scaled')
plt.ylabel(r'$z$')


plt.subplot(5,1,2)
plot = [plt.contourf(Xc, Yc, np.transpose((C_sol[:,np.argwhere(t>24.9999)[0,0]]/phi[:,0]).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = 0, vmax = 1,levels=100)]
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.plot(Xc_col[np.argwhere(C_sol[:,np.argwhere(t>24.9999)[0,0]]<0)[:,0]],Yc_col[np.argwhere(C_sol[:,np.argwhere(t>24.9999)[0,0]]<0)[:,0]],'ro',label='Sw>1')
plt.plot(Xc_col[np.argwhere(C_sol[:,np.argwhere(t>24.9999)[0,0]]>(phi[:,0]+1e-9))[:,0]],Yc_col[np.argwhere(C_sol[:,np.argwhere(t>24.9999)[0,0]]>(phi[:,0]+1e-9))[:,0]],'gX',label='Sw>1')
plt.axis('scaled')
plt.ylabel(r'$z$')

plt.subplot(5,1,3)
plot = [plt.contourf(Xc, Yc, np.transpose((C_sol[:,np.argwhere(t>49.9999)[0,0]]/phi[:,0]).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = 0, vmax = 1,levels=100)]
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.plot(Xc_col[np.argwhere(C_sol[:,np.argwhere(t>49.9999)[0,0]]<0)[:,0]],Yc_col[np.argwhere(C_sol[:,np.argwhere(t>49.9999)[0,0]]<0)[:,0]],'ro',label='Sw>1')
plt.plot(Xc_col[np.argwhere(C_sol[:,np.argwhere(t>49.9999)[0,0]]>(phi[:,0]+1e-9))[:,0]],Yc_col[np.argwhere(C_sol[:,np.argwhere(t>49.9999)[0,0]]>(phi[:,0]+1e-9))[:,0]],'gX',label='Sw>1')
plt.axis('scaled')
plt.ylabel(r'$z$')

plt.subplot(5,1,4)
plot = [plt.contourf(Xc, Yc, np.transpose((C_sol[:,np.argwhere(t>99.999)[0,0]]/phi[:,0]).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = 0, vmax = 1,levels=100)]
plt.ylabel(r'$z$')
plt.plot(Xc_col[np.argwhere(C_sol[:,np.argwhere(t>99.9999)[0,0]]<0)[:,0]],Yc_col[np.argwhere(C_sol[:,np.argwhere(t>99.9999)[0,0]]<0)[:,0]],'ro',label='Sw>1')
plt.plot(Xc_col[np.argwhere(C_sol[:,np.argwhere(t>99.9999)[0,0]]>(phi[:,0]+1e-9))[:,0]],Yc_col[np.argwhere(C_sol[:,np.argwhere(t>99.9999)[0,0]]>(phi[:,0]+1e-9))[:,0]],'gX',label='Sw>1')

plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.axis('scaled')

plt.subplot(5,1,5)
plot = [plt.contourf(Xc, Yc, np.transpose((C_sol[:,np.argwhere(t>249.999)[0,0]]/phi[:,0]).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = 0, vmax = 1,levels=100)]
plt.plot(Xc_col[np.argwhere(C_sol[:,np.argwhere(t>249.9999)[0,0]]<0)[:,0]],Yc_col[np.argwhere(C_sol[:,np.argwhere(t>249.9999)[0,0]]<0)[:,0]],'ro',label='Sw>1')
plt.plot(Xc_col[np.argwhere(C_sol[:,np.argwhere(t>249.9999)[0,0]]>(phi[:,0]+1e-9))[:,0]],Yc_col[np.argwhere(C_sol[:,np.argwhere(t>249.9999)[0,0]]>(phi[:,0]+1e-9))[:,0]],'gX',label='Sw>1')

plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.axis('scaled')
plt.ylabel(r'$z$')

plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.axis('scaled')
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
plot = [plt.contourf(Xc, Yc, np.transpose((C_sol[:,np.argwhere(t>9.9999)[0,0]]).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = np.min(C_sol), vmax = np.max(C_sol),levels=100)]
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.plot(Xc_col[np.argwhere(C_sol[:,np.argwhere(t>9.9999)[0,0]]<0)[:,0]],Yc_col[np.argwhere(C_sol[:,np.argwhere(t>9.9999)[0,0]]<0)[:,0]],'ro',label='Sw>1')
plt.plot(Xc_col[np.argwhere(C_sol[:,np.argwhere(t>9.9999)[0,0]]>(phi[:,0]+1e-9))[:,0]],Yc_col[np.argwhere(C_sol[:,np.argwhere(t>9.9999)[0,0]]>(phi[:,0]+1e-9))[:,0]],'gX',label='Sw>1')
plt.axis('scaled')
plt.ylabel(r'$z$')

plt.subplot(5,1,2)
plot = [plt.contourf(Xc, Yc, np.transpose((C_sol[:,np.argwhere(t>24.9999)[0,0]]).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = np.min(C_sol), vmax = np.max(C_sol),levels=100)]
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.plot(Xc_col[np.argwhere(C_sol[:,np.argwhere(t>24.9999)[0,0]]<0)[:,0]],Yc_col[np.argwhere(C_sol[:,np.argwhere(t>24.9999)[0,0]]<0)[:,0]],'ro',label='Sw>1')
plt.plot(Xc_col[np.argwhere(C_sol[:,np.argwhere(t>24.9999)[0,0]]>(phi[:,0]+1e-9))[:,0]],Yc_col[np.argwhere(C_sol[:,np.argwhere(t>24.9999)[0,0]]>(phi[:,0]+1e-9))[:,0]],'gX',label='Sw>1')
plt.axis('scaled')
plt.ylabel(r'$z$')

plt.subplot(5,1,3)
plot = [plt.contourf(Xc, Yc, np.transpose((C_sol[:,np.argwhere(t>49.9999)[0,0]]).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = np.min(C_sol), vmax = np.max(C_sol),levels=100)]
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.plot(Xc_col[np.argwhere(C_sol[:,np.argwhere(t>49.9999)[0,0]]<0)[:,0]],Yc_col[np.argwhere(C_sol[:,np.argwhere(t>49.9999)[0,0]]<0)[:,0]],'ro',label='Sw>1')
plt.plot(Xc_col[np.argwhere(C_sol[:,np.argwhere(t>49.9999)[0,0]]>(phi[:,0]+1e-9))[:,0]],Yc_col[np.argwhere(C_sol[:,np.argwhere(t>49.9999)[0,0]]>(phi[:,0]+1e-9))[:,0]],'gX',label='Sw>1')
plt.axis('scaled')
plt.ylabel(r'$z$')

plt.subplot(5,1,4)
plot = [plt.contourf(Xc, Yc, np.transpose((C_sol[:,np.argwhere(t>99.999)[0,0]]).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = np.min(C_sol), vmax = np.max(C_sol),levels=100)]
plt.ylabel(r'$z$')
plt.plot(Xc_col[np.argwhere(C_sol[:,np.argwhere(t>99.9999)[0,0]]<0)[:,0]],Yc_col[np.argwhere(C_sol[:,np.argwhere(t>99.9999)[0,0]]<0)[:,0]],'ro',label='Sw>1')
plt.plot(Xc_col[np.argwhere(C_sol[:,np.argwhere(t>99.9999)[0,0]]>(phi[:,0]+1e-9))[:,0]],Yc_col[np.argwhere(C_sol[:,np.argwhere(t>99.9999)[0,0]]>(phi[:,0]+1e-9))[:,0]],'gX',label='Sw>1')

plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.axis('scaled')

plt.subplot(5,1,5)
plot = [plt.contourf(Xc, Yc, np.transpose((C_sol[:,np.argwhere(t>249.999)[0,0]]).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = np.min(C_sol), vmax = np.max(C_sol),levels=100)]
plt.plot(Xc_col[np.argwhere(C_sol[:,np.argwhere(t>249.9999)[0,0]]<0)[:,0]],Yc_col[np.argwhere(C_sol[:,np.argwhere(t>249.9999)[0,0]]<0)[:,0]],'ro',label='Sw>1')
plt.plot(Xc_col[np.argwhere(C_sol[:,np.argwhere(t>249.9999)[0,0]]>(phi[:,0]+1e-9))[:,0]],Yc_col[np.argwhere(C_sol[:,np.argwhere(t>249.9999)[0,0]]>(phi[:,0]+1e-9))[:,0]],'gX',label='Sw>1')

plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.axis('scaled')
plt.ylabel(r'$z$')


plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.axis('scaled')
plt.ylabel(r'$z$')
plt.xlabel(r'$x$')
plt.subplots_adjust(wspace=0, hspace=0)
mm = plt.cm.ScalarMappable(cmap=cm.Blues)
mm.set_array(np.linspace(np.min(C_sol),np.max(C_sol),100))
cbar = fig.colorbar(mm, ax=axes)
cbar.ax.set_title(r'$\theta$',y=1.03)
plt.savefig(f'../Figures/{simulation_name}_theta_combined.pdf',bbox_inches='tight', dpi = 600)

