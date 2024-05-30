#Coding the Richard's equation: Schematic figure
#Woods and Hesse
#Mohammad Afzal Shadab
#Date modified: 03/29/2022

import sys
sys.path.insert(1, '../../solver')


# import personal libraries and class
from classes import *    

from two_components_aux import *
from solve_lbvpfun_SPD import solve_lbvp_SPD
from complex_domain import find_faces, find_bnd_cells, find_all_faces, find_all_x_faces,find_all_y_faces, find_crater_dofs
from comp_fluxfun import comp_flux
from scipy.integrate import solve_ivp
from comp_sat_unsat_bnd_flux_fun import comp_sat_unsat_bnd_flux, find_top_bot_cells, find_left_right_cells#,comp_sat_unsat_bnd_flux_xface
from comp_face_coords_fun import comp_face_coords
from find_plate_dofs import find_plate_dofs
from spin_up import spin_up_C
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
simulation_name = f'Paper-schematic'
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

jet_width = 1#0.0469343437 #
jet_intended_location = 2.5#5#0.72995

#injection
Param.xleft_inj = jet_intended_location-jet_width/2  
Param.xright_inj= jet_intended_location+jet_width/2


#temporal
tmax = 0.1#0.07#0.0621#2 #5.7#6.98  #time scaling with respect to fc
t_interest = np.linspace(0,tmax,21)   #swr,sgr=0

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
Grid.xmin =  0; Grid.xmax =1; Grid.Nx = 25; 
Grid.ymin =  0; Grid.ymax =1; Grid.Ny = 25;
Grid = build_grid(Grid)
[D,G,I] = build_ops(Grid)
Avg     = comp_mean_matrix(Grid) #comp_algebraic_mean(Grid)#

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


phi = phi_L*np.ones((Grid.N,1))
BC.C_g         = 0.0*phi[dof_inj-1,:]  #saturated boundary condition

C = phi*0.1

[dof_inact,_] = find_crater_dofs(0.2,0.4,0.15,Grid,Xc_col,Yc_col)
C[dof_inact-1,0] = phi_L

[dof_inact,_] = find_crater_dofs(0.8,0.4,0.15,Grid,Xc_col,Yc_col)
C[dof_inact-1,0] = phi_L

[dof_inact,_] = find_crater_dofs(0.5,1,0.4,Grid,Xc_col,Yc_col)
C[dof_inact-1,0] = phi_L


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
    
    res = D@flux_vert  #since the gradients are only zero and 1    
    res_vert = res.copy()
    ######
    #Taking out the domain to cut off single phase region
    dof_act  = Grid.dof[C_old[:,0] / (phi[:,0]*(1-s_gr)) < sat_threshold]
    dof_inact= np.setdiff1d(Grid.dof,dof_act) #saturated cells
    if len(dof_act)< Grid.N: #when atleast one layer is present
        #############################################
        dof_f_saturated = find_faces(dof_inact,D,Grid)       

        Param.dof_neu   = np.hstack([Grid.dof_xmin, Grid.dof_xmax, Grid.dof_ymin, Grid.dof_ymax])
        dof_act = np.concatenate([dof_act,np.intersect1d(dof_inact,Param.dof_neu)]) #setting outflow

        #Eliminate inactive cells by putting them to constraint matrix
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
        fs = np.zeros((Grid.N,1))
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

t = np.array(t)


#saving the tensors
np.savez(f'{simulation_name}_C{C_L}_{Grid.Nx}by{Grid.Ny}_t{tmax}.npz', t=t,C_sol =C_sol,phi=phi,Xc=Xc,Yc=Yc,Xc_col=Xc_col,Yc_col=Yc_col,Grid_Nx=Grid.Nx,Grid_Ny=Grid.Ny,Grid_xc=Grid.xc,Grid_yc=Grid.yc,Grid_xf=Grid.xf,Grid_yf=Grid.yf,flux_sol = flux_sol)



###############################################
light_red  = [1.0,0.5,0.5]
light_blue = [0.5,0.5,1.0]
light_black= [0.5,0.5,0.5]

fig = plt.figure(figsize=(15,4) , dpi=100)
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



#Plotting the schematic
fig, axes = plt.subplots(nrows=2, ncols=3,sharex=True,sharey=True, figsize=(12,15) , dpi=100)


plt.subplot(2,3,1)
dof_inact= Grid.dof[C_sol[:,0] / (phi[:,0]*(1-s_gr)) >= sat_threshold] #saturated cells
dof_act= np.setdiff1d(Grid.dof,dof_inact) #saturated cells
dof_f_saturated = find_faces(dof_inact,D,Grid) 
dof_sat_faces = find_all_faces(dof_inact,D,Grid) 
[X_sat,Y_sat] = comp_face_coords(dof_sat_faces,Grid)
[X_bnd,Y_bnd] = comp_face_coords(dof_f_saturated,Grid)
[X_all,Y_all] = comp_face_coords(Grid.dof_f,Grid)
dof_f_saturated = find_faces(dof_inact,D,Grid) 
dof_f_saturated_domain_bnd = np.intersect1d(Grid.dof_f_ymax,dof_f_saturated)
[X_sat_dom_bnd,Y_sat_dom_bnd] = comp_face_coords(dof_f_saturated_domain_bnd,Grid)

plot = [plt.contourf(Xc, Yc, np.transpose((C_sol[:,0]/phi[:,0]).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin=0,vmax=1.05,levels=100, extend="max")]
plt.plot(X_all,Y_all,'k-',linewidth=0.4)
plt.plot(X_bnd,Y_bnd,'r-',linewidth=2)
plt.plot(X_sat_dom_bnd,Y_sat_dom_bnd,'g-',linewidth=2)
#circle1 = plt.Circle((0.2,0.4),0.15, color='b', fill=False,linewidth=2,label='Saturated'); plt.gca().add_patch(circle1)
#circle2 = plt.Circle((0.8,0.4),0.15, color='b', fill=False,linewidth=2); plt.gca().add_patch(circle2)
#circle3 = plt.Circle((0.5,1),0.4, color='b', fill=False,linewidth=2); plt.gca().add_patch(circle3)
plt.ylabel(r'$z$')
plt.axis('scaled')
plt.yticks([0,0.25,0.5,0.75,1])
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax+0.005,Grid.ymin])


plt.subplot(2,3,4)

plt.plot(Xc_col[dof_inact-1,0],Yc_col[dof_inact-1,0],'ro', markerfacecolor="None",markersize=2)
plt.plot(Xc_col[dof_act-1,0],Yc_col[dof_act-1,0],'go', markerfacecolor="None",markersize=2)

plt.plot(Xc_col[dof_inact[0]-1,0],Yc_col[dof_inact[0]-1,0],'ro',label='Saturated cell', markerfacecolor="None",markersize=2)
plt.plot(Xc_col[dof_act[0]-1,0],Yc_col[dof_act[0]-1,0],'go',label='Unsaturated cell', markerfacecolor="None",markersize=2)

plt.plot(X_all,Y_all,'k-',linewidth=0.4)
plt.plot(X_sat,Y_sat,'r-',linewidth=0.4)
plt.plot(X_bnd,Y_bnd,'r-',linewidth=2)
plt.plot(X_sat_dom_bnd,Y_sat_dom_bnd,'g-',linewidth=2)

plt.plot(X_sat[0,0],Y_sat[0,0],'r-',linewidth=0.4,label='Saturated face')
plt.plot(X_bnd[0,0],Y_bnd[0,0],'r-',linewidth=2,label=r'$\partial \Omega_s$')
plt.plot(X_sat_dom_bnd[0,0],Y_sat_dom_bnd[0,0],'g-',linewidth=2,label=r'$\partial \Omega_s \cap \partial \Omega$')

#plt.legend(loc='best', framealpha=1)
plt.ylabel(r'$z$')
plt.xlabel(r'$x$')
plt.axis('scaled')
plt.xticks([0,0.25,0.5,0.75,1])
plt.yticks([0,0.25,0.5,0.75,1])
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax+0.005,Grid.ymin])



plt.subplot(2,3,2)
dof_inact= Grid.dof[C_sol[:,5] / (phi[:,0]*(1-s_gr)) >= sat_threshold] #saturated cells
dof_act= np.setdiff1d(Grid.dof,dof_inact) #saturated cells
dof_f_saturated = find_faces(dof_inact,D,Grid) 
dof_sat_faces = find_all_faces(dof_inact,D,Grid) 
[X_sat,Y_sat] = comp_face_coords(dof_sat_faces,Grid)
[X_bnd,Y_bnd] = comp_face_coords(dof_f_saturated,Grid)
[X_all,Y_all] = comp_face_coords(Grid.dof_f,Grid)
dof_f_saturated_domain_bnd = np.intersect1d(Grid.dof_f_ymax,dof_f_saturated)
[X_sat_dom_bnd,Y_sat_dom_bnd] = comp_face_coords(dof_f_saturated_domain_bnd,Grid)

plot = [plt.contourf(Xc, Yc, np.transpose(((C_sol[:,5]/phi[:,0]).reshape(Grid.Nx,Grid.Ny))),cmap="Blues",vmin=0.0,vmax=1,levels=100, extend="max")]
plt.plot(X_sat,Y_sat,'r-',linewidth=0.4)
plt.plot(X_bnd,Y_bnd,'r-',linewidth=2)
plt.plot(X_all,Y_all,'k-',linewidth=0.4)
plt.plot(X_sat_dom_bnd,Y_sat_dom_bnd,'g-',linewidth=2)
#circle1 = plt.Circle((0.2,0.4),0.15, color='b', fill=False,linewidth=2,label='Saturated'); plt.gca().add_patch(circle1)
#circle2 = plt.Circle((0.8,0.4),0.15, color='b', fill=False,linewidth=2); plt.gca().add_patch(circle2)
#circle3 = plt.Circle((0.5,1),0.4, color='b', fill=False,linewidth=2); plt.gca().add_patch(circle3)
plt.axis('scaled')
plt.xticks([0,0.25,0.5,0.75,1])
plt.yticks([0,0.25,0.5,0.75,1])
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax+0.005,Grid.ymin])

plt.subplot(2,3,5)
plt.plot(X_all,Y_all,'k-',linewidth=0.4)
plt.plot(X_sat,Y_sat,'r-',linewidth=0.4)
plt.plot(X_bnd,Y_bnd,'r-',linewidth=2)
plt.plot(X_sat_dom_bnd,Y_sat_dom_bnd,'g-',linewidth=2)
plt.plot(Xc_col[dof_inact-1,0],Yc_col[dof_inact-1,0],'ro',label='Saturated', markerfacecolor="None",markersize=2)
plt.plot(Xc_col[dof_act-1,0],Yc_col[dof_act-1,0],'go',label='Unsaturated', markerfacecolor="None",markersize=2)
plt.xlabel(r'$x$')
plt.axis('scaled')
plt.xticks([0,0.25,0.5,0.75,1])
plt.yticks([0,0.25,0.5,0.75,1])
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax+0.005,Grid.ymin])


plt.subplot(2,3,3)
dof_inact= Grid.dof[C_sol[:,10] / (phi[:,0]*(1-s_gr)) >= sat_threshold] #saturated cells
dof_act= np.setdiff1d(Grid.dof,dof_inact) #saturated cells
dof_f_saturated = find_faces(dof_inact,D,Grid) 
dof_sat_faces = find_all_faces(dof_inact,D,Grid) 
[X_sat,Y_sat] = comp_face_coords(dof_sat_faces,Grid)
[X_bnd,Y_bnd] = comp_face_coords(dof_f_saturated,Grid)
[X_all,Y_all] = comp_face_coords(Grid.dof_f,Grid)
dof_f_saturated_domain_bnd = np.intersect1d(Grid.dof_f_ymax,dof_f_saturated)
[X_sat_dom_bnd,Y_sat_dom_bnd] = comp_face_coords(dof_f_saturated_domain_bnd,Grid)

plot = [plt.contourf(Xc, Yc, np.transpose(((C_sol[:,10]/phi[:,0]).reshape(Grid.Nx,Grid.Ny))),cmap="Blues",vmin=0,vmax=1,levels=100, extend="max")]
plt.plot(X_sat,Y_sat,'r-',linewidth=0.4)
plt.plot(X_bnd,Y_bnd,'r-',linewidth=2)
plt.plot(X_all,Y_all,'k-',linewidth=0.4)
plt.plot(X_sat_dom_bnd,Y_sat_dom_bnd,'g-',linewidth=2)
#circle1 = plt.Circle((0.2,0.4),0.15, color='b', fill=False,linewidth=2,label='Saturated'); plt.gca().add_patch(circle1)
#circle2 = plt.Circle((0.8,0.4),0.15, color='b', fill=False,linewidth=2); plt.gca().add_patch(circle2)
#circle3 = plt.Circle((0.5,1),0.4, color='b', fill=False,linewidth=2); plt.gca().add_patch(circle3)
plt.axis('scaled')
plt.xticks([0,0.25,0.5,0.75,1])
plt.yticks([0,0.25,0.5,0.75,1])
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax+0.005,Grid.ymin])


plt.subplot(2,3,6)
plt.plot(X_all,Y_all,'k-',linewidth=0.4)
plt.plot(X_sat,Y_sat,'r-',linewidth=0.4)
plt.plot(X_bnd,Y_bnd,'r-',linewidth=2)
plt.plot(X_sat_dom_bnd,Y_sat_dom_bnd,'g-',linewidth=2)
plt.plot(Xc_col[dof_inact-1,0],Yc_col[dof_inact-1,0],'ro',label='Saturated', markerfacecolor="None",markersize=2)
plt.plot(Xc_col[dof_act-1,0],Yc_col[dof_act-1,0],'go',label='Unsaturated', markerfacecolor="None",markersize=2)

plt.xlabel(r'$x$')
plt.axis('scaled')
plt.xticks([0,0.25,0.5,0.75,1])
plt.yticks([0,0.25,0.5,0.75,1])
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax+0.005,Grid.ymin])

plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.tight_layout()
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.savefig(f'../Figures/{simulation_name}_schematic_without_legend.pdf',bbox_inches='tight', dpi = 100)

mm = plt.cm.ScalarMappable(cmap=cm.Blues)
mm.set_array(np.linspace(0,1,100))
cbar = fig.colorbar(mm, ax=axes[1],aspect=10)
cbar.ax.set_title(r'$s_w$',y=1.03)
plt.savefig(f'../Figures/{simulation_name}_schematic_without_legend_with_colorbar.pdf',bbox_inches='tight', dpi = 100)
