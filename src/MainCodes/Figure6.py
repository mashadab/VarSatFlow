#Seepage face
#Woods and Hesse
#Mohammad Afzal Shadab
#Date modified: 03/29/2022

import sys
sys.path.insert(1, '../../Solver')

# import personal libraries and class
from classes import *    

from two_components_aux import *
from complex_domain import find_faces, find_bnd_cells, find_all_faces, find_all_x_faces,find_all_y_faces
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
qH =0.084
#parameters
direction = 'z' #x or z

if direction == 'x':
    tilt_angle = 90  #angle of the slope in degrees
else:
    tilt_angle = 0   #angle of the slope in degrees
    
    
case_no =  3               #case number
simulation_name = f'seepage_face'
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
tmax = 5   #time scaling with respect to fc
t_interest = np.linspace(0,tmax,401) 

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
Grid.xmin =  0; Grid.xmax =1; Grid.Nx = 75; 
Grid.ymin =  0; Grid.ymax =1; Grid.Ny = 75;
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

'''
BC.dof_dir   = np.hstack([Grid.dof_ymax,hor_bnd])
BC.dof_f_dir = np.hstack([Grid.dof_f_ymax,Grid.dof_f_xmax[:-1]])
BC.C_g       = np.hstack([C_L*np.ones_like(Grid.dof_ymax),C_L*np.ones_like(Grid.dof_xmax)[:-1]])

BC.dof_dir = Grid.dof_ymin[int(len(Grid.dof_ymin)/2-len(Grid.dof_ymin)/100):int(len(Grid.dof_ymin)/2+len(Grid.dof_ymin)/100)+1]
BC.dof_f_dir = Grid.dof_f_ymin[int(len(Grid.dof_ymin)/2-len(Grid.dof_ymin)/100):int(len(Grid.dof_ymin)/2+len(Grid.dof_ymin)/100)+1]
BC.C_g  = (C_L*np.ones(len(Grid.dof_ymin)))[int(len(Grid.dof_ymin)/2-len(Grid.dof_ymin)/100):int(len(Grid.dof_ymin)/2+len(Grid.dof_ymin)/100)+1]
'''

BC.dof_dir = np.array([])
BC.dof_f_dir = np.array([])
BC.C_g  = np.array([])

'''

BC.dof_dir   = np.array([Grid.dof_xmax])
BC.dof_f_dir = np.array([Grid.dof_f_xmax])
BC.C_g       = phi_L*(sat_threshold-1e-10)*np.transpose([np.ones_like(Grid.dof_xmax)]) #placeholder
'''
height = qH
dof_inj   = Grid.dof_xmin[np.argwhere(Grid.ymax-Grid.yc <= height)]
dof_f_inj = Grid.dof_f_xmin[np.argwhere(Grid.ymax-Grid.yc <= height)]


BC.dof_neu   = np.ravel(dof_inj)
BC.dof_f_neu = np.ravel(dof_f_inj)
BC.qb        = np.ones_like(dof_inj)
#BC.qb[0:int(len(Grid.dof_xmin)/2),0]      = 0

[B,N,fn] = build_bnd(BC, Grid, I)


phi = phi_L*np.ones((Grid.N,1))


'''
#Marc's permeability field ###########
corr_length = 100; phi_mean = 0; amplitude = 1;s = 25061977

[phi_array,Xc,Yc] = GeneratePermField2D(Grid,corr_length,corr_length,amplitude,phi_mean,'exp',s)

phi_array = 10**(phi_array);
phi = np.reshape(np.transpose(phi_array), (Grid.N,-1))
#phi = phi_L*np.ones((Grid.N,1))

fig = plt.figure(figsize=(15,4) , dpi=100)
plot = [plt.contourf(Xc, Yc, np.transpose(np.log10(K).reshape(Grid.Nx,Grid.Ny)),cmap="coolwarm",levels=100)]
plt.xlabel(r'$x$')
plt.ylabel(r'$z$')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.axis('scaled')
mm = plt.cm.ScalarMappable(cmap=cm.coolwarm)
mm.set_array(np.log10(K))
clb = plt.colorbar(mm, pad=0.1,aspect=10)
clb.set_title(r'$\log_{10}(K) $', y=1.03)
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
plt.savefig(f'../Figures/{simulation_name}_{Grid.Nx}by{Grid.Ny}_Marcs-logK_corr{corr_length}.pdf',bbox_inches='tight', dpi = 600)
'''
############################################

'''
#Adding low porosity
low_phi_dof = np.union1d(np.intersect1d(np.argwhere(Xc_col >=0.5)[:,0],np.argwhere(Xc_col < 0.75 - (1-Yc_col)/3)), \
                         np.intersect1d(np.argwhere(Xc_col < 0.5)[:,0],np.argwhere(Xc_col > 0.25 + (1-Yc_col)/3)))

phi[low_phi_dof,0] = 0.1
'''

C = phi*s_wr

C_sol = C.copy()
flux_sol = np.zeros((Grid.Nf,1))
fs = np.zeros((Grid.N,1))
t    =[0.0]
time = 0
v = np.ones((Grid.Nf,1))
i = 0

#Spinup
#Grid,C,time,time,t_interest,i,tmax,t,C_sol = spin_up_C('paper_seepage-face-q0.08457464_new_flux_C0.3164247822051133_75by75_t5.npz',Grid,tmax+0.01,5,t_interest)

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
        ###############################################################
        #changing the flux BC
        height = len(np.intersect1d(dof_inact,Grid.dof_xmin))*Grid.A[1][0] #calculating the height of saturated region
        qcell = qH/height
        dof_inj   = Grid.dof_xmin[np.argwhere(Grid.ymax-Grid.yc <= height)]
        dof_f_inj = Grid.dof_f_xmin[np.argwhere(Grid.ymax-Grid.yc <= height)]
    
        BC.dof_neu   = np.ravel(dof_inj)
        BC.dof_f_neu = np.ravel(dof_f_inj)
        BC.qb        = qcell*np.ones_like(dof_inj)
        [B,N,fn] = build_bnd(BC, Grid, I)
        
        ###############################################################

        #############################################
        dof_f_saturated = find_faces(dof_inact,D,Grid)       
        ##### 
        #For seepage face
        dof_act_without_inlet= np.setdiff1d(dof_act,BC.dof_neu)
        #dof_act_without_inlet= np.union1d(dof_act,Grid.dof_xmax)
        dof_seepage= np.intersect1d(dof_inact,Grid.dof_xmax)


        #Eliminate inactive cells by putting them to constraint matrix
        BC_P.dof_dir = np.hstack([dof_act_without_inlet,dof_seepage])           
        BC_P.dof_f_dir= np.array([])
        BC_P.g       =  -Yc_col[BC_P.dof_dir-1]*np.cos(tilt_angle*deg2rad) \
                        -Xc_col[BC_P.dof_dir-1]*np.sin(tilt_angle*deg2rad)  

      
        BC_P.dof_neu   = BC.dof_neu.copy()
        BC_P.dof_f_neu = BC.dof_f_neu.copy()
        BC_P.qb        = BC.qb.copy()
        
        [B_P,N_P,fn_P] = build_bnd(BC_P,Grid,I)
        Kd  = comp_harmonicmean(Avg_open,f_Cm(phi,m)) * (Avg_open @ f_Cn(C_old,phi,n))
        
                                                                                                                                                                                                                                                                                              
        Kd  = sp.dia_matrix((Kd[:,0],  np.array([0])), shape=(Grid.Nf, Grid.Nf))
        L = - D @ Kd @ G
        u = solve_lbvp(L,fn_P,B_P,BC_P.g,N_P)   # Non dimensional water potential

        ####### New boundary condition for outflow
        Param.dof_neu   = np.hstack([Grid.dof_xmin, Grid.dof_xmax, Grid.dof_ymin, Grid.dof_ymax])
        Param.dof_f_neu = np.hstack([Grid.dof_f_xmin, Grid.dof_f_xmax, Grid.dof_f_ymin, Grid.dof_f_ymax])        
        Param.dof_dir   = np.array([]) 
        Param.dof_f_dir = np.array([]) 
        q_w = comp_flux(D, Kd, G, u, fs, Grid, Param)
        
         
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
        dt = tmax/(Nt*100000)
    elif time+dt >= t_interest[np.max(np.argwhere(time+dt >= t_interest))] and time < t_interest[np.max(np.argwhere(time+dt >= t_interest))]:
        dt = t_interest[np.max(np.argwhere(time+dt >= t_interest))] - time   #To have the results at a specific time

    RHS = C_old - dt*(res - fn)  #since the gradients are only zero and 1  
    
    C = solve_lbvp(I,RHS,B,BC.C_g,N)
    time = time + dt    
    if np.isin(time,t_interest):
        C_sol = np.concatenate((C_sol,C),axis=1)
        flux_sol = np.concatenate((flux_sol,flux_vert),axis=1)
        t.append(time)
        np.savez(f'{simulation_name}_C{C_L}_{Grid.Nx}by{Grid.Ny}_t{tmax}.npz', t=np.array(t),C_sol =C_sol,phi=phi,Xc=Xc,Yc=Yc,Xc_col=Xc_col,Yc_col=Yc_col,Grid_Nx=Grid.Nx,Grid_Ny=Grid.Ny,Grid_xc=Grid.xc,Grid_yc=Grid.yc,Grid_xf=Grid.xf,Grid_yf=Grid.yf,flux_sol = flux_sol)

        if len(dof_act)< Grid.N:
            print(i,time,'Saturated cells',Grid.N-len(dof_act))        
        else:    
            print(i,time)
    i = i+1

t = np.array(t)


#saving the tensors
np.savez(f'{simulation_name}_porosity.npz', phi = phi, K = K, m = m, Grid_Nx=Grid.Nx,Grid_Ny=Grid.Ny)

'''
#for loading data
data = np.load('paper_seepage-face-q0.2114366_new_flux_C0.3164247822051133_75by75_t5.npz')
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


#Importing results from PbK analytic solution
zeropt1 = np.loadtxt('./Seepage-face-analytic/q0.0846.csv',delimiter=',')
zeropt1_expt = np.loadtxt('./Seepage-face-analytic/q0.0846_Expt.csv',delimiter=',')

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
plt.plot(zeropt1[:,0],Grid.ymax-zeropt1[:,1],'g--')
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
#plt.plot(1.5*analytic[:,0]+Param.xleft_plate,Param.ytop_place-1.5*analytic[:,1],'r--',linewidth=2,label='Analytic')
#plt.plot(1.5*expt[:,0]+Param.xleft_plate,Param.ytop_place-1.5*expt[:,1],'b--',linewidth=2,label='Expt')
#plt.plot(X_plate,Y_plate,'k-',linewidth=2)
#plt.legend(loc='best',frameon=False)
plt.ylabel(r'$z$')
plt.xlabel(r'$x$')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax, Grid.ymin])
plt.axis('scaled')
mm = plt.cm.ScalarMappable(cmap='Blues')
mm.set_array(C/phi)
mm.set_clim(np.min((C/phi)),np.max((C/phi)))
clb = plt.colorbar(mm,aspect=15)
clb.ax.set_title(r'$s_w$',y=1.03)
plt.plot(zeropt1[:,0],Grid.ymax-zeropt1[:,1],'r--',linewidth=3)
#plt.plot(1-zeropt1_expt[:,0]/0.4,Grid.ymax-zeropt1_expt[:,1]/0.4,'k--')
plt.savefig(f'../Figures/{simulation_name}_Marcs_way_Sw_withoutmesh_dashed.pdf',bbox_inches='tight', dpi = 600)


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
#plt.plot(1.5*analytic[:,0]+Param.xleft_plate,Param.ytop_place-1.5*analytic[:,1],'r--',linewidth=2,label='Analytic')
#plt.plot(1.5*expt[:,0]+Param.xleft_plate,Param.ytop_place-1.5*expt[:,1],'b--',linewidth=2,label='Expt')
#plt.plot(X_plate,Y_plate,'k-',linewidth=2)
#plt.legend(loc='best',frameon=False)
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






#Combined figure

Expt_Q = np.array([4.1675e-07, 3.334e-06, 8.335e-06, 4.1675e-07, 1.667e-06,6.668e-06,\
                   4.1675e-07,1.3336e-05,1.5003e-05,1.667e-05,1.8337e-05,2.08375e-05]) #flowrate m^3/s
Expt_h_seep = np.array([0.010, 0.0240, 0.0520, 0.009,0.015,0.039,\
                        0.008,0.062,0.071,0.08,0.081,0.088]) #seepage face height in m
Expt_K_hyd = 0.009700000000000 #hydraulic condm/s
Expt_L   = 0.4 #Length in m
Expt_W   = 0.0254 #Depth in third dimension in m
Expt_q_dimless = Expt_Q/(Expt_K_hyd*Expt_L*Expt_W)
Expt_h_seep    = Expt_h_seep/Expt_L

Expt_Q2mm = np.array([4.1675e-07, 8.335e-07, 1.25025e-06, 1.66700e-06, 2.08375e-06, 2.50050e-06, 3.33400e-06, 4.16750e-06, 5.83450e-06,\
                      5.001e-07, 3.8341e-06, 5.001e-06, 6.668e-06, 8.335e-06, 1.0002e-05, 1.1669e-05, 1.3336e-05,\
                     5.0010e-07,5.8345e-06,1.5003e-05,1.6670e-05,2.0004e-05,2.1671e-05,2.3338e-05]) #flowrate m^3/s
Expt_h_seep2mm = np.array([0.0040,0.0060,0.0070,0.0070,0.0070,0.0080,0.010,0.011,0.014,\
                           0.006,0.009,0.015,0.020,0.022,0.024,0.026,0.029,\
                           0.0050,0.013,0.032,0.034,0.041,0.042,0.050]) #seepage face height in m
Expt_K_hyd2mm = 0.03860
Expt_q_dimless2mm = Expt_Q2mm/(Expt_K_hyd2mm*Expt_L*Expt_W)
Expt_h_seep2mm    = Expt_h_seep2mm/Expt_L


Analy_q    = np.array([0., 0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45, 0.5 ,\
                       0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95, 1.  ])
Analy_seep = np.array([0, 0.03, 0.0742381, 0.1113437, 0.1484415, 0.1854798, 0.222413, 0.2592087, 0.2958039, 0.3321507, 0.3682058,\
                       0.4039311,0.4392946, 0.4742698, 0.5088362, 0.5429783, 0.5766099, 0.6099228, 0.64274, 0.6751104, 0.7070354])

dof_seep  = Grid.dof[C_sol[:,-1] / (phi[:,0]*(1-s_gr)) >0.5]
Yc_col[np.intersect1d(dof_seep,Grid.dof_xmax)-1]
    
Sim_q    = np.array([0.4,0.211,0.0097,0.3,0.0846])

Sim_seep = 1-np.array([0.7,0.83333333,0.985,0.78,0.94])  #50%
  
    
fig = plt.figure(figsize=(10,10) , dpi=75)
plt.ylabel(r'Seepage face height, $H_0/L$')
plt.xlabel(r'Discharge by width, $Q/(KLw)$')
plt.plot(Analy_q,Analy_seep,'r-',linewidth=3)
plt.plot(Sim_q,Sim_seep,'bs',label='Simulations, Present',markersize=20, markerfacecolor='none',markeredgewidth=3)
plt.plot(Analy_q[0:1],Analy_seep[0:1],'r-',label='Analytic, Kochina (2015)',linewidth=3)
plt.axis('scaled')
plt.xlim([0,0.5])
plt.ylim([0,0.5])
plt.legend(loc='upper right',frameon=False)
plt.tight_layout()
plt.savefig(f'../Figures/seepage-face-heightwithout_expt50percent.pdf',bbox_inches='tight', dpi = 600)











