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
from comp_sat_unsat_bnd_flux_fun import comp_sat_unsat_bnd_flux, find_top_bot_cells
from comp_face_coords_fun import comp_face_coords
from find_plate_dofs import find_plate_dofs

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
                3: [[0.39,0.4],[0.0,0.2]]#[[0.4,0.5],[0.0,0.2]], #Contact discontinuity C1  (working)   
             }
        return switcher.get(i,"Invalid Case")

#parameters
case_no =  3              #case number
simulation_name = f'Woods_Hesse_upwinding_hyperbolic_Richards_final_fs5by8'
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

#injection
Param.xleft_inj= 3.5-0.1;  Param.xright_inj= 3.5+0.1

#plate
Param.xleft_plate= 0.5; Param.xright_plate=6.5
Param.ytop_place = 3.0; Param.ybot_plate=  3.3

#temporal
tmax = 14
t_interest = np.linspace(0,tmax,500)   #swr,sgr=0

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
Grid.xmin =  0; Grid.xmax =7; Grid.Nx = 140; 
Grid.ymin =  0; Grid.ymax =4; Grid.Ny = 80;
Grid = build_grid(Grid)
[D,G,I] = build_ops(Grid)
Avg     = comp_algebraic_mean(Grid)

[Xc,Yc] = np.meshgrid(Grid.xc,Grid.yc)                 #building the (x,y) matrix
Xc_col  = np.reshape(np.transpose(Xc), (Grid.N,-1))    #building the single X vector
Yc_col  = np.reshape(np.transpose(Yc), (Grid.N,-1))    #building the single Y vector

#injection
dof_inj   = Grid.dof_ymin[  np.intersect1d(np.argwhere(Grid.xc>= Param.xleft_inj),np.argwhere(Grid.xc <= Param.xright_inj))]
dof_f_inj = Grid.dof_f_ymin[np.intersect1d(np.argwhere(Grid.xc>= Param.xleft_inj),np.argwhere(Grid.xc <= Param.xright_inj))]

###########
#plate 

[dof_plate, dof_out, dof_f_plate_bnd] = find_plate_dofs(Param.xleft_plate, Param.xright_plate, Param.ytop_place, Param.ybot_plate, Grid, Xc_col, Yc_col, D)

#Step 1: Modify the gradient to import natural BC at the plate boundary
G  = zero_rows(G,dof_f_plate_bnd-1)
D  = np.transpose(zero_rows(np.transpose(D),dof_f_plate_bnd-1))


#to evaluate fluxes
#flux: bottom left
[dof_dummy1, dof_dummy2, dof_f_left_half] = find_plate_dofs(Grid.xmin, Param.xleft_plate, Param.ybot_plate, Param.ybot_plate+Grid.dy, Grid, Xc_col, Yc_col, D)
[dof_dummy1, dof_dummy2, dof_f_right_half]= find_plate_dofs(Param.xright_plate,Grid.xmax, Param.ybot_plate, Param.ybot_plate+Grid.dy, Grid, Xc_col, Yc_col, D)

#keeping only y faces
dof_f_left_half  = dof_f_left_half[dof_f_left_half>Grid.Nfx] 
dof_f_right_half = dof_f_right_half[dof_f_right_half>Grid.Nfx] 

##########

#boundary condition

BC.dof_dir   = np.hstack([dof_inj,Grid.dof_ymax])
BC.dof_f_dir = np.hstack([dof_f_inj,Grid.dof_ymax])
BC.C_g       = np.hstack([C_L*np.ones_like(dof_inj),C_L*np.ones_like(Grid.dof_ymax)])

BC.dof_neu   = np.array([])
BC.dof_f_neu = np.array([])
BC.qb = np.array([])

[B,N,fn] = build_bnd(BC, Grid, I)


phi = phi_L*np.ones((Grid.N,1))

C = phi*s_wr

C_sol = C.copy()
flux_sol = np.zeros((Grid.Nf,1))

t    =[0.0]
time = 0
v = np.ones((Grid.Nf,1))

i = 0

while time<tmax:

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
    if len(dof_act)< Grid.N: #when atleast one layer is present
        #############################################
        dof_f_saturated = find_faces(dof_inact,D,Grid)       
        #Step 1: Modify the gradient to import natural BC at the crater
        #G_small = zero_rows(G,dof_f_saturated-1)
        
        #Step 2 Eliminate inactive cells by putting them to constraint matrix
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
        
        #upwinding boundary y-directional flux
        #finding boundary faces
        dof_ysat_faces = dof_f_saturated[dof_f_saturated>=Grid.Nfx]
        
        #removing boundary faces
        dof_ysat_faces = np.setdiff1d(dof_ysat_faces,np.append(Grid.dof_f_ymin,Grid.dof_f_ymax))
        
        ytop,ybot               = find_top_bot_cells(dof_ysat_faces,D,Grid)
        q_w[dof_ysat_faces-1,0] = comp_sat_unsat_bnd_flux(q_w[dof_ysat_faces-1],flux_vert[dof_ysat_faces-1],ytop,ybot,C,phi,sat_threshold)
        
        #find all saturated faces
        dof_sat_faces = find_all_faces(dof_inact,D,Grid)  
        
        flux_vert[dof_sat_faces-1] = q_w[dof_sat_faces-1]
        
        res = D @ flux_vert
        

    #dt  = np.abs((phi - C_old)/(D@flux)) #Calculating the time step from the filling of volume
    #dt   = CFL*np.abs((phi - phi*s_gr - C_old)/(D@flux)) #Calculating the time step from the filling of volume
    dt   = CFL*np.abs((phi - phi*s_gr - C_old)/(res)) #Calculating the time step from the filling of volume
    
    dt[dt<1e-6] = 0.0
    dt  =  np.min(dt[dt>0])
    if i<1000: 
        dt = tmax/(Nt*1000)
    elif time+dt >= t_interest[np.max(np.argwhere(time+dt >= t_interest))] and time < t_interest[np.max(np.argwhere(time+dt >= t_interest))]:
        dt = t_interest[np.max(np.argwhere(time+dt >= t_interest))] - time   #To have the results at a specific time
    
    #RHS = C_old - dt*D@flux  #since the gradients are only zero and 1    
    RHS = C_old - dt*res  #since the gradients are only zero and 1  
    
    C = solve_lbvp(I,RHS,B,BC.C_g,N)
    #C[C<0] = 0
    #C = np.transpose([solve_lbvp(I,RHS,B,BC.C_g,N)])
    time = time + dt    
    #C[C>=phi] = phi[C>=phi]       #constraining the composition
    #C[dof_plate,:] = np.nan
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


#Woords and Hesse (2010) Validation
Qa = np.sum(flux_vert[dof_f_left_half-1])  #summing the flux on the left side
Qb = np.sum(flux_vert[dof_f_right_half-1]) #summing the flux on the right side
La =-Param.xleft_plate  + (Param.xleft_inj+Param.xright_inj)/2
Lb = Param.xright_plate - (Param.xleft_inj+Param.xright_inj)/2


#saving the tensors
np.savez(f'{simulation_name}_C{C_L}_{Grid.Nx}by{Grid.Ny}_t{tmax}.npz', t=t,C_sol =C_sol,phi=phi,Xc=Xc,Yc=Yc,Xc_col=Xc_col,Yc_col=Yc_col,Grid_Nx=Grid.Nx,Grid_Ny=Grid.Ny,Grid_xc=Grid.xc,Grid_yc=Grid.yc,Grid_xf=Grid.xf,Grid_yf=Grid.yf,Qa=Qa,Qb=Qb,La=La,Lb=Lb,flux_sol = flux_sol)

print(f'The Hesse and Woods validation for La={La},Lb={Lb} is: \n fb=La/(La+Lb)={La/(La+Lb)} \n fb=Qa/(Qa+Qb)={Qa/(Qa+Qb)}')

'''
#for loading data
data = np.load('Final_Woods_Hesse_upwinding_hyperbolic_Richards_final_fs4by8_C0.39_140by80_t14.npz')
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
Qa     =data['Qa']
Qb     =data['Qb']
La     =data['La']
Lb     =data['Lb']
flux_sol = data['flux_sol']
[dummy,endstop] = np.shape(C_sol)
C[:,0] = C_sol[:,-1]
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
ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(C_sol[:,:], plot[:],t[:]), interval=1/fps)
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
ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(C_sol[:,:], plot[:0],t[:]), interval=1/fps)

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
ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(C_sol[:,:], plot[:],t[:]), interval=1/fps)

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
[X_plate,Y_plate] = comp_face_coords(dof_f_plate_bnd,Grid)

fig = plt.figure(figsize=(15,7.5) , dpi=100)
plot = [plt.contourf(Xc, Yc, np.transpose((C/phi).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = -0.00001, vmax = 1.00001)]
plt.plot(X_sat,Y_sat,'r-',linewidth=0.4)
plt.plot(X_bnd,Y_bnd,'r-',linewidth=2)
plt.plot(X_all,Y_all,'k-',linewidth=0.4)
plt.plot(X_plate,Y_plate,'k-',linewidth=2)
plt.colorbar()
plt.ylabel(r'$z$')
plt.xlabel(r'$x$')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.axis('scaled')
plt.plot(Xc_col[np.argwhere(C_sol[:,-1]<0)[:,0]],Yc_col[np.argwhere(C_sol[:,-1]<0)[:,0]],'ro',label='Sw>1')
plt.plot(Xc_col[np.argwhere(C_sol[:,-1]>(phi[:,0]+1e-9))[:,0]],Yc_col[np.argwhere(C_sol[:,-1]>(phi[:,0]+1e-9))[:,0]],'gX',label='Sw>1')
plt.savefig(f'../Figures/{simulation_name}_Marcs_way_Sw.pdf',bbox_inches='tight', dpi = 600)


fig = plt.figure(figsize=(15,7.5) , dpi=100)
plot = [plt.contourf(Xc, Yc, np.transpose((C/phi).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin = -0.00001, vmax = 1.00001,levels=1000)]
plt.plot(X_plate,Y_plate,'k-',linewidth=2)
plt.colorbar()
plt.ylabel(r'$z$')
plt.xlabel(r'$x$')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax, Grid.ymin])
plt.axis('scaled')
plt.savefig(f'../Figures/{simulation_name}_Marcs_way_Sw_withoutmesh.pdf',bbox_inches='tight', dpi = 600)


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
#Streamlines with head

#streamlines
from comp_streamfun import comp_streamfun
fig = plt.figure(figsize=(15,15) , dpi=100)
Xp,Yp = np.meshgrid(Grid.xf,Grid.yf)

#q_w[Grid.dof_f_ymax[0]-1,0] = 0
#q_w[Grid.dof_f_ymax[-1]-1,0] = 0

fig = plt.figure(figsize=(12,10) , dpi=100)
plot = [plt.contourf(Xc, Yc, np.transpose((C/phi).reshape(Grid.Nx,Grid.Ny)),cmap="Blues",vmin=0,vmax=1)]
mm = plt.cm.ScalarMappable(cmap='Blues')
mm.set_array(C_sol[:,-1]/phi[:,0])
mm.set_clim(0,1)
clb = plt.colorbar(mm,aspect=15)
clb.ax.set_title(r'$s_w$',y=1.03)
plt.ylabel(r'$z$')
plt.xlabel(r'$x$')
plt.xlim([Grid.xmin, Grid.xmax])
plt.ylim([Grid.ymax,Grid.ymin])
plt.axis('scaled')
PSI, psi_min, psi_max = comp_streamfun(np.transpose([flux_sol[:,-100]]),Grid)
cutoff_min = -0.05#psi_min + 0.4*np.abs(psi_min)
cutoff_max = 0.07#psi_max - 0.4*np.abs(psi_max)
#PSI[PSI<cutoff_min] = np.nan
#PSI[PSI>cutoff_max] = np.nan
plt.contour(Xp, Yp, PSI, 20,colors='w')
plt.savefig(f'../Figures/{simulation_name}_Marcs_withstreamlines.pdf',bbox_inches='tight', dpi = 600)

'''

