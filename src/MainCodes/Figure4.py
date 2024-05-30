#Coding the Richard's equation for water multiple saturated region
#Mohammad Afzal Shadab
#Date modified: 01/15/22

import sys
sys.path.insert(1, '../../solver')

# import personal libraries and class
from classes import *    

from two_components_aux import *
from complex_domain import find_faces, find_bnd_cells
from comp_fluxfun import comp_flux
from scipy.integrate import solve_ivp

plt.rcParams.update({'font.family': "Serif"})

brown  = [181/255 , 101/255, 29/255]
red    = [255/255 ,255/255 ,255/255 ]
blue   = [ 30/255 ,144/255 , 255/255 ]
green  = [  0/255 , 166/255 ,  81/255]
orange = [247/255 , 148/255 ,  30/255]
purple = [102/255 ,  45/255 , 145/255]
brown  = [155/255 ,  118/255 ,  83/255]
tan    = [199/255 , 178/255 , 153/255]
gray   = [100/255 , 100/255 , 100/255]

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

def analytical(case_no,eta_L,eta_R,phi_L,C_L,phi_R,C_R, m, n):
        print('Special case: Backward Shock')
        eta  = np.linspace(eta_L,eta_R,20000)
        if case_no ==1:
            Sw = C_L/phi_L*np.ones_like(eta)             
            s = (f_Cn(np.array([C_L]),phi_L,n)*f_Cm(phi_L,m)/phi_L-f_Cn(np.array([C_R]),phi_R,n)*f_Cm(phi_R,m)/phi_R)[0]/(C_L/phi_L - C_R/phi_R)#Shock speed
            print(s)
            Sw[eta>=s] = C_R/phi_R
        
        else: #Not plotting
            Sw = np.nan*np.ones_like(eta)
        return eta,Sw
    

def case(i):
        switcher={
                #Separate regions
                #Region 2: Three phase region
                3: [[0.4,0.5],[0.0,0.2]]#[[0.4,0.5],[0.0,0.2]], #Contact discontinuity C1  (working)   
             }
        return switcher.get(i,"Invalid Case")


#parameters
case_no =  3              #case number
simulation_name = f'case{case_no}_hyperbolic_Richards_'
m = 3 #Cozeny-Karman coefficient for numerator K = K0 (1-phi_i)^m
n = 2 #Corey-Brooks coefficient krw = krw0 * sw^n
s_wr = 0.0 #Residual water saturation
s_gr = 0.0 #Residual gas saturation


#test details
[u_L,u_R] = case(case_no) #left and right states
[C_L,phi_L] = u_L #water sat, porosity
[C_R,phi_R] = u_R #water sat, porosity
C_R = s_gr*phi_R #resetting the saturation
xm = 1.0 #the location of jump

#temporal
tmax = 1  #time scaling with respect to fc
t_interest = [0,0.3,0.6249999999999999 +0.005,0.7,0.871335520204737,1.0]   #swr,sgr=0


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
    #fC[C>=phi]  = 0.0      
    return fC

#spatial
Grid.xmin = 0; Grid.xmax =2.0; Grid.Nx = 400; Grid.Ny=1
Grid = build_grid(Grid)
[D,G,I] = build_ops(Grid)
Avg     = Grid.dx/2*abs(G)

#boundary condition
BC.dof_dir = np.array([Grid.dof_xmin,Grid.dof_xmax])
BC.dof_f_dir = np.array([Grid.dof_f_xmin, Grid.dof_f_xmax])
BC.C_g  = np.array([C_L,C_R])
'''
BC.dof_dir = np.array([])
BC.dof_f_dir = np.array([])
BC.C_g  = np.array([])
'''

BC.dof_neu   = np.array([])
BC.dof_f_neu = np.array([])
BC.qb = np.array([])

[B,N,fn] = build_bnd(BC, Grid, I)

phi = phi_L*np.ones((Grid.N,1))
phi[Grid.xc>xm,:] = phi_R  

C = phi*s_wr

C_sol = C.copy()

t   = [0.0]
z_l = [0.0]
z_r = [0.0]
s_l = []
s_r = []
qs_sol = []

v = np.ones((Grid.Nx+1,1)) #flow is always left to right

i = 0

while t[i]<tmax:
    if C[1]>=phi_L:
        BC.C_g[0] = C[1]
    C_old = C.copy() 
    flux = (comp_harmonicmean(Avg,f_Cm(phi,m))*(flux_upwind(v, Grid)@f_Cn(C_old,phi,n)))

    ######
    #Taking out the domain to cut off single phase region
    dof_inact  = Grid.dof[C_old[:,0] / (phi[:,0]*(1-s_gr)) > 0.999999]
    dof_act    = np.setdiff1d(Grid.dof,dof_inact)
    dof_f_saturated = find_faces(dof_inact,D,Grid)
    [dof_bnd_inact,dof_bnd_act,dof_bnd] = find_bnd_cells(dof_act,dof_inact,dof_f_saturated,D,Grid)
    
    if len(dof_inact)>1: #if more than 2 cells are non-zero
        #############
        #Solving the Laplace equation for fully saturated region
        dof_inact = np.array([i for i in range(dof_inact[0],dof_inact[len(dof_inact)-1]+1)])
        dof_act   = np.setdiff1d(Grid.dof,dof_inact)
        dof_f_saturated = find_faces(dof_inact,D,Grid)
        [dof_bnd_inact,dof_bnd_act,dof_bnd] = find_bnd_cells(dof_act,dof_inact,dof_f_saturated,D,Grid)
        
        Grid_P.xmin = Grid.xc[dof_inact[0]-1]; Grid_P.xmax=Grid.xc[dof_inact[len(dof_inact)-1]-1]; Grid_P.Nx = len(dof_inact); Grid_P.Ny=1
        Grid_P = build_grid(Grid_P)
        [D_sat,G_sat,I_sat] = build_ops(Grid_P)
        
        BC_P.dof_dir = np.array([Grid_P.dof_xmin,Grid_P.dof_xmax])
        BC_P.dof_f_dir = np.array([Grid_P.dof_f_xmin, Grid_P.dof_f_xmax])
        BC_P.g = -np.array([Grid_P.xc[0],Grid_P.xc[len(Grid_P.xc)-1]])  #because pressure is just zero
        
        BC_P.dof_neu = np.array([])
        BC_P.dof_f_neu = np.array([])
        BC_P.qb = np.array([])
        [B_P,N_P,fn_P] = build_bnd(BC_P,Grid_P,I_sat)
        
        dof_f_inact = np.array([i for i in range(dof_f_saturated[0],dof_f_saturated[len(dof_f_saturated)-1]+1)])
        flux_sat = sp.dia_matrix((flux[dof_f_inact-1,0],  np.array([0])), shape=(len(dof_f_inact), len(dof_f_inact)))
        
        L = - D_sat @ flux_sat @ G_sat
        u = solve_lbvp(L,fn_P,B_P,BC_P.g,N_P)   # Non dimensional water potential PHI/(rho * g * L)
        
        #q_w = comp_flux(D_sat,flux_sat,G_sat,u,np.zeros((len(u),1)),Grid_P,BC_P)
        print(i,'Calculated',np.max(-flux_sat @ G_sat @ u))
        flux[dof_f_inact-1,0] = np.max(-flux_sat @ G_sat @ u)#q_w[:,0]
        #print('saturated flux:',i,len(dof_inact), 'cells',dof_f_saturated-1,flux[dof_f_saturated-1,:])
        #############
        
    else:
        flux[dof_f_saturated-1,:] = 0
        #print(i,'Single saturated flux:',dof_f_saturated-1,flux[dof_f_saturated-1,:])
    
    dt   = CFL*np.abs((phi - phi*s_gr - C_old)/(D@flux)) #Calculating the time step from the filling of volume
    
    dt[dt<1e-14] = 0.0
    dt  =  np.min(dt[dt>0])
    if i<10: 
        dt = tmax/Nt
    elif t[i]+dt >= t_interest[np.max(np.argwhere(t[i]+dt >= t_interest))] and t[i] < t_interest[np.max(np.argwhere(t[i]+dt >= t_interest))]:
        dt = t_interest[np.max(np.argwhere(t[i]+dt >= t_interest))] - t[i]   #To have the results at a specific time
    
    RHS = C_old - dt*D@flux  #since the gradients are only zero and 1    
    C = solve_lbvp(I,RHS,B,BC.C_g,N)
    
    q_s  =  np.abs((z_l[i] - z_r[i])/(z_l[i]/f_Cm(np.array([phi_L]),m)[0] - z_r[i]/f_Cm(np.array([phi_R]),m)[0]+1e-32)) 
    
    dz_l =  dt*(q_s - flux[1,0] )/(phi_L*(1-C_L/phi_L))
    dz_r =  dt*(q_s - flux[-2,0])/(phi_R*(1-C_R/phi_R))

    s_l_dummy =  (q_s - flux[1,0] )/(phi_L*(1-C_L/phi_L))
    s_r_dummy =  (q_s - flux[-2,0])/(phi_R*(1-C_R/phi_R))
    
    print(i,t[i],s_l_dummy,s_r_dummy )
    
    C_sol = np.concatenate((C_sol,C),axis=1)
    t.append(t[i] + dt)
    z_l.append(z_l[i] + dz_l)
    z_r.append(z_r[i] + dz_r)
    s_l.append(s_l_dummy)
    s_r.append(s_r_dummy)
    qs_sol.append(q_s)
    i = i+1

t = np.array(t)

[eta_analy,Sw_analy] = analytical(case_no,Grid.xc[0]/t[len(t)-1],Grid.xc[len(Grid.xc)-1]/t[len(t)-1],phi_L,C_L,phi_R,C_R, m, n)

###############################################
light_red  = [1.0,0.5,0.5]
light_blue = [0.5,0.5,1.0]
light_black= [0.5,0.5,0.5]

#C vs x/t
fig = plt.figure(figsize=(8,8) , dpi=100)
plot = [plt.plot(Grid.xc/t[len(t)-1],C_sol[:,len(t)-1],'b--',label=r'$\mathcal{C}$')]
#plot = [plt.plot(Grid.xc/t[int((len(t)-1)/2)],C_sol[:,int((len(t)-1)/2)],'b--',label=r'$\mathcal{C}$')]
if np.isnan(Sw_analy[1]) == False:
    plot = [plt.plot(eta_analy,Sw_analy,'-',c=light_red,label=r'$S_w$ (Analytical)')]
plot = [plt.plot(Grid.xc/t[len(t)-1],C_sol[:,len(t)-1]/phi[:,0],'r--',label=r'$S_w$')]

plt.xlabel(r'$\eta = \zeta/\tau$')
plt.xlim([Grid.xmin/t[len(t)-1], Grid.xmax/t[len(t)-1]])
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.legend(loc='upper right')
plt.savefig(f"../Figures/{simulation_name}_Nx{Grid.Nx}_CL{C_L}CR{C_R}__tf{t[len(t)-1]}.pdf")

################


#####RUN FROM HERE FOR ANALYTICAL


# %% Define derivative function

def rhs(t, y): 
    return [((y[0]-y[1])/(y[0]/f_Cm(np.array([phi_L]),m)-y[1]/f_Cm(np.array([phi_R]),m))-f_Cm(np.array([phi_L]),m)*f_Cn(np.array([C_L]),np.array([phi_L]),n))[0]/(phi_L*(1-s_gr)-C_L), \
            ((y[0]-y[1])/(y[0]/f_Cm(np.array([phi_L]),m)-y[1]/f_Cm(np.array([phi_R]),m))-f_Cm(np.array([phi_R]),m)*f_Cn(np.array([C_R]),np.array([phi_R]),n))[0]/(phi_R*(1-s_gr)-C_R)]
res = solve_ivp(rhs, (0, t[-1]), [-1e-14,1e-15],t_eval=t)
y   = res.y
qs_int = (y[0]-y[1])/(y[0]/f_Cm(np.array([phi_L]),m)-y[1]/f_Cm(np.array([phi_R]),m))
s_l_int = (qs_int -f_Cm(np.array([phi_L]),m)*f_Cn(np.array([C_L]),np.array([phi_L]),n))/(phi_L*(1-s_gr)-C_L)
s_r_int = (qs_int -f_Cm(np.array([phi_R]),m)*f_Cn(np.array([C_R]),np.array([phi_R]),n))/(phi_R*(1-s_gr)-C_R)

##############

'''
A = (phi_L - C_L)
B = (phi_R - C_R)
C = (phi_L**m*(C_L/phi_L)**n)
D = (phi_R**m*(C_R/phi_R)**n)

aa = A - A*D/(phi_L**m)
bb = -A-B+A*D/(phi_R**m)+B*C/(phi_L**m)
cc = B - B*C/phi_R**m
'''

aa = phi_L/phi_R*(1-C_L/phi_L-s_gr)/(1-C_R/phi_R-s_gr)*(1- f_Cm(np.array([phi_R]),m) * f_Cn(np.array([C_R]),np.array([phi_R]),n) / f_Cm(np.array([phi_L]),m))
bb =-phi_L/phi_R*(1-C_L/phi_L-s_gr)/(1-C_R/phi_R-s_gr)*(1- f_Cm(np.array([phi_R]),m) * f_Cn(np.array([C_R]),np.array([phi_R]),n) / f_Cm(np.array([phi_R]),m)) - (1 - f_Cm(np.array([phi_L]),m) * f_Cn(np.array([C_L]),np.array([phi_L]),n) / f_Cm(np.array([phi_L]),m))
cc = 1 - f_Cm(np.array([phi_L]),m) * f_Cn(np.array([C_L]),np.array([phi_L]),n) / f_Cm(np.array([phi_R]),m)

k = (-bb-np.sqrt(bb**2-4*aa*cc))/(2*aa)

def qs(x):
    return (x-1)/(x/f_Cm(np.array([phi_L]),m) - 1/f_Cm(np.array([phi_R]),m))

s_l_analy =(qs(k) - f_Cm(np.array([phi_L]),m)*f_Cn(np.array([C_L]),np.array([phi_L]),n))[0]/(phi_L*(1-s_gr-C_L/phi_L))
s_r_analy =(qs(k) - f_Cm(np.array([phi_R]),m)*f_Cn(np.array([C_R]),np.array([phi_R]),n))[0]/(phi_R*(1-C_R/phi_R-s_gr))
#print(k,q_w[0,0],qs(k),qs_int[-1],s_l_analy,s_l_int[-1],s_r_analy,s_r_int[-1])

# First set up the figure, the axis
fig = plt.figure(figsize=(15,7.5) , dpi=100)
ax1 = fig.add_subplot(1, 6, 1)
ax2 = fig.add_subplot(1, 6, 2)
ax3 = fig.add_subplot(1, 6, 3)
ax4 = fig.add_subplot(1, 6, 4)
ax5 = fig.add_subplot(1, 6, 5)
ax6 = fig.add_subplot(1, 6, 6)

#ax1.set_xlim([T_top-dt_a-dt_d-273.16,T_top+dt_a+dt_d-273.16])
#ax1.set_ylim(70, grid.ymax)
#ax1.set_ylim([grid.ymin,grid.ymax])
ax1.set_ylabel(r'Dimensionless depth $z/z_0$')
ax1.set_ylim([Grid.xmax,Grid.xmin])
ax1.set_xlim([0,1])

ax2.set_xlim([0,1])
ax2.set_ylim([Grid.xmax,Grid.xmin])
ax2.axes.yaxis.set_visible(False)
ax4.set_xlabel(r'Volume fractions $\phi$')


ax3.set_xlim([0,1])
ax3.axes.yaxis.set_visible(False)
ax3.set_ylim([Grid.xmax,Grid.xmin])


ax4.set_xlim([0,1])
ax4.axes.yaxis.set_visible(False)
ax4.set_ylim([Grid.xmax,Grid.xmin])

ax5.set_xlim([0,1])
ax5.axes.yaxis.set_visible(False)
ax5.set_ylim([Grid.xmax,Grid.xmin])

ax6.set_xlim([0,1])
ax6.axes.yaxis.set_visible(False)
ax6.set_ylim([Grid.xmax,Grid.xmin])

manager = plt.get_current_fig_manager()
manager.window.showMaximized()

#tf = (xm - Grid.xmin)*(C_L)/(phi_L**m*(C_L/phi_L)**n)
Sf = (f_Cm(np.array([phi_L]),m)[0]*f_Cn(np.array([C_L]),np.array([phi_L]),n))[0] / (phi_L*(C_L/phi_L - s_wr))
tf = (xm - Grid.xmin)/Sf
tp = tf + (Grid.xmin-xm)/s_l_analy

xf = Grid.xmin + Sf*t_interest[0]#(phi_L**m*(C_L/phi_L)**n)/(C_L)*0   #at t = 0
S_w_analy_int = (1-s_gr)*np.ones((Grid.Nx,1))
S_w_analy_int[Grid.xc<=xf] = C_L/phi_L
S_w_analy_int[Grid.xc>=xf] = C_R/phi_R

S_w_analy_int_combined = []
S_w_analy_int_combined = S_w_analy_int.copy()

ax1.fill_betweenx(Grid.xc,1, facecolor=red,label=r'$\phi_g$')
ax1.fill_betweenx(Grid.xc,(1-phi+phi*S_w_analy_int)[:,0], facecolor=blue,label=r'$\phi_w$')
ax1.fill_betweenx(Grid.xc,(1-phi)[:,0], facecolor=brown,label=r'$\phi_s$')

xf = Grid.xmin + Sf*t_interest[1]
S_w_analy_int = (1-s_gr)*np.ones((Grid.Nx,1))
S_w_analy_int[Grid.xc<=xf] = C_L/phi_L
S_w_analy_int[Grid.xc>=xf] = C_R/phi_R

S_w_analy_int_combined = np.hstack([S_w_analy_int_combined,S_w_analy_int])

ax2.fill_betweenx(Grid.xc,1, facecolor=red,label=r'$\phi_g$')
ax2.fill_betweenx(Grid.xc,(1-phi+phi*S_w_analy_int)[:,0], facecolor=blue,label=r'$\phi_w$')
ax2.fill_betweenx(Grid.xc,(1-phi)[:,0], facecolor=brown,label=r'$\phi_s$')


res = solve_ivp(rhs, (0, t_interest[2] - tf), [-1e-14,1e-15],t_eval=[t_interest[2]-tf])
S_w_analy_int = (1-s_gr)*np.ones((Grid.Nx,1))
S_w_analy_int[Grid.xc<=xm+res.y[0,0]] = C_L/phi_L
S_w_analy_int[Grid.xc>=xm+res.y[1,0]] = C_R/phi_R

S_w_analy_int_combined = np.hstack([S_w_analy_int_combined,S_w_analy_int])

ax3.fill_betweenx(Grid.xc,1, facecolor=red,label=r'$\phi_{gas}$')
ax3.fill_betweenx(Grid.xc,(1-phi+phi*S_w_analy_int)[:,0], facecolor=blue,label=r'$\phi_{water}$')
ax3.fill_betweenx(Grid.xc,(1-phi)[:,0], facecolor=brown,label=r'$\phi_{soil}$')

res = solve_ivp(rhs, (0, t_interest[3]-tf), [-1e-14,1e-15],t_eval=[t_interest[3]-tf])
S_w_analy_int = (1-s_gr)*np.ones((Grid.Nx,1))
S_w_analy_int[Grid.xc<=xm+res.y[0,0]] = C_L/phi_L
S_w_analy_int[Grid.xc>=xm+res.y[1,0]] = C_R/phi_R

S_w_analy_int_combined = np.hstack([S_w_analy_int_combined,S_w_analy_int])

ax4.fill_betweenx(Grid.xc,1, facecolor=red,label=r'$\phi_g$')
ax4.fill_betweenx(Grid.xc,(1-phi+phi*S_w_analy_int)[:,0], facecolor=blue,label=r'$\phi_w$')
ax4.fill_betweenx(Grid.xc,(1-phi)[:,0], facecolor=brown,label=r'$\phi_s$')

res = solve_ivp(rhs, (0, tp-tf), [-1e-14,1e-15],t_eval=[tp-tf])
S_w_analy_int = (1-s_gr)*np.ones((Grid.Nx,1))
S_w_analy_int[Grid.xc<=xm+res.y[0,0]] = C_L/phi_L
S_w_analy_int[Grid.xc>=xm+res.y[1,0]] = C_R/phi_R

S_w_analy_int_combined = np.hstack([S_w_analy_int_combined,S_w_analy_int])

ax5.fill_betweenx(Grid.xc,1, facecolor=red,label=r'$\phi_{gas}$')
ax5.fill_betweenx(Grid.xc,(1-phi+phi*S_w_analy_int)[:,0], facecolor=blue,label=r'$\phi_{water}$')
ax5.fill_betweenx(Grid.xc,(1-phi)[:,0], facecolor=brown,label=r'$\phi_{soil}$')

ytop = res.y[0,0]
ybot = res.y[1,0]

def rhs(t, y): 
    return [0, \
            ((y[0]-y[1])/(y[0]/f_Cm(np.array([phi_L]),m)-y[1]/f_Cm(np.array([phi_R]),m))-f_Cm(np.array([phi_R]),m)*f_Cn(np.array([C_R]),np.array([phi_R]),n))[0]/(phi_R*(1-s_gr)-C_R)]

res = solve_ivp(rhs, (0, t_interest[5]-tp), [ytop,ybot],t_eval=[t_interest[5]-tp])
S_w_analy_int = (1-s_gr)*np.ones((Grid.Nx,1))
S_w_analy_int[Grid.xc<=xm+res.y[0,0]] = C_L/phi_L
S_w_analy_int[Grid.xc>=xm+res.y[1,0]] = C_R/phi_R

S_w_analy_int_combined = np.hstack([S_w_analy_int_combined,S_w_analy_int])

ax6.fill_betweenx(Grid.xc,1, facecolor=red,label=r'$\phi_{gas}$')
ax6.fill_betweenx(Grid.xc,(1-phi+phi*S_w_analy_int)[:,0], facecolor=blue,label=r'$\phi_{water}$')
ax6.fill_betweenx(Grid.xc,(1-phi)[:,0], facecolor=brown,label=r'$\phi_{soil}$')

ax1.legend(loc='lower left', shadow=False, fontsize='medium')

ax1.set_title(r'$t^*$=%.2f'%t_interest[0])
ax2.set_title(r'%.2f'%t_interest[1])
ax3.set_title(r'%.2f ($t_s$)'%t_interest[2])
ax4.set_title(r'%.2f'%t_interest[3])
ax5.set_title(r'%.2f ($t_p$)'%tp)
ax6.set_title(r'%.2f'%t_interest[5])
plt.subplots_adjust(wspace=0.25, hspace=0)
plt.savefig(f"../Figures/shock_Nx{Grid.Nx}_CL{C_L}CR{C_R}_phiL{phi_L}phiR{phi_R}m{m}_n{n}.pdf")


plt.figure(figsize=(8,8) , dpi=100)
for i in range(0,len(t_interest)):
    plt.plot(C_sol[:,(np.argwhere(t==t_interest[i]))[0,0]]/phi[:,0],Grid.xc, linestyle='-',label=r'$t^\'=$%.2f'%t_interest[i])
    plt.plot(S_w_analy_int_combined[:,i],Grid.xc , c = 'k',linestyle='--')
plt.ylabel(r'$z/z_0$')
plt.xlabel(r'$s_w$')
plt.xlim([-0.1,1.1])
plt.ylim([Grid.xmax,Grid.xmin+0.01])
plt.legend(loc='best', shadow=False, fontsize='medium')
#plt.ylim([0.99*(np.min(qs2)//f_Cm(np.array([phi_L]),m)),1.05*(qs(k)/f_Cm(np.array([phi_L]),m))])
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig(f"../Figures/swvsZatT_shock_{simulation_name}_Nx{Grid.Nx}_CL{C_L}CR{C_R}_phiL{phi_L}phiR{phi_R}__tf{t[len(t)-1]}.pdf")


# First set up the figure, the axis

fig = plt.figure(figsize=(15,7.5) , dpi=100)
ax1 = fig.add_subplot(1, 6, 1)
ax2 = fig.add_subplot(1, 6, 2)
ax3 = fig.add_subplot(1, 6, 3)
ax4 = fig.add_subplot(1, 6, 4)
ax5 = fig.add_subplot(1, 6, 5)
ax6 = fig.add_subplot(1, 6, 6)

#ax1.set_xlim([T_top-dt_a-dt_d-273.16,T_top+dt_a+dt_d-273.16])
#ax1.set_ylim(70, grid.ymax)
#ax1.set_ylim([grid.ymin,grid.ymax])
ax1.set_ylabel(r'Dimensionless depth $z/z_0$')
ax1.set_ylim([Grid.xmax,0])
ax1.set_xlim([-0.05,1.05])

ax2.set_xlim([-0.05,1.05])
ax2.set_ylim([Grid.xmax,0])
ax2.axes.yaxis.set_visible(False)
ax4.set_xlabel(r'Water saturation $s_w$')

ax3.set_xlim([-0.05,1.05])
ax3.axes.yaxis.set_visible(False)
ax3.set_ylim([Grid.xmax,0])
#ax3.set_ylim([grid.ymin,grid.ymax])

ax4.set_xlim([-0.05,1.05])
ax4.axes.yaxis.set_visible(False)
ax4.set_ylim([Grid.xmax,0])

ax5.set_xlim([-0.05,1.05])
ax5.axes.yaxis.set_visible(False)
ax5.set_ylim([Grid.xmax,0])

ax6.set_xlim([-0.05,1.05])
ax6.axes.yaxis.set_visible(False)
ax6.set_ylim([Grid.xmax,0])

manager = plt.get_current_fig_manager()
manager.window.showMaximized()

ax1.plot(C_sol[:,(np.argwhere(t==t_interest[0]))[0,0]]/phi[:,0],Grid.xc,'r-',label=r'N')
ax1.plot(S_w_analy_int_combined[:,0],Grid.xc , c = 'k',linestyle='--',label=r'A')
ax1.legend(loc='lower left', shadow=False, fontsize='medium')

ax2.plot(C_sol[:,(np.argwhere(t==t_interest[1]))[0,0]]/phi[:,0],Grid.xc, 'r-',label=r'Analytical')
ax2.plot(S_w_analy_int_combined[:,1],Grid.xc , c = 'k',linestyle='--',label=r'Numerical')

ax3.plot(C_sol[:,(np.argwhere(t==t_interest[2]))[0,0]]/phi[:,0],Grid.xc, 'r-',label=r'Analytical')
ax3.plot(S_w_analy_int_combined[:,2],Grid.xc , c = 'k',linestyle='--',label=r'Numerical')

ax4.plot(C_sol[:,(np.argwhere(t==t_interest[3]))[0,0]]/phi[:,0],Grid.xc, 'r-',label=r'Analytical')
ax4.plot(S_w_analy_int_combined[:,3],Grid.xc , c = 'k',linestyle='--',label=r'Numerical')

ax5.plot(C_sol[:,(np.argwhere(t==t_interest[4]))[0,0]]/phi[:,0],Grid.xc, 'r-',label=r'Analytical')
ax5.plot(S_w_analy_int_combined[:,4],Grid.xc , c = 'k',linestyle='--',label=r'Numerical')

ax6.plot(C_sol[:,(np.argwhere(t==t_interest[5]))[0,0]]/phi[:,0],Grid.xc,'r-',label=r'Analytical')
ax6.plot(S_w_analy_int_combined[:,5],Grid.xc , c = 'k',linestyle='--',label=r'Numerical')

ax1.set_title(r'$t^*$=%.2f'%t_interest[0])
ax2.set_title(r'%.2f'%t_interest[1])
ax3.set_title(r'%.2f ($t_s$)'%t_interest[2])
ax4.set_title(r'%.2f'%t_interest[3])
ax5.set_title(r'%.2f ($t_p$)'%tp)
ax6.set_title(r'%.2f'%t_interest[5])
plt.subplots_adjust(wspace=0.25, hspace=0)
plt.savefig(f"../Figures/swvsZpanelshock_Nx{Grid.Nx}_CL{C_L}CR{C_R}_phiL{phi_L}phiR{phi_R}m{m}_n{n}.pdf")


'''
#New Contour plot video
print('Saving animation')
fps = 10000 # frame per sec
#frn = endstop # frame number of the animation
[N,frn] = np.shape(C_sol) # frame number of the animation from the saved file

def update_plot(frame_number, C_sol, plot,t):  
    
    fig.clear() 
    plot[0] = plt.fill_betweenx(Grid.xc,1, facecolor='white',label=r'$\phi_{gas}$')
    plot[0] = plt.fill_betweenx(Grid.xc, 1-phi[:,0] + C_sol[:,frame_number], facecolor=blue,label=r'$\phi_{water}$')
    plot[0] = plt.fill_betweenx(Grid.xc, 1 - phi[:,0], facecolor=brown,label=r'$\phi_{soil}$')
    
    #plt.axis('scaled')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    #plt.clim(250,270)
    plt.ylabel(r'Dimensionless depth $z/z_0$')
    plt.xlabel(r'Volume fraction $\phi$')
    plt.ylim([Grid.xmax,Grid.xmin])
    plt.xlim([0,1])
    #plt.axis('scaled')
    #plt.clim(250,270)
    plt.title(r"$t'$= %0.2f" % t[frame_number],loc = 'center', fontsize=18)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.legend(loc='lower left',borderaxespad=0.)


fig = plt.figure(figsize=(4,7.5) , dpi=100)

plot = [plt.fill_betweenx(Grid.xc,1, facecolor='white',label=r'$\phi_{gas}$')]
plot = [plt.fill_betweenx(Grid.xc, 1-phi[:,0]+ C_sol[:,0], facecolor=blue,label=r'$\phi_{water}$')]
plot = [plt.fill_betweenx(Grid.xc, 1 - phi[:,0], facecolor=brown,label=r'$\phi_{soil}$')]


#plot = plt.plot(Grid.xc,T(H_sol[:,0],C_sol[:,0], Ste, Cpr),'r--',label=r'$\mathcal{T}$')
plt.ylabel(r'Dimensionless depth $z/z_0$')
plt.xlabel(r'Volume fraction $\phi$')
plt.ylim([Grid.xmax,Grid.xmin])
#plt.ylim([np.min([np.min(H_sol),np.min(C_sol)])-0.05,np.max([np.max(H_sol),np.max(C_sol)])+0.05])
plt.xlim([0,1])
plt.title(r"$\tau$= %0.2f" % t[i],loc = 'center', fontsize=18)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.legend(loc='lower left',borderaxespad=0.)

ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(C_sol[:,::40], plot[::40],t[::40]), interval=1/fps)

ani.save(f"../Figures/{simulation_name}_video_Nx{Grid.Nx}_tf{t[frn-1]}.mov", writer='ffmpeg', fps=30)

'''
