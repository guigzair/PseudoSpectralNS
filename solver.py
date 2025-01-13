import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from torch_geometric.data import Data
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as animation
import Helpers as hp

size = 14
params = {
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': 'cm',  # Computer Modern font
	'legend.fontsize':size,
    'axes.labelsize' : size,
	'axes.titlesize' : size +2,
    'xtick.labelsize' : size+1,
    'ytick.labelsize' : size+1
}
plt.rcParams.update(params)


################################################################
######################### Device ###############################
################################################################

print(f"Is GPU available ? : {torch.cuda.is_available()}")
print(f"Number of GPU devices : {torch.cuda.device_count()}")
# print(f"Current GPU devive : {torch.cuda.current_device()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device : {device}")
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_default_dtype(torch.float32)

################################################################
####################### Parameters #############################
################################################################


class Parameters(torch.nn.Module):
    def __init__(self, dt = 0.01, nu = 1e-10, n_nu = 4, eta = 0, n_eta = 2, mu = 0.1, n_mu = 0, device  = device):
        super().__init__()
        self.device = device
        self.dt = torch.tensor(dt).to(device)   # timestep
        self.nu = torch.tensor(nu).to(device)   # hyper-viscosity
        self.n_nu = torch.tensor(n_nu).to(device) 
        self.eta = torch.tensor(eta).to(device)   # viscosity
        self.n_eta = torch.tensor(n_eta).to(device) 
        self.mu = torch.tensor(mu).to(device)  # linear drag
        self.n_mu = torch.tensor(n_mu).to(device) 

class Grid(Parameters):
    def __init__(self,N = 256, L = 2 * torch.pi,  **kwargs):
        super().__init__(**kwargs)
        
        self.N = N
        self.L = L
        self.xlin = torch.tensor(np.linspace(0, self.L - 1/self.N, num=self.N, endpoint=False)).to(self.device) 
        self.xx, self.yy = torch.meshgrid(self.xlin, self.xlin) 
        self.klin = 2.0 * torch.pi / self.L * torch.arange(-self.N/2, self.N/2).to(self.device) 
        self.kmax = torch.max(self.klin).to(self.device) 
        self.kx, self.ky = torch.meshgrid(self.klin, self.klin)
        self.kx = torch.fft.ifftshift(self.kx).to(self.device) 
        self.ky = torch.fft.ifftshift(self.ky).to(self.device) 
        self.kSq = self.kx**2 + self.ky**2
        self.kSq_inv = 1.0 / self.kSq
        self.kSq_inv[self.kSq==0] = 1
        self.dealias = (1 - torch.sign( torch.sqrt(self.kSq) - 1/3 * (self.N-1))) / 2
        self.dealais = self.dealias.to(self.device)

# Probably a better way to convert all parameters to device
# device = 'cpu'
grid = Grid( device = device)



################################################################
######################### Solve ################################
################################################################




def initField(Amp = 0.5, grid = grid):
    psi_hat = ((2 * torch.rand(*grid.xx.shape) - 1) + (2 * torch.rand(*grid.xx.shape) - 1)* 1j)
    psi_hat = hp.fftn(torch.real(hp.ifftn(psi_hat)))
    return psi_hat

def peakedisotropicspectrum(k_peak = 6, E = 0.5, grid = grid):
    k_0 = k_peak * 2 * torch.pi/grid.L
    K = torch.sqrt(grid.kSq)
    modk = (K * (1 + (K / k_0)**4))**(-1)
    modk[grid.kSq==0] = 0
    phases = torch.normal(mean = torch.zeros(*grid.kx.shape), std = torch.ones(*grid.kx.shape)) +  1j * torch.normal(mean = torch.zeros(*grid.kx.shape), std = torch.ones(*grid.kx.shape))
    phases = phases.to(grid.device)
    psi_hat = modk * phases / (grid.N * grid.N * torch.exp(1j * (grid.kx * grid.xx[0,0] + grid.ky * grid.yy[0,0])))
    E_0 = hp.getEnergy(psi_hat, grid.kSq) 
    psi_hat = psi_hat  * torch.sqrt(E/E_0)
    psi_hat = hp.fftn(torch.real(hp.ifftn(psi_hat, grid)), grid) # real initial condition
    return psi_hat



def forcing(grid, dt, eps = 0.1 ):
    forcing_wavenumber = 14.0 * 2 * torch.pi/grid.L  # the forcing wavenumber, `k_f`, for a spectrum that is a ring in wavenumber space
    forcing_bandwidth  = 1.5  * 2 * torch.pi/grid.L  # the width of the forcing spectrum, `δ_f`

    K = torch.sqrt(grid.kSq)
    forcing_spectrum = torch.exp(-(K - forcing_wavenumber)**2 / (2 * forcing_bandwidth**2))
    forcing_spectrum[grid.kSq == 0] = 0

    rand_val = torch.rand(*grid.kx.shape).to(grid.device)
    F_hat = torch.sqrt(forcing_spectrum) * torch.exp(1j * 2 * torch.pi * rand_val) 
    F_hat = hp.fftn(torch.real(hp.ifftn(F_hat, grid)), grid) # real forcing

    F_hat = F_hat / (grid.N * grid.N * torch.exp(1j * (grid.kx * grid.xx[0,0] + grid.ky * grid.yy[0,0])))
    
    eps_0 = hp.getEnergy(F_hat, grid.kSq)
    F_hat = F_hat * torch.sqrt(eps/eps_0)/ torch.sqrt(dt)
    return F_hat

# hp.getEnergy(forcing(grid, grid.dt), grid.kSq)

def Linear(grid):
    L_hat = - grid.nu * grid.kSq ** grid.n_nu 
    L_hat = L_hat - grid.eta * grid.kSq ** grid.n_eta
    L_hat = L_hat - grid.mu 
    return L_hat

def NonLinear(psi_hat, grid, F_hat):

    u = torch.real(hp.ifftn(1j * grid.ky * psi_hat, grid)) 
    v = torch.real(hp.ifftn(- 1j * grid.kx * psi_hat, grid)) 
    zeta = torch.real(hp.ifftn( grid.kSq * psi_hat, grid)) 

    NL_hat = 1j * grid.kx * hp.fftn( u * zeta, grid ) + 1j * grid.ky * hp.fftn( v * zeta , grid)
    NL_hat =  (-NL_hat + F_hat) * grid.kSq_inv 
    return NL_hat

def time_step(psi_hat, dt, grid = grid):
    L_hat = Linear(grid)
    F_hat = forcing(grid, grid.dt, eps = 0.1) 

    PNL_hat = torch.zeros(L_hat.shape, dtype = torch.complex128).to(device)

    psi_hat_0 = psi_hat.clone().detach()

    # weights and coefficients for classical explicit rk4 method
    order = torch.tensor([0.5, 0.5, 1]).to(device)
    pond = torch.tensor([1/6, 1/3, 1/3, 1/6]).to(device)

    for irk4 in range(len(order)) : 
        NL_hat = NonLinear(psi_hat, grid, F_hat) 
        PNL_hat += pond[irk4] * NL_hat

        # imex
        psi_hat = (psi_hat_0 + dt * order[irk4] * NL_hat)/(1 - dt * order[irk4] * L_hat)

        # dealiasing
        psi_hat = grid.dealias * psi_hat

    NL_hat = NonLinear(psi_hat, grid, F_hat) 
    PNL_hat += pond[-1] * NL_hat

    # imex
    psi_hat = (psi_hat_0 + dt * PNL_hat)/(1 - dt * L_hat)

    # dealiasing
    psi_hat = grid.dealias * psi_hat

    return psi_hat


###########################################################################################
################################## Solving ################################################
###########################################################################################

torch.manual_seed(1234)
psi_hat = peakedisotropicspectrum().to(device) * 0

frames = []
E = []
E2 = []
Z = []
for t in range(2000):
    if t % 100 == 0:
        print(t)
    psi_hat = time_step(psi_hat, grid.dt)
    E.append(hp.getEnergy(psi_hat, grid.kSq))
    E2.append(hp.getEnergy2(psi_hat, grid))
    Z.append(hp.getEnstrophy(psi_hat, grid.kSq))
    # if t % 25 == 0:
    #     # print(hp.getCFL(psi_hat, grid.dt, grid))
    #     frames.append(torch.real(hp.ifftn(psi_hat * grid.kSq, grid)).to('cpu').detach().numpy())    

zeta = torch.real(hp.ifftn(psi_hat * grid.kSq, grid)).to('cpu').detach().numpy()


###########################################################################################
################################## Plot ###################################################
###########################################################################################

grid = Grid(device = 'cpu')
E = torch.tensor(E).to('cpu')
E2 = torch.tensor(E2).to('cpu')
Z = torch.tensor(Z).to('cpu')

# F_hat = forcing(grid, grid.dt)
# zeta = torch.real(hp.ifftn(F_hat , grid)).to('cpu').detach().numpy()
fig, ax = plt.subplots()
ax.set_aspect('equal')
cs = ax.contourf(grid.xx, grid.yy, zeta, levels = 20)
cbar = fig.colorbar(cs)


fig, ax = plt.subplots()
ax.plot(E/E[0], label = "Energy")
ax.plot(Z/Z[0], label = "Enstrophy")
ax.set_xlabel(r'Time')
ax.set_ylabel(r'Energy \& Enstrophy')
ax.legend()
ax.grid()


k_r, E_k = hp.getRadialSpectrum(psi_hat.to('cpu'), grid.kSq) 
fig, ax = plt.subplots()
ax.plot(k_r, E_k, label = "Energy")
ax.set_xlabel(r'$k_r$')
ax.set_ylabel(r'$\int |\hat{E}(k_r)|k_r dk_\theta$')
ax.loglog()
ax.grid()


# Animation
# print("Animation")
# fig, ax = plt.subplots()
# ax.set_aspect('equal')
# ax.set_xlabel(r'$x$')
# ax.set_ylabel(r'$y$')
# ax.set_title(r'$\zeta$')
# ims = []
# for i in range(len(frames)):
#     im = ax.contourf(grid.xx, grid.yy, frames[i], levels = 50)
#     if i == 0:
#         ax.contourf(grid.xx, grid.yy, zeta, levels = 50)  # show an initial one first
#     ims.append([im])

# ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
#                                 repeat_delay=1000)
# ani.save("animation.gif",writer = animation.PillowWriter(fps=10) )
