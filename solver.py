import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from torch_geometric.data import Data
from mpl_toolkits.axes_grid1 import make_axes_locatable

# size = 14
# params = {
#     'text.usetex': True,
#     'font.family': 'serif',
#     'font.serif': 'cm',  # Computer Modern font
# 	'legend.fontsize':size,
#     'axes.labelsize' : size,
# 	'axes.titlesize' : size +2,
#     'xtick.labelsize' : size+1,
#     'ytick.labelsize' : size+1
# }
# plt.rcParams.update(params)

################################################################
######################### Device ###############################
################################################################

print(f"Is GPU available ? : {torch.cuda.is_available()}")
print(f"Number of GPU devices : {torch.cuda.device_count()}")
print(f"Current GPU devive : {torch.cuda.current_device()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device : {device}")
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_default_dtype(torch.float32)

################################################################
####################### Parameters #############################
################################################################

class Params(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key != 'mesh':
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)

# Parameters
params = {
    't_end': 0.1,  # End time
    'Lx': 1.0, # Domain size
    'Ly': 1.0,  
    'N_x': 160, # Number of grid points
    'N_y': 160,
    'CFL' : 0.1, # CFL number
    'n_modes': 32,  # Number of modes
    'nu' : 0.1, # small-scale (hyper)-viscosity coefficient
    'n_nu' : 1,  # order of the small-scale (hyper)-viscosity
    'mu' : 0.01,  # viscosity coefficient
    'n_mu' : 0,  # order of the viscosity
    "F" : "kolmogorov" # Forcing type
}

params = torch.nn.ParameterDict(params)


################################################################
######################### Solve ################################
################################################################

# Simulation parameters
N         = 400     # Spatial resolution
t         = 0       # current time of the simulation
tEnd      = 1       # time at which simulation ends
dt        = 0.01   # timestep
tOut      = 0.01    # draw frequency
nu        = 2e-7   # viscosity
n_nu      = 2
mu        = 0.1   # viscosity
n_mu      = 0

# Domain [0,1] x [0,1]
L = 2 * np.pi   
xlin = np.linspace(0,L, num=N+1)  # Note: x=0 & x=1 are the same point!
xlin = xlin[0:N]                  # chop off periodic point
xx, yy = np.meshgrid(xlin, xlin)

# Intial Condition (vortex)
psi = -np.sin(2*np.pi*(yy+xx))
psi_hat = np.fft.fftn( psi )
vy =  np.sin(2*np.pi*xx*2) 

# Fourier Space Variables
klin = 2.0 * np.pi / L * np.arange(-N/2, N/2)
kmax = np.max(klin)
kx, ky = np.meshgrid(klin, klin)
kx = np.fft.ifftshift(kx)
ky = np.fft.ifftshift(ky)
kSq = kx**2 + ky**2
kSq_inv = 1.0 / kSq
kSq_inv[kSq==0] = 1

# dealias with the 2/3 rule
dealias = (np.abs(kx) < (2./3.)*kmax) & (np.abs(ky) < (2./3.)*kmax)

def initField(Amp = 1e-6):
    psi_hat = Amp * ((2 * np.random.rand(*xx.shape) - 1) + (2 * np.random.rand(*xx.shape) - 1)* 1j)
    psi_hat = np.fft.fftn(np.real(np.fft.ifftn(psi_hat)))
    return psi_hat


def forcing(eps = 0.1 ):
    forcing_wavenumber = 25.0 * 2 * np.pi/L  # the forcing wavenumber, `k_f`, for a spectrum that is a ring in wavenumber space
    forcing_bandwidth  = 1.5  * 2 * np.pi/L  # the width of the forcing spectrum, `δ_f`

    K = np.sqrt(kSq)
    forcing_spectrum = np.exp(-(K - forcing_wavenumber)**2 / (2 * forcing_bandwidth**2))
    forcing_spectrum[kSq == 0] = 0
    eps_0 = np.sum(np.abs(forcing_spectrum)**2) / (L**2)

    forcing_spectrum = forcing_spectrum * eps/eps_0

    rand_val = np.random.rand(*kx.shape)
    F_hat = np.sqrt(forcing_spectrum) * np.exp(1j * 2 * np.pi * rand_val) / np.sqrt(dt)
    return F_hat



def Linear():
    L_hat = - nu * kSq ** n_nu 
    L_hat = L_hat - mu 
    return L_hat

def NonLinear(psi_hat):
    # forcing
    F_hat = forcing(eps = 0.1)

    u = np.real(np.fft.ifftn(-1j * ky * psi_hat))
    v = np.real(np.fft.ifftn(1j * kx * psi_hat))
    zeta = np.real(np.fft.ifftn( kSq * psi_hat))

    NL_hat = 1j * kx * np.fft.fftn( u * zeta ) + 1j * ky * np.fft.fftn( v * zeta )
    NL_hat = kSq_inv * (NL_hat - F_hat)
    return NL_hat

def time_step(psi_hat, dt):
    L_hat = Linear()
    NL_hat = NonLinear(psi_hat)

    # imex
    psi_hat = (psi_hat - dt * NL_hat)/(1 - dt * L_hat)

    # dealiasing
    psi_hat = dealias * psi_hat

    return psi_hat

np.random.seed(1234)
psi_hat = initField()
for _ in range(1000):
    psi_hat = time_step(psi_hat, dt)

psi = np.real(np.fft.ifftn(psi_hat ))


fig, ax = plt.subplots()
ax.set_aspect('equal')
cs = ax.contourf(xx, yy, psi, levels = 20)
cbar = fig.colorbar(cs)


