import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from torch_geometric.data import Data
from mpl_toolkits.axes_grid1 import make_axes_locatable

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