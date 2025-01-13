import torch
import numpy as np
import matplotlib.pyplot as plt

# # Simulation parameters
# N         = 64     # Spatial resolution

# # Domain [0,1] x [0,1]
# L = 2 * np.pi   
# xlin = np.linspace(0,L, num=N, endpoint=False) 
# xx, yy = np.meshgrid(xlin, xlin)

# # Intial Condition (vortex)
# psi = np.cos( xx)
# d_psi = -np.sin(xx)
# # Fourier Space Variables
# klin = 2.0 * np.pi / L * np.arange(-N/2, N/2)
# kmax = np.max(klin)
# kx, ky = np.meshgrid(klin, klin)
# kx = np.fft.ifftshift(kx)
# ky = np.fft.ifftshift(ky)


def fftn(x, grid):
    return torch.fft.fftn(x) / (grid.N * grid.N * torch.exp(1j * (grid.kx * grid.xx[0,0] + grid.ky * grid.yy[0,0])))

def ifftn(x, grid):
    return torch.fft.ifftn(x * (grid.N * grid.N * torch.exp(1j * (grid.kx * grid.xx[0,0] + grid.ky * grid.yy[0,0]))))

def getEnstrophy(psi_hat, kSq):
    Z = 0.5 * torch.sum(torch.abs(kSq * psi_hat)**2)
    return Z    
    
def getEnergy(psi_hat, kSq):
    E = 0.5 * torch.sum(kSq * torch.abs(psi_hat)**2)# /N**2
    return E

def getEnergy2(psi_hat, grid):
    u = torch.real(ifftn(1j * grid.ky * psi_hat, grid))
    v = torch.real(ifftn(- 1j * grid.kx * psi_hat, grid))
    E = 0.5 * torch.sum(u**2 + v**2)/grid.N**2
    return E

def getCFL(psi_hat, dt, grid):
    u = torch.real(ifftn(1j * grid.ky * psi_hat, grid))
    v = torch.real(ifftn(- 1j * grid.kx * psi_hat, grid))
    CFL = torch.max(torch.sqrt(u**2 + v**2) * dt) * grid.N
    return CFL

def getRadialSpectrum(psi_hat, kSq):
    E_hat = torch.sqrt(kSq) * torch.abs(psi_hat)
    E_flat = E_hat.flatten()
    k_flat = kSq.flatten()
    k_flat_unique = torch.unique(k_flat)
    Radial_spectrum = torch.zeros(*k_flat_unique.shape)
    for i in range(k_flat_unique.shape[0]):
        Radial_spectrum[i] = torch.sum(E_flat[k_flat == k_flat_unique[i]])
    return torch.sqrt(k_flat_unique), Radial_spectrum

# nx, Lx = 64, 2 * np.pi
# x = np.linspace(-Lx/3,2 * Lx / 3, num=nx, endpoint=False) 
# klin = 2.0 * np.pi / Lx * np.arange(-nx/2, nx/2)
# klin = np.fft.ifftshift(klin)
# u  = lambda x : np.sin(2 * x) + 0.5 * np.cos(5 * x)
# u_hat = np.fft.fftn(u(x)) 
# plt.plot(klin[:10], np.real(u_hat)[:10])
# plt.plot(klin[:10], np.imag(u_hat)[:10])
# plt.grid()
# plt.show()
# u_rec = np.fft.ifftn(u_hat * (nx * np.exp(1j * klin* x[0])))
# plt.plot(x, u_rec)
# plt.plot(x, u(x))
# plt.show()
# # derivatives
# d_u = lambda x : 2 * np.cos(2 * x) - 2.5 * np.sin(5 * x)
# plt.plot(x, d_u(x))
# du_rec = np.fft.ifftn(u_hat * (1j * klin) * (nx * np.exp(1j * klin* x[0])))
# plt.plot(x, du_rec)


# fig, ax = plt.subplots()
# cs = ax.contourf(xx, yy, psi)
# cbar = fig.colorbar(cs)
# deli = 5
# psi_hat = fftn(psi) 
# fig, ax = plt.subplots()
# cs = ax.contourf(kx[:deli,:deli], ky[:deli,:deli], np.real(psi_hat)[:deli,:deli])
# cbar = fig.colorbar(cs)
# fig, ax = plt.subplots()
# cs = ax.contourf(kx[:deli,:deli], ky[:deli,:deli], np.imag(psi_hat)[:deli,:deli])
# cbar = fig.colorbar(cs)

# fig, ax = plt.subplots()
# cs = ax.contourf(xx, yy, d_psi)
# cbar = fig.colorbar(cs)
# dx_psi_hat = ifftn(psi_hat * (1j * kx))
# fig, ax = plt.subplots()
# cs = ax.contourf(xx, yy, d_psi)
# cbar = fig.colorbar(cs)
