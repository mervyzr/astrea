import concurrent.futures
from itertools import repeat

import numpy as np

from functions import fv

##############################################################################
# Self-gravity module
##############################################################################

# (FFT) Poisson solver
def poisson_solver(grid, sim_variables, G=1., eps=1e-6):
    cells, ds = sim_variables.cells, sim_variables.ds
    rhos = grid[...,sim_variables.rho]

    # FFT densities
    rhos_k = np.fft.fftn(rhos)

    # Construct k-vectors for each dimension from FFT
    compute_k = lambda n, d: 2 * np.pi * np.fft.fftfreq(n, d=d)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        jobs = executor.map(compute_k, cells, [dh for dh in ds.values()])

        # Build |k|^2 on full grid
        kx, ky = np.meshgrid(*[kvector for kvector in jobs], indexing='ij')
        rhos_k2 = kx**2 + ky**2

    # Solve Poisson equation in Fourier space
    phi_k = np.zeros_like(rhos_k, dtype=np.complex128)
    mask = rhos_k2 > 0
    phi_k[mask] = -(rhos_k/(rhos_k2 + eps))[mask]
    if G != 1.:
        phi_k *= 4 * np.pi * G

    # Enforce zero-mean potential (k=0 mode)
    phi_k[~mask] = 0

    # Inverse FFT to real space
    return np.fft.ifftn(phi_k).real


# Compute g = -∇Φ using central differences (2nd-order)
def get_acceleration(potentials, sim_variables):
    ds, axes = sim_variables.ds, sim_variables.axes

    def axis_acc(phi, ax):
        padded_phi = fv.add_boundary(phi, sim_variables, axis=ax)
        return -(fv.slice_(padded_phi, axis=ax, start=2) - fv.slice_(padded_phi, axis=ax, end=-2))/(2 * ds[ax])

    with concurrent.futures.ThreadPoolExecutor() as executor:
        jobs = executor.map(axis_acc, repeat(potentials), axes)
        g_accs = np.stack([g_acc for g_acc in jobs], axis=0)

    return g_accs