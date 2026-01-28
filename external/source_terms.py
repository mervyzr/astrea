import concurrent.futures
from itertools import repeat

import numpy as np

##############################################################################
# Source terms
##############################################################################

# Poisson solver for self-gravity
def poisson_solver(grid, sim_variables, G=1.):
    cells, ds = sim_variables.cells, sim_variables.ds
    rhos = grid[...,sim_variables.rho]

    # FFT densities
    rhos_k = np.fft.fftn(rhos)

    # Construct k-vectors for each dimension from FFT
    compute_k = lambda n, d: 2 * np.pi * np.fft.fftfreq(n, d=d)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        jobs = executor.map(compute_k, cells, [value for _, value in ds.items()])

        # Build |k|^2 on full grid
        rhos_k2 = np.sum([kvector**2 for kvector in jobs], axis=0)

    # Solve Poisson equation in Fourier space
    phi_k = np.zeros_like(rhos_k, dtype=np.complex128)
    mask = rhos_k2 > 0
    phi_k[mask] = -4 * np.pi * G * (rhos_k/rhos_k2)[mask]

    # Enforce zero-mean potential (k=0 mode)
    phi_k[~mask] = 0

    # Inverse FFT to real space
    return np.fft.ifftn(phi_k).real


# Compute g = -∇Φ using central differences (2nd-order)
def calcute_gravity(potentials, sim_variables):
    axes = sim_variables.axes

    fwd_acc = lambda g, ax: np.roll(g, -1, axis=ax)
    bwd_acc = lambda g, ax: np.roll(g, 1, axis=ax)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        fwd_jobs = executor.map(fwd_acc, repeat(potentials), axes)
        bwd_jobs = executor.map(bwd_acc, repeat(potentials), axes)

        fwd_potentials = np.stack([fwd_job for fwd_job in fwd_jobs], axis=0)
        bwd_potentials = np.stack([bwd_job for bwd_job in bwd_jobs], axis=0)

    ds = np.array([sim_variables.ds[ax] for ax in axes])
    ds = ds.reshape((-1,) + (1,) * potentials.ndim)

    return -(fwd_potentials - bwd_potentials)/(2.0 * ds)