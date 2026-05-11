import concurrent.futures
from itertools import repeat

import numpy as np

from functions import grid as gutils

##############################################################################
# Gravity module, for self-gravity and external gravity
##############################################################################

# Initialise the grid for external gravity
def initialise(sim_variables):
    config, cells, multidimensional, dimensions, coordinates = sim_variables.config, sim_variables.cells, sim_variables.multidimensional, sim_variables.dimensions, sim_variables.coordinates
    gx, gy, gz = sim_variables.gx, sim_variables.gy, sim_variables.gz

    match = lambda match_type, substrings: match_type(substring in config for substring in substrings)

    source_grid = np.zeros(list(cells)+[dimensions,], dtype=float, order='C')

    if match(any, ["rayleigh", "taylor", "rti"]):
        source_grid[...,gy] = -sim_variables.test_specifics['grav_acc']

    elif "torus" in config:
        x_centre = np.average(coordinates[0])
        physical_grid_x = gutils.make_physical_grid(coordinates[0], cells[0])
        if multidimensional:
            y_centre = np.average(coordinates[1])
            physical_grid_y = gutils.make_physical_grid(coordinates[1], cells[1])
            if dimensions > 2:
                z_centre = np.average(coordinates[2])
                physical_grid_z = gutils.make_physical_grid(coordinates[2], cells[2])

                x, y, z = np.meshgrid(physical_grid_x, physical_grid_y, physical_grid_z, indexing='ij')
                r = np.sqrt((x-x_centre)**2 + (y-y_centre)**2 + (z-z_centre)**2)

                source_grid[...,gx] = -sim_variables.test_specifics['GM']/r**3 * x
                source_grid[...,gy] = -sim_variables.test_specifics['GM']/r**3 * y
                source_grid[...,gz] = -sim_variables.test_specifics['GM']/r**3 * z

            else:
                x, y = np.meshgrid(physical_grid_x, physical_grid_y, indexing='ij')
                r = np.sqrt((x-x_centre)**2 + (y-y_centre)**2)

                source_grid[...,gx] = -sim_variables.test_specifics['GM']/r**2 * x
                source_grid[...,gy] = -sim_variables.test_specifics['GM']/r**2 * y

    return source_grid


# (FFT) Poisson solver for self-gravity; works on uniform grids
def poisson_solver(grid, sim_variables, G=1., eps=1e-6):
    cells = sim_variables.cells
    ds = [dh for dh in sim_variables.ds.values()]
    rhos = grid[...,sim_variables.rho]

    # FFT densities
    rhos_k = np.fft.fftn(rhos)

    # Construct k-vectors for each dimension from FFT
    compute_k = lambda n, dh: 2 * np.pi * np.fft.fftfreq(n, d=dh)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        jobs = executor.map(compute_k, cells, ds)
        kvectors = np.meshgrid(*[kvector for kvector in jobs], indexing='ij')

    # Build higher-order |k|^2 on full grid
    compute_k2 = lambda kvector, dh: 4/(dh**2) * np.sin(.5* kvector * dh)**2
    with concurrent.futures.ThreadPoolExecutor() as executor:
        jobs = executor.map(compute_k2, kvectors, ds)
        rhos_k2 = np.sum([k2 for k2 in jobs], axis=0)

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


# Compute g = -∇Φ using central differences (4th-order)
def get_acceleration(potentials, sim_variables):
    ds, axes = sim_variables.ds, sim_variables.axes

    def axis_acc(phi, ax):
        if sim_variables.higher_order:
            padded_phi_2 = gutils.add_boundary(phi, sim_variables, stencil=2, axis=ax)
            padded_phi = gutils.slice_(padded_phi_2, ax, *[1,-1])
            return -(gutils.slice_(padded_phi_2, ax, start=4) + 8*gutils.slice_(padded_phi, ax, start=2) - 8*gutils.slice_(padded_phi, ax, end=-2) + gutils.slice_(padded_phi_2, ax, end=-4))/(12 * ds[ax])
        else:
            padded_phi = gutils.add_boundary(phi, sim_variables, axis=ax)
            return -(gutils.slice_(padded_phi, axis=ax, start=2) - gutils.slice_(padded_phi, axis=ax, end=-2))/(2 * ds[ax])

    with concurrent.futures.ThreadPoolExecutor() as executor:
        jobs = executor.map(axis_acc, repeat(potentials), axes)
        g_accs = np.stack([g_acc for g_acc in jobs], axis=0)

    return g_accs


# Update step for gravity with conservative grid, given the timestep dt
def update(grid, dt, sim_variables, source_terms=0):
    rho, momentums, energy = sim_variables.rho, 1+sim_variables.axes, sim_variables.energy

    original_momentum = np.copy(grid[...,momentums])

    if sim_variables.self_gravity:
        Phi = poisson_solver(grid, sim_variables)
        g_self = np.moveaxis(get_acceleration(Phi, sim_variables), 0, -1)
    else:
        g_self = 0

    g_accs = g_self + source_terms

    grid[...,momentums] += dt * grid[...,rho][...,None] * g_accs
    grid[...,energy] += dt * np.sum(original_momentum * g_accs, axis=-1)

    return grid