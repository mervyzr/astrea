import concurrent.futures
from itertools import repeat

import numpy as np

##############################################################################
# Generic functions used throughout the finite volume code
##############################################################################

EPSILON = np.finfo('float64').eps


# Magic function to make errors disappear (!! physics would most likely be messed up so be very careful using this function !!)
def nan_to_num(arr):
    return np.nan_to_num(arr, copy=True, nan=0., posinf=1e16, neginf=-1e16)


# For handling division-by-zero warnings during array divisions
# !! MONITOR THE PHYSICS WHEN USING THIS; ZEROS IN DIVISOR MIGHT MEAN YOUR CODE IS INCORRECT !!
def divide(dividend, divisor, eps=EPSILON):
    #return np.divide(np.real(dividend), np.real(divisor+eps))
    return np.divide(np.real(dividend), np.real(divisor), out=np.full_like(dividend, 1/eps), where=divisor!=0)


# For handling log zero and log negative values
# !! MONITOR THE PHYSICS WHEN USING THIS; NEGATIVE OR ZERO VALUES MIGHT MEAN YOUR CODE IS INCORRECT INSTEAD !!
def log(arr, eps=EPSILON):
    positive = np.log(np.full(arr.shape, eps))
    return np.log(arr, out=positive, where=arr>0)


# There are situations where oscillations may produce negative densities/pressures
# This function is for handling those scenarios; ideally there should be no negative values
# !! MONITOR THE PHYSICS WHEN USING THIS; IMAGINARY PARTS DISCARDED, MONITOR FOR RANDOM OSCILLATIONS !!
def sqrt(arr):
    return np.sqrt(np.real(arr), out=np.zeros_like(arr), where=arr>=0)


# For handling norms; typically would always be using the last axis
def norm(arr):
    return np.linalg.norm(arr, axis=-1)


# Slice grid along axis
def slice_(grid, axis, start=0, end=None, step=1, *args):
    slc = [slice(None)] * grid.ndim

    if args and (2 <= len(args) <= 3):
        try:
            start, end, step = args
        except ValueError:
            try:
                start, end = args
                step = 1
            except ValueError:
                start, end, step = 0, grid.shape[axis], 1

    if not end:
        end = grid.shape[axis]

    slc[axis] = slice(start, end, step)
    return grid[tuple(slc)]


# Finite difference derivative (second order) of a padded grid
# [ W(i+1) - W(i) ] - [ W(i) - W(i-1) ] = W(i+1) - 2W(i) + W(i-1)
def laplacian(grid, sim_variables, axis):
    padded_grid = add_boundary(grid, sim_variables, axis=axis)
    return 1/(sim_variables.ds[axis]**2) * (np.diff(slice_(padded_grid, axis, start=1), axis=axis) - np.diff(slice_(padded_grid, axis, end=-1), axis=axis))


# Add boundary conditions
def add_boundary(grid, sim_variables, stencil=1, axis=0):
    arr = np.copy(grid)
    padding = [(0,0)] * grid.ndim
    padding[axis] = (stencil,stencil)
    return np.pad(arr, padding, mode=sim_variables.boundary)


# Convert between pressure P and total energy density e_tot; P is also related to the internal energy density e_int: P = (gamma-1) * e_int
# Do note that the energy densities e are related to the energies E: e_tot = rho * E_tot, e_int = rho * E_int
def convert_variable(variable, grid, sim_variables):
    rho, pressure, vels, Bfields = sim_variables.rho, sim_variables.pressure, sim_variables.vels, sim_variables.Bfields
    energy, momentums = pressure, vels
    gamma, permeability = sim_variables.gamma, sim_variables.constants.mu_0

    if variable.lower().startswith('p'):
        return grid[...,pressure]/(gamma-1) + .5*(grid[...,rho]*norm(grid[...,vels])**2) + .5*(norm(grid[...,Bfields])**2)/permeability
    elif variable.lower().startswith('e') or 'energy' in variable.lower():
        return (gamma-1) * (grid[...,energy] - .5 * (grid[...,rho]*norm(divide(grid[...,momentums], grid[...,rho][...,None]))**2 + (norm(grid[...,Bfields])**2)/permeability))


# Handler for conversion
def convert(variable_form, grid, sim_variables):
    converter = high_order_convert if sim_variables.higher_order else point_convert
    return converter(variable_form, grid, sim_variables)


# Pointwise (exact) conversion of conservative variables q <-> primitive variables w (up to 2nd-order accurate)
def point_convert(variable_form, grid, sim_variables):
    rho, pressure, energy, vels, momentums = sim_variables.rho, sim_variables.pressure, sim_variables.energy, sim_variables.vels, sim_variables.momentums
    arr = np.copy(grid)

    if variable_form.lower().startswith("p"):
        arr[...,energy] = convert_variable('pressure', grid, sim_variables)
        arr[...,momentums] = grid[...,vels] * grid[...,rho][...,None]
    elif variable_form.lower().startswith("c"):
        arr[...,pressure] = convert_variable('energy', grid, sim_variables)
        arr[...,vels] = divide(grid[...,momentums], grid[...,rho][...,None])
    return arr


# Variable inversion using the conversion of the grid (base) and the Taylor expansion terms (expansion) through a Laplacian (2nd-deriv, 2nd-order) approx. for each axis (up to 4th-order accurate)
def inversion_per_axis(variable_form, grid, sim_variables, axis):
    original_expansion = (sim_variables.ds[axis]**2)/24 * laplacian(grid, sim_variables, axis)
    converted_avg = point_convert(variable_form, grid, sim_variables)
    converted_expansion = (sim_variables.ds[axis]**2)/24 * laplacian(converted_avg, sim_variables, axis)
    return original_expansion, converted_expansion


# Converting cell-averaged conservative variables <q>_{i,j} <-> cell-averaged primitive variables <w>_{i,j}
def high_order_convert(variable_form, grid, sim_variables):
    base, expansion = np.copy(grid), np.zeros_like(grid)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        jobs = executor.map(inversion_per_axis, repeat(variable_form), repeat(grid), repeat(sim_variables), sim_variables.axes)

        for original_expansion, converted_expansion in jobs:
            base -= original_expansion
            expansion += converted_expansion

    return point_convert(variable_form, base, sim_variables) + expansion


# Converting face-averaged conservative variables <q>_{i+1/2,j} <-> face-averaged primitive variables <w>_{i+1/2,j}
def convert_intf(variable_form, grid, sim_variables, axis):
    base, expansion = np.copy(grid), np.zeros_like(grid)

    if sim_variables.higher_order and sim_variables.multidimensional:
        ortho_axes = sim_variables.axes[sim_variables.axes != axis]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            jobs = executor.map(inversion_per_axis, repeat(variable_form), repeat(grid), repeat(sim_variables), ortho_axes)

            for original_expansion, converted_expansion in jobs:
                base -= original_expansion
                expansion += converted_expansion

    new_grid = point_convert(variable_form, base, sim_variables) + expansion

    if sim_variables.magnetic:
        new_grid[...,5+sim_variables.axes] = grid[...,5+sim_variables.axes]

    return new_grid


# Converting cell-centred variables q_{i,j}     <-> cell-averaged variables <q>_{i,j} through a Laplacian (2nd-deriv, 2nd-order) approx. (up to 4th-order accurate), OR
# converting face-centred variables q_{i+1/2,j} <-> face-averaged variables <q>_{i+1/2,j}
def avg_cntr_convert(grid_form, grid, sim_variables, **kwargs):
    base = np.copy(grid)
    axes = np.array([])

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # for computation at interfaces
        if kwargs:
            if sim_variables.higher_order and sim_variables.multidimensional:
                axes = sim_variables.axes[sim_variables.axes != kwargs['axis']]
        # for computation at centres
        else:
            axes = sim_variables.axes

        if axes.size != 0:
            jobs = executor.map(laplacian, repeat(grid), repeat(sim_variables), axes)

            for job_idx, expansion in enumerate(jobs):
                if grid_form.lower().startswith('a'):
                    base -= (sim_variables.ds[axes[job_idx]]**2)/24 * expansion
                elif grid_form.lower().startswith('c'):
                    base += (sim_variables.ds[axes[job_idx]]**2)/24 * expansion

    return base


# Higher-order approximations at the interfaces for multi-dimensional higher-order schemes
def approx_face_avg(interfaces, sim_variables, axis):
    inner_func = lambda func, _grid_form, _grid, _sim_variables, _kwargs: func(_grid_form, _grid, _sim_variables, **_kwargs)
    with concurrent.futures.ThreadPoolExecutor() as inner_executor:
        return list(inner_executor.map(inner_func, repeat(avg_cntr_convert), repeat('avg'), interfaces, repeat(sim_variables), repeat({'axis':axis, 'pos':'intf'})))


# Compute the max eigenvalues for calculating the time evolution
def compute_eigmax(characteristics, axis):
    # Local max eigenvalue for each cell (1- or 3-Riemann invariant; shock wave or rarefaction wave)
    local_max_eigvals = np.max(np.abs(characteristics), axis=-1)

    # Local max eigenvalue between consecutive pairs of cell
    max_eigvals = np.maximum(slice_(local_max_eigvals, axis, end=-1), slice_(local_max_eigvals, axis, start=1))

    # Maximum wave speed (max eigenvalue) for time evolution
    return np.max(max_eigvals)


# Calculate the Roe-averaged primitive variables at the interface from the minus- & plus-interface states for use in Roe solver in order to better capture shocks [Roe & Pike, 1984; Brio & Wu, 1988; LeVeque, 2002; Stone et al., 2008]
def compute_Roe_average(interfaces, sim_variables):
    rho, pressure, vels, Bfields = sim_variables.rho, sim_variables.pressure, sim_variables.vels, sim_variables.Bfields

    plus_interface, minus_interface = interfaces
    avg = np.zeros_like(plus_interface)
    rho_plus, rho_minus = np.sqrt(plus_interface[...,rho]), np.sqrt(minus_interface[...,rho])

    avg[...,rho] = rho_minus * rho_plus
    avg[...,vels] = divide((plus_interface[...,vels] * rho_plus[...,None]) + (minus_interface[...,vels] * rho_minus[...,None]), (rho_minus + rho_plus)[...,None])
    avg[...,pressure] = divide((rho_plus * plus_interface[...,pressure]) + (rho_minus * minus_interface[...,pressure]), rho_minus + rho_plus)
    avg[...,Bfields] = divide((plus_interface[...,Bfields] * rho_minus[...,None]) + (minus_interface[...,Bfields] * rho_plus[...,None]), (rho_minus + rho_plus)[...,None])
    return avg


# Function for checking the numerical errors when computing the (primitive) Jacobian matrices, characteristic waves (eigenvalues/diagonal matrix), and left and right eigenvectors
def compute_characteristic_errors(grid, sim_variables, axis, check='identity'):
    from functions import constructor

    left_eigenvectors, right_eigenvectors = constructor.make_eigenvectors(grid, sim_variables, axis)
    _axis = tuple(np.arange(-sim_variables.dimensions, 0))

    # Jacobian check: A = R @ λ @ L (stricter)
    if check.lower() == "jacobian":
        characteristics = constructor.make_characteristics(grid, sim_variables, axis)

        i, j = np.diag_indices(characteristics.shape[-1])
        Lambda = np.zeros(sim_variables.cells + [len(i),len(j)])
        Lambda[...,i,j] = characteristics

        jacobian = constructor.make_Jacobian(grid, sim_variables, axis=axis)
        jacobian = np.delete(jacobian, 5+axis, axis=-2)
        jacobian = np.delete(jacobian, 5+axis, axis=-1)

        err = np.linalg.norm(jacobian - (right_eigenvectors @ Lambda @ left_eigenvectors), axis=_axis)

    # Identity check: L @ R = I
    elif check.lower() == "identity":
        err = np.linalg.norm((left_eigenvectors @ right_eigenvectors) - np.eye(right_eigenvectors.shape[-1]), axis=_axis)

    return err.max()