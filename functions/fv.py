from itertools import repeat
import concurrent.futures as cfutures

import numpy as np

##############################################################################
# Generic functions used throughout the finite volume code
##############################################################################

# Generic Gaussian function
def gauss_func(x, params):
    return params['y_offset'] + params['ampl']*np.exp(-((x-params['peak_pos'])**2)/params['fwhm'])


# Generic sin function
def sine_func(x, params):
    return params['y_offset'] + params['ampl']*np.sin(params['freq']*np.pi*x)


# For handling division-by-zero warnings during array divisions
# !! MONITOR THE PHYSICS WHEN USING THIS; ZEROS IN DIVISOR MIGHT MEAN YOUR CODE IS INCORRECT INSTEAD !!
def divide(dividend, divisor):
    return np.divide(dividend, divisor, out=np.zeros_like(dividend), where=divisor!=0)


# For handling log zero and log negative values
# !! MONITOR THE PHYSICS WHEN USING THIS; NEGATIVE OR ZERO VALUES MIGHT MEAN YOUR CODE IS INCORRECT INSTEAD !!
def log(arr):
    return np.log(arr, out=np.zeros_like(arr), where=arr>0)


# There are situations where oscillations may produce negative densities/pressures
# This function is for handling those scenarios; ideally there should be no negative values
# !! MONITOR THE PHYSICS WHEN USING THIS; NEGATIVE VALUES MIGHT MEAN YOUR CODE IS INCORRECT INSTEAD !!
def sqrt(arr):
    return np.sqrt(arr, out=np.zeros_like(arr), where=arr>=0)


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
def derivative(grid, axis=0):
    return np.diff(slice_(grid, axis, start=1), axis=axis) - np.diff(slice_(grid, axis, end=-1), axis=axis)


# Taylor expansion of higher-order terms
def taylor_expand(grid, sim_variables, axis):
    return 1/24 * derivative(add_boundary(grid, sim_variables.boundary, axis=axis), axis=axis)


# Add boundary conditions
def add_boundary(grid, boundary, stencil=1, axis=0):
    arr = np.copy(grid)
    padding = [(0,0)] * grid.ndim
    padding[axis] = (stencil,stencil)
    return np.pad(arr, padding, mode=boundary)


# Convert between pressure P and total energy density e_tot; P is also related to the internal energy density e_int: P = (gamma-1) * e_int
# Do note that the energy densities e are related to the energies E: e_tot = rho * E_tot, e_int = rho * E_int
def convert_variable(variable, grid, sim_variables):
    rho, pressure, vels, Bfields = sim_variables.rho, sim_variables.pressure, sim_variables.vels, sim_variables.Bfields
    energy, momentums = pressure, vels

    if variable.lower().startswith('p'):
        return grid[...,pressure]/(sim_variables.gamma-1) + .5 * (grid[...,rho]*norm(grid[...,vels])**2 + (norm(grid[...,Bfields])**2)/sim_variables.permeability)
    elif variable.lower().startswith('e') or 'energy' in variable.lower():
        return (sim_variables.gamma-1) * (grid[...,energy] - .5 * (grid[...,rho]*norm(divide(grid[...,momentums], grid[...,rho][...,None]))**2 + (norm(grid[...,Bfields])**2)/sim_variables.permeability))


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


# Higher-order approximation for higher-order conversion between primitive & conservative variables (high_order_convert), and for centred & averaged variables (convert_interface)
def approximate_per_axis(variable_form, grid, sim_variables, axis):
    _base = add_boundary(grid, sim_variables.boundary, axis=axis)
    base = 1/24 * derivative(_base, axis=axis)

    _expansion = point_convert(variable_form, _base, sim_variables)
    expansion = 1/24 * derivative(_expansion, axis=axis)
    return base, expansion


# Converting cell-averaged conservative variables <q> <-> cell-averaged primitive variables <w> through a Laplacian (2nd-deriv, 2nd-order) approx. (up to 4th-order accurate)
def high_order_convert(variable_form, grid, sim_variables):
    axes = sim_variables.axes
    base, expansion = np.copy(grid), np.zeros_like(grid)

    with cfutures.ThreadPoolExecutor() as executor:
        jobs = executor.map(approximate_per_axis, repeat(variable_form), repeat(grid), repeat(sim_variables), axes)
        base -= np.sum([job[0] for job in jobs], axis=0)
        expansion += np.sum([job[1] for job in jobs], axis=0)

    return point_convert(variable_form, base, sim_variables) + expansion


# Converting face-averaged conservative variables <q>_{i+1/2,j} <-> face-averaged primitive variables <w>_{i+1/2,j}
def convert_interface(variable_form, interfaces, axis, sim_variables):
    axes, Bx, By = sim_variables.axes, sim_variables.Bx, sim_variables.By
    base, expansion = np.copy(interfaces), np.zeros_like(interfaces)

    if sim_variables.higher_order and sim_variables.multidimensional:
        ortho_axes = axes[axes != axis]

        with cfutures.ThreadPoolExecutor() as executor:
            jobs = executor.map(approximate_per_axis, repeat(variable_form), repeat(interfaces), repeat(sim_variables), ortho_axes)
            base -= np.sum([job[0] for job in jobs], axis=0)
            expansion -= np.sum([job[1] for job in jobs], axis=0)

    new_interfaces = point_convert(variable_form, base, sim_variables) + expansion

    if sim_variables.magnetic:
        new_interfaces[...,(Bx,By)] = interfaces[...,(Bx,By)]
    return new_interfaces


# 'Inverse reconstruct' the mag. fields' cell-averaged values from the (staggered grid) face-averaged values [Felker & Stone, 2018]
def inverse_reconstruct(grid, sim_variables):
    axes = sim_variables.axes
    new_grid = np.copy(grid)

    def per_axis(_grid, _sim_variables, axis):
        ortho_axes = _sim_variables.axes[_sim_variables.axes != axis]

        if _sim_variables.higher_order:
            face_cntrd = np.copy(_grid)

            if _sim_variables.multidimensional:
                # Approximate the face-averaged values to face-centred values (eq. 38)
                with cfutures.ThreadPoolExecutor() as inner_executor:
                    jobs = inner_executor.map(taylor_expand, repeat(_grid), repeat(_sim_variables), ortho_axes)
                    face_cntrd -= np.sum([job for job in jobs], axis=0)

            # Interpolate the face-centred values to cell-centred values (eq. 39)
            face_cntrd_padded_2 = add_boundary(face_cntrd, _sim_variables.boundary, stencil=2, axis=axis)
            face_cntrd_padded = slice_(face_cntrd_padded_2, axis, *[1,-1])
            cell_cntrd = -1/16 * (slice_(face_cntrd_padded, axis, end=-2) + slice_(face_cntrd_padded_2, axis, start=4)) \
                        + 9/16 * (face_cntrd + slice_(face_cntrd_padded, axis, start=2))

            # Apply Laplacian operator to convert cell-centred values to cell-averaged values (eq. 40)
            cell_avgd = np.copy(cell_cntrd) + taylor_expand(cell_cntrd, sim_variables, axis=axis)

            if _sim_variables.multidimensional:
                with cfutures.ThreadPoolExecutor() as inner_executor:
                    jobs = inner_executor.map(taylor_expand, repeat(cell_cntrd), repeat(_sim_variables), ortho_axes)
                    cell_avgd += np.sum([job for job in jobs], axis=0)

        elif _sim_variables.subgrid in ['plm', 'l', 'linear']:
            padded_grid = add_boundary(_grid, _sim_variables.boundary, axis=axis)
            cell_avgd = slice_(.5 * (slice_(padded_grid, axis, start=1) + slice_(padded_grid, axis, end=-1)), axis, start=1)

        else:
            cell_avgd = _grid

        return cell_avgd

    # Update the grid values with the updated B-field values
    with cfutures.ThreadPoolExecutor() as executor:
        for axis, Bfield in enumerate(executor.map(per_axis, repeat(grid), repeat(sim_variables), axes)):
            new_grid[...,5+axes[axis]] = Bfield[...,5+axes[axis]]

    return new_grid


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