import numpy as np

##############################################################################
# Generic functions used throughout the finite volume code
##############################################################################

rho, vx, vy, vz, pressure, Bx, By, Bz = range(8)
vels, Bfields = slice(1,4), slice(5,8)
energy, momentums = pressure, vels


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


# Slice ndarray along axis
def slice_along_axis(arr, axis, *args, **kwargs):
    slc = [slice(None)] * arr.ndim

    if args and (2 <= len(args) <= 3):
        try:
            start, end, step = args
        except ValueError:
            try:
                start, end = args
                step = 1
            except ValueError:
                start, end, step = 0, len(arr.shape[axis]), 1
        slc[axis] = slice(start, end, step)

    elif kwargs:
        try:
            end = kwargs['end']
        except KeyError:
            end = arr.shape[axis]
        try:
            start = kwargs['start']
        except KeyError:
            start = 0
        try:
            step = kwargs['step']
        except KeyError:
            step = 1
        slc[axis] = slice(start, end, step)

    return arr[tuple(slc)]


# Finite difference derivative (second order) of a padded grid
# [ W(i+1) - W(i) ] - [ W(i) - W(i-1) ] = W(i+1) - 2W(i) + W(i-1)
def derivative(grid, axis=0):
    return np.diff(slice_along_axis(grid, axis, start=1), axis=axis) - np.diff(slice_along_axis(grid, axis, end=-1), axis=axis)


# Add boundary conditions
def add_boundary(grid, boundary, stencil=1, axis=0):
    arr = np.copy(grid)
    padding = [(0,0)] * grid.ndim
    padding[axis] = (stencil,stencil)
    return np.pad(arr, padding, mode=boundary)


# Convert between pressure P and total energy density e_tot; P is also related to the internal energy density e_int: P = (gamma-1) * e_int
# Do note that the energy densities e are related to the energies E: e_tot = rho * E_tot, e_int = rho * E_int
def convert_variable(variable, grid, sim_variables, staggered=False, permeability=1):
    if staggered:
        arr = inverse_reconstruct(grid, sim_variables)
    else:
        arr = grid

    if variable.lower().startswith('p'):
        return arr[...,pressure]/(sim_variables.gamma-1) + .5 * (arr[...,rho]*norm(arr[...,vels])**2 + (norm(arr[...,Bfields])**2)/permeability)
    elif variable.lower().startswith('e') or 'energy' in variable.lower():
        return (sim_variables.gamma-1) * (arr[...,energy] - .5 * (arr[...,rho]*norm(divide(arr[...,vels], arr[...,rho][...,None]))**2 + (norm(arr[...,Bfields])**2)/permeability))


# Pointwise (exact) conversion of primitive variables w to conservative variables q (up to 2nd-order accurate)
def point_convert_primitive(grid, sim_variables, staggered=False):
    arr = np.copy(grid)
    arr[...,energy] = convert_variable('pressure', grid, sim_variables, staggered=staggered)
    arr[...,momentums] = grid[...,vels] * grid[...,rho][...,None]
    return arr


# Pointwise (exact) conversion of conservative variables q to primitive variables w (up to 2nd-order accurate)
def point_convert_conservative(grid, sim_variables, staggered=False):
    arr = np.copy(grid)
    arr[...,pressure] = convert_variable('energy', grid, sim_variables, staggered=staggered)
    arr[...,vels] = divide(grid[...,momentums], grid[...,rho][...,None])
    return arr


# Converting (cell-/face-averaged) primitive variables w to (cell-/face-averaged) conservative variables q through a Laplacian (2nd-deriv, 2nd-order) approx.
def high_order_convert_primitive(grid, sim_variables, staggered=False, compute_face=False):
    w, q = np.copy(grid), np.zeros_like(grid)

    if compute_face:
        _range = range(1, sim_variables.dimension)
    else:
        _range = range(sim_variables.dimension)

    for ax in _range:
        _w = add_boundary(grid, sim_variables.boundary, axis=ax)
        w -= 1/24 * derivative(_w, axis=ax)

        _q = point_convert_primitive(_w, sim_variables, staggered=staggered)
        q += 1/24 * derivative(_q, axis=ax)

    conservative_grid = point_convert_primitive(w, sim_variables, staggered=staggered) + q
    if staggered:
        conservative_grid[...,(Bx,By)] = grid[...,(Bx,By)]
    return conservative_grid


# Converting (cell-/face-averaged) conservative variables q to (cell-/face-averaged) primitive variables q through a Laplacian (2nd-deriv, 2nd-order) approx.
def high_order_convert_conservative(grid, sim_variables, staggered=False, compute_face=False):
    w, q = np.zeros_like(grid), np.copy(grid)

    if compute_face:
        _range = range(1, sim_variables.dimension)
    else:
        _range = range(sim_variables.dimension)

    for ax in _range:
        _q = add_boundary(grid, sim_variables.boundary, axis=ax)
        q -= 1/24 * derivative(_q, axis=ax)

        _w = point_convert_conservative(_q, sim_variables, staggered=staggered)
        w += 1/24 * derivative(_w, axis=ax)

    primitive_grid = point_convert_conservative(q, sim_variables, staggered=staggered) + w
    if staggered:
        primitive_grid[...,(Bx,By)] = grid[...,(Bx,By)]
    return primitive_grid


# Convert between CENTRED cell/face variables and AVERAGED cell/face variables (i.e. FD <-> FV) (at higher order) with the Laplacian operator and centred difference coefficients (up to 2nd derivative because parabolic function)
def high_order_convert(var_pos, grid_rep, grid, sim_variables):
    new_grid = np.copy(grid)

    if "face" in var_pos:
        _range = range(1, sim_variables.dimension)
    else:
        _range = range(sim_variables.dimension)

    for ax in _range:
        padded_grid = add_boundary(grid, sim_variables.boundary, axis=ax)

        if grid_rep.startswith("a"):
            new_grid -= 1/24 * derivative(padded_grid, axis=ax)
        else:
            new_grid += 1/24 * derivative(padded_grid, axis=ax)
    return new_grid


# Compute the 4th-order interface-centred fluxes from the interface-averaged fluxes via higher order approximation
def high_order_compute_flux(_cntr_flux, _avg_flux, sim_variables):
    cntr_flux, avg_flux = np.copy(_cntr_flux), np.copy(_avg_flux)

    if sim_variables.higher_order:
        for ax in range(1, sim_variables.dimension):
            padded_avg_flux = add_boundary(avg_flux, sim_variables.boundary, axis=ax)
            cntr_flux -= 1/24 * derivative(padded_avg_flux, ax)
    return cntr_flux


# 'Inverse reconstruct' the centred grid cell-averages from the staggered grid face-averages [Felker & Stone, 2018]
def inverse_reconstruct(grid, sim_variables):
    new_grid = np.copy(grid)

    for axis in sim_variables.axes:
        if sim_variables.higher_order:
            # Approximate the face-averaged values to face-centred values (eq. 38)
            face_cntrd = high_order_convert('face', 'avg', grid, sim_variables)

            # Interpolate the face-centred values to cell-centred values (eq. 39)
            face_cntrd_padded_2 = add_boundary(face_cntrd, sim_variables.boundary, stencil=2, axis=axis)
            face_cntrd_padded = slice_along_axis(face_cntrd_padded_2, axis, *[1,-1])
            cell_cntrd = -1/16 * (slice_along_axis(face_cntrd_padded, axis, end=-2) + slice_along_axis(face_cntrd_padded_2, axis, start=4)) + 9/16 * (face_cntrd + slice_along_axis(face_cntrd_padded, axis, start=2))

            # Apply Laplacian operator to convert cell-centred values to cell-averaged values (eq. 40)
            cell_avgd = high_order_convert('cell', 'cntr', cell_cntrd, sim_variables)
        elif sim_variables.subgrid in ['plm', 'l', 'linear']:
            padded_grid = add_boundary(grid, sim_variables.boundary, axis=axis)
            cell_avgd = slice_along_axis(.5 * (slice_along_axis(padded_grid, axis, start=1) + slice_along_axis(padded_grid, axis, end=-1)), axis, end=-1)
        else:
            cell_avgd = grid

        # Update the grid values with the updated B-field values
        new_grid[...,5+axis] = cell_avgd[...,5+axis]

    return new_grid


# Compute the max eigenvalues for calculating the time evolution
def compute_eigmax(characteristics, axis):
    # Local max eigenvalue for each cell (1- or 3-Riemann invariant; shock wave or rarefaction wave)
    local_max_eigvals = np.max(np.abs(characteristics), axis=-1)

    # Local max eigenvalue between consecutive pairs of cell
    max_eigvals = np.maximum(slice_along_axis(local_max_eigvals, axis, end=-1), slice_along_axis(local_max_eigvals, axis, start=1))

    # Maximum wave speed (max eigenvalue) for time evolution
    return np.max(max_eigvals)