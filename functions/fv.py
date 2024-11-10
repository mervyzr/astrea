import numpy as np

##############################################################################
# Generic functions used throughout the finite volume code
##############################################################################

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


# Generic Gaussian function
def gauss_func(x, params):
    return params['y_offset'] + params['ampl']*np.exp(-((x-params['peak_pos'])**2)/params['fwhm'])


# Generic sin function
def sin_func(x, params):
    return params['y_offset'] + params['ampl']*np.sin(params['freq']*np.pi*x)


# Finite difference derivative (second order)
def derivative(grid, ax):
    width = grid.shape[ax]
    return np.diff(grid.take(range(1,width), axis=ax), axis=ax) - np.diff(grid.take(range(0,width-1), axis=ax), axis=ax)


# Add boundary conditions
def add_boundary(grid, boundary, stencil=1, axis=0):
    arr = np.copy(grid)
    padding = [(0,0)] * grid.ndim
    padding[axis] = (stencil,stencil)
    return np.pad(arr, padding, mode=boundary)


# Pointwise (exact) conversion of primitive variables w to conservative variables q (up to 2nd-order accurate)
def point_convert_primitive(grid, sim_variables):
    arr = np.copy(grid)
    rhos, vecs, pressures, B_fields = grid[...,0], grid[...,1:4], grid[...,4], grid[...,5:8]
    arr[...,4] = (pressures/(sim_variables.gamma-1)) + .5*(rhos*norm(vecs)**2 + norm(B_fields)**2)
    arr[...,1:4] = vecs * rhos[...,None]
    return arr


# Pointwise (exact) conversion of conservative variables q to primitive variables w (up to 2nd-order accurate)
def point_convert_conservative(grid, sim_variables):
    arr = np.copy(grid)
    rhos, moms, energies, B_fields = grid[...,0], grid[...,1:4], grid[...,4], grid[...,5:8]
    vecs = divide(moms, rhos[...,None])
    arr[...,4] = (sim_variables.gamma-1) * (energies - .5*(rhos*norm(vecs)**2 + norm(B_fields)**2))
    arr[...,1:4] = vecs
    return arr


# Conversion between averaged and centred variable "modes" with the Laplacian operator and centred difference coefficients (up to 2nd derivative because parabolic function)
# Attempts to raise the order of accuracy for the Laplacian to 4th-, 6th- and even 8th-order were made, but not too feasible because the averaging function
# is limited by the time-stepping and the limiting functions (currently max is 4th order)
def convert_mode(grid, sim_variables, _type="cell"):
    dimension, boundary, permutations = sim_variables.dimension, sim_variables.boundary, sim_variables.permutations
    new_grid = np.copy(grid)

    if "face" in _type:
        _range = range(1, dimension)
    else:
        _range = range(1)

    if grid.ndim - dimension == 2:
        for ax in _range:
            _new_grid = add_boundary(grid, boundary, axis=ax+1)
            new_grid -= np.sum(1/24 * derivative(_new_grid, ax+1), axis=0)
    else:
        for axes in permutations:
            reversed_axes = np.argsort(axes)
            for ax in _range:
                _new_grid = add_boundary(grid.transpose(axes), boundary, axis=ax)
                new_grid -= 1/24 * derivative(_new_grid, ax).transpose(reversed_axes)
    return new_grid


# Converting (cell-/face-averaged) primitive variables w to (cell-/face-averaged) conservative variables q through a higher-order approx.
def convert_primitive(grid, sim_variables, _type="cell"):
    dimension, boundary, permutations = sim_variables.dimension, sim_variables.boundary, sim_variables.permutations
    w, q = np.copy(grid), np.zeros_like(grid)

    if "face" in _type:
        _range = range(1, dimension)
    else:
        _range = range(1)

    if grid.ndim - dimension == 2:
        for ax in _range:
            _w = add_boundary(grid, boundary, axis=ax+1)
            w -= np.sum(1/24 * derivative(_w, ax+1), axis=0)

            _q = point_convert_primitive(_w, sim_variables)
            q += np.sum(1/24 * derivative(_q, ax+1), axis=0)
    else:
        for axes in permutations:
            reversed_axes = np.argsort(axes)
            for ax in _range:
                _w = add_boundary(grid.transpose(axes), boundary, axis=ax)
                w -= 1/24 * derivative(_w, ax).transpose(reversed_axes)

                _q = point_convert_primitive(_w, sim_variables)
                q += 1/24 * derivative(_q, ax).transpose(reversed_axes)
    return point_convert_primitive(w, sim_variables) + q


# Converting (cell-/face-averaged) conservative variables q to (cell-/face-averaged) primitive variables q through a higher-order approx.
def convert_conservative(grid, sim_variables, _type="cell"):
    dimension, boundary, permutations = sim_variables.dimension, sim_variables.boundary, sim_variables.permutations
    w, q = np.zeros_like(grid), np.copy(grid)

    if "face" in _type:
        _range = range(1, dimension)
    else:
        _range = range(1)

    if grid.ndim - dimension == 2:
        for ax in _range:
            _q = add_boundary(grid, boundary, axis=ax+1)
            q -= np.sum(1/24 * derivative(_q, ax+1), axis=0)

            _w = point_convert_conservative(_q, sim_variables)
            w += np.sum(1/24 * derivative(_w, ax+1), axis=0)
    else:
        for axes in permutations:
            reversed_axes = np.argsort(axes)
            for ax in _range:
                _q = add_boundary(grid.transpose(axes), boundary, axis=ax)
                q -= 1/24 * derivative(_q, ax).transpose(reversed_axes)

                _w = point_convert_conservative(_q, sim_variables)
                w += 1/24 * derivative(_w, ax).transpose(reversed_axes)
    return point_convert_conservative(q, sim_variables) + w


# Get the characteristics and max eigenvalues for calculating the time evolution
def compute_eigen(Jacobian, ax):
    characteristics = np.linalg.eigvals(Jacobian)

    # Local max eigenvalue for each cell (1- or 3-Riemann invariant; shock wave or rarefaction wave)
    local_max_eigvals = np.max(np.abs(characteristics), axis=-1)

    # Local max eigenvalue between consecutive pairs of cell
    max_eigvals = np.maximum(local_max_eigvals.take(range(0,local_max_eigvals.shape[ax]-1), axis=ax), local_max_eigvals.take(range(1,local_max_eigvals.shape[ax]), axis=ax))

    # Maximum wave speed (max eigenvalue) for time evolution
    eigmax = np.max(max_eigvals)

    return characteristics, eigmax


# Compute the 4th-order interface-averaged fluxes from the interface-averaged fluxes via higher order approximation
def compute_high_approx_flux(cntr_flux, avg_flux, sim_variables):
    dimension, boundary = sim_variables.dimension, sim_variables.boundary
    _cntr_flux, _avg_flux = np.copy(cntr_flux), np.copy(avg_flux)

    for ax in range(1, dimension):
        padded_arr = add_boundary(_avg_flux, boundary, axis=ax+1)
        _cntr_flux -= np.sum(1/24 * derivative(padded_arr, ax+1), axis=0)
    return _cntr_flux