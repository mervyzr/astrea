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
def norm(arr, axis=-1):
    return np.linalg.norm(arr, axis=axis)


# Generic Gaussian function
def gauss_func(x, params):
    peak_pos = (x[0]+x[-1])/2
    return params['y_offset'] + params['ampl']*np.exp(-((x-peak_pos)**2)/params['fwhm'])


# Generic sin function
def sin_func(x, params):
    return params['y_offset'] + params['ampl']*np.sin(params['freq']*np.pi*x)


# Generic sinc function
def sinc_func(x, params):
    return params['y_offset'] + params['ampl']*np.sinc(x*params['freq']/np.pi)


# Add boundary conditions
def add_boundary(grid, boundary, stencil=1):
    arr = np.copy(grid)
    padding = [(0,0)] * grid.ndim
    padding[0] = (stencil,stencil)
    return np.pad(arr, padding, mode=boundary)


# Pointwise (exact) conversion of primitive variables w to conservative variables q (up to 2nd-order accurate)
def point_convert_primitive(grid, sim_variables):
    arr = np.copy(grid)
    rhos, vecs, pressures, B_fields = grid[...,0], grid[...,1:4], grid[...,4], grid[...,5:8]
    arr[...,4] = (pressures/(sim_variables.gamma-1)) + (.5*rhos*norm(vecs)**2) + (.5*norm(B_fields)**2)
    arr[...,1:4] = (vecs.T * rhos.T).T
    return arr


# Pointwise (exact) conversion of conservative variables q to primitive variables w (up to 2nd-order accurate)
def point_convert_conservative(grid, sim_variables):
    arr = np.copy(grid)
    rhos, energies, B_fields = grid[...,0], grid[...,4], grid[...,5:8]
    vecs = divide(grid[...,1:4].T, grid[...,0].T).T
    arr[...,4] = (sim_variables.gamma-1) * (energies - (.5*rhos*norm(vecs)**2) - (.5*norm(B_fields)**2))
    arr[...,1:4] = vecs
    return arr


# Converting (cell-averaged) primitive variables w to (cell-averaged) conservative variables q through a higher-order approx.
def convert_primitive(grid, sim_variables):
    boundary, permutations = sim_variables.boundary, sim_variables.permutations
    w, q = np.copy(grid), np.zeros_like(grid)

    for axes in permutations:
        reversed_axes = np.argsort(axes)
        _w = add_boundary(grid.transpose(axes), boundary)
        w -= (np.diff(_w[1:], axis=0) - np.diff(_w[:-1], axis=0)).transpose(reversed_axes)/24

        _q = point_convert_primitive(_w, sim_variables)
        q += (np.diff(_q[1:], axis=0) - np.diff(_q[:-1], axis=0)).transpose(reversed_axes)/24
    return point_convert_primitive(w, sim_variables) + q


# Converting (cell-averaged) conservative variables q to (cell-averaged) primitive variables w through a higher-order approx.
def convert_conservative(grid, sim_variables):
    boundary, permutations = sim_variables.boundary, sim_variables.permutations
    w, q = np.zeros_like(grid), np.copy(grid)

    for axes in permutations:
        reversed_axes = np.argsort(axes)
        _q = add_boundary(grid.transpose(axes), boundary)
        q -= (np.diff(_q[1:], axis=0) - np.diff(_q[:-1], axis=0)).transpose(reversed_axes)/24

        _w = point_convert_conservative(_q, sim_variables)
        w += (np.diff(_w[1:], axis=0) - np.diff(_w[:-1], axis=0)).transpose(reversed_axes)/24
    return point_convert_conservative(q, sim_variables) + w


# Convert interface-averaged states/fluxes to interface-centred states/fluxes via higher order approximation
def convert_interface(expand_arr, boundary, main_arr=None):
    _arr = np.copy(expand_arr)

    try:
        _ = main_arr.shape
    except (AttributeError, NameError):
        arr = np.copy(expand_arr)
    else:
        arr = np.copy(main_arr)

    for _axis in range(1, arr.ndim-1):
        padding = [(0,0)] * arr.ndim
        padding[_axis] = (1,1)

        _padded_arr = np.pad(_arr, padding, mode=boundary)
        length = _padded_arr.shape[_axis]
        arr -= (np.diff(_padded_arr.take(range(1,length), axis=_axis), axis=_axis) - np.diff(_padded_arr.take(range(0,length-1), axis=_axis), axis=_axis))/24

    return arr


# Get the characteristics and max eigenvalues for calculating the time evolution
def compute_eigen(jacobian):
    characteristics = np.linalg.eigvals(jacobian)

    # Local max eigenvalue for each cell (1- or 3-Riemann invariant; shock wave or rarefaction wave)
    local_max_eigvals = np.max(np.abs(characteristics), axis=-1)

    # Local max eigenvalue between consecutive pairs of cell
    max_eigvals = np.max([local_max_eigvals[:-1], local_max_eigvals[1:]], axis=0)

    # Maximum wave speed (max eigenvalue) for time evolution
    eigmax = np.max(max_eigvals)

    return characteristics, eigmax