import numpy as np

##############################################################################
# Generic functions used throughout the finite volume code
##############################################################################

# For handling division-by-zero warnings during array divisions
def divide(dividend, divisor):
    return np.divide(dividend, divisor, out=np.zeros_like(dividend), where=divisor!=0)


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


# Converting (cell-/face-averaged) primitive variables w to conservative variables q through a higher-order approx.
def convert_primitive(grid, sim_variables):
    boundary, permutations = sim_variables.boundary, sim_variables.permutations
    w, q = np.copy(grid), np.zeros_like(grid)

    for axes in permutations:
        _w = add_boundary(grid.transpose(axes), boundary)
        w -= (np.diff(_w[1:], axis=0) - np.diff(_w[:-1], axis=0))/24

        _q = point_convert_primitive(_w, sim_variables)
        q += (np.diff(_q[1:], axis=0) - np.diff(_q[:-1], axis=0))/24
    return point_convert_primitive(w, sim_variables) + q


# Converting (cell-/face-averaged) conservative variables q to primitive variables w through a higher-order approx.
def convert_conservative(grid, sim_variables):
    boundary, permutations = sim_variables.boundary, sim_variables.permutations
    w, q = np.zeros_like(grid), np.copy(grid)

    for axes in permutations:
        _q = add_boundary(grid.transpose(axes), boundary)
        q -= (np.diff(_q[1:], axis=0) - np.diff(_q[:-1], axis=0))/24

        _w = point_convert_conservative(_q, sim_variables)
        w += (np.diff(_w[1:], axis=0) - np.diff(_w[:-1], axis=0))/24
    return point_convert_conservative(q, sim_variables) + w