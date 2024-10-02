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


# Add boundary conditions
def add_boundary(grid, boundary, stencil=1):
    arr = np.copy(grid)
    padding = [(0,0)] * grid.ndim
    padding[0] = (stencil,stencil)
    return np.pad(arr, padding, mode=boundary)


# Conversion between averaged and centred variable "modes" with Laplacian operator (4th-order accuracy with 2nd-order centred difference)
def convert_mode(grid, sim_variables, _type="cell"):
    dimension, boundary, permutations = sim_variables.dimension, sim_variables.boundary, sim_variables.permutations
    new_grid = np.copy(grid)

    if _type == "face" or _type == "interface":
        for axes in permutations:
            reversed_axes = np.argsort(axes)  # Only necessary for 3D
            for ax in range(1, dimension):
                padding = [(0,0)] * grid.ndim
                padding[ax] = (1,1)

                padded_grid = np.pad(grid.transpose(axes), padding, mode=boundary)
                length = padded_grid.shape[ax]
                new_grid -= 1/24 * (np.diff(padded_grid.take(range(1,length), axis=ax), axis=ax) - np.diff(padded_grid.take(range(0,length-1), axis=ax), axis=ax)).transpose(reversed_axes)
    else:
        for axes in permutations:
            reversed_axes = np.argsort(axes)  # Only necessary for 3D
            padded_grid = add_boundary(grid.transpose(axes), boundary)
            new_grid -= 1/24 * (np.diff(padded_grid[1:], axis=0) - np.diff(padded_grid[:-1], axis=0)).transpose(reversed_axes)
    return new_grid


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


# Converting (cell-/face-averaged) primitive variables w to (cell-/face-averaged) conservative variables q through a higher-order approx.
def convert_primitive(grid, sim_variables, _type="cell"):
    dimension, boundary, permutations = sim_variables.dimension, sim_variables.boundary, sim_variables.permutations
    w, q = np.copy(grid), np.zeros_like(grid)

    if _type == "face" or _type == "interface":
        for axes in permutations:
            reversed_axes = np.argsort(axes)
            for ax in range(1, dimension):
                padding = [(0,0)] * grid.ndim
                padding[ax] = (1,1)

                _w = np.pad(grid.transpose(axes), padding, mode=boundary)
                length = _w.shape[ax]
                w -= 1/24 * (np.diff(_w.take(range(1,length), axis=ax), axis=ax) - np.diff(_w.take(range(0,length-1), axis=ax), axis=ax)).transpose(reversed_axes)

                _q = point_convert_primitive(_w, sim_variables)
                q += 1/24 * (np.diff(_q.take(range(1,length), axis=ax), axis=ax) - np.diff(_q.take(range(0,length-1), axis=ax), axis=ax)).transpose(reversed_axes)
    else:
        for axes in permutations:
            reversed_axes = np.argsort(axes)  # Only necessary for 3D
            _w = add_boundary(grid.transpose(axes), boundary)
            w -= 1/24 * (np.diff(_w[1:], axis=0) - np.diff(_w[:-1], axis=0)).transpose(reversed_axes)

            _q = point_convert_primitive(_w, sim_variables)
            q += 1/24 * (np.diff(_q[1:], axis=0) - np.diff(_q[:-1], axis=0)).transpose(reversed_axes)
    return point_convert_primitive(w, sim_variables) + q


"""# Converting (cell-averaged) primitive variables w to (cell-averaged) conservative variables q through a higher-order approx.
def convert_primitive(grid, sim_variables):
    boundary, permutations = sim_variables.boundary, sim_variables.permutations
    w, q = np.copy(grid), np.zeros_like(grid)

    for axes in permutations:
        reversed_axes = np.argsort(axes)  # Only necessary for 3D
        _w = add_boundary(grid.transpose(axes), boundary)
        w -= 1/24 * (np.diff(_w[1:], axis=0) - np.diff(_w[:-1], axis=0)).transpose(reversed_axes)

        _q = point_convert_primitive(_w, sim_variables)
        q += 1/24 * (np.diff(_q[1:], axis=0) - np.diff(_q[:-1], axis=0)).transpose(reversed_axes)
    return point_convert_primitive(w, sim_variables) + q"""


# Converting (cell-/face-averaged) conservative variables q to (cell-/face-averaged) primitive variables q through a higher-order approx.
def convert_conservative(grid, sim_variables, _type="cell"):
    dimension, boundary, permutations = sim_variables.dimension, sim_variables.boundary, sim_variables.permutations
    w, q = np.zeros_like(grid), np.copy(grid)

    if _type == "face" or _type == "interface":
        for axes in permutations:
            reversed_axes = np.argsort(axes)
            for ax in range(1, dimension):
                padding = [(0,0)] * grid.ndim
                padding[ax] = (1,1)

                _q = np.pad(grid.transpose(axes), padding, mode=boundary)
                length = _q.shape[ax]
                q -= 1/24 * (np.diff(_q.take(range(1,length), axis=ax), axis=ax) - np.diff(_q.take(range(0,length-1), axis=ax), axis=ax)).transpose(reversed_axes)

                _w = point_convert_conservative(_q, sim_variables)
                w += 1/24 * (np.diff(_w.take(range(1,length), axis=ax), axis=ax) - np.diff(_w.take(range(0,length-1), axis=ax), axis=ax)).transpose(reversed_axes)
    else:
        for axes in permutations:
            reversed_axes = np.argsort(axes)  # Only necessary for 3D
            _q = add_boundary(grid.transpose(axes), boundary)
            q -= 1/24 * (np.diff(_q[1:], axis=0) - np.diff(_q[:-1], axis=0)).transpose(reversed_axes)

            _w = point_convert_conservative(_q, sim_variables)
            w += 1/24 * (np.diff(_w[1:], axis=0) - np.diff(_w[:-1], axis=0)).transpose(reversed_axes)
    return point_convert_conservative(q, sim_variables) + w


"""# Converting (cell-averaged) conservative variables q to (cell-averaged) primitive variables w through a higher-order approx.
def convert_conservative(grid, sim_variables):
    boundary, permutations = sim_variables.boundary, sim_variables.permutations
    w, q = np.zeros_like(grid), np.copy(grid)

    for axes in permutations:
        reversed_axes = np.argsort(axes)  # Only necessary for 3D
        _q = add_boundary(grid.transpose(axes), boundary)
        q -= 1/24 * (np.diff(_q[1:], axis=0) - np.diff(_q[:-1], axis=0)).transpose(reversed_axes)

        _w = point_convert_conservative(_q, sim_variables)
        w += 1/24 * (np.diff(_w[1:], axis=0) - np.diff(_w[:-1], axis=0)).transpose(reversed_axes)
    return point_convert_conservative(q, sim_variables) + w"""


# Compute the 4th-order interface-averaged fluxes from the interface-averaged fluxes via higher order approximation
def compute_high_approx_flux(cntr_flux, avg_flux, boundary):
    arr, _arr = np.copy(cntr_flux), np.copy(avg_flux)

    for ax in range(1, _arr.ndim-1):
        # Pad the orthogonal interface-averaged fluxes
        padding = [(0,0)] * _arr.ndim
        padding[ax] = (1,1)

        # Pad and expand the orthogonal interface-averaged fluxes
        padded_arr = np.pad(_arr, padding, boundary)
        length = padded_arr.shape[ax]

        # Subtract the Laplacian approximation of the interface-averaged fluxes from the interface-centred fluxes
        arr -= 1/24 * (np.diff(padded_arr.take(range(1,length), axis=ax), axis=ax) - np.diff(padded_arr.take(range(0,length-1), axis=ax), axis=ax))
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





















