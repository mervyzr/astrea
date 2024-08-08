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
    peakPos = (x[0]+x[-1])/2
    return params['y_offset'] + params['ampl']*np.exp(-((x-peakPos)**2)/params['fwhm'])


# Generic sin function
def sin_func(x, params):
    return params['y_offset'] + params['ampl']*np.sin(params['freq']*np.pi*x)


# Generic sinc function
def sinc_func(x, params):
    return params['y_offset'] + params['ampl']*np.sinc(x*params['freq']/np.pi)


# Add boundary conditions
def addBoundary(tube, boundary, stencil=1):
    arr = np.copy(tube)
    padding = [(0,0)] * tube.ndim
    padding[0] = (stencil,stencil)
    return np.pad(arr, padding, mode=boundary)


# Pointwise (exact) conversion of primitive variables w to conservative variables q (up to 2nd-order accurate)
def pointConvertPrimitive(tube, gamma):
    arr = np.copy(tube)
    rhos, vecs, pressures, Bfield = tube[...,0], tube[...,1:4], tube[...,4], tube[...,5:8]
    arr[...,4] = (pressures/(gamma-1)) + (.5*rhos*norm(vecs)**2) + (.5*norm(Bfield)**2)
    arr[...,1:4] = (vecs.T * rhos.T).T
    return arr


# Pointwise (exact) conversion of conservative variables q to primitive variables w (up to 2nd-order accurate)
def pointConvertConservative(tube, gamma):
    arr = np.copy(tube)
    rhos, energies, Bfield = tube[...,0], tube[...,4], tube[...,5:8]
    vecs = np.divide(tube[...,1:4].T, tube[...,0].T, out=np.zeros_like(tube[...,1:4].T), where=tube[...,0].T!=0).T
    arr[...,4] = (gamma-1) * (energies - (.5*rhos*norm(vecs)**2) - (.5*norm(Bfield)**2))
    arr[...,1:4] = vecs
    return arr


# Converting (cell-/face-averaged) primitive variables w to conservative variables q through a higher-order approx.
def convertPrimitive(tube, simVariables):
    gamma, boundary, permutations = simVariables.gamma, simVariables.boundary, simVariables.permutations
    w, q = np.copy(tube), np.zeros_like(tube)

    for axes in permutations:
        _w = addBoundary(tube.transpose(axes), boundary)
        w -= (np.diff(_w[1:], axis=0) - np.diff(_w[:-1], axis=0))/24

        _q = pointConvertPrimitive(_w, gamma)
        q += (np.diff(_q[1:], axis=0) - np.diff(_q[:-1], axis=0))/24
    return pointConvertPrimitive(w, gamma) + q


# Converting (cell-/face-averaged) conservative variables q to primitive variables w through a higher-order approx.
def convertConservative(tube, simVariables):
    gamma, boundary, permutations = simVariables.gamma, simVariables.boundary, simVariables.permutations
    w, q = np.zeros_like(tube), np.copy(tube)

    for axes in permutations:
        _q = addBoundary(tube.transpose(axes), boundary)
        q -= (np.diff(_q[1:], axis=0) - np.diff(_q[:-1], axis=0))/24

        _w = pointConvertConservative(_q, gamma)
        w += (np.diff(_w[1:], axis=0) - np.diff(_w[:-1], axis=0))/24
    return pointConvertConservative(q, gamma) + w