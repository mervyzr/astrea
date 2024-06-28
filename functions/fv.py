import numpy as np

##############################################################################

# For handling division-by-zero warnings during array divisions
def divide(dividend, divisor):
    return np.divide(dividend, divisor, out=np.zeros_like(dividend), where=divisor!=0)


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


# Initialise the discrete solution array with initial conditions and primitive variables w
# Returns the solution array in conserved variables q
def initialise(simVariables):
    config, N, gamma, precision = simVariables.config, simVariables.cells, simVariables.gamma, simVariables.precision
    start, end, shock, params = simVariables.startPos, simVariables.endPos, simVariables.shockPos, simVariables.misc
    initialLeft, initialRight = simVariables.initialLeft, simVariables.initialRight

    arr = np.zeros((N, len(initialRight)), dtype=precision)
    arr[:] = initialRight

    midpoint = (end+start)/2
    if config == "sedov" or config.startswith('sq'):
        half_width = int(N/2 * ((shock-midpoint)/(end-midpoint)))
        left_edge, right_edge = int(N/2-half_width), int(N/2+half_width)
        arr[left_edge:right_edge] = initialLeft
    else:
        split_point = int(N * ((shock-start)/(end-start)))
        arr[:split_point] = initialLeft

    if "shu" in config or "osher" in config:
        xi = np.linspace(shock, end, N-split_point)
        arr[split_point:,0] = sin_func(xi, params)
    elif config == "sin" or config == "sinc" or config.startswith('gauss'):
        xi = np.linspace(start, end, N)
        if config == "sin":
            arr[:,0] = sin_func(xi, params)
        elif config == "sinc":
            arr[:,0] = sinc_func(xi, params)
        else:
            arr[:,0] = gauss_func(xi, params)

    return pointConvertPrimitive(arr, gamma)


# Make boundary conditions
def makeBoundary(tube, boundary, stencil=1):
    arr = np.copy(tube)
    return np.pad(arr, [(stencil,stencil), (0,0)], mode=boundary)


# Pointwise (exact) conversion of primitive variables w to conservative variables q (up to 2nd-order accurate)
def pointConvertPrimitive(tube, gamma):
    arr = np.copy(tube)
    rhos, vecs, pressures, Bfield = tube[:,0], tube[:,1:4], tube[:,4], tube[:,5:8]
    arr[:,4] = (pressures/(gamma-1)) + (.5*rhos*np.linalg.norm(vecs, axis=1)**2) + (.5*np.linalg.norm(Bfield, axis=1)**2)
    arr[:,1:4] = (vecs.T * rhos).T
    arr[:,5:8] = Bfield
    return arr


# Pointwise (exact) conversion of conservative variables q to primitive variables w (up to 2nd-order accurate)
def pointConvertConservative(tube, gamma):
    arr = np.copy(tube)
    rhos, vecs, energies, Bfield = tube[:,0], (tube[:,1:4].T / tube[:,0]).T, tube[:,4], tube[:,5:8]
    arr[:,4] = (gamma-1) * (energies - (.5*rhos*np.linalg.norm(vecs, axis=1)**2) - (.5*np.linalg.norm(Bfield, axis=1)**2))
    arr[:,1:4] = vecs
    arr[:,5:8] = Bfield
    return arr


# Converting (cell-/face-averaged) primitive variables w to conservative variables q through a higher-order approx.
def convertPrimitive(tube, gamma, boundary):
    arr = makeBoundary(tube, boundary)
    w = tube - (np.diff(arr[1:], axis=0) - np.diff(arr[:-1], axis=0))/24  # 2nd-order Taylor expansion (Laplacian)
    q = pointConvertPrimitive(arr, gamma)
    return pointConvertPrimitive(w, gamma) + (np.diff(q[1:], axis=0) - np.diff(q[:-1], axis=0))/24


# Converting (cell-/face-averaged) conservative variables q to primitive variables w through a higher-order approx.
def convertConservative(tube, gamma, boundary):
    arr = makeBoundary(tube, boundary)
    q = tube - (np.diff(arr[1:], axis=0) - np.diff(arr[:-1], axis=0))/24  # 2nd-order Taylor expansion (Laplacian)
    w = pointConvertConservative(arr, gamma)
    return pointConvertConservative(q, gamma) + (np.diff(w[1:], axis=0) - np.diff(w[:-1], axis=0))/24


# Make flux as a function of cell-averaged (primitive) variables
def makeFlux(tube, gamma):
    rhos, vecs, pressures, Bfield = tube[:,0], tube[:,1:4], tube[:,4], tube[:,5:8]
    arr = np.zeros_like(tube)

    arr[:,0] = rhos*vecs[:,0]
    arr[:,1] = rhos*(vecs[:,0]**2) + pressures + (.5*np.linalg.norm(Bfield, axis=1)**2) - Bfield[:,0]**2
    arr[:,2] = rhos*vecs[:,0]*vecs[:,1] - Bfield[:,0]*Bfield[:,1]
    arr[:,3] = rhos*vecs[:,0]*vecs[:,2] - Bfield[:,0]*Bfield[:,2]
    arr[:,4] = (vecs[:,0]*((.5*rhos*np.linalg.norm(vecs, axis=1)**2) + ((gamma*pressures)/(gamma-1)))) + (vecs[:,0]*np.linalg.norm(Bfield, axis=1)**2) - (Bfield[:,0]*np.sum(Bfield*vecs, axis=1))
    arr[:,6] = Bfield[:,1]*vecs[:,0] - Bfield[:,0]*vecs[:,1]
    arr[:,7] = Bfield[:,2]*vecs[:,0] - Bfield[:,0]*vecs[:,2]
    return arr


"""# Entropy-stable flux calculation based on left and right interpolated primitive variables [Winters & Gassner, 2015]
def makeFlux(interpolatedValues, gamma):
    wL, wR = interpolatedValues
    arr = np.zeros_like(wL)

    def make_z(w):
        _arr = np.copy(w)
        x = np.sqrt(divide(w[:,0], w[:,4]))

        _arr[:,0] = x
        _arr[:,1:4] = (x*w[:,1:4].T).T
        _arr[:,4] = np.sqrt(w[:,0]*w[:,4])

        return _arr

    z_wL, z_wR = make_z(wL), make_z(wR)
    avg_z = .5 * (z_wL + z_wR)
    ln_z = divide((z_wL - z_wR), (np.log(z_wL, out=np.zeros_like(z_wL), where=z_wL!=0) - np.log(z_wR, out=np.zeros_like(z_wR), where=z_wR!=0)))

    rho_hat = avg_z[:,0]*ln_z[:,4]
    P1_hat = divide(avg_z[:,4], avg_z[:,0])
    P2_hat = ((gamma+1)/(2*gamma)) * divide(ln_z[:,4], ln_z[:,0]) + ((gamma-1)/(2*gamma)) * divide(avg_z[:,4], avg_z[:,0])
    vx_hat, vy_hat, vz_hat = divide(avg_z[:,1], avg_z[:,0]), divide(avg_z[:,2], avg_z[:,0]), divide(avg_z[:,3], avg_z[:,0])
    vx_dot = divide((z_wL[:,0]*z_wL[:,1] + z_wR[:,0]*z_wR[:,1]), (z_wL[:,0]**2 + z_wR[:,0]**2))
    vy_dot = divide((z_wL[:,0]*z_wL[:,2] + z_wR[:,0]*z_wR[:,2]), (z_wL[:,0]**2 + z_wR[:,0]**2))
    vz_dot = divide((z_wL[:,0]*z_wL[:,3] + z_wR[:,0]*z_wR[:,3]), (z_wL[:,0]**2 + z_wR[:,0]**2))
    Bx_hat, By_hat, Bz_hat = avg_z[:,5], avg_z[:,6], avg_z[:,7]
    Bx_dot, By_dot, Bz_dot = .5 * (z_wL[:,5]**2+z_wR[:,5]**2), .5 * (z_wL[:,6]**2+z_wR[:,6]**2), .5 * (z_wL[:,7]**2+z_wR[:,7]**2)
    BxBy, BxBz = .5 * (z_wL[:,5]*z_wL[:,6] + z_wR[:,5]*z_wR[:,6]), .5 * (z_wL[:,5]*z_wL[:,7] + z_wR[:,5]*z_wR[:,7])

    arr[:,0] = rho_hat*vx_hat
    arr[:,1] = P1_hat + rho_hat*vx_hat**2 - Bx_dot + .5*(Bx_dot+By_dot+Bz_dot)
    arr[:,2] = rho_hat*vx_hat*vy_hat - BxBy
    arr[:,3] = rho_hat*vx_hat*vz_hat - BxBz
    arr[:,4] = (gamma/(gamma-1))*vx_hat*P2_hat + .5*rho_hat*vx_hat*(vx_hat**2 + (vy_hat**2)*(vz_hat**2)) + vx_dot*(By_hat**2 + Bz_hat**2) - Bx_hat*(vy_dot*By_hat + vz_dot*Bz_hat)
    arr[:,6] = vx_dot*By_hat - vy_dot*Bx_hat
    arr[:,7] = vx_dot*Bz_hat - vz_dot*Bx_hat
    return arr"""


# Jacobian matrix based on primitive variables
def makeJacobian(tube, gamma):
    rho, vx, pressure, Bfield = tube[:,0], tube[:,1], tube[:,4], tube[:,5:8]
    gridLength, variables = len(tube), len(tube[0])

    # Create empty square arrays for each cell
    arr = np.zeros((gridLength, variables, variables))
    i,j = np.diag_indices(variables)

    # Replace matrix with values
    arr[:,i,j] = vx[:,None]  # diagonal elements
    arr[:,0,1] = rho
    arr[:,1,4] = 1/rho
    arr[:,4,1] = gamma*pressure

    arr[:,1,6] = Bfield[:,1]/rho
    arr[:,1,7] = Bfield[:,2]/rho
    arr[:,2,6] = -Bfield[:,0]/rho
    arr[:,3,7] = -Bfield[:,0]/rho
    arr[:,6,1] = Bfield[:,1]
    arr[:,6,2] = -Bfield[:,0]
    arr[:,7,1] = Bfield[:,2]
    arr[:,7,3] = -Bfield[:,0]
    return arr