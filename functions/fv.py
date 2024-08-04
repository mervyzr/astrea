from itertools import permutations

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


# Initialise the discrete solution array with initial conditions and primitive variables w. Returns the solution array in conserved variables q
def initialise(simVariables, convert=False):
    config, N, gamma, dim, precision = simVariables.config, simVariables.cells, simVariables.gamma, simVariables.dim, simVariables.precision
    start, end, shock, params = simVariables.startPos, simVariables.endPos, simVariables.shockPos, simVariables.misc
    initialLeft, initialRight = simVariables.initialLeft, simVariables.initialRight

    _i = (N,) * dim
    _i += (len(initialRight),)
    arr = np.zeros(_i, dtype=precision)
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
            arr[...,0] = sin_func(xi, params)
        elif config == "sinc":
            arr[...,0] = sinc_func(xi, params)
        else:
            arr[...,0] = gauss_func(xi, params)
    
    if convert:
        return pointConvertPrimitive(arr, gamma)
    else:
        return arr


# Make boundary conditions
def makeBoundary(tube, boundary, stencil=1):
    arr = np.copy(tube)
    padding = [(0,0)] * tube.ndim
    padding[0] = (stencil,stencil)
    return np.pad(arr, padding, mode=boundary)


# Pointwise (exact) conversion of primitive variables w to conservative variables q (up to 2nd-order accurate)
def pointConvertPrimitive(tube, gamma):
    arr = np.copy(tube)
    rhos, vecs, pressures, Bfield = tube[...,0], tube[...,1:4], tube[...,4], tube[...,5:8]
    arr[...,4] = (pressures/(gamma-1)) + (.5*rhos*np.linalg.norm(vecs, axis=-1)**2) + (.5*np.linalg.norm(Bfield, axis=-1)**2)
    arr[...,1:4] = (vecs.T * rhos.T).T
    return arr


# Pointwise (exact) conversion of conservative variables q to primitive variables w (up to 2nd-order accurate)
def pointConvertConservative(tube, gamma):
    arr = np.copy(tube)
    rhos, energies, Bfield = tube[...,0], tube[...,4], tube[...,5:8]
    vecs = np.divide(tube[...,1:4].T, tube[...,0].T, out=np.zeros_like(tube[...,1:4].T), where=tube[...,0].T!=0).T
    arr[...,4] = (gamma-1) * (energies - (.5*rhos*np.linalg.norm(vecs, axis=-1)**2) - (.5*np.linalg.norm(Bfield, axis=-1)**2))
    arr[...,1:4] = vecs
    return arr


# Converting (cell-/face-averaged) primitive variables w to conservative variables q through a higher-order approx.
def convertPrimitive(tube, simVariables):
    w, q = np.copy(tube), np.zeros_like(tube)
    for axes in simVariables.permutations:
        _w = makeBoundary(tube.transpose(axes), simVariables.boundary)
        w -= (np.diff(_w[1:], axis=0) - np.diff(_w[:-1], axis=0))/24

        _q = pointConvertPrimitive(_w, simVariables.gamma)
        q += (np.diff(_q[1:], axis=0) - np.diff(_q[:-1], axis=0))/24
    return pointConvertPrimitive(w, simVariables.gamma) + q


# Converting (cell-/face-averaged) conservative variables q to primitive variables w through a higher-order approx.
def convertConservative(tube, simVariables):
    w, q = np.zeros_like(tube), np.copy(tube)
    for axes in simVariables.permutations:
        _q = makeBoundary(tube.transpose(axes), simVariables.boundary)
        q -= (np.diff(_q[1:], axis=0) - np.diff(_q[:-1], axis=0))/24

        _w = pointConvertConservative(_q, simVariables.gamma)
        w += (np.diff(_w[1:], axis=0) - np.diff(_w[:-1], axis=0))/24
    return pointConvertConservative(q, simVariables.gamma) + w


# Make flux as a function of cell-averaged (primitive) variables
def makeFluxTerm(tube, gamma):
    rhos, vecs, pressures, Bfield = tube[...,0], tube[...,1:4], tube[...,4], tube[...,5:8]
    arr = np.zeros_like(tube)

    arr[...,0] = rhos*vecs[...,0]
    arr[...,1] = rhos*(vecs[...,0]**2) + pressures + (.5*np.linalg.norm(Bfield, axis=-1)**2) - Bfield[...,0]**2
    arr[...,2] = rhos*vecs[...,0]*vecs[...,1] - Bfield[...,0]*Bfield[...,1]
    arr[...,3] = rhos*vecs[...,0]*vecs[...,2] - Bfield[...,0]*Bfield[...,2]
    arr[...,4] = (vecs[...,0]*((.5*rhos*np.linalg.norm(vecs, axis=-1)**2) + ((gamma*pressures)/(gamma-1)))) + (vecs[...,0]*np.linalg.norm(Bfield, axis=-1)**2) - (Bfield[...,0]*np.sum(Bfield*vecs, axis=-1))
    arr[...,6] = Bfield[...,1]*vecs[...,0] - Bfield[...,0]*vecs[...,1]
    arr[...,7] = Bfield[...,2]*vecs[...,0] - Bfield[...,0]*vecs[...,2]
    return arr


# Jacobian matrix based on primitive variables
def makeJacobian(tube, gamma):
    rho, vx, pressure, Bfield = tube[...,0], tube[...,1], tube[...,4], tube[...,5:8]/np.sqrt(4*np.pi)
    
    # Create empty square arrays for each cell
    _arr = np.zeros_like(tube)
    arr = np.repeat(_arr[..., np.newaxis], _arr.shape[-1], axis=-1)
    i, j = np.diag_indices(_arr.shape[-1])

    # Replace matrix with values
    arr[...,i,j] = vx[...,None]  # diagonal elements
    arr[...,0,1] = rho
    arr[...,1,4] = 1/rho
    arr[...,4,1] = gamma*pressure

    arr[...,1,6] = np.divide(Bfield[...,1], rho, out=np.zeros_like(Bfield[...,1]), where=rho!=0)
    arr[...,1,7] = np.divide(Bfield[...,2], rho, out=np.zeros_like(Bfield[...,2]), where=rho!=0)
    arr[...,2,6] = np.divide(-Bfield[...,0], rho, out=np.zeros_like(-Bfield[...,0]), where=rho!=0)
    arr[...,3,7] = np.divide(-Bfield[...,0], rho, out=np.zeros_like(-Bfield[...,0]), where=rho!=0)
    arr[...,6,1] = Bfield[...,1]
    arr[...,6,2] = -Bfield[...,0]
    arr[...,7,1] = Bfield[...,2]
    arr[...,7,3] = -Bfield[...,0]
    return arr


# Make the right eigenvector for adiabatic magnetohydrodynamics in Osher-Solomon flux
def makeOSRightEigenvectors(tubes, gamma):
    rhos, pressures, Bfields = tubes[...,0], tubes[...,4], tubes[...,5:8]/np.sqrt(4*np.pi)

    # Define the right eigenvectors for each cell in each tube
    _rightEigenvectors = np.zeros_like(tubes)
    rightEigenvectors = np.repeat(_rightEigenvectors[..., np.newaxis], _rightEigenvectors.shape[-1], axis=-1)

    # Define speed
    soundSpeed = np.sqrt(gamma * divide(pressures, rhos))
    alfvenSpeed = np.sqrt(divide(np.linalg.norm(Bfields, axis=2)**2, rhos))
    alfvenSpeedx = divide(Bfields[...,0], np.sqrt(rhos))

    fastMagnetosonicWave = .5 * (soundSpeed**2 + alfvenSpeed**2 + np.sqrt(((soundSpeed**2 + alfvenSpeed**2)**2) - (4*(soundSpeed**2)*(alfvenSpeedx**2))))
    slowMagnetosonicWave = .5 * (soundSpeed**2 + alfvenSpeed**2 - np.sqrt(((soundSpeed**2 + alfvenSpeed**2)**2) - (4*(soundSpeed**2)*(alfvenSpeedx**2))))

    # Define frequently used components
    S = np.sign(Bfields[...,0])
    alpha_f = np.ones_like(soundSpeed)
    alpha_s = np.zeros_like(soundSpeed)
    alpha_f[fastMagnetosonicWave != slowMagnetosonicWave] = (np.sqrt(divide(soundSpeed**2 - slowMagnetosonicWave**2, fastMagnetosonicWave**2 - slowMagnetosonicWave**2)))[fastMagnetosonicWave != slowMagnetosonicWave]
    alpha_s[fastMagnetosonicWave != slowMagnetosonicWave] = (np.sqrt(divide(fastMagnetosonicWave**2 - soundSpeed**2, fastMagnetosonicWave**2 - slowMagnetosonicWave**2)))[fastMagnetosonicWave != slowMagnetosonicWave]
    beta_y = divide(Bfields[...,1], np.sqrt(Bfields[...,1]**2 + Bfields[...,2]**2))
    beta_z = divide(Bfields[...,2], np.sqrt(Bfields[...,1]**2 + Bfields[...,2]**2))
    C_ff = fastMagnetosonicWave * alpha_f
    C_ss = slowMagnetosonicWave * alpha_s
    Q_f = C_ff * S
    Q_s = C_ss * S
    A_f = soundSpeed * alpha_f * np.sqrt(rhos)
    A_s = soundSpeed * alpha_s * np.sqrt(rhos)

    # Generate the right eigenvectors
    # First row
    rightEigenvectors[...,0,0] = rhos * alpha_f
    rightEigenvectors[...,0,2] = rhos * alpha_s
    rightEigenvectors[...,0,3] = 1
    rightEigenvectors[...,0,4] = 1
    rightEigenvectors[...,0,5] = rhos * alpha_s
    rightEigenvectors[...,0,7] = rhos * alpha_f
    # Second row
    rightEigenvectors[...,1,0] = -C_ff
    rightEigenvectors[...,1,2] = -C_ss
    rightEigenvectors[...,1,5] = C_ss
    rightEigenvectors[...,1,7] = C_ff
    # Third row
    rightEigenvectors[...,2,0] = Q_s * beta_y
    rightEigenvectors[...,2,1] = -beta_z
    rightEigenvectors[...,2,2] = -Q_f * beta_y
    rightEigenvectors[...,2,5] = Q_f * beta_y
    rightEigenvectors[...,2,6] = beta_z
    rightEigenvectors[...,2,7] = -Q_s * beta_y
    # Fourth row
    rightEigenvectors[...,3,0] = Q_s * beta_z
    rightEigenvectors[...,3,1] = beta_y
    rightEigenvectors[...,3,2] = -Q_f * beta_z
    rightEigenvectors[...,3,5] = Q_f * beta_z
    rightEigenvectors[...,3,6] = -beta_y
    rightEigenvectors[...,3,7] = -Q_s * beta_z
    # Fifth row
    rightEigenvectors[...,4,0] = rhos * alpha_f * soundSpeed**2
    rightEigenvectors[...,4,2] = rhos * alpha_s * soundSpeed**2
    rightEigenvectors[...,4,5] = rhos * alpha_s * soundSpeed**2
    rightEigenvectors[...,4,7] = rhos * alpha_f * soundSpeed**2
    # Seventh row
    rightEigenvectors[...,6,0] = A_s * beta_y
    rightEigenvectors[...,6,1] = -beta_z * S * np.sqrt(rhos)
    rightEigenvectors[...,6,2] = -A_f * beta_y
    rightEigenvectors[...,6,5] = -A_f * beta_y
    rightEigenvectors[...,6,6] = -beta_z * S * np.sqrt(rhos)
    rightEigenvectors[...,6,7] = A_s * beta_y
    # Eighth row
    rightEigenvectors[...,7,0] = A_s * beta_z
    rightEigenvectors[...,7,1] = -beta_y * S * np.sqrt(rhos)
    rightEigenvectors[...,7,2] = -A_f * beta_z
    rightEigenvectors[...,7,5] = -A_f * beta_z
    rightEigenvectors[...,7,6] = -beta_y * S * np.sqrt(rhos)
    rightEigenvectors[...,7,7] = A_s * beta_z

    return rightEigenvectors


# Make the right eigenvector for adiabatic magnetohydrodynamics in entropy-stable flux (primitive variables)
def makeESRightEigenvectors(tube, gamma):
    rhos, pressures, Bfields = tube[...,0], tube[...,4], tube[...,5:8]
    vx, vy, vz = tube[...,1], tube[...,2], tube[...,3]

    def divide(dividend, divisor):
        return np.divide(dividend, divisor, out=np.zeros_like(dividend), where=divisor!=0)

    # Define the right eigenvectors for each cell in each tube
    _rightEigenvectors = np.zeros_like(tube)
    rightEigenvectors = np.repeat(_rightEigenvectors[..., np.newaxis], _rightEigenvectors.shape[-1], axis=-1)

    # Define speeds
    soundSpeed = np.sqrt(gamma * divide(pressures, rhos))
    alfvenSpeed = np.sqrt(divide(np.linalg.norm(Bfields, axis=1)**2, rhos))
    alfvenSpeedx = divide(tube[...,5], np.sqrt(rhos))
    fastMagnetosonicWave = .5 * (soundSpeed**2 + alfvenSpeed**2 + np.sqrt(((soundSpeed**2 + alfvenSpeed**2)**2) - (4*(soundSpeed**2)*(alfvenSpeedx**2))))
    slowMagnetosonicWave = .5 * (soundSpeed**2 + alfvenSpeed**2 - np.sqrt(((soundSpeed**2 + alfvenSpeed**2)**2) - (4*(soundSpeed**2)*(alfvenSpeedx**2))))

    # Define frequently used components
    S = np.sign(tube[...,5])
    S[S == 0] = 1
    alpha_f = np.sqrt(divide(soundSpeed**2 - slowMagnetosonicWave**2, fastMagnetosonicWave**2 - slowMagnetosonicWave**2))
    alpha_s = np.sqrt(divide(fastMagnetosonicWave**2 - soundSpeed**2, fastMagnetosonicWave**2 - slowMagnetosonicWave**2))
    b_perpend = np.sqrt(divide(tube[...,6]**2 + tube[...,7]**2, rhos))
    beta2 = divide(tube[...,6], np.sqrt(tube[...,6]**2 + tube[...,7]**2))
    beta3 = divide(tube[...,7], np.sqrt(tube[...,6]**2 + tube[...,7]**2))

    psi_plus_slow = (
        .5*alpha_s*rhos*np.linalg.norm(tube[...,1:4], axis=1)**2
        - soundSpeed*alpha_f*rhos*b_perpend
        + (alpha_s*rhos*soundSpeed**2)/(gamma-1)
        + alpha_s*slowMagnetosonicWave*rhos*vx
        + alpha_f*fastMagnetosonicWave*rhos*S*(vy*beta2 + vz*beta3)
        )
    psi_minus_slow = (
        .5*alpha_s*rhos*np.linalg.norm(tube[...,1:4], axis=1)**2
        - soundSpeed*alpha_f*rhos*b_perpend
        + (alpha_s*rhos*soundSpeed**2)/(gamma-1)
        - alpha_s*slowMagnetosonicWave*rhos*vx
        - alpha_f*fastMagnetosonicWave*rhos*S*(vy*beta2 + vz*beta3)
        )
    psi_plus_fast = (
        .5*alpha_f*rhos*np.linalg.norm(tube[...,1:4], axis=1)**2
        + soundSpeed*alpha_s*rhos*b_perpend
        + (alpha_f*rhos*soundSpeed**2)/(gamma-1)
        + alpha_f*fastMagnetosonicWave*rhos*vx
        - alpha_s*slowMagnetosonicWave*rhos*S*(vy*beta2 + vz*beta3)
        )
    psi_minus_fast = (
        .5*alpha_f*rhos*np.linalg.norm(tube[...,1:4], axis=1)**2
        + soundSpeed*alpha_s*rhos*b_perpend
        + (alpha_f*rhos*soundSpeed**2)/(gamma-1)
        - alpha_f*fastMagnetosonicWave*rhos*vx
        + alpha_s*slowMagnetosonicWave*rhos*S*(vy*beta2 + vz*beta3)
        )

    # Generate the right eigenvectors
    # First column (Fast+ magnetoacoustic wave)
    rightEigenvectors[...,0,0] = rhos * alpha_f
    rightEigenvectors[...,1,0] = rhos * alpha_f * (vx + fastMagnetosonicWave)
    rightEigenvectors[...,2,0] = rhos * (alpha_f*vy - alpha_s*slowMagnetosonicWave*beta2*S)
    rightEigenvectors[...,3,0] = rhos * (alpha_f*vz - alpha_s*slowMagnetosonicWave*beta3*S)
    rightEigenvectors[...,4,0] = psi_plus_fast
    rightEigenvectors[...,6,0] = alpha_s * soundSpeed * beta2 * np.sqrt(rhos)
    rightEigenvectors[...,7,0] = alpha_s * soundSpeed * beta3 * np.sqrt(rhos)
    # Second column (Alfven+ wave)
    rightEigenvectors[...,2,1] = beta3 * rhos**1.5
    rightEigenvectors[...,3,1] = -beta2 * rhos**1.5
    rightEigenvectors[...,4,1] = (beta3*vy - beta2*vz) * rhos**1.5
    rightEigenvectors[...,6,1] = -rhos * beta3
    rightEigenvectors[...,7,1] = rhos * beta2
    # Third column (Slow+ magnetoacoustic wave)
    rightEigenvectors[...,0,2] = rhos * alpha_s
    rightEigenvectors[...,1,2] = rhos * alpha_s * (vx + slowMagnetosonicWave)
    rightEigenvectors[...,2,2] = rhos * (alpha_s*vy + alpha_f*fastMagnetosonicWave*beta2*S)
    rightEigenvectors[...,3,2] = rhos * (alpha_s*vz + alpha_f*fastMagnetosonicWave*beta3*S)
    rightEigenvectors[...,4,2] = psi_plus_slow
    rightEigenvectors[...,6,2] = -alpha_f * soundSpeed * beta2 * np.sqrt(rhos)
    rightEigenvectors[...,7,2] = -alpha_f * soundSpeed * beta3 * np.sqrt(rhos)
    # Fourth column (Entropy wave)
    rightEigenvectors[...,0,3] = 1
    rightEigenvectors[...,1,3] = vx
    rightEigenvectors[...,2,3] = vy
    rightEigenvectors[...,3,3] = vz
    rightEigenvectors[...,4,3] = .5 * np.linalg.norm(tube[...,1:4], axis=1)**2
    # Fifth column (Divergence wave)
    rightEigenvectors[...,4,4] = tube[...,5]
    rightEigenvectors[...,5,4] = 1
    # Sixth column (Slow- magnetoacoustic wave)
    rightEigenvectors[...,0,5] = rhos * alpha_s
    rightEigenvectors[...,1,5] = rhos * alpha_s * (vx - slowMagnetosonicWave)
    rightEigenvectors[...,2,5] = rhos * (alpha_s*vy - alpha_f*fastMagnetosonicWave*beta2*S)
    rightEigenvectors[...,3,5] = rhos * (alpha_s*vz - alpha_f*fastMagnetosonicWave*beta3*S)
    rightEigenvectors[...,4,5] = psi_minus_slow
    rightEigenvectors[...,6,5] = -alpha_f * soundSpeed * beta2 * np.sqrt(rhos)
    rightEigenvectors[...,7,5] = -alpha_f * soundSpeed * beta3 * np.sqrt(rhos)
    # Seventh column (Alfven- wave)
    rightEigenvectors[...,2,6] = -beta3 * rhos**1.5
    rightEigenvectors[...,3,6] = beta2 * rhos**1.5
    rightEigenvectors[...,4,6] = (beta2*vz - beta3*vy) * rhos**1.5
    rightEigenvectors[...,6,6] = -rhos * beta3
    rightEigenvectors[...,7,6] = rhos * beta2
    # Eighth column (Fast- magnetoacoustic wave)
    rightEigenvectors[...,0,7] = rhos * alpha_f
    rightEigenvectors[...,1,7] = rhos * alpha_f * (vx - fastMagnetosonicWave)
    rightEigenvectors[...,2,7] = rhos * (alpha_f*vy + alpha_s*slowMagnetosonicWave*beta2*S)
    rightEigenvectors[...,3,7] = rhos * (alpha_f*vz + alpha_s*slowMagnetosonicWave*beta3*S)
    rightEigenvectors[...,4,7] = psi_minus_fast
    rightEigenvectors[...,6,7] = alpha_s * soundSpeed * beta2 * np.sqrt(rhos)
    rightEigenvectors[...,7,7] = alpha_s * soundSpeed * beta3 * np.sqrt(rhos)

    # Scale the right eigenvectors with a diagonal scaling matrix, so as to prevent degeneracies [Barth, 1999]
    diag_scaler = np.zeros_like(rightEigenvectors)
    diag_scaler[...,0,0] = 1/(2*gamma*rhos)
    diag_scaler[...,1,1] = divide(pressures, 2*rhos**2)
    diag_scaler[...,2,2] = 1/(2*gamma*rhos)
    diag_scaler[...,3,3] = (rhos*(gamma-1))/gamma
    diag_scaler[...,4,4] = divide(pressures, rhos)
    diag_scaler[...,5,5] = 1/(2*gamma*rhos)
    diag_scaler[...,6,6] = divide(pressures, 2*rhos**2)
    diag_scaler[...,7,7] = 1/(2*gamma*rhos)
    R_dot = rightEigenvectors @ np.sqrt(diag_scaler)

    return R_dot