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

    arr[...,1,6] = Bfield[...,1]/rho
    arr[...,1,7] = Bfield[...,2]/rho
    arr[...,2,6] = -Bfield[...,0]/rho
    arr[...,3,7] = -Bfield[...,0]/rho
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


# Entropy-stable flux calculation based on left and right interpolated primitive variables [Derigs et al., 2016]
def makeEntropyFlux(wS, gamma):
    wLs, wRs = wS
    arr = np.zeros_like(wLs)

    def divide(dividend, divisor):
        return np.divide(dividend, divisor, out=np.zeros_like(dividend), where=divisor!=0)

    # Compute the jump term
    def jump(term):
        return term[1] - term[0]

    # Compute arithmetic mean
    def arith_mean(term):
        return .5 * (term[0] + term[1])

    # Stable numerical procedure for computing logarithmic mean [Ismail & Roe, 2009]
    def lon(term):
        L, R = term
        zeta = np.divide(L, R, out=np.zeros_like(L), where=R!=0)
        f = np.divide(zeta-1, zeta+1, out=np.zeros_like(zeta), where=(zeta+1)!=0)
        u = f*f

        if (u < 1e-2).any():
            F = 1 + u/3 + u*u/5 + u*u*u/7
        else:
            F = np.log(zeta)/2/f
        return (L+R)/(2*F)

    # Define frequently used terms
    z1 = np.array([np.sqrt(divide(wLs[:,0], wLs[:,4])), np.sqrt(divide(wRs[:,0], wRs[:,4]))])
    z5 = np.array([np.sqrt(wLs[:,0]*wLs[:,4]), np.sqrt(wRs[:,0]*wRs[:,4])])
    vx, vy, vz = np.array([wLs[:,1], wRs[:,1]]), np.array([wLs[:,2], wRs[:,2]]), np.array([wLs[:,3], wRs[:,3]])

    # Compute the averages
    rho_hat = arith_mean(z1) * lon(z5)
    P1_hat = divide(arith_mean(z5), arith_mean(z1))
    P2_hat = ((gamma+1)/(2*gamma))*(divide(lon(z5), lon(z1))) + ((gamma-1)/(2*gamma))*(divide(arith_mean(z5), arith_mean(z1)))
    u1_hat = divide(arith_mean((vx.T*z1.T).T), arith_mean(z1))
    v1_hat = divide(arith_mean((vy.T*z1.T).T), arith_mean(z1))
    w1_hat = divide(arith_mean((vz.T*z1.T).T), arith_mean(z1))
    u2_hat = divide(arith_mean((vx.T*(z1**2).T).T), arith_mean(z1**2))
    v2_hat = divide(arith_mean((vy.T*(z1**2).T).T), arith_mean(z1**2))
    w2_hat = divide(arith_mean((vz.T*(z1**2).T).T), arith_mean(z1**2))
    B1_hat = arith_mean(np.array([wLs[:,5], wRs[:,5]]))
    B1_dot = arith_mean(np.array([wLs[:,5]**2, wRs[:,5]**2]))
    B2_hat = arith_mean(np.array([wLs[:,6], wRs[:,6]]))
    B2_dot = arith_mean(np.array([wLs[:,6]**2, wRs[:,6]**2]))
    B3_hat = arith_mean(np.array([wLs[:,7], wRs[:,7]]))
    B3_dot = arith_mean(np.array([wLs[:,7]**2, wRs[:,7]**2]))
    B1B2 = arith_mean(np.array([wLs[:,5]*wLs[:,6], wRs[:,5]*wRs[:,6]]))
    B1B3 = arith_mean(np.array([wLs[:,5]*wLs[:,7], wRs[:,5]*wRs[:,7]]))

    # Update the entropy-conserving flux vector; suitable for smooth solutions [Winters & Gassner, 2015]
    arr[:,0] = rho_hat * u1_hat
    arr[:,1] = P1_hat + rho_hat*u1_hat**2 + .5*(B1_dot+B2_dot+B3_dot) - B1_dot
    arr[:,2] = rho_hat*u1_hat*v1_hat - B1B2
    arr[:,3] = rho_hat*u1_hat*w1_hat - B1B3
    arr[:,4] = (gamma/(gamma-1))*u1_hat*P2_hat + .5*rho_hat*u1_hat*(u1_hat**2 + v1_hat**2 + w1_hat**2) + u2_hat*(B2_hat**2 + B3_hat**2) - B1_hat*(v2_hat*B2_hat + w2_hat*B3_hat)
    arr[:,6] = u2_hat*B2_hat - v2_hat*B1_hat
    arr[:,7] = u2_hat*B3_hat - w2_hat*B1_hat

    return arr


# Calculate the entropy vector (jump between the left and right states)
def getEntropyVector(w, g):
    arr = np.copy(w)
    factor = w[:,0]/w[:,4]

    arr[:,0] = ((g-np.log(w[:,4]*w[:,0]**-g))/(g-1)) - (.5*w[:,0]*np.linalg.norm(w[:,1:4], axis=1)**2)/w[:,4]
    arr[:,1:4] = (w[:,1:4].T * factor).T
    arr[:,4] = -factor
    arr[:,5:8] = (w[:,5:8].T * factor).T
    return arr