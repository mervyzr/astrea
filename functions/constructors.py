import numpy as np

from functions import fv

##############################################################################
# Functions for constructing objects such as the grid, eigenvectors, Jacobian and flux terms
##############################################################################

# Initialise the discrete solution array with initial conditions and primitive variables w. Returns the solution array in conserved variables q
def initialise(simVariables):
    config, N, gamma, dim, precision = simVariables.config, simVariables.cells, simVariables.gamma, simVariables.dim, simVariables.precision
    start, end, shock, params = simVariables.startPos, simVariables.endPos, simVariables.shockPos, simVariables.misc
    initialLeft, initialRight = simVariables.initialLeft, simVariables.initialRight

    _i = (N,) * int(dim)
    _i += (len(initialRight),)
    arr = np.zeros(_i, dtype=precision)
    arr[:] = initialRight

    midpoint = (end+start)/2

    if dim >= 2:
        x = y = np.arange(N)
        cx = cy = int(N/2)

        if config == "sedov":
            r = int(N/2 * ((shock-midpoint)/(end-midpoint)))

            mask = (x[np.newaxis,:]-cx)**2 + (y[:,np.newaxis]-cy)**2 < r**2
            arr[mask] = initialLeft
        elif config.startswith("gauss"):
            pass

    else:
        if config == "sedov" or config.startswith('sq'):
            half_width = int(N/2 * ((shock-midpoint)/(end-midpoint)))
            left_edge, right_edge = int(N/2-half_width), int(N/2+half_width)
            arr[left_edge:right_edge] = initialLeft
        else:
            split_point = int(N * ((shock-start)/(end-start)))
            arr[:split_point] = initialLeft

        if "shu" in config or "osher" in config:
            xi = np.linspace(shock, end, N-split_point)
            arr[split_point:,0] = fv.sin_func(xi, params)
        elif config == "sin" or config == "sinc" or config.startswith('gauss'):
            xi = np.linspace(start, end, N)
            if config == "sin":
                arr[...,0] = fv.sin_func(xi, params)
            elif config == "sinc":
                arr[...,0] = fv.sinc_func(xi, params)
            else:
                arr[...,0] = fv.gauss_func(xi, params)
        
        if dim != 1:
            layer = 1
            _arr = np.pad(arr, ((int(N*layer),int(N*layer)),(0,0)), mode="constant")
            _arr = _arr.reshape(2*layer+1,N,len(initialRight))
            arr = _arr.transpose(1,0,2)

    return fv.pointConvertPrimitive(arr, gamma)


# Make flux as a function of cell-averaged (primitive) variables
def makeFluxTerm(tube, gamma, axis):
    axis %= 3
    rhos, vecs, pressures, Bfields = tube[...,0], tube[...,1:4], tube[...,4], tube[...,5:8]
    arr = np.zeros_like(tube)

    arr[...,0] = rhos*vecs[...,axis]
    arr[...,axis+1] = rhos*(vecs[...,axis]**2) + pressures + (.5*fv.norm(Bfields)**2) - Bfields[...,axis]**2
    arr[...,(axis+1)%3+1] = rhos*vecs[...,axis]*vecs[...,(axis+1)%3] - Bfields[...,axis]*Bfields[...,(axis+1)%3]
    arr[...,(axis+2)%3+1] = rhos*vecs[...,axis]*vecs[...,(axis+2)%3] - Bfields[...,axis]*Bfields[...,(axis+2)%3]
    arr[...,4] = (vecs[...,axis] * ((.5*rhos*fv.norm(vecs)**2) + ((gamma*pressures)/(gamma-1)) + (fv.norm(Bfields)**2))) - (Bfields[...,axis]*np.sum(Bfields*vecs, axis=-1))
    arr[...,(axis+1)%3+5] = Bfields[...,(axis+1)%3]*vecs[...,axis] - Bfields[...,axis]*vecs[...,(axis+1)%3]
    arr[...,(axis+2)%3+5] = Bfields[...,(axis+2)%3]*vecs[...,axis] - Bfields[...,axis]*vecs[...,(axis+2)%3]
    return arr


# Jacobian matrix based on primitive variables
def makeJacobian(tube, gamma, axis):
    axis %= 3
    rho, v, pressure, Bfields = tube[...,0], tube[...,axis+1], tube[...,4], tube[...,5:8]/np.sqrt(4*np.pi)
    
    # Create empty square arrays for each cell
    _arr = np.zeros_like(tube)
    arr = np.repeat(_arr[..., np.newaxis], _arr.shape[-1], axis=-1)
    i, j = np.diag_indices(_arr.shape[-1])

    # Replace matrix with values
    arr[...,i,j] = v[...,None]  # diagonal elements
    arr[...,0,axis+1] = rho
    arr[...,axis+1,4] = 1/rho
    arr[...,4,axis+1] = gamma*pressure

    arr[...,axis+1,(axis+1)%3+5] = fv.divide(Bfields[...,(axis+1)%3], rho)
    arr[...,axis+1,(axis+2)%3+5] = fv.divide(Bfields[...,(axis+2)%3], rho)
    arr[...,(axis+1)%3+1,(axis+1)%3+5] = fv.divide(-Bfields[...,axis], rho)
    arr[...,(axis+2)%3+1,(axis+2)%3+5] = fv.divide(-Bfields[...,axis], rho)
    arr[...,(axis+1)%3+5,axis+1] = Bfields[...,(axis+1)%3]
    arr[...,(axis+1)%3+5,(axis+1)%3+1] = -Bfields[...,axis]
    arr[...,(axis+2)%3+5,axis+1] = Bfields[...,(axis+2)%3]
    arr[...,(axis+2)%3+5,(axis+2)%3+1] = -Bfields[...,axis]
    return arr


# Calculate the Roe-averaged primitive variables from the left- & right-interface states for use in Roe solver in order to better capture shocks [Brio & Wu, 1988; LeVeque, 2002; Stone et al., 2008]
def makeRoeAverage(wS, qS, gamma):
    wL, wR = wS
    qL, qR = qS

    avg = np.zeros_like(wL)
    rho_L, rho_R = np.sqrt(wL[...,0]), np.sqrt(wR[...,0])

    avg[...,0] = rho_L * rho_R
    avg[...,1:4] = fv.divide((rho_L.T * wL[...,1:4].T) + (rho_R.T * wR[...,1:4].T), (rho_L + rho_R).T).T
    avg[...,6:8] = fv.divide((rho_R.T * wL[...,6:8].T) + (rho_L.T * wR[...,6:8].T), (rho_L + rho_R).T).T

    H_L, H_R = fv.divide(qL[...,4] + wL[...,4] + .5*fv.norm(wL[...,5:8])**2, wL[...,0]), fv.divide(qR[...,4] + wR[...,4] + .5*fv.norm(wR[...,5:8])**2, wR[...,0])
    H = fv.divide(rho_L*H_L + rho_R*H_R, rho_L + rho_R)
    avg[...,4] = ((gamma-1)/gamma) * (avg[...,0]*H - .5*(avg[...,0]*fv.norm(avg[...,1:4])**2 + fv.norm(avg[...,5:8])**2))

    return avg


# Make the right eigenvector for adiabatic magnetohydrodynamics in Osher-Solomon flux
def makeOSRightEigenvectors(tubes, gamma):
    rhos, pressures, Bfields = tubes[...,0], tubes[...,4], tubes[...,5:8]/np.sqrt(4*np.pi)

    # Define the right eigenvectors for each cell in each tube
    _rightEigenvectors = np.zeros_like(tubes)
    rightEigenvectors = np.repeat(_rightEigenvectors[..., np.newaxis], _rightEigenvectors.shape[-1], axis=-1)

    # Define speed
    soundSpeed = np.sqrt(gamma * fv.divide(pressures, rhos))
    alfvenSpeed = np.sqrt(fv.divide(fv.norm(Bfields)**2, rhos))
    alfvenSpeedx = fv.divide(Bfields[...,0], np.sqrt(rhos))

    fastMagnetosonicWave = .5 * (soundSpeed**2 + alfvenSpeed**2 + np.sqrt(((soundSpeed**2 + alfvenSpeed**2)**2) - (4*(soundSpeed**2)*(alfvenSpeedx**2))))
    slowMagnetosonicWave = .5 * (soundSpeed**2 + alfvenSpeed**2 - np.sqrt(((soundSpeed**2 + alfvenSpeed**2)**2) - (4*(soundSpeed**2)*(alfvenSpeedx**2))))

    # Define frequently used components
    S = np.sign(Bfields[...,0])
    alpha_f = np.ones_like(soundSpeed)
    alpha_s = np.zeros_like(soundSpeed)
    alpha_f[fastMagnetosonicWave != slowMagnetosonicWave] = (np.sqrt(fv.divide(soundSpeed**2 - slowMagnetosonicWave**2, fastMagnetosonicWave**2 - slowMagnetosonicWave**2)))[fastMagnetosonicWave != slowMagnetosonicWave]
    alpha_s[fastMagnetosonicWave != slowMagnetosonicWave] = (np.sqrt(fv.divide(fastMagnetosonicWave**2 - soundSpeed**2, fastMagnetosonicWave**2 - slowMagnetosonicWave**2)))[fastMagnetosonicWave != slowMagnetosonicWave]
    beta_y = fv.divide(Bfields[...,1], np.sqrt(Bfields[...,1]**2 + Bfields[...,2]**2))
    beta_z = fv.divide(Bfields[...,2], np.sqrt(Bfields[...,1]**2 + Bfields[...,2]**2))
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
    rhos, vs, pressures, Bfields = tube[...,0], tube[...,1:4], tube[...,4], tube[...,5:8]
    vx, vy, vz = tube[...,1], tube[...,2], tube[...,3]

    # Define the right eigenvectors for each cell in each tube
    _rightEigenvectors = np.zeros_like(tube)
    rightEigenvectors = np.repeat(_rightEigenvectors[..., np.newaxis], _rightEigenvectors.shape[-1], axis=-1)

    # Define speeds
    soundSpeed = np.sqrt(gamma * fv.divide(pressures, rhos))
    alfvenSpeed = np.sqrt(fv.divide(fv.norm(Bfields)**2, rhos))
    alfvenSpeedx = fv.divide(tube[...,5], np.sqrt(rhos))
    fastMagnetosonicWave = .5 * (soundSpeed**2 + alfvenSpeed**2 + np.sqrt(((soundSpeed**2 + alfvenSpeed**2)**2) - (4*(soundSpeed**2)*(alfvenSpeedx**2))))
    slowMagnetosonicWave = .5 * (soundSpeed**2 + alfvenSpeed**2 - np.sqrt(((soundSpeed**2 + alfvenSpeed**2)**2) - (4*(soundSpeed**2)*(alfvenSpeedx**2))))

    # Define frequently used components
    S = np.sign(tube[...,5])
    S[S == 0] = 1
    alpha_f = np.sqrt(fv.divide(soundSpeed**2 - slowMagnetosonicWave**2, fastMagnetosonicWave**2 - slowMagnetosonicWave**2))
    alpha_s = np.sqrt(fv.divide(fastMagnetosonicWave**2 - soundSpeed**2, fastMagnetosonicWave**2 - slowMagnetosonicWave**2))
    b_perpend = np.sqrt(fv.divide(tube[...,6]**2 + tube[...,7]**2, rhos))
    beta2 = fv.divide(tube[...,6], np.sqrt(tube[...,6]**2 + tube[...,7]**2))
    beta3 = fv.divide(tube[...,7], np.sqrt(tube[...,6]**2 + tube[...,7]**2))

    psi_plus_slow = (
        .5 * alpha_s * rhos * fv.norm(vs)**2
        - soundSpeed * alpha_f * rhos * b_perpend
        + (alpha_s * rhos * soundSpeed**2)/(gamma - 1)
        + alpha_s * slowMagnetosonicWave * rhos * vx
        + alpha_f * fastMagnetosonicWave * rhos * S * (vy*beta2 + vz*beta3)
        )
    psi_minus_slow = (
        .5 * alpha_s * rhos * fv.norm(vs)**2
        - soundSpeed * alpha_f * rhos * b_perpend
        + (alpha_s * rhos * soundSpeed**2)/(gamma - 1)
        - alpha_s * slowMagnetosonicWave * rhos * vx
        - alpha_f * fastMagnetosonicWave * rhos * S * (vy*beta2 + vz*beta3)
        )
    psi_plus_fast = (
        .5 * alpha_f * rhos * fv.norm(vs)**2
        + soundSpeed * alpha_s * rhos * b_perpend
        + (alpha_f * rhos * soundSpeed**2)/(gamma - 1)
        + alpha_f * fastMagnetosonicWave * rhos * vx
        - alpha_s * slowMagnetosonicWave * rhos * S * (vy*beta2 + vz*beta3)
        )
    psi_minus_fast = (
        .5 * alpha_f * rhos * fv.norm(vs)**2
        + soundSpeed * alpha_s * rhos * b_perpend
        + (alpha_f * rhos * soundSpeed**2)/(gamma - 1)
        - alpha_f * fastMagnetosonicWave * rhos * vx
        + alpha_s * slowMagnetosonicWave * rhos * S * (vy*beta2 + vz*beta3)
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
    rightEigenvectors[...,4,3] = .5 * fv.norm(vs)**2
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
    diag_scaler[...,1,1] = fv.divide(pressures, 2*rhos**2)
    diag_scaler[...,2,2] = 1/(2*gamma*rhos)
    diag_scaler[...,3,3] = (rhos*(gamma-1))/gamma
    diag_scaler[...,4,4] = fv.divide(pressures, rhos)
    diag_scaler[...,5,5] = 1/(2*gamma*rhos)
    diag_scaler[...,6,6] = fv.divide(pressures, 2*rhos**2)
    diag_scaler[...,7,7] = 1/(2*gamma*rhos)
    R_dot = rightEigenvectors @ np.sqrt(diag_scaler)

    return R_dot