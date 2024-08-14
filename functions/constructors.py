import numpy as np

from functions import fv

##############################################################################
# Functions for constructing objects such as the grid, eigenvectors, Jacobian and flux terms
##############################################################################

# Initialise the discrete solution array with initial conditions and primitive variables w. Returns the solution array in conserved variables q
def initialise(sim_variables):
    config, N, gamma, dimension, precision = sim_variables.config, sim_variables.cells, sim_variables.gamma, sim_variables.dimension, sim_variables.precision
    start_pos, end_pos, shock_pos, params = sim_variables.start_pos, sim_variables.end_pos, sim_variables.shock_pos, sim_variables.misc
    initial_left, initial_right = sim_variables.initial_left, sim_variables.initial_right

    _i = (N,) * int(dimension)
    _i += (len(initial_right),)
    arr = np.zeros(_i, dtype=precision)
    arr[:] = initial_right

    midpoint = (end_pos+start_pos)/2

    if dimension >= 2:
        x = y = np.arange(N)
        cx = cy = int(N/2)

        if config == "sedov":
            r = int(N/2 * ((shock_pos-midpoint)/(end_pos-midpoint)))

            mask = (x[np.newaxis,:]-cx)**2 + (y[:,np.newaxis]-cy)**2 < r**2
            arr[mask] = initial_left
        elif config.startswith("gauss"):
            pass

    else:
        if config == "sedov" or config.startswith('sq'):
            half_width = int(N/2 * ((shock_pos-midpoint)/(end_pos-midpoint)))
            left_edge, right_edge = int(N/2-half_width), int(N/2+half_width)
            arr[left_edge:right_edge] = initial_left
        else:
            split_point = int(N * ((shock_pos-start_pos)/(end_pos-start_pos)))
            arr[:split_point] = initial_left

        if "shu" in config or "osher" in config:
            xi = np.linspace(shock_pos, end_pos, N-split_point)
            arr[split_point:,0] = fv.sin_func(xi, params)
        elif config == "sin" or config == "sinc" or config.startswith('gauss'):
            xi = np.linspace(start_pos, end_pos, N)
            if config == "sin":
                arr[...,0] = fv.sin_func(xi, params)
            elif config == "sinc":
                arr[...,0] = fv.sinc_func(xi, params)
            else:
                arr[...,0] = fv.gauss_func(xi, params)
        
        if dimension != 1:
            layer = 1
            _arr = np.pad(arr, ((int(N*layer),int(N*layer)),(0,0)), mode="constant")
            _arr = _arr.reshape(2*layer+1,N,len(initial_right))
            arr = _arr.transpose(1,0,2)

    return fv.point_convert_primitive(arr, gamma)


# Make flux as a function of cell-averaged (primitive) variables
def make_flux_term(tube, gamma, axis):
    axis %= 3
    rhos, vecs, pressures, B_fields = tube[...,0], tube[...,1:4], tube[...,4], tube[...,5:8]
    arr = np.zeros_like(tube)

    arr[...,0] = rhos*vecs[...,axis]
    arr[...,axis+1] = rhos*(vecs[...,axis]**2) + pressures + (.5*fv.norm(B_fields)**2) - B_fields[...,axis]**2
    arr[...,(axis+1)%3+1] = rhos*vecs[...,axis]*vecs[...,(axis+1)%3] - B_fields[...,axis]*B_fields[...,(axis+1)%3]
    arr[...,(axis+2)%3+1] = rhos*vecs[...,axis]*vecs[...,(axis+2)%3] - B_fields[...,axis]*B_fields[...,(axis+2)%3]
    arr[...,4] = (vecs[...,axis] * ((.5*rhos*fv.norm(vecs)**2) + ((gamma*pressures)/(gamma-1)) + (fv.norm(B_fields)**2))) - (B_fields[...,axis]*np.sum(B_fields*vecs, axis=-1))
    arr[...,(axis+1)%3+5] = B_fields[...,(axis+1)%3]*vecs[...,axis] - B_fields[...,axis]*vecs[...,(axis+1)%3]
    arr[...,(axis+2)%3+5] = B_fields[...,(axis+2)%3]*vecs[...,axis] - B_fields[...,axis]*vecs[...,(axis+2)%3]
    return arr


# Jacobian matrix based on primitive variables
def make_Jacobian(tube, gamma, axis):
    axis %= 3
    rho, v, pressure, B_fields = tube[...,0], tube[...,axis+1], tube[...,4], tube[...,5:8]/np.sqrt(4*np.pi)
    
    # Create empty square arrays for each cell
    _arr = np.zeros_like(tube)
    arr = np.repeat(_arr[..., np.newaxis], _arr.shape[-1], axis=-1)
    i, j = np.diag_indices(_arr.shape[-1])

    # Replace matrix with values
    arr[...,i,j] = v[...,None]  # diagonal elements
    arr[...,0,axis+1] = rho
    arr[...,axis+1,4] = 1/rho
    arr[...,4,axis+1] = gamma*pressure

    arr[...,axis+1,(axis+1)%3+5] = fv.divide(B_fields[...,(axis+1)%3], rho)
    arr[...,axis+1,(axis+2)%3+5] = fv.divide(B_fields[...,(axis+2)%3], rho)
    arr[...,(axis+1)%3+1,(axis+1)%3+5] = fv.divide(-B_fields[...,axis], rho)
    arr[...,(axis+2)%3+1,(axis+2)%3+5] = fv.divide(-B_fields[...,axis], rho)
    arr[...,(axis+1)%3+5,axis+1] = B_fields[...,(axis+1)%3]
    arr[...,(axis+1)%3+5,(axis+1)%3+1] = -B_fields[...,axis]
    arr[...,(axis+2)%3+5,axis+1] = B_fields[...,(axis+2)%3]
    arr[...,(axis+2)%3+5,(axis+2)%3+1] = -B_fields[...,axis]
    return arr


# Calculate the Roe-averaged primitive variables from the left- & right-interface states for use in Roe solver in order to better capture shocks [Brio & Wu, 1988; LeVeque, 2002; Stone et al., 2008]
def make_Roe_average(wS, qS, gamma):
    wL, wR = wS
    qL, qR = qS

    avg = np.zeros_like(wL)
    rhoL, rhoR = np.sqrt(wL[...,0]), np.sqrt(wR[...,0])

    avg[...,0] = rhoL * rhoR
    avg[...,1:4] = fv.divide((rhoL.T * wL[...,1:4].T) + (rhoR.T * wR[...,1:4].T), (rhoL + rhoR).T).T
    avg[...,6:8] = fv.divide((rhoR.T * wL[...,6:8].T) + (rhoL.T * wR[...,6:8].T), (rhoL + rhoR).T).T

    HL, HR = fv.divide(qL[...,4] + wL[...,4] + .5*fv.norm(wL[...,5:8])**2, wL[...,0]), fv.divide(qR[...,4] + wR[...,4] + .5*fv.norm(wR[...,5:8])**2, wR[...,0])
    H = fv.divide(rhoL*HL + rhoR*HR, rhoL + rhoR)
    avg[...,4] = ((gamma-1)/gamma) * (avg[...,0]*H - .5*(avg[...,0]*fv.norm(avg[...,1:4])**2 + fv.norm(avg[...,5:8])**2))

    return avg


# Make the right eigenvector for adiabatic magnetohydrodynamics in Osher-Solomon flux
def make_OS_right_eigenvectors(tubes, gamma):
    rhos, pressures, B_fields = tubes[...,0], tubes[...,4], tubes[...,5:8]/np.sqrt(4*np.pi)

    # Define the right eigenvectors for each cell in each tube
    _right_eigenvectors = np.zeros_like(tubes)
    right_eigenvectors = np.repeat(_right_eigenvectors[..., np.newaxis], _right_eigenvectors.shape[-1], axis=-1)

    # Define speed
    sound_speed = np.sqrt(gamma * fv.divide(pressures, rhos))
    alfven_speed = np.sqrt(fv.divide(fv.norm(B_fields)**2, rhos))
    alfven_speed_x = fv.divide(B_fields[...,0], np.sqrt(rhos))

    fast_magnetosonic_wave = .5 * (sound_speed**2 + alfven_speed**2 + np.sqrt(((sound_speed**2 + alfven_speed**2)**2) - (4*(sound_speed**2)*(alfven_speed_x**2))))
    slow_magnetosonic_wave = .5 * (sound_speed**2 + alfven_speed**2 - np.sqrt(((sound_speed**2 + alfven_speed**2)**2) - (4*(sound_speed**2)*(alfven_speed_x**2))))

    # Define frequently used components
    S = np.sign(B_fields[...,0])
    alpha_f = np.ones_like(sound_speed)
    alpha_s = np.zeros_like(sound_speed)
    alpha_f[fast_magnetosonic_wave != slow_magnetosonic_wave] = (np.sqrt(fv.divide(sound_speed**2 - slow_magnetosonic_wave**2, fast_magnetosonic_wave**2 - slow_magnetosonic_wave**2)))[fast_magnetosonic_wave != slow_magnetosonic_wave]
    alpha_s[fast_magnetosonic_wave != slow_magnetosonic_wave] = (np.sqrt(fv.divide(fast_magnetosonic_wave**2 - sound_speed**2, fast_magnetosonic_wave**2 - slow_magnetosonic_wave**2)))[fast_magnetosonic_wave != slow_magnetosonic_wave]
    beta_y = fv.divide(B_fields[...,1], np.sqrt(B_fields[...,1]**2 + B_fields[...,2]**2))
    beta_z = fv.divide(B_fields[...,2], np.sqrt(B_fields[...,1]**2 + B_fields[...,2]**2))
    C_ff = fast_magnetosonic_wave * alpha_f
    C_ss = slow_magnetosonic_wave * alpha_s
    Q_f = C_ff * S
    Q_s = C_ss * S
    A_f = sound_speed * alpha_f * np.sqrt(rhos)
    A_s = sound_speed * alpha_s * np.sqrt(rhos)

    # Generate the right eigenvectors
    # First row
    right_eigenvectors[...,0,0] = rhos * alpha_f
    right_eigenvectors[...,0,2] = rhos * alpha_s
    right_eigenvectors[...,0,3] = 1
    right_eigenvectors[...,0,4] = 1
    right_eigenvectors[...,0,5] = rhos * alpha_s
    right_eigenvectors[...,0,7] = rhos * alpha_f
    # Second row
    right_eigenvectors[...,1,0] = -C_ff
    right_eigenvectors[...,1,2] = -C_ss
    right_eigenvectors[...,1,5] = C_ss
    right_eigenvectors[...,1,7] = C_ff
    # Third row
    right_eigenvectors[...,2,0] = Q_s * beta_y
    right_eigenvectors[...,2,1] = -beta_z
    right_eigenvectors[...,2,2] = -Q_f * beta_y
    right_eigenvectors[...,2,5] = Q_f * beta_y
    right_eigenvectors[...,2,6] = beta_z
    right_eigenvectors[...,2,7] = -Q_s * beta_y
    # Fourth row
    right_eigenvectors[...,3,0] = Q_s * beta_z
    right_eigenvectors[...,3,1] = beta_y
    right_eigenvectors[...,3,2] = -Q_f * beta_z
    right_eigenvectors[...,3,5] = Q_f * beta_z
    right_eigenvectors[...,3,6] = -beta_y
    right_eigenvectors[...,3,7] = -Q_s * beta_z
    # Fifth row
    right_eigenvectors[...,4,0] = rhos * alpha_f * sound_speed**2
    right_eigenvectors[...,4,2] = rhos * alpha_s * sound_speed**2
    right_eigenvectors[...,4,5] = rhos * alpha_s * sound_speed**2
    right_eigenvectors[...,4,7] = rhos * alpha_f * sound_speed**2
    # Seventh row
    right_eigenvectors[...,6,0] = A_s * beta_y
    right_eigenvectors[...,6,1] = -beta_z * S * np.sqrt(rhos)
    right_eigenvectors[...,6,2] = -A_f * beta_y
    right_eigenvectors[...,6,5] = -A_f * beta_y
    right_eigenvectors[...,6,6] = -beta_z * S * np.sqrt(rhos)
    right_eigenvectors[...,6,7] = A_s * beta_y
    # Eighth row
    right_eigenvectors[...,7,0] = A_s * beta_z
    right_eigenvectors[...,7,1] = -beta_y * S * np.sqrt(rhos)
    right_eigenvectors[...,7,2] = -A_f * beta_z
    right_eigenvectors[...,7,5] = -A_f * beta_z
    right_eigenvectors[...,7,6] = -beta_y * S * np.sqrt(rhos)
    right_eigenvectors[...,7,7] = A_s * beta_z

    return right_eigenvectors


# Make the right eigenvector for adiabatic magnetohydrodynamics in entropy-stable flux (primitive variables)
def make_ES_right_eigenvectors(tube, gamma):
    rhos, vs, pressures, B_fields = tube[...,0], tube[...,1:4], tube[...,4], tube[...,5:8]
    vx, vy, vz = tube[...,1], tube[...,2], tube[...,3]

    # Define the right eigenvectors for each cell in each tube
    _right_eigenvectors = np.zeros_like(tube)
    right_eigenvectors = np.repeat(_right_eigenvectors[..., np.newaxis], _right_eigenvectors.shape[-1], axis=-1)

    # Define speeds
    sound_speed = np.sqrt(gamma * fv.divide(pressures, rhos))
    alfven_speed = np.sqrt(fv.divide(fv.norm(B_fields)**2, rhos))
    alfven_speed_x = fv.divide(tube[...,5], np.sqrt(rhos))
    fast_magnetosonic_wave = .5 * (sound_speed**2 + alfven_speed**2 + np.sqrt(((sound_speed**2 + alfven_speed**2)**2) - (4*(sound_speed**2)*(alfven_speed_x**2))))
    slow_magnetosonic_wave = .5 * (sound_speed**2 + alfven_speed**2 - np.sqrt(((sound_speed**2 + alfven_speed**2)**2) - (4*(sound_speed**2)*(alfven_speed_x**2))))

    # Define frequently used components
    S = np.sign(tube[...,5])
    S[S == 0] = 1
    alpha_f = np.sqrt(fv.divide(sound_speed**2 - slow_magnetosonic_wave**2, fast_magnetosonic_wave**2 - slow_magnetosonic_wave**2))
    alpha_s = np.sqrt(fv.divide(fast_magnetosonic_wave**2 - sound_speed**2, fast_magnetosonic_wave**2 - slow_magnetosonic_wave**2))
    b_perpend = np.sqrt(fv.divide(tube[...,6]**2 + tube[...,7]**2, rhos))
    beta2 = fv.divide(tube[...,6], np.sqrt(tube[...,6]**2 + tube[...,7]**2))
    beta3 = fv.divide(tube[...,7], np.sqrt(tube[...,6]**2 + tube[...,7]**2))

    psi_plus_slow = (
        .5 * alpha_s * rhos * fv.norm(vs)**2
        - sound_speed * alpha_f * rhos * b_perpend
        + (alpha_s * rhos * sound_speed**2)/(gamma - 1)
        + alpha_s * slow_magnetosonic_wave * rhos * vx
        + alpha_f * fast_magnetosonic_wave * rhos * S * (vy*beta2 + vz*beta3)
        )
    psi_minus_slow = (
        .5 * alpha_s * rhos * fv.norm(vs)**2
        - sound_speed * alpha_f * rhos * b_perpend
        + (alpha_s * rhos * sound_speed**2)/(gamma - 1)
        - alpha_s * slow_magnetosonic_wave * rhos * vx
        - alpha_f * fast_magnetosonic_wave * rhos * S * (vy*beta2 + vz*beta3)
        )
    psi_plus_fast = (
        .5 * alpha_f * rhos * fv.norm(vs)**2
        + sound_speed * alpha_s * rhos * b_perpend
        + (alpha_f * rhos * sound_speed**2)/(gamma - 1)
        + alpha_f * fast_magnetosonic_wave * rhos * vx
        - alpha_s * slow_magnetosonic_wave * rhos * S * (vy*beta2 + vz*beta3)
        )
    psi_minus_fast = (
        .5 * alpha_f * rhos * fv.norm(vs)**2
        + sound_speed * alpha_s * rhos * b_perpend
        + (alpha_f * rhos * sound_speed**2)/(gamma - 1)
        - alpha_f * fast_magnetosonic_wave * rhos * vx
        + alpha_s * slow_magnetosonic_wave * rhos * S * (vy*beta2 + vz*beta3)
        )

    # Generate the right eigenvectors
    # First column (Fast+ magnetoacoustic wave)
    right_eigenvectors[...,0,0] = rhos * alpha_f
    right_eigenvectors[...,1,0] = rhos * alpha_f * (vx + fast_magnetosonic_wave)
    right_eigenvectors[...,2,0] = rhos * (alpha_f*vy - alpha_s*slow_magnetosonic_wave*beta2*S)
    right_eigenvectors[...,3,0] = rhos * (alpha_f*vz - alpha_s*slow_magnetosonic_wave*beta3*S)
    right_eigenvectors[...,4,0] = psi_plus_fast
    right_eigenvectors[...,6,0] = alpha_s * sound_speed * beta2 * np.sqrt(rhos)
    right_eigenvectors[...,7,0] = alpha_s * sound_speed * beta3 * np.sqrt(rhos)
    # Second column (Alfven+ wave)
    right_eigenvectors[...,2,1] = beta3 * rhos**1.5
    right_eigenvectors[...,3,1] = -beta2 * rhos**1.5
    right_eigenvectors[...,4,1] = (beta3*vy - beta2*vz) * rhos**1.5
    right_eigenvectors[...,6,1] = -rhos * beta3
    right_eigenvectors[...,7,1] = rhos * beta2
    # Third column (Slow+ magnetoacoustic wave)
    right_eigenvectors[...,0,2] = rhos * alpha_s
    right_eigenvectors[...,1,2] = rhos * alpha_s * (vx + slow_magnetosonic_wave)
    right_eigenvectors[...,2,2] = rhos * (alpha_s*vy + alpha_f*fast_magnetosonic_wave*beta2*S)
    right_eigenvectors[...,3,2] = rhos * (alpha_s*vz + alpha_f*fast_magnetosonic_wave*beta3*S)
    right_eigenvectors[...,4,2] = psi_plus_slow
    right_eigenvectors[...,6,2] = -alpha_f * sound_speed * beta2 * np.sqrt(rhos)
    right_eigenvectors[...,7,2] = -alpha_f * sound_speed * beta3 * np.sqrt(rhos)
    # Fourth column (Entropy wave)
    right_eigenvectors[...,0,3] = 1
    right_eigenvectors[...,1,3] = vx
    right_eigenvectors[...,2,3] = vy
    right_eigenvectors[...,3,3] = vz
    right_eigenvectors[...,4,3] = .5 * fv.norm(vs)**2
    # Fifth column (Divergence wave)
    right_eigenvectors[...,4,4] = tube[...,5]
    right_eigenvectors[...,5,4] = 1
    # Sixth column (Slow- magnetoacoustic wave)
    right_eigenvectors[...,0,5] = rhos * alpha_s
    right_eigenvectors[...,1,5] = rhos * alpha_s * (vx - slow_magnetosonic_wave)
    right_eigenvectors[...,2,5] = rhos * (alpha_s*vy - alpha_f*fast_magnetosonic_wave*beta2*S)
    right_eigenvectors[...,3,5] = rhos * (alpha_s*vz - alpha_f*fast_magnetosonic_wave*beta3*S)
    right_eigenvectors[...,4,5] = psi_minus_slow
    right_eigenvectors[...,6,5] = -alpha_f * sound_speed * beta2 * np.sqrt(rhos)
    right_eigenvectors[...,7,5] = -alpha_f * sound_speed * beta3 * np.sqrt(rhos)
    # Seventh column (Alfven- wave)
    right_eigenvectors[...,2,6] = -beta3 * rhos**1.5
    right_eigenvectors[...,3,6] = beta2 * rhos**1.5
    right_eigenvectors[...,4,6] = (beta2*vz - beta3*vy) * rhos**1.5
    right_eigenvectors[...,6,6] = -rhos * beta3
    right_eigenvectors[...,7,6] = rhos * beta2
    # Eighth column (Fast- magnetoacoustic wave)
    right_eigenvectors[...,0,7] = rhos * alpha_f
    right_eigenvectors[...,1,7] = rhos * alpha_f * (vx - fast_magnetosonic_wave)
    right_eigenvectors[...,2,7] = rhos * (alpha_f*vy + alpha_s*slow_magnetosonic_wave*beta2*S)
    right_eigenvectors[...,3,7] = rhos * (alpha_f*vz + alpha_s*slow_magnetosonic_wave*beta3*S)
    right_eigenvectors[...,4,7] = psi_minus_fast
    right_eigenvectors[...,6,7] = alpha_s * sound_speed * beta2 * np.sqrt(rhos)
    right_eigenvectors[...,7,7] = alpha_s * sound_speed * beta3 * np.sqrt(rhos)

    # Scale the right eigenvectors with a diagonal scaling matrix, so as to prevent degeneracies [Barth, 1999]
    diag_scaler = np.zeros_like(right_eigenvectors)
    diag_scaler[...,0,0] = 1/(2*gamma*rhos)
    diag_scaler[...,1,1] = fv.divide(pressures, 2*rhos**2)
    diag_scaler[...,2,2] = 1/(2*gamma*rhos)
    diag_scaler[...,3,3] = (rhos*(gamma-1))/gamma
    diag_scaler[...,4,4] = fv.divide(pressures, rhos)
    diag_scaler[...,5,5] = 1/(2*gamma*rhos)
    diag_scaler[...,6,6] = fv.divide(pressures, 2*rhos**2)
    diag_scaler[...,7,7] = 1/(2*gamma*rhos)
    R_dot = right_eigenvectors @ np.sqrt(diag_scaler)

    return R_dot