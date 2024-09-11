import numpy as np

from functions import fv

##############################################################################
# Functions for constructing objects such as the grid, eigenvectors, Jacobian and flux terms
##############################################################################

# Initialise the discrete solution array with initial conditions and primitive variables w. Returns the solution array in conserved variables q
def initialise(sim_variables, convert=False):
    config, N, gamma, dimension, precision = sim_variables.config, sim_variables.cells, sim_variables.gamma, sim_variables.dimension, sim_variables.precision
    start_pos, end_pos, shock_pos, params = sim_variables.start_pos, sim_variables.end_pos, sim_variables.shock_pos, sim_variables.misc
    initial_left, initial_right = sim_variables.initial_left, sim_variables.initial_right

    _i = (N,) * int(dimension)
    _i += (len(initial_right),)
    arr = np.zeros(_i, dtype=precision)
    arr[:] = initial_right

    if dimension >= 2:
        x, y = np.meshgrid(np.linspace(start_pos, end_pos, N), np.linspace(start_pos, end_pos, N))
        centre = (end_pos+start_pos)/2

        if config == "sedov":
            mask = np.where(((x-centre)**2 + (y-centre)**2) <= shock_pos**2)
            arr[mask] = initial_left
        elif config.startswith("gauss"):
            dst = np.sqrt(x**2 + y**2)
            mask = params['y_offset'] + params['ampl']*np.exp(-((dst-centre)**2)/params['fwhm'])
            arr[...,0] = mask
        elif config in ["ivc", "vortex"]:
            arr[...,0] = ()**(1/(gamma-1))
            pass
        else:
            pass
    else:
        x = np.linspace(start_pos, end_pos, N)

        if config == "sedov" or config.startswith('sq'):
            mask = np.where(np.abs(x) <= shock_pos)
        else:
            mask = np.where(x <= shock_pos)

        arr[mask] = initial_left

        if "shu" in config or "osher" in config:
            arr[np.where(x>shock_pos),0] = fv.sin_func(x[x>shock_pos], params)
        elif config == "sin":
            arr[...,0] = fv.sin_func(x, params)
        elif config == "sinc":
            arr[...,0] = fv.sinc_func(x, params)
        elif config.startswith('gauss'):
            arr[...,0] = fv.gauss_func(x, params)

        if dimension > 1:
            layer = 2
            arr = np.repeat(arr[np.newaxis,...], 2*layer+1, axis=0)

    if convert:
        return fv.point_convert_primitive(arr, sim_variables)
    else:
        return arr


# Make flux as a function of cell-averaged (primitive) variables
def make_flux_term(grid, gamma, axis):
    axis %= 3
    rhos, vecs, pressures, B_fields = grid[...,0], grid[...,1:4], grid[...,4], grid[...,5:8]
    arr = np.zeros_like(grid)

    arr[...,0] = rhos*vecs[...,axis]
    arr[...,(axis+0)%3+1] = rhos*(vecs[...,axis]**2) + pressures + (.5*fv.norm(B_fields)**2) - B_fields[...,axis]**2
    arr[...,(axis+1)%3+1] = rhos*vecs[...,axis]*vecs[...,(axis+1)%3] - B_fields[...,axis]*B_fields[...,(axis+1)%3]
    arr[...,(axis+2)%3+1] = rhos*vecs[...,axis]*vecs[...,(axis+2)%3] - B_fields[...,axis]*B_fields[...,(axis+2)%3]
    arr[...,4] = (vecs[...,axis] * ((.5*rhos*fv.norm(vecs)**2) + ((gamma*pressures)/(gamma-1)) + (fv.norm(B_fields)**2))) - (B_fields[...,axis]*np.sum(B_fields*vecs, axis=-1))
    arr[...,(axis+1)%3+5] = B_fields[...,(axis+1)%3]*vecs[...,axis] - B_fields[...,axis]*vecs[...,(axis+1)%3]
    arr[...,(axis+2)%3+5] = B_fields[...,(axis+2)%3]*vecs[...,axis] - B_fields[...,axis]*vecs[...,(axis+2)%3]
    return arr


# Jacobian matrix based on primitive variables
def make_Jacobian(grid, gamma, axis):
    axis %= 3
    rhos, v, pressures, B_fields = grid[...,0], grid[...,axis+1], grid[...,4], grid[...,5:8]/np.sqrt(4*np.pi)
    
    # Create empty square arrays for each cell
    _arr = np.zeros_like(grid)
    arr = np.repeat(_arr[..., np.newaxis], _arr.shape[-1], axis=-1)
    i, j = np.diag_indices(_arr.shape[-1])

    # Replace matrix with values
    arr[...,i,j] = v[...,None]  # diagonal elements
    arr[...,0,axis+1] = rhos
    arr[...,axis+1,4] = 1/rhos
    arr[...,4,axis+1] = gamma * pressures

    arr[...,axis+5,axis+5] = 0
    arr[...,axis+1,(axis+1)%3+5] = fv.divide(B_fields[...,(axis+1)%3], rhos)
    arr[...,axis+1,(axis+2)%3+5] = fv.divide(B_fields[...,(axis+2)%3], rhos)
    arr[...,(axis+1)%3+1,(axis+1)%3+5] = -fv.divide(B_fields[...,axis], rhos)
    arr[...,(axis+2)%3+1,(axis+2)%3+5] = -fv.divide(B_fields[...,axis], rhos)
    arr[...,(axis+1)%3+5,axis+1] = B_fields[...,(axis+1)%3]
    arr[...,(axis+2)%3+5,axis+1] = B_fields[...,(axis+2)%3]
    arr[...,(axis+1)%3+5,(axis+1)%3+1] = -B_fields[...,axis]
    arr[...,(axis+2)%3+5,(axis+2)%3+1] = -B_fields[...,axis]
    return arr


# Calculate the Roe-averaged primitive variables from the left- & right-interface states for use in Roe solver in order to better capture shocks [Brio & Wu, 1988; LeVeque, 2002; Stone et al., 2008]
def make_Roe_average(left_interface, right_interface):
    avg = np.zeros_like(left_interface)
    rhoL, rhoR = np.sqrt(left_interface[...,0]), np.sqrt(right_interface[...,0])

    avg[...,0] = rhoL * rhoR
    avg[...,1:4] = fv.divide((rhoL.T * left_interface[...,1:4].T) + (rhoR.T * right_interface[...,1:4].T), (rhoL + rhoR).T).T
    avg[...,4] = fv.divide((rhoL * left_interface[...,4]) + (rhoR * right_interface[...,4]), rhoL + rhoR)
    avg[...,5:8] = fv.divide((rhoR.T * left_interface[...,5:8].T) + (rhoL.T * right_interface[...,5:8].T), (rhoL + rhoR).T).T

    return avg


# Make the right eigenvector for adiabatic magnetohydrodynamics in Osher-Solomon flux
def make_OS_right_eigenvectors(tubes, gamma):
    rhos, pressures, B_fields = tubes[...,0], tubes[...,4], tubes[...,5:8]/np.sqrt(4*np.pi)

    # Define the right eigenvectors for each cell in each grid
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
def make_ES_right_eigenvectors(grid, gamma):
    rhos, vs, pressures, B_fields = grid[...,0], grid[...,1:4], grid[...,4], grid[...,5:8]
    vx, vy, vz = grid[...,1], grid[...,2], grid[...,3]

    # Define the right eigenvectors for each cell in each grid
    _right_eigenvectors = np.zeros_like(grid)
    right_eigenvectors = np.repeat(_right_eigenvectors[..., np.newaxis], _right_eigenvectors.shape[-1], axis=-1)

    # Define speeds
    sound_speed = np.sqrt(gamma * fv.divide(pressures, rhos))
    alfven_speed = np.sqrt(fv.divide(fv.norm(B_fields)**2, rhos))
    alfven_speed_x = fv.divide(grid[...,5], np.sqrt(rhos))
    fast_magnetosonic_wave = .5 * (sound_speed**2 + alfven_speed**2 + np.sqrt(((sound_speed**2 + alfven_speed**2)**2) - (4*(sound_speed**2)*(alfven_speed_x**2))))
    slow_magnetosonic_wave = .5 * (sound_speed**2 + alfven_speed**2 - np.sqrt(((sound_speed**2 + alfven_speed**2)**2) - (4*(sound_speed**2)*(alfven_speed_x**2))))

    # Define frequently used components
    S = np.sign(grid[...,5])
    S[S == 0] = 1
    alpha_f = np.sqrt(fv.divide(sound_speed**2 - slow_magnetosonic_wave**2, fast_magnetosonic_wave**2 - slow_magnetosonic_wave**2))
    alpha_s = np.sqrt(fv.divide(fast_magnetosonic_wave**2 - sound_speed**2, fast_magnetosonic_wave**2 - slow_magnetosonic_wave**2))
    b_perpend = np.sqrt(fv.divide(grid[...,6]**2 + grid[...,7]**2, rhos))
    beta2 = fv.divide(grid[...,6], np.sqrt(grid[...,6]**2 + grid[...,7]**2))
    beta3 = fv.divide(grid[...,7], np.sqrt(grid[...,6]**2 + grid[...,7]**2))

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
    right_eigenvectors[...,4,4] = grid[...,5]
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