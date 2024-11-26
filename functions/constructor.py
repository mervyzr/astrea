import numpy as np

from functions import fv

##############################################################################
# Functions for constructing objects such as the grid, eigenvectors, Jacobian and flux terms
##############################################################################

# Initialise the discrete POINTWISE solution array with initial conditions and primitive variables w, and transform into discrete AVERAGES <w>
# Gives option to convert to conservative variables <q>
def initialise(sim_variables, convert=False):

    def make_physical_grid(_start_pos, _end_pos, _N):
        dx = abs(_end_pos-_start_pos)/_N
        half_cell = dx/2
        return np.linspace(_start_pos-half_cell, _end_pos+half_cell, _N+2)[1:-1]

    config, N, gamma, dimension, precision = sim_variables.config, sim_variables.cells, sim_variables.gamma, sim_variables.dimension, sim_variables.precision
    start_pos, end_pos, shock_pos, params = sim_variables.start_pos, sim_variables.end_pos, sim_variables.shock_pos, sim_variables.misc
    initial_left, initial_right = sim_variables.initial_left, sim_variables.initial_right

    _i = (N,) * dimension
    _i += (len(initial_right),)
    computational_grid = np.zeros(_i, dtype=precision)
    computational_grid[:] = initial_right

    physical_grid = make_physical_grid(start_pos, end_pos, N)

    if dimension == 2:
        x, y = np.meshgrid(physical_grid, physical_grid, indexing='ij')
        centre = (end_pos+start_pos)/2

        if config == "sedov":
            mask = np.where(((x-centre)**2 + (y-centre)**2) <= shock_pos**2)
            computational_grid[mask] = initial_left

        elif config.startswith("gauss"):
            r = np.sqrt((x-centre)**2 + (y-centre)**2)
            mask = params['y_offset'] + params['ampl']*np.exp(-((r-centre)**2)/params['fwhm'])
            computational_grid[...,0] = mask

        elif config in ["khi", "kelvin-helmholtz"] or ("kelvin" in config or "helmholtz" in config):
            computational_grid[np.where(y <= shock_pos)] = initial_left
            computational_grid[...,2] = params['perturb_ampl'] * np.sin(params['freq']*np.pi*x/(end_pos-start_pos))

        elif config in ["ivc", "vortex", "isentropic vortex"]:
            x_centre, y_centre = (np.min(x)+np.max(x))/2, (np.min(y)+np.max(y))/2

            r = np.sqrt((x-x_centre)**2 + (y-y_centre)**2)
            T = 1 - (((gamma-1)*params['vortex_str']**2)/(2*gamma*(params['freq']*np.pi)**2))*np.exp(1-r**2)

            computational_grid[...,0] = T**(1/(gamma-1))
            computational_grid[...,1] = (params['vortex_str']/(params['freq']*np.pi)) * np.exp((1-r**2)/2)
            computational_grid[...,2] = (params['vortex_str']/(params['freq']*np.pi)) * np.exp((1-r**2)/2)
            computational_grid[...,4] = T**(gamma/(gamma-1))

        elif "ll" in config or "lax-liu" in config:
            computational_grid[np.where(x <= shock_pos)] = initial_left
            computational_grid[np.where((x <= shock_pos) & (y >= shock_pos))] = params['bottom_left']
            computational_grid[np.where((x > shock_pos) & (y >= shock_pos))] = params['bottom_right']

        else:
            computational_grid[np.where(x < shock_pos)] = initial_left
    else:
        x = physical_grid

        if config == "sedov" or config.startswith('sq'):
            mask = np.where(np.abs(x) <= shock_pos)
        else:
            mask = np.where(x <= shock_pos)

        computational_grid[mask] = initial_left

        if "shu" in config or "osher" in config:
            computational_grid[np.where(x>shock_pos),0] = fv.sine_func(x[x>shock_pos], params)
        elif config.startswith("sin"):
            computational_grid[...,0] = fv.sine_func(x, params)
        elif config.startswith('gauss'):
            computational_grid[...,0] = fv.gauss_func(x, params)

    if convert:
        grid = sim_variables.convert_primitive(computational_grid, sim_variables)
    else:
        grid = computational_grid
    return fv.high_order_average(grid, sim_variables)


# Make flux as a function of cell-averaged (primitive) variables
def make_flux(grid, gamma, axis):
    rhos, vels, pressures, B_fields = grid[...,0], grid[...,1:4], grid[...,4], grid[...,5:8]
    abscissa, ordinate, applicate = axis%3, (axis+1)%3, (axis+2)%3
    arr = np.zeros_like(grid)

    arr[...,0] = rhos * vels[...,axis]
    arr[...,abscissa+1] = rhos*vels[...,axis]**2 + pressures + .5*fv.norm(B_fields)**2 - B_fields[...,axis]**2
    arr[...,ordinate+1] = rhos*vels[...,axis]*vels[...,ordinate] - B_fields[...,axis]*B_fields[...,ordinate]
    arr[...,applicate+1] = rhos*vels[...,axis]*vels[...,applicate] - B_fields[...,axis]*B_fields[...,applicate]
    arr[...,4] = vels[...,axis]*(.5*rhos*fv.norm(vels)**2 + (gamma*pressures)/(gamma-1) + fv.norm(B_fields)**2) - B_fields[...,axis]*np.sum(vels*B_fields, axis=-1)
    arr[...,ordinate+5] = B_fields[...,ordinate]*vels[...,axis] - B_fields[...,axis]*vels[...,ordinate]
    arr[...,applicate+5] = B_fields[...,applicate]*vels[...,axis] - B_fields[...,axis]*vels[...,applicate]
    return arr


# Jacobian matrix based on primitive variables
def make_Jacobian(grid, gamma, axis):
    rhos, v, pressures, B_fields = grid[...,0], grid[...,axis+1], grid[...,4], grid[...,5:8]
    abscissa, ordinate, applicate = axis%3, (axis+1)%3, (axis+2)%3
    
    # Create empty square arrays for each cell
    _arr = np.zeros_like(grid)
    arr = np.repeat(_arr[...,None], _arr.shape[-1], axis=-1)
    i, j = np.diag_indices(_arr.shape[-1])

    # Replace matrix with values
    # Hydrodynamic components
    arr[...,i,j] = v[...,None]  # diagonal elements
    arr[...,0,axis+1] = rhos
    arr[...,axis+1,4] = 1/rhos
    arr[...,4,axis+1] = gamma * pressures

    # Magnetic field components
    arr[...,abscissa+1,ordinate+5] = fv.divide(B_fields[...,ordinate], rhos)
    arr[...,abscissa+1,applicate+5] = fv.divide(B_fields[...,applicate], rhos)
    arr[...,ordinate+1,ordinate+5] = arr[...,applicate+1,applicate+5] = -fv.divide(B_fields[...,abscissa], rhos)
    arr[...,ordinate+5,abscissa+1] = B_fields[...,ordinate]
    arr[...,applicate+5,abscissa+1] = B_fields[...,applicate]
    arr[...,ordinate+5,ordinate+1] = arr[...,applicate+5,applicate+1] = -B_fields[...,abscissa]
    return arr


# Calculate the Roe-averaged primitive variables at the interface from the minus- & plus-interface states for use in Roe solver in order to better capture shocks [Roe & Pike, 1984; Brio & Wu, 1988; LeVeque, 2002; Stone et al., 2008]
def make_Roe_average(left_interface, right_interface):
    avg = np.zeros_like(left_interface)
    rho_minus, rho_plus = np.sqrt(right_interface[...,0]), np.sqrt(left_interface[...,0])

    avg[...,0] = rho_minus * rho_plus
    avg[...,1:4] = fv.divide((left_interface[...,1:4] * rho_plus[...,None]) + (right_interface[...,1:4] * rho_minus[...,None]), (rho_minus + rho_plus)[...,None])
    avg[...,4] = fv.divide((rho_plus * left_interface[...,4]) + (rho_minus * right_interface[...,4]), rho_minus + rho_plus)
    avg[...,5:8] = fv.divide((left_interface[...,5:8] * rho_minus[...,None]) + (right_interface[...,5:8] * rho_plus[...,None]), (rho_minus + rho_plus)[...,None])

    return avg


# Make the right eigenvectors for adiabatic magnetohydrodynamics [Derigs]
def make_right_eigenvectors(axis, grids, gamma):
    abscissa, ordinate, applicate = axis%3, (axis+1)%3, (axis+2)%3
    rhos, vels, pressures, B_fields = grids[...,0], grids[...,1:4], grids[...,4], grids[...,5:8]/np.sqrt(4*np.pi)
    vx, vy, vz = vels[...,abscissa], vels[...,ordinate], vels[...,applicate]
    Bx, By, Bz = B_fields[...,abscissa], B_fields[...,ordinate], B_fields[...,applicate]

    # Define the right eigenvectors for each cell in each grid
    _right_eigenvectors = np.zeros_like(grids)
    right_eigenvectors = np.repeat(_right_eigenvectors[...,None], _right_eigenvectors.shape[-1], axis=-1)

    # Define speed
    sound_speed = np.sqrt(gamma * fv.divide(pressures, rhos))
    alfven_speed = fv.divide(fv.norm(B_fields), np.sqrt(rhos))
    alfven_speed_x = fv.divide(Bx, np.sqrt(rhos))
    fast_magnetosonic_wave = np.sqrt(.5 * (sound_speed**2 + alfven_speed**2 + np.sqrt(((sound_speed**2 + alfven_speed**2)**2) - (4*(sound_speed**2)*(alfven_speed_x**2)))))
    slow_magnetosonic_wave = np.sqrt(.5 * (sound_speed**2 + alfven_speed**2 - np.sqrt(((sound_speed**2 + alfven_speed**2)**2) - (4*(sound_speed**2)*(alfven_speed_x**2)))))

    # Define frequently used components
    S = np.sign(Bx)
    S[S == 0] = 1
    alpha_f = np.sqrt(fv.divide(sound_speed**2 - slow_magnetosonic_wave**2, fast_magnetosonic_wave**2 - slow_magnetosonic_wave**2))
    alpha_s = np.sqrt(fv.divide(fast_magnetosonic_wave**2 - sound_speed**2, fast_magnetosonic_wave**2 - slow_magnetosonic_wave**2))
    b_perpend = np.sqrt(fv.divide(By**2 + Bz**2, rhos))
    beta2 = fv.divide(By, np.sqrt(By**2 + Bz**2))
    beta3 = fv.divide(Bz, np.sqrt(By**2 + Bz**2))

    psi_plus_slow = (
        .5 * alpha_s * rhos * fv.norm(vels)**2
        - sound_speed * alpha_f * rhos * b_perpend
        + (alpha_s * rhos * sound_speed**2)/(gamma - 1)
        + alpha_s * slow_magnetosonic_wave * rhos * vx
        + alpha_f * fast_magnetosonic_wave * rhos * S * (vy*beta2 + vz*beta3)
        )
    psi_minus_slow = (
        .5 * alpha_s * rhos * fv.norm(vels)**2
        - sound_speed * alpha_f * rhos * b_perpend
        + (alpha_s * rhos * sound_speed**2)/(gamma - 1)
        - alpha_s * slow_magnetosonic_wave * rhos * vx
        - alpha_f * fast_magnetosonic_wave * rhos * S * (vy*beta2 + vz*beta3)
        )
    psi_plus_fast = (
        .5 * alpha_f * rhos * fv.norm(vels)**2
        + sound_speed * alpha_s * rhos * b_perpend
        + (alpha_f * rhos * sound_speed**2)/(gamma - 1)
        + alpha_f * fast_magnetosonic_wave * rhos * vx
        - alpha_s * slow_magnetosonic_wave * rhos * S * (vy*beta2 + vz*beta3)
        )
    psi_minus_fast = (
        .5 * alpha_f * rhos * fv.norm(vels)**2
        + sound_speed * alpha_s * rhos * b_perpend
        + (alpha_f * rhos * sound_speed**2)/(gamma - 1)
        - alpha_f * fast_magnetosonic_wave * rhos * vx
        + alpha_s * slow_magnetosonic_wave * rhos * S * (vy*beta2 + vz*beta3)
        )

    # Generate the right eigenvectors
    # First column (Fast- magnetoacoustic wave)
    right_eigenvectors[...,0,0] = rhos * alpha_f
    right_eigenvectors[...,abscissa+1,0] = rhos * alpha_f * (vx - fast_magnetosonic_wave)
    right_eigenvectors[...,ordinate+1,0] = rhos * (alpha_f*vy + alpha_s*slow_magnetosonic_wave*beta2*S)
    right_eigenvectors[...,applicate+1,0] = rhos * (alpha_f*vz + alpha_s*slow_magnetosonic_wave*beta3*S)
    right_eigenvectors[...,4,0] = psi_minus_fast
    right_eigenvectors[...,ordinate+5,0] = alpha_s * sound_speed * beta2 * np.sqrt(rhos)
    right_eigenvectors[...,applicate+5,0] = alpha_s * sound_speed * beta3 * np.sqrt(rhos)
    # Second column (Alfven- wave)
    right_eigenvectors[...,ordinate+1,1] = -beta3 * rhos**1.5
    right_eigenvectors[...,applicate+1,1] = beta2 * rhos**1.5
    right_eigenvectors[...,4,1] = (beta2*vz - beta3*vy) * rhos**1.5
    right_eigenvectors[...,ordinate+5,1] = -rhos * beta3
    right_eigenvectors[...,applicate+5,1] = rhos * beta2
    # Third column (Slow- magnetoacoustic wave)
    right_eigenvectors[...,0,2] = rhos * alpha_s
    right_eigenvectors[...,abscissa+1,2] = rhos * alpha_s * (vx - slow_magnetosonic_wave)
    right_eigenvectors[...,ordinate+1,2] = rhos * (alpha_s*vy - alpha_f*fast_magnetosonic_wave*beta2*S)
    right_eigenvectors[...,applicate+1,2] = rhos * (alpha_s*vz - alpha_f*fast_magnetosonic_wave*beta3*S)
    right_eigenvectors[...,4,2] = psi_minus_slow
    right_eigenvectors[...,ordinate+5,2] = -alpha_f * sound_speed * beta2 * np.sqrt(rhos)
    right_eigenvectors[...,applicate+5,2] = -alpha_f * sound_speed * beta3 * np.sqrt(rhos)
    # Fourth column (Entropy wave)
    right_eigenvectors[...,0,3] = 1
    right_eigenvectors[...,abscissa+1,3] = vx
    right_eigenvectors[...,ordinate+1,3] = vy
    right_eigenvectors[...,applicate+1,3] = vz
    right_eigenvectors[...,4,3] = .5 * fv.norm(vels)**2
    # Fifth column (Divergence wave)
    right_eigenvectors[...,4,4] = Bx
    right_eigenvectors[...,abscissa+5,4] = 1
    # Sixth column (Slow+ magnetoacoustic wave)
    right_eigenvectors[...,0,5] = rhos * alpha_s
    right_eigenvectors[...,abscissa+1,5] = rhos * alpha_s * (vx + slow_magnetosonic_wave)
    right_eigenvectors[...,ordinate+1,5] = rhos * (alpha_s*vy + alpha_f*fast_magnetosonic_wave*beta2*S)
    right_eigenvectors[...,applicate+1,5] = rhos * (alpha_s*vz + alpha_f*fast_magnetosonic_wave*beta3*S)
    right_eigenvectors[...,4,5] = psi_plus_slow
    right_eigenvectors[...,ordinate+5,5] = -alpha_f * sound_speed * beta2 * np.sqrt(rhos)
    right_eigenvectors[...,applicate+5,5] = -alpha_f * sound_speed * beta3 * np.sqrt(rhos)
    # Seventh column (Alfven+ wave)
    right_eigenvectors[...,ordinate+1,6] = beta3 * rhos**1.5
    right_eigenvectors[...,applicate+1,6] = -beta2 * rhos**1.5
    right_eigenvectors[...,4,6] = (beta3*vy - beta2*vz) * rhos**1.5
    right_eigenvectors[...,ordinate+5,6] = -rhos * beta3
    right_eigenvectors[...,applicate+5,6] = rhos * beta2
    # Eighth column (Fast+ magnetoacoustic wave)
    right_eigenvectors[...,0,7] = rhos * alpha_f
    right_eigenvectors[...,abscissa+1,7] = rhos * alpha_f * (vx + fast_magnetosonic_wave)
    right_eigenvectors[...,ordinate+1,7] = rhos * (alpha_f*vy - alpha_s*slow_magnetosonic_wave*beta2*S)
    right_eigenvectors[...,applicate+1,7] = rhos * (alpha_f*vz - alpha_s*slow_magnetosonic_wave*beta3*S)
    right_eigenvectors[...,4,7] = psi_plus_fast
    right_eigenvectors[...,ordinate+5,7] = alpha_s * sound_speed * beta2 * np.sqrt(rhos)
    right_eigenvectors[...,applicate+5,7] = alpha_s * sound_speed * beta3 * np.sqrt(rhos)

    return right_eigenvectors


# Make the right eigenvector for adiabatic magnetohydrodynamics in entropy-stable flux (primitive variables)
def make_ES_right_eigenvectors(axis, grids, gamma):
    abscissa, ordinate, applicate = axis%3, (axis+1)%3, (axis+2)%3
    rhos, vels, pressures, B_fields = grids[...,0], grids[...,1:4], grids[...,4], grids[...,5:8]/np.sqrt(4*np.pi)
    vx, vy, vz = vels[...,abscissa], vels[...,ordinate], vels[...,applicate]

    # Define the right eigenvectors for each cell in each grid
    _right_eigenvectors = np.zeros_like(grids)
    right_eigenvectors = np.repeat(_right_eigenvectors[...,None], _right_eigenvectors.shape[-1], axis=-1)

    # Define speeds
    sound_speed = np.sqrt(gamma * fv.divide(pressures, rhos))
    alfven_speed = fv.divide(fv.norm(B_fields), np.sqrt(rhos))
    alfven_speed_x = fv.divide(grids[...,abscissa+5], np.sqrt(rhos))
    fast_magnetosonic_wave = np.sqrt(.5 * (sound_speed**2 + alfven_speed**2 + np.sqrt(((sound_speed**2 + alfven_speed**2)**2) - (4*(sound_speed**2)*(alfven_speed_x**2)))))
    slow_magnetosonic_wave = np.sqrt(.5 * (sound_speed**2 + alfven_speed**2 - np.sqrt(((sound_speed**2 + alfven_speed**2)**2) - (4*(sound_speed**2)*(alfven_speed_x**2)))))

    # Define frequently used components
    S = np.sign(grids[...,abscissa+5])
    S[S == 0] = 1
    alpha_f = np.sqrt(fv.divide(sound_speed**2 - slow_magnetosonic_wave**2, fast_magnetosonic_wave**2 - slow_magnetosonic_wave**2))
    alpha_s = np.sqrt(fv.divide(fast_magnetosonic_wave**2 - sound_speed**2, fast_magnetosonic_wave**2 - slow_magnetosonic_wave**2))
    b_perpend = np.sqrt(fv.divide(grids[...,ordinate+5]**2 + grids[...,applicate+5]**2, rhos))
    beta2 = fv.divide(grids[...,ordinate+5], np.sqrt(grids[...,ordinate+5]**2 + grids[...,applicate+5]**2))
    beta3 = fv.divide(grids[...,applicate+5], np.sqrt(grids[...,ordinate+5]**2 + grids[...,applicate+5]**2))

    psi_plus_slow = (
        .5 * alpha_s * rhos * fv.norm(vels)**2
        - sound_speed * alpha_f * rhos * b_perpend
        + (alpha_s * rhos * sound_speed**2)/(gamma - 1)
        + alpha_s * slow_magnetosonic_wave * rhos * vx
        + alpha_f * fast_magnetosonic_wave * rhos * S * (vy*beta2 + vz*beta3)
        )
    psi_minus_slow = (
        .5 * alpha_s * rhos * fv.norm(vels)**2
        - sound_speed * alpha_f * rhos * b_perpend
        + (alpha_s * rhos * sound_speed**2)/(gamma - 1)
        - alpha_s * slow_magnetosonic_wave * rhos * vx
        - alpha_f * fast_magnetosonic_wave * rhos * S * (vy*beta2 + vz*beta3)
        )
    psi_plus_fast = (
        .5 * alpha_f * rhos * fv.norm(vels)**2
        + sound_speed * alpha_s * rhos * b_perpend
        + (alpha_f * rhos * sound_speed**2)/(gamma - 1)
        + alpha_f * fast_magnetosonic_wave * rhos * vx
        - alpha_s * slow_magnetosonic_wave * rhos * S * (vy*beta2 + vz*beta3)
        )
    psi_minus_fast = (
        .5 * alpha_f * rhos * fv.norm(vels)**2
        + sound_speed * alpha_s * rhos * b_perpend
        + (alpha_f * rhos * sound_speed**2)/(gamma - 1)
        - alpha_f * fast_magnetosonic_wave * rhos * vx
        + alpha_s * slow_magnetosonic_wave * rhos * S * (vy*beta2 + vz*beta3)
        )

    # Generate the right eigenvectors
    # First column (Fast- magnetoacoustic wave)
    right_eigenvectors[...,0,0] = rhos * alpha_f
    right_eigenvectors[...,abscissa+1,0] = rhos * alpha_f * (vx - fast_magnetosonic_wave)
    right_eigenvectors[...,ordinate+1,0] = rhos * (alpha_f*vy + alpha_s*slow_magnetosonic_wave*beta2*S)
    right_eigenvectors[...,applicate+1,0] = rhos * (alpha_f*vz + alpha_s*slow_magnetosonic_wave*beta3*S)
    right_eigenvectors[...,4,0] = psi_minus_fast
    right_eigenvectors[...,ordinate+5,0] = alpha_s * sound_speed * beta2 * np.sqrt(rhos)
    right_eigenvectors[...,applicate+5,0] = alpha_s * sound_speed * beta3 * np.sqrt(rhos)
    # Second column (Alfven- wave)
    right_eigenvectors[...,ordinate+1,1] = -beta3 * rhos**1.5
    right_eigenvectors[...,applicate+1,1] = beta2 * rhos**1.5
    right_eigenvectors[...,4,1] = (beta2*vz - beta3*vy) * rhos**1.5
    right_eigenvectors[...,ordinate+5,1] = -rhos * beta3
    right_eigenvectors[...,applicate+5,1] = rhos * beta2
    # Third column (Slow- magnetoacoustic wave)
    right_eigenvectors[...,0,2] = rhos * alpha_s
    right_eigenvectors[...,abscissa+1,2] = rhos * alpha_s * (vx - slow_magnetosonic_wave)
    right_eigenvectors[...,ordinate+1,2] = rhos * (alpha_s*vy - alpha_f*fast_magnetosonic_wave*beta2*S)
    right_eigenvectors[...,applicate+1,2] = rhos * (alpha_s*vz - alpha_f*fast_magnetosonic_wave*beta3*S)
    right_eigenvectors[...,4,2] = psi_minus_slow
    right_eigenvectors[...,ordinate+5,2] = -alpha_f * sound_speed * beta2 * np.sqrt(rhos)
    right_eigenvectors[...,applicate+5,2] = -alpha_f * sound_speed * beta3 * np.sqrt(rhos)
    # Fourth column (Entropy wave)
    right_eigenvectors[...,0,3] = 1
    right_eigenvectors[...,abscissa+1,3] = vx
    right_eigenvectors[...,ordinate+1,3] = vy
    right_eigenvectors[...,applicate+1,3] = vz
    right_eigenvectors[...,4,3] = .5 * fv.norm(vels)**2
    # Fifth column (Divergence wave)
    right_eigenvectors[...,4,4] = grids[...,abscissa+5]
    right_eigenvectors[...,abscissa+5,4] = 1
    # Sixth column (Slow+ magnetoacoustic wave)
    right_eigenvectors[...,0,5] = rhos * alpha_s
    right_eigenvectors[...,abscissa+1,5] = rhos * alpha_s * (vx + slow_magnetosonic_wave)
    right_eigenvectors[...,ordinate+1,5] = rhos * (alpha_s*vy + alpha_f*fast_magnetosonic_wave*beta2*S)
    right_eigenvectors[...,applicate+1,5] = rhos * (alpha_s*vz + alpha_f*fast_magnetosonic_wave*beta3*S)
    right_eigenvectors[...,4,5] = psi_plus_slow
    right_eigenvectors[...,ordinate+5,5] = -alpha_f * sound_speed * beta2 * np.sqrt(rhos)
    right_eigenvectors[...,applicate+5,5] = -alpha_f * sound_speed * beta3 * np.sqrt(rhos)
    # Seventh column (Alfven+ wave)
    right_eigenvectors[...,ordinate+1,6] = beta3 * rhos**1.5
    right_eigenvectors[...,applicate+1,6] = -beta2 * rhos**1.5
    right_eigenvectors[...,4,6] = (beta3*vy - beta2*vz) * rhos**1.5
    right_eigenvectors[...,ordinate+5,6] = -rhos * beta3
    right_eigenvectors[...,applicate+5,6] = rhos * beta2
    # Eighth column (Fast+ magnetoacoustic wave)
    right_eigenvectors[...,0,7] = rhos * alpha_f
    right_eigenvectors[...,abscissa+1,7] = rhos * alpha_f * (vx + fast_magnetosonic_wave)
    right_eigenvectors[...,ordinate+1,7] = rhos * (alpha_f*vy - alpha_s*slow_magnetosonic_wave*beta2*S)
    right_eigenvectors[...,applicate+1,7] = rhos * (alpha_f*vz - alpha_s*slow_magnetosonic_wave*beta3*S)
    right_eigenvectors[...,4,7] = psi_plus_fast
    right_eigenvectors[...,ordinate+5,7] = alpha_s * sound_speed * beta2 * np.sqrt(rhos)
    right_eigenvectors[...,applicate+5,7] = alpha_s * sound_speed * beta3 * np.sqrt(rhos)

    # Scale the right eigenvectors with a diagonal scaling matrix, so as to prevent degeneracies [Barth, 1999]
    diag_scaler = np.zeros_like(right_eigenvectors)
    diag_scaler[...,0,0] = 1/(2*gamma*rhos)
    diag_scaler[...,abscissa+1,abscissa+1] = fv.divide(pressures, 2*rhos**2)
    diag_scaler[...,ordinate+1,ordinate+1] = 1/(2*gamma*rhos)
    diag_scaler[...,applicate+1,applicate+1] = (rhos*(gamma-1))/gamma
    diag_scaler[...,4,4] = fv.divide(pressures, rhos)
    diag_scaler[...,abscissa+5,abscissa+5] = 1/(2*gamma*rhos)
    diag_scaler[...,ordinate+5,ordinate+5] = fv.divide(pressures, 2*rhos**2)
    diag_scaler[...,applicate+5,applicate+5] = 1/(2*gamma*rhos)
    R_dot = right_eigenvectors @ np.sqrt(diag_scaler)

    return R_dot