import numpy as np

from functions import fv

##############################################################################
# Functions for constructing objects such as the grid, eigenvectors, Jacobian and flux terms
##############################################################################

# Initialise the discrete POINTWISE solution array with initial conditions and primitive variables w, and transform into discrete AVERAGES <w>
# Gives option to convert to conservative variables <q>
# For magnetohydrodynamics, this returns a staggered grid
def initialise(sim_variables, convert=False):

    rho, vx, vy, vz, pressure, Bx, By, Bz = range(8)

    def make_physical_grid(_start_pos, _end_pos, _N):
        dx = abs(_end_pos-_start_pos)/_N
        half_cell = dx/2
        return np.linspace(_start_pos-half_cell, _end_pos+half_cell, _N+2)[1:-1]

    config, cells, gamma, dimension, precision, magnetic = sim_variables.config, sim_variables.cells, sim_variables.gamma, sim_variables.dimension, sim_variables.precision, sim_variables.magnetic
    start_pos, end_pos, shock_pos, params = sim_variables.start_pos, sim_variables.end_pos, sim_variables.shock_pos, sim_variables.misc
    initial_left, initial_right = sim_variables.initial_left, sim_variables.initial_right

    _i = (cells,) * dimension
    _i += (len(initial_right),)
    computational_grid = np.zeros(_i, dtype=precision)
    computational_grid[:] = initial_right

    physical_grid = make_physical_grid(start_pos, end_pos, cells)

    if dimension == 2:
        x, y = np.meshgrid(physical_grid, physical_grid, indexing='ij')
        centre = (end_pos+start_pos)/2

        if config == "sedov" or "blast" in config:
            mask = np.where(((x-centre)**2 + (y-centre)**2) <= (shock_pos-centre)**2)
            computational_grid[mask] = initial_left

        elif config.startswith("gauss"):
            r = np.sqrt((x-centre)**2 + (y-centre)**2)
            mask = params['y_offset'] + params['ampl']*np.exp(-((r-centre)**2)/params['fwhm'])
            computational_grid[...,rho] = mask

        elif config in ["khi", "kelvin-helmholtz"] or ("kelvin" in config or "helmholtz" in config):
            computational_grid[np.where(y <= shock_pos)] = initial_left
            computational_grid[...,vy] = params['perturb_ampl'] * np.sin(params['freq']*np.pi*x/(end_pos-start_pos))

        elif config in ["ivc", "vortex", "isentropic vortex"]:
            r = np.sqrt((x-centre)**2 + (y-centre)**2)
            b, freq = params['vortex_str'], params['freq']

            T = (1 - (((gamma-1)*b**2)/(freq*gamma*(2*np.pi)**2) * np.exp(1 - r**2)))**(1/(gamma-1))

            computational_grid[...,rho] = T
            computational_grid[...,vx] = 1 - (b/(freq*np.pi) * np.exp((1-r**2)/freq) * (y-centre))
            computational_grid[...,vy] = b/(freq*np.pi) * np.exp((1-r**2)/freq) * (x-centre)
            computational_grid[...,pressure] = T**(gamma)

        elif "gresho" in config:
            r = np.sqrt((x-centre)**2 + (y-centre)**2)
            core, ring = np.where((0 <= r) & (r < .2)), np.where((.2 <= r) & (r < .4))
            rx, ry = -np.sin(np.arctan2(y,x)), np.cos(np.arctan2(y,x))

            v_phi = 5 * r
            computational_grid[...,vx][core] = (v_phi * rx)[core]
            computational_grid[...,vy][core] = (v_phi * ry)[core]

            v_phi = 2 - 5*r
            computational_grid[...,vx][ring] = (v_phi * rx)[ring]
            computational_grid[...,vy][ring] = (v_phi * ry)[ring]

            v_phi = 1
            p0 = (initial_left[...,rho] * v_phi)/(gamma*params['mach']**2) - .5
            computational_grid[...,pressure] = p0 - 2 + 4*np.log(2)
            computational_grid[...,pressure][core] = (p0 + (25/2)*r**2)[core]
            computational_grid[...,pressure][ring] = (p0 + (25/2)*r**2 + 4*(1 - 5*r + np.log(5*r)))[ring]

        elif "ll" in config or "lax-liu" in config:
            computational_grid[np.where(x <= shock_pos)] = initial_left
            computational_grid[np.where((x <= shock_pos) & (y >= shock_pos))] = params['bottom_left']
            computational_grid[np.where((x > shock_pos) & (y >= shock_pos))] = params['bottom_right']

        elif config in ["orszag-tang", "orszag", "tang", "ot"]:
            computational_grid[...,vx] = -np.sin(2*np.pi*y)
            computational_grid[...,vy] = np.sin(2*np.pi*x)
            computational_grid[...,Bx] = -params['ampl'] * np.sin(2*np.pi*y)
            computational_grid[...,By] = params['ampl'] * np.sin(4*np.pi*x)

        elif "rotor" in config:
            mask = np.where(((x-centre)**2 + (y-centre)**2) <= (shock_pos-centre)**2)
            computational_grid[mask] = initial_left
            computational_grid[mask][...,vx] = (-params['omega']*(y-centre)/shock_pos)[mask]
            computational_grid[mask][...,vy] = (params['omega']*(x-centre)/shock_pos)[mask]

        elif "noh" in config:
            mask = np.where(((x-start_pos)**2 + (y-start_pos)**2) > (shock_pos-start_pos)**2)
            computational_grid[mask][...,vx] = -np.sin(x)[mask]
            computational_grid[mask][...,vy] = -np.cos(x)[mask]

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
            computational_grid[np.where(x > shock_pos), rho] = fv.sine_func(x[x > shock_pos], params)
        elif config.startswith("sin"):
            computational_grid[...,rho] = fv.sine_func(x, params)
        elif config.startswith('gauss'):
            computational_grid[...,rho] = fv.gauss_func(x, params)

    if convert:
        return fv.point_convert_primitive(computational_grid, sim_variables, staggered=magnetic)
    else:
        return computational_grid


# Make flux as a function of cell-averaged (primitive) variables
def make_flux(grid, sim_variables, axis):
    gamma, permeability = sim_variables.gamma, sim_variables.permeability

    rhos, vels, pressures, B_fields = grid[...,0], grid[...,1:4], grid[...,4], grid[...,5:8]
    abscissa, ordinate, applicate = (axis + np.array(range(3)))%3
    arr = np.zeros_like(grid)

    arr[...,0] = rhos * vels[...,abscissa]
    arr[...,abscissa+1] = rhos*vels[...,abscissa]**2 + pressures + .5*fv.norm(B_fields)**2 - (B_fields[...,abscissa]**2)/permeability
    arr[...,ordinate+1] = rhos*vels[...,abscissa]*vels[...,ordinate] - (B_fields[...,abscissa]*B_fields[...,ordinate])/permeability
    arr[...,applicate+1] = rhos*vels[...,abscissa]*vels[...,applicate] - (B_fields[...,abscissa]*B_fields[...,applicate])/permeability
    arr[...,4] = vels[...,abscissa]*(.5*rhos*fv.norm(vels)**2 + (gamma*pressures)/(gamma-1) + fv.norm(B_fields)**2) - (B_fields[...,abscissa]*np.sum(vels*B_fields, axis=-1))/permeability
    arr[...,ordinate+5] = B_fields[...,ordinate]*vels[...,abscissa] - B_fields[...,abscissa]*vels[...,ordinate]
    arr[...,applicate+5] = B_fields[...,applicate]*vels[...,abscissa] - B_fields[...,abscissa]*vels[...,applicate]

    return arr


# Jacobian matrix based on primitive variables [Winters & Gassner, 2016]
def make_Jacobian(grid, sim_variables, axis):
    gamma, permeability = sim_variables.gamma, sim_variables.permeability

    rhos, vels, pressures, B_fields = grid[...,0], grid[...,1:4], grid[...,4], grid[...,5:8]
    abscissa, ordinate, applicate = (axis + np.array(range(3)))%3

    # Create empty square arrays for each cell
    _arr = np.zeros_like(grid)
    arr = np.repeat(_arr[...,None], _arr.shape[-1], axis=-1)
    i, j = np.diag_indices(_arr.shape[-1])

    # Input matrix with values at position [row i, col j]; positions refer to x-axis arrangement, but will permute based on the axis
    # Hydrodynamic components
    arr[...,i,j] = vels[...,abscissa][...,None]  # diagonal elements
    arr[...,0,abscissa+1] = rhos  # [0,1]
    arr[...,abscissa+1,4] = 1/rhos  # [1,4]
    arr[...,4,abscissa+1] = gamma * pressures  # [4,1]

    # Magnetic field components
    arr[...,ordinate+1,ordinate+5] = arr[...,applicate+1,applicate+5] = -fv.divide(B_fields[...,abscissa], rhos*permeability)  # [2,6] = [3,7]
    arr[...,abscissa+1,ordinate+5] = fv.divide(B_fields[...,ordinate], rhos*permeability)  # [1,6]
    arr[...,abscissa+1,applicate+5] = fv.divide(B_fields[...,applicate], rhos*permeability)  # [1,7]

    arr[...,ordinate+5,ordinate+1] = arr[...,applicate+5,applicate+1] = -B_fields[...,abscissa]  # [6,2] = [7,3]
    arr[...,ordinate+5,abscissa+1] = B_fields[...,ordinate]  # [6,1]
    arr[...,applicate+5,abscissa+1] = B_fields[...,applicate]  # [7,1]

    return arr


# Calculate the Roe-averaged primitive variables at the interface from the minus- & plus-interface states for use in Roe solver in order to better capture shocks [Roe & Pike, 1984; Brio & Wu, 1988; LeVeque, 2002; Stone et al., 2008]
def make_Roe_average(left_interface, right_interface):
    rho, pressure, vels, Bfields = 0, 4, slice(1,4), slice(5,8)
    avg = np.zeros_like(left_interface)
    rho_minus, rho_plus = np.sqrt(right_interface[...,rho]), np.sqrt(left_interface[...,rho])

    avg[...,rho] = rho_minus * rho_plus
    avg[...,vels] = fv.divide((left_interface[...,vels] * rho_plus[...,None]) + (right_interface[...,vels] * rho_minus[...,None]), (rho_minus + rho_plus)[...,None])
    avg[...,pressure] = fv.divide((rho_plus * left_interface[...,pressure]) + (rho_minus * right_interface[...,pressure]), rho_minus + rho_plus)
    avg[...,Bfields] = fv.divide((left_interface[...,Bfields] * rho_minus[...,None]) + (right_interface[...,Bfields] * rho_plus[...,None]), (rho_minus + rho_plus)[...,None])

    return avg


# Make the right eigenvectors for adiabatic magnetohydrodynamics [Derigs]
def make_right_eigenvectors(axis, grids, gamma):
    abscissa, ordinate, applicate = (axis + np.array(range(3)))%3
    rhos, vels, pressures, B_fields = grids[...,0], grids[...,1:4], grids[...,4], grids[...,5:8]
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
    abscissa, ordinate, applicate = (axis + np.array(range(3)))%3
    rhos, vels, pressures, B_fields = grids[...,0], grids[...,1:4], grids[...,4], grids[...,5:8]
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