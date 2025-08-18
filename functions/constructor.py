import numpy as np

from functions import fv

##############################################################################
# Functions for constructing objects such as the grid, eigenvectors, Jacobian and flux terms
##############################################################################

# Initialise the discrete POINTWISE solution array with initial conditions and primitive variables w, and transform into discrete AVERAGES <w>
# For magnetohydrodynamics, this returns a staggered grid
def initialise(sim_variables):

    # Create a physical grid for a single axis
    def make_physical_grid(_axis, _cells):
        dh = np.abs(np.diff(_axis)[0])/_cells
        half_cell = dh/2
        return np.linspace(_axis[0]-half_cell, _axis[1]+half_cell, _cells+2)[1:-1]

    config, cells, gamma, dimension, precision = sim_variables.config, sim_variables.cells, sim_variables.gamma, sim_variables.dimension, sim_variables.precision
    rho, vx, vy, pressure, Bx, By = sim_variables.rho, sim_variables.vx, sim_variables.vy, sim_variables.pressure, sim_variables.Bx, sim_variables.By
    x_axis, y_axis, params = sim_variables.x_axis, sim_variables.y_axis, sim_variables.misc
    initial_left, initial_right = sim_variables.initial_left, sim_variables.initial_right
    x_shock_pos, y_shock_pos = sim_variables.shock_pos

    computational_grid = np.zeros(list(cells)+[len(initial_right),], dtype=precision)
    computational_grid[:] = initial_right

    physical_grid_x = make_physical_grid(x_axis, cells[0])

    if dimension == 2:
        physical_grid_y = make_physical_grid(y_axis, cells[1])
        x, y = np.meshgrid(physical_grid_x, physical_grid_y, indexing='ij')
        x_centre, y_centre = np.average(x_axis), np.average(y_axis)

        if config == "sedov" or "blast" in config:
            mask = np.where(((x-x_centre)**2 + (y-y_centre)**2) <= (x_shock_pos-x_centre)**2)
            computational_grid[mask] = initial_left

        elif config.startswith("gauss"):
            r = np.sqrt((x-x_centre)**2 + (y-y_centre)**2)
            mask = params['y_offset'] + params['ampl']*np.exp(-(r**2)/params['fwhm'])
            computational_grid[...,rho] = mask

        elif config in ["khi", "kelvin-helmholtz"] or ("kelvin" in config or "helmholtz" in config):
            computational_grid[np.where(y <= y_shock_pos)] = initial_left
            computational_grid[...,vy] = params['perturb_ampl'] * np.sin(params['freq']*np.pi*x/np.diff(x_axis))

        elif config in ["ivc", "isentropic"]:
            r = np.sqrt((x-x_centre)**2 + (y-y_centre)**2)
            b, freq = params['vortex_str'], params['freq']

            T = (1 - (((gamma-1)*b**2)/(freq*gamma*(2*np.pi)**2) * np.exp(1 - r**2)))**(1/(gamma-1))

            computational_grid[...,rho] = T
            computational_grid[...,vx] = 1 - (b/(freq*np.pi) * np.exp((1-r**2)/freq) * (y-y_centre))
            computational_grid[...,vy] = b/(freq*np.pi) * np.exp((1-r**2)/freq) * (x-x_centre)
            computational_grid[...,pressure] = T**(gamma)

        elif "gresho" in config:
            r = np.sqrt((x-x_centre)**2 + (y-y_centre)**2)
            core, ring = np.where((0 <= r) & (r < .2)), np.where((.2 <= r) & (r < .4))
            rx, ry = -np.sin(np.arctan2(y-y_centre,x-x_centre)), np.cos(np.arctan2(y-y_centre,x-x_centre))
            p0 = initial_left[...,rho]/(gamma*params['mach']**2)

            computational_grid[...,pressure] = p0 - 2 + 4*np.log(2)

            v_phi = 5 * r
            computational_grid[...,vx][core] = (v_phi * rx)[core]
            computational_grid[...,vy][core] = (v_phi * ry)[core]
            computational_grid[...,pressure][core] = (p0 + (25/2)*r**2)[core]

            v_phi = 2 - 5*r
            computational_grid[...,vx][ring] = (v_phi * rx)[ring]
            computational_grid[...,vy][ring] = (v_phi * ry)[ring]
            computational_grid[...,pressure][ring] = (p0 + (25/2)*r**2 + 4*(1 - 5*r + np.log(5*r)))[ring]

        elif "ll" in config or "lax-liu" in config:
            computational_grid[np.where(x < x_shock_pos)] = initial_left
            computational_grid[np.where((x < x_shock_pos) & (y < y_shock_pos))] = params['bottom_left']
            computational_grid[np.where((x >= x_shock_pos) & (y < y_shock_pos))] = params['bottom_right']

        elif config in ["orszag-tang", "orszag", "tang", "ot"]:
            computational_grid[...,vx] = -np.sin(2*np.pi*y)
            computational_grid[...,vy] = np.sin(2*np.pi*x)
            computational_grid[...,Bx] = -params['ampl'] * np.sin(2*np.pi*y)
            computational_grid[...,By] = params['ampl'] * np.sin(4*np.pi*x)

        elif "rotor" in config:
            mask = np.where(((x-x_centre)**2 + (y-y_centre)**2) <= (x_shock_pos-x_centre)**2)
            computational_grid[mask] = initial_left
            computational_grid[...,vx][mask] = (-params['omega']*(y-y_centre)/y_shock_pos)[mask]
            computational_grid[...,vy][mask] = (params['omega']*(x-x_centre)/x_shock_pos)[mask]

        elif "sheet" in config or "current" in config:
            computational_grid[...,vx] = params['ampl'] * np.sin(2*np.pi*y)
            mask = np.where((-x_shock_pos < x) & (x < x_shock_pos))
            computational_grid[...,By][mask] = -computational_grid[...,By][mask]

        elif "noh" in config:
            mask = np.where(((x-x_axis[0])**2 + (y-y_axis[0])**2) > (x_shock_pos-x_axis[0])**2)
            computational_grid[...,vx][mask] = -np.sin(x-x_shock_pos)[mask]
            computational_grid[...,vy][mask] = -np.cos(x-x_shock_pos)[mask]

        else:
            computational_grid[np.where(x < x_shock_pos)] = initial_left

    else:
        x = physical_grid_x

        if config == "sedov" or config.startswith('sq'):
            mask = np.where(np.abs(x) <= x_shock_pos)
        else:
            mask = np.where(x <= x_shock_pos)

        computational_grid[mask] = initial_left

        if "shu" in config or "osher" in config:
            computational_grid[np.where(x > x_shock_pos), rho] = fv.sine_func(x[x > x_shock_pos], params)
        elif config.startswith("sin"):
            computational_grid[...,rho] = fv.sine_func(x, params)
        elif config.startswith('gauss'):
            computational_grid[...,rho] = fv.gauss_func(x, params)

    return computational_grid


# Make flux as a function of cell-averaged (primitive) variables
def make_flux(grid, sim_variables, axis):
    abscissa, ordinate, applicate = (axis + np.array(range(3)))%3
    gamma, permeability = sim_variables.gamma, sim_variables.permeability

    rhos, vels, pressures, Bfields = grid[...,sim_variables.rho], grid[...,sim_variables.vels], grid[...,sim_variables.pressure], grid[...,sim_variables.Bfields]
    arr = np.zeros_like(grid)

    arr[...,0] = rhos * vels[...,abscissa]
    arr[...,1+abscissa] = rhos*vels[...,abscissa]**2 + pressures + .5*fv.norm(Bfields)**2 - (Bfields[...,abscissa]**2)/permeability
    arr[...,1+ordinate] = rhos*vels[...,abscissa]*vels[...,ordinate] - (Bfields[...,abscissa]*Bfields[...,ordinate])/permeability
    arr[...,1+applicate] = rhos*vels[...,abscissa]*vels[...,applicate] - (Bfields[...,abscissa]*Bfields[...,applicate])/permeability
    arr[...,4] = vels[...,abscissa]*(.5*rhos*fv.norm(vels)**2 + (gamma*pressures)/(gamma-1) + fv.norm(Bfields)**2) - (Bfields[...,abscissa]*np.sum(vels*Bfields, axis=-1))/permeability
    arr[...,5+ordinate] = Bfields[...,ordinate]*vels[...,abscissa] - Bfields[...,abscissa]*vels[...,ordinate]
    arr[...,5+applicate] = Bfields[...,applicate]*vels[...,abscissa] - Bfields[...,abscissa]*vels[...,applicate]

    return arr


# Jacobian matrix based on primitive variables [Winters & Gassner, 2016]
def make_Jacobian(grid, sim_variables, axis):
    abscissa, ordinate, applicate = (axis + np.array(range(3)))%3
    gamma, permeability = sim_variables.gamma, sim_variables.permeability

    rhos, vels, pressures, Bfields = grid[...,sim_variables.rho], grid[...,sim_variables.vels], grid[...,sim_variables.pressure], grid[...,sim_variables.Bfields]

    # Create empty square arrays for each cell
    _arr = np.zeros_like(grid)
    arr = np.repeat(_arr[...,None], _arr.shape[-1], axis=-1)
    i, j = np.diag_indices(_arr.shape[-1])

    # Input matrix with values at position [row i, col j]; positions refer to x-axis arrangement, but will permute based on the axis
    # Hydrodynamic components
    arr[...,i,j] = vels[...,abscissa][...,None]  # diagonal elements
    arr[...,0,1+abscissa] = rhos  # [0,1]
    arr[...,1+abscissa,4] = 1/rhos  # [1,4]
    arr[...,4,1+abscissa] = gamma * pressures  # [4,1]

    # Magnetic field components
    arr[...,1+ordinate,5+ordinate] = arr[...,applicate+1,applicate+5] = -fv.divide(Bfields[...,abscissa], rhos*permeability)  # [2,6] = [3,7]
    arr[...,1+abscissa,5+ordinate] = fv.divide(Bfields[...,ordinate], rhos*permeability)  # [1,6]
    arr[...,1+abscissa,5+applicate] = fv.divide(Bfields[...,applicate], rhos*permeability)  # [1,7]

    arr[...,5+ordinate,1+ordinate] = arr[...,applicate+5,applicate+1] = -Bfields[...,abscissa]  # [6,2] = [7,3]
    arr[...,5+ordinate,1+abscissa] = Bfields[...,ordinate]  # [6,1]
    arr[...,5+applicate,1+abscissa] = Bfields[...,applicate]  # [7,1]

    return arr


# Compute wavespeeds for a grid
def make_wavespeeds(grid, sim_variables, axis):
    gamma, permeability = sim_variables.gamma, sim_variables.permeability
    rho, pressure, Bfields = sim_variables.rho, sim_variables.pressure, sim_variables.Bfields

    sound_speed = np.sqrt(fv.divide(gamma*grid[...,pressure], grid[...,rho]))
    alfven_speed = fv.divide(fv.norm(grid[...,Bfields]), np.sqrt(grid[...,rho]*permeability))
    alfven_speed_x = fv.divide(grid[...,5+axis], np.sqrt(grid[...,rho]*permeability))
    fast_magnetosonic_wave = np.sqrt(.5 * (sound_speed**2 + alfven_speed**2 + np.sqrt((sound_speed**2 + alfven_speed**2)**2 - (2 * sound_speed * alfven_speed_x)**2)))
    slow_magnetosonic_wave = np.sqrt(.5 * (sound_speed**2 + alfven_speed**2 - np.sqrt((sound_speed**2 + alfven_speed**2)**2 - (2 * sound_speed * alfven_speed_x)**2)))

    return sound_speed, alfven_speed_x, fast_magnetosonic_wave, slow_magnetosonic_wave


# Make the right eigenvectors for adiabatic magnetohydrodynamics [Derigs et al., 2016]
def make_right_eigenvectors(grids, sim_variables, axis):
    abscissa, ordinate, applicate = (axis + np.array(range(3)))%3
    gamma = sim_variables.gamma

    rhos, vels, Bfields = grids[...,sim_variables.rho], grids[...,sim_variables.vels], grids[...,sim_variables.Bfields]
    vx, vy, vz = vels[...,abscissa], vels[...,ordinate], vels[...,applicate]
    Bx, By, Bz = Bfields[...,abscissa], Bfields[...,ordinate], Bfields[...,applicate]

    # Define the right eigenvectors for each cell in each grid
    _right_eigenvectors = np.zeros_like(grids)
    right_eigenvectors = np.repeat(_right_eigenvectors[...,None], _right_eigenvectors.shape[-1], axis=-1)

    # Compute wavespeeds
    sound_speed, alfven_speed_x, fast_magnetosonic_wave, slow_magnetosonic_wave = make_wavespeeds(grids, sim_variables, axis)

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
def make_ES_right_eigenvectors(grids, sim_variables, axis):
    abscissa, ordinate, applicate = (axis + np.array(range(3)))%3
    gamma = sim_variables.gamma

    rhos, vels, pressures, Bfields = grids[...,sim_variables.rho], grids[...,sim_variables.vels], grids[...,sim_variables.pressure], grids[...,sim_variables.Bfields]
    vx, vy, vz = vels[...,abscissa], vels[...,ordinate], vels[...,applicate]
    Bx, By, Bz = Bfields[...,abscissa], Bfields[...,ordinate], Bfields[...,applicate]

    # Define the right eigenvectors for each cell in each grid
    _right_eigenvectors = np.zeros_like(grids)
    right_eigenvectors = np.repeat(_right_eigenvectors[...,None], _right_eigenvectors.shape[-1], axis=-1)

    # Compute wavespeeds
    a, bx, cf, cs = make_wavespeeds(grids, sim_variables, axis)

    # Define frequently used components
    sigma = np.sign(Bx)
    sigma[sigma == 0] = 1
    alpha_f = np.sqrt(fv.divide(a**2 - cs**2, cf**2 - cs**2))
    alpha_s = np.sqrt(fv.divide(cf**2 - a**2, cf**2 - cs**2))
    b_perpend = np.sqrt(fv.divide(By**2 + Bz**2, rhos))
    beta = lambda b_axis: fv.divide(b_axis, b_perpend)

    psi_plus_slow = (
        .5 * alpha_s * rhos * fv.norm(vels)**2
        - a * alpha_f * rhos * b_perpend
        + (alpha_s * rhos * a**2)/(gamma-1)
        + alpha_s * cs * rhos * vx
        + alpha_f * cf * rhos * sigma * (vy*beta(By) * vz*beta(Bz))
    )
    psi_minus_slow = (
        .5 * alpha_s * rhos * fv.norm(vels)**2
        - a * alpha_f * rhos * b_perpend
        + (alpha_s * rhos * a**2)/(gamma-1)
        - alpha_s * cs * rhos * vx
        - alpha_f * cf * rhos * sigma * (vy*beta(By) * vz*beta(Bz))
    )
    psi_plus_fast = (
        .5 * alpha_f * rhos * fv.norm(vels)**2
        + a * alpha_s * rhos * b_perpend
        + (alpha_f * rhos * a**2)/(gamma-1)
        + alpha_f * cf * rhos * vx
        - alpha_s * cs * rhos * sigma * (vy*beta(By) * vz*beta(Bz))
    )
    psi_minus_fast = (
        .5 * alpha_f * rhos * fv.norm(vels)**2
        + a * alpha_s * rhos * b_perpend
        + (alpha_f * rhos * a**2)/(gamma-1)
        - alpha_f * cf * rhos * vx
        + alpha_s * cs * rhos * sigma * (vy*beta(By) * vz*beta(Bz))
    )

    # Generate the right eigenvectors
    # First column (Fast+ magnetoacoustic wave)
    right_eigenvectors[...,0,0] = alpha_f * rhos
    right_eigenvectors[...,1+abscissa,0] = alpha_f * rhos * (vx + cf)
    right_eigenvectors[...,1+ordinate,0] = rhos * (alpha_f*vy - alpha_s*cs*beta(By)*sigma)
    right_eigenvectors[...,1+applicate,0] = rhos * (alpha_f*vz - alpha_s*cs*beta(Bz)*sigma)
    right_eigenvectors[...,4,0] = psi_plus_fast
    right_eigenvectors[...,5+ordinate,0] = alpha_s * a * beta(By) * np.sqrt(rhos)
    right_eigenvectors[...,5+applicate,0] = alpha_s * a * beta(Bz) * np.sqrt(rhos)
    # Second column (Alfven+ wave)
    right_eigenvectors[...,1+ordinate,1] = beta(Bz) * rhos**1.5
    right_eigenvectors[...,1+applicate,1] = -beta(By) * rhos**1.5
    right_eigenvectors[...,4,1] = -(beta(By)*vz - beta(Bz)*vy) * rhos**1.5
    right_eigenvectors[...,5+ordinate,1] = -rhos * beta(Bz)
    right_eigenvectors[...,5+applicate,1] = rhos * beta(By)
    # Third column (Slow+ magnetoacoustic wave)
    right_eigenvectors[...,0,2] = alpha_s * rhos
    right_eigenvectors[...,1+abscissa,2] = alpha_s * rhos * (vx + cs)
    right_eigenvectors[...,1+ordinate,2] = rhos * (alpha_s*vy + alpha_f*cf*beta(By)*sigma)
    right_eigenvectors[...,1+applicate,2] = rhos * (alpha_s*vz + alpha_f*cf*beta(Bz)*sigma)
    right_eigenvectors[...,4,2] = psi_plus_slow
    right_eigenvectors[...,5+ordinate,2] = -alpha_f * a * beta(By) * np.sqrt(rhos)
    right_eigenvectors[...,5+applicate,2] = -alpha_f * a * beta(Bz) * np.sqrt(rhos)
    # Fourth column (Entropy wave)
    right_eigenvectors[...,0,3] = 1
    right_eigenvectors[...,1+abscissa,3] = vx
    right_eigenvectors[...,1+ordinate,3] = vy
    right_eigenvectors[...,1+applicate,3] = vz
    right_eigenvectors[...,4,3] = .5 * fv.norm(vels)**2
    # Fifth column (Divergence wave)
    right_eigenvectors[...,4,4] = Bx
    right_eigenvectors[...,5+abscissa,4] = 1
    # Sixth column (Slow- magnetoacoustic wave)
    right_eigenvectors[...,0,5] = alpha_s * rhos
    right_eigenvectors[...,1+abscissa,5] = alpha_s * rhos * (vx - cs)
    right_eigenvectors[...,1+ordinate,5] = rhos * (alpha_s*vy - alpha_f*cf*beta(By)*sigma)
    right_eigenvectors[...,1+applicate,5] = rhos * (alpha_s*vz - alpha_f*cf*beta(Bz)*sigma)
    right_eigenvectors[...,4,5] = psi_minus_slow
    right_eigenvectors[...,5+ordinate,5] = -alpha_f * a * beta(By) * np.sqrt(rhos)
    right_eigenvectors[...,5+applicate,5] = -alpha_f * a * beta(Bz) * np.sqrt(rhos)
    # Seventh column (Alfven- wave)
    right_eigenvectors[...,1+ordinate,6] = -beta(Bz) * rhos**1.5
    right_eigenvectors[...,1+applicate,6] = beta(By) * rhos**1.5
    right_eigenvectors[...,4,6] = (beta(By)*vz - beta(Bz)*vy) * rhos**1.5
    right_eigenvectors[...,5+ordinate,6] = -rhos * beta(Bz)
    right_eigenvectors[...,5+applicate,6] = rhos * beta(By)
    # Eighth column (Fast- magnetoacoustic wave)
    right_eigenvectors[...,0,7] = alpha_f * rhos
    right_eigenvectors[...,1+abscissa,0] = alpha_f * rhos * (vx - cf)
    right_eigenvectors[...,1+ordinate,0] = rhos * (alpha_f*vy + alpha_s*cs*beta(By)*sigma)
    right_eigenvectors[...,1+applicate,0] = rhos * (alpha_f*vz + alpha_s*cs*beta(Bz)*sigma)
    right_eigenvectors[...,4,0] = psi_minus_fast
    right_eigenvectors[...,5+ordinate,0] = alpha_s * a * beta(By) * np.sqrt(rhos)
    right_eigenvectors[...,5+applicate,0] = alpha_s * a * beta(Bz) * np.sqrt(rhos)

    # Scale the right eigenvectors with a diagonal scaling matrix, so as to prevent degeneracies [Barth, 1999]
    diag_scaler = np.zeros_like(right_eigenvectors)
    diag_scaler[...,0,0] = 1/(2*gamma*rhos)
    diag_scaler[...,1+abscissa,1+abscissa] = fv.divide(pressures, 2*rhos**3)
    diag_scaler[...,1+ordinate,1+ordinate] = 1/(2*gamma*rhos)
    diag_scaler[...,1+applicate,1+applicate] = (rhos*(gamma-1))/gamma
    diag_scaler[...,4,4] = fv.divide(pressures, rhos)
    diag_scaler[...,5+abscissa,5+abscissa] = 1/(2*gamma*rhos)
    diag_scaler[...,5+ordinate,5+ordinate] = fv.divide(pressures, 2*rhos**3)
    diag_scaler[...,5+applicate,5+applicate] = 1/(2*gamma*rhos)
    R_dot = right_eigenvectors @ np.sqrt(diag_scaler)

    return R_dot