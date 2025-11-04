import numpy as np

from functions import fv
from functions.generic import verbose_timer

##############################################################################
# Functions for constructing objects such as the grid, eigenvectors, Jacobian and flux terms
##############################################################################

# Initialise the discrete POINTWISE solution array with initial conditions and primitive variables w, and transform into discrete AVERAGES <w>
# For magnetohydrodynamics, this returns a staggered grid
@verbose_timer
def initialise(sim_variables):

    # Create a physical grid for a single axis
    def make_physical_grid(_coord, _cells):
        start_pos, end_pos = _coord
        dh = np.abs(np.diff(_coord)[0])/_cells
        half_cell = dh/2
        return np.linspace(start_pos-half_cell, end_pos+half_cell, _cells+2)[1:-1]

    config, cells, gamma, dimensions, multidimensional, precision = sim_variables.config, sim_variables.cells, sim_variables.gamma, sim_variables.dimensions, sim_variables.multidimensional, sim_variables.precision
    rho, vx, vy, vz, pressure, Bx, By, Bz = sim_variables.rho, sim_variables.vx, sim_variables.vy, sim_variables.vz, sim_variables.pressure, sim_variables.Bx, sim_variables.By, sim_variables.Bz
    axis_coord, shock_pos, params = sim_variables.axis_coord, sim_variables.shock_pos, sim_variables.misc
    initial_left, initial_right = sim_variables.initial_left, sim_variables.initial_right
    axes = sim_variables.axes


    computational_grid = np.zeros(list(cells)+[len(initial_right),], dtype=precision)
    computational_grid[:] = initial_right

    centre = np.average(axis_coord)
    physical_grid_x = make_physical_grid(axis_coord, cells[0])

    if multidimensional:
        physical_grid_y = make_physical_grid(axis_coord, cells[1])

        if dimensions > 2:
            physical_grid_z = make_physical_grid(axis_coord, cells[2])
            x, y, z = np.meshgrid(physical_grid_x, physical_grid_y, physical_grid_z, indexing='ij')
            r = np.sqrt((x-centre)**2 + (y-centre)**2 + (z-centre)**2)

            if "sedov" in config or "blast" in config:
                mask = np.where(r**2 <= (shock_pos-centre)**2)
                computational_grid[mask] = initial_left
                if config.startswith("mhd"):
                    computational_grid[...,5+axes] = params['ampl']

            elif config.startswith("sin"):
                computational_grid[...,rho] = params['y_offset'] + params['ampl']*np.sin(params['freq']*np.pi*r)

            elif config.startswith("gauss"):
                mask = params['y_offset'] + params['ampl']*np.exp(-(r**2)/params['fwhm'])
                computational_grid[...,rho] = mask

            elif config in ["orszag-tang", "orszag", "tang", "ot"]:
                computational_grid[...,vx] = -np.sin(2*np.pi*y)
                computational_grid[...,vy] = np.sin(2*np.pi*x)
                computational_grid[...,Bx] = params['ampl'] * -np.sin(2*np.pi*y)
                computational_grid[...,By] = params['ampl'] * np.sin(4*np.pi*x)

            elif "vortex" in config and config.startswith("mhd"):
                factor = np.exp(params['q'] * (1 - r**2))
                computational_grid[...,vx] = 1 - (y-centre)*params['kappa']*factor
                computational_grid[...,vy] = 1 + (x-centre)*params['kappa']*factor
                computational_grid[...,pressure] = 1 + (1/(4*params['q'])) * ((1 - 2*params['q']*(r**2 - (z-centre)**2)) * params['mu']**2 - params['kappa']**2) * factor**2
                computational_grid[...,Bx] = -(y-centre)*params['mu']*factor
                computational_grid[...,By] = (x-centre)*params['mu']*factor

        else:
            x, y = np.meshgrid(physical_grid_x, physical_grid_y, indexing='ij')
            r = np.sqrt((x-centre)**2 + (y-centre)**2)

            if "sedov" in config or "blast" in config:
                mask = np.where(r**2 <= (shock_pos-centre)**2)
                computational_grid[mask] = initial_left
                if config.startswith("mhd"):
                    computational_grid[...,5+axes] = params['ampl']

            elif config.startswith("sin"):
                computational_grid[...,rho] = params['y_offset'] + params['ampl']*np.sin(params['freq']*np.pi*r)

            elif config.startswith("gauss"):
                computational_grid[...,rho] = params['y_offset'] + params['ampl']*np.exp(-(r**2)/params['fwhm'])

            elif "kelvin" in config or "helmholtz" in config or "khi" in config:
                layer = np.where(np.abs(y) <= shock_pos)
                computational_grid[layer] = initial_left
                computational_grid[...,vy] = params['ampl'] * np.sin(params['freq']*np.pi*x/np.diff(axis_coord))
                perturbation = np.random.uniform(-params['perturb_ampl'], params['perturb_ampl'], size=(computational_grid[...,(vx,vy)][layer].shape))
                computational_grid[...,(vx,vy)][layer] += perturbation
                if config.startswith('m'):
                    computational_grid[...,Bx] = params['Bx']

            elif config in ["ivc", "isentropic"]:
                b, freq = params['vortex_str'], params['freq']

                dv = lambda _array: (b*np.exp(.5*(1-r**2))*_array)/(np.sqrt(freq)*np.pi)
                computational_grid[...,vx] = 1 + dv(-(y-centre))
                computational_grid[...,vy] = 1 + dv(x-centre)

                db = lambda _array: (b*np.exp(.5*(1-r**2))*_array)/(freq*np.pi)
                computational_grid[...,Bx] = db(-(y-centre))
                computational_grid[...,By] = db(x-centre)

                dp = ((1+r**2) * np.exp(1-r**2) * b**2)/(2 * (freq*np.pi)**2)
                computational_grid[...,pressure] = 1 + dp

            elif "gresho" in config:
                core, ring = np.where((0 <= r) & (r < .2)), np.where((.2 <= r) & (r < .4))
                rx, ry = -np.sin(np.arctan2(y-centre,x-centre)), np.cos(np.arctan2(y-centre,x-centre))
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
                computational_grid[np.where(x < shock_pos)] = initial_left
                computational_grid[np.where((x < shock_pos) & (y < shock_pos))] = params['bottom_left']
                computational_grid[np.where((x >= shock_pos) & (y < shock_pos))] = params['bottom_right']

            elif config in ["orszag-tang", "orszag", "tang", "ot"]:
                computational_grid[...,vx] = -np.sin(2*np.pi*y)
                computational_grid[...,vy] = np.sin(2*np.pi*x)
                computational_grid[...,Bx] = -params['ampl'] * np.sin(2*np.pi*y)
                computational_grid[...,By] = params['ampl'] * np.sin(4*np.pi*x)

            elif "vortex" in config and config.startswith("mhd"):
                computational_grid[...,vx] = 1 - (((y-centre)*params['kappa'])/(2*np.pi) * np.exp((1-r**2)/2))
                computational_grid[...,vy] = 1 + (((x-centre)*params['kappa'])/(2*np.pi) * np.exp((1-r**2)/2))
                computational_grid[...,pressure] = 1 + (((1-r**2)*params['kappa']**2 - params['mu']**2)/(8*np.pi**2) * np.exp(1-r**2))
                computational_grid[...,Bx] = (-(y-centre)*params['mu'])/(2*np.pi) * np.exp((1-r**2)/2)
                computational_grid[...,By] = ((x-centre)*params['mu'])/(2*np.pi) * np.exp((1-r**2)/2)

            elif "rotor" in config:
                f = (params['ring_pos'] - r)/(params['ring_pos'] - shock_pos)

                ring = np.where(r**2 <= (params['ring_pos']-centre)**2)
                computational_grid[...,rho][ring] = (1 + 9*f)[ring]
                computational_grid[...,vx][ring] = ((-f*params['omega']*(y-centre)*shock_pos)/r)[ring]
                computational_grid[...,vy][ring] = ((f*params['omega']*(x-centre)*shock_pos)/r)[ring]

                core = np.where(r**2 <= (shock_pos-centre)**2)
                computational_grid[core] = initial_left
                computational_grid[...,vx][core] = (-params['omega']*(y-centre))[core]
                computational_grid[...,vy][core] = (params['omega']*(x-centre))[core]

            elif "sheet" in config or "current" in config:
                computational_grid[...,vx] = params['ampl'] * np.sin(2*np.pi*y)
                mask = np.where((-shock_pos < x) & (x < shock_pos))
                computational_grid[...,By][mask] = -computational_grid[...,By][mask]

            elif "noh" in config:
                mask = np.where(((x-axis_coord[0])**2 + (y-axis_coord[0])**2) > (shock_pos-axis_coord[0])**2)
                computational_grid[...,vx][mask] = -np.sin(x-shock_pos)[mask]
                computational_grid[...,vy][mask] = -np.cos(x-shock_pos)[mask]

            elif "cloud" in config:
                mask = np.where(((x-.8)**2 + (y-.5)**2) < .15**2)
                computational_grid[np.where(x < shock_pos)] = initial_left
                computational_grid[...,rho][mask] = 10

            elif "jet" in config:
                mask = np.where((np.abs(x) < .05) & (y <= shock_pos))
                computational_grid[...,vy][mask] = 800

            elif "circular" in config or "polarised" in config or "alfven" in config or config == "cpaw":
                computational_grid[...,vx] = -params['A']/np.sqrt(2) * np.sin(2*np.pi*(x+y))
                computational_grid[...,vy] = params['A']/np.sqrt(2) * np.sin(2*np.pi*(x+y))
                computational_grid[...,vz] = params['A'] * np.cos(2*np.pi*(x+y))
                computational_grid[...,Bx] = params['ampl']/np.sqrt(2) + (params['A']/np.sqrt(2) * np.sin(2*np.pi*(x+y)))
                computational_grid[...,By] = params['ampl']/np.sqrt(2) - (params['A']/np.sqrt(2) * np.sin(2*np.pi*(x+y)))
                computational_grid[...,Bz] = -params['A'] * np.cos(2*np.pi*(x+y))

            else:
                computational_grid[np.where(x < shock_pos)] = initial_left

    else:
        x = physical_grid_x

        if "sedov" in config or config.startswith('sq') or "blast" in config:
            mask = np.where(np.abs(x) <= shock_pos)
        else:
            mask = np.where(x <= shock_pos)
        computational_grid[mask] = initial_left

        if config.startswith("mhd"):
            computational_grid[...,5+axes] = params['ampl']

        if "shu" in config or "osher" in config:
            computational_grid[np.where(x > shock_pos), rho] = fv.sine_func(x[x > shock_pos], params)
        elif config.startswith("sin"):
            computational_grid[...,rho] = fv.sine_func(x, params)
        elif config.startswith('gauss'):
            computational_grid[...,rho] = fv.gauss_func(x, params)

    sim_variables.magnetic = computational_grid[...,sim_variables.Bfields].any()

    return computational_grid


# Make flux as a function of cell-averaged (primitive) variables
def make_flux(grid, sim_variables, axis):
    abscissa, ordinate, applicate = (axis + np.array(range(3)))%3
    gamma, permeability = sim_variables.gamma, sim_variables.permeability

    # In code units
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

    # In code units
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

    sound_speed = np.sqrt(fv.divide(gamma * grid[...,pressure], grid[...,rho]))
    alfven_speed = fv.divide(fv.norm(grid[...,Bfields]), np.sqrt(grid[...,rho] * permeability))
    alfven_speed_x = fv.divide(grid[...,5+axis], np.sqrt(grid[...,rho] * permeability))
    fast_magnetosonic_wave = np.sqrt(.5 * (sound_speed**2 + alfven_speed**2 + np.sqrt((sound_speed**2 + alfven_speed**2)**2 - (2 * sound_speed * alfven_speed_x)**2)))
    slow_magnetosonic_wave = np.sqrt(.5 * (sound_speed**2 + alfven_speed**2 - np.sqrt((sound_speed**2 + alfven_speed**2)**2 - (2 * sound_speed * alfven_speed_x)**2)))

    return sound_speed, alfven_speed_x, fast_magnetosonic_wave, slow_magnetosonic_wave


# Make the left & right eigenvectors for adiabatic magnetohydrodynamics [Powell, 1994; Roe & Balsara, 1996; Stone et al., 2008; Derigs et al., 2016]
# Stone and Roe & Balsara only uses the 7-wave formulation, while Powell adds an 8th "divergence" wave to correct for the longitudinal magnetic field for divergence cleaning. Derigs modifies it further for entropy-stable formulation
# This formulation adopts the 7-wave formulation while adding the 8th-wave from Powell
def make_eigenvectors(grids, sim_variables, axis, vectors="both"):
    abscissa, ordinate, applicate = (axis + np.array(range(3)))%3

    # In code units
    rhos, Bfields = grids[...,sim_variables.rho], grids[...,sim_variables.Bfields]
    Bx, By, Bz = Bfields[...,abscissa], Bfields[...,ordinate], Bfields[...,applicate]

    # Define the left & right eigenvectors for each cell in each grid
    if vectors.casefold().startswith(("b", "l")):
        left_eigenvectors = np.repeat(np.zeros_like(grids)[...,None], grids.shape[-1], axis=-1)
    if vectors.casefold().startswith(("b", "r")):
        right_eigenvectors = np.repeat(np.zeros_like(grids)[...,None], grids.shape[-1], axis=-1)

    # Compute wavespeeds
    cs, cAx, caF, caS = make_wavespeeds(grids, sim_variables, axis)
    degenerate = np.where(cAx == cs)
    Nf = Ns = 1/(2*cs**2)

    # Define frequently used components
    S = np.sign(Bx)
    alpha_f = np.sqrt(fv.divide(cs**2 - caS**2, caF**2 - caS**2))
    alpha_s = np.sqrt(fv.divide(caF**2 - cs**2, caF**2 - caS**2))
    alpha_f[degenerate], alpha_s[degenerate] = 1, 0
    beta_y = fv.divide(By, np.sqrt(By**2 + Bz**2))
    beta_z = fv.divide(Bz, np.sqrt(By**2 + Bz**2))

    Cff, Css = caF * alpha_f, caS * alpha_s
    Qff, Qss = Cff * S, Css * S
    Af, As = cs * alpha_f * np.sqrt(rhos), cs * alpha_s * np.sqrt(rhos)

    # Generate the LEFT eigenvectors
    if vectors.casefold().startswith(("b", "l")):
        # First row (Fast- magnetoacoustic wave)
        left_eigenvectors[...,0,1+abscissa] = -Nf * Cff
        left_eigenvectors[...,0,1+ordinate] = Nf * Qss * beta_y
        left_eigenvectors[...,0,1+applicate] = Nf * Qss * beta_z
        left_eigenvectors[...,0,4] = fv.divide(Nf * alpha_f, rhos)
        left_eigenvectors[...,0,5+ordinate] = fv.divide(Nf * As * beta_y, rhos)
        left_eigenvectors[...,0,5+applicate] = fv.divide(Nf * As * beta_z, rhos)
        # Second row (Alfven- wave)
        left_eigenvectors[...,1,1+ordinate] = -beta_z/2
        left_eigenvectors[...,1,1+applicate] = beta_y/2
        left_eigenvectors[...,1,5+ordinate] = -fv.divide(beta_z * S, 2 * np.sqrt(rhos))
        left_eigenvectors[...,1,5+applicate] = fv.divide(beta_y * S, 2 * np.sqrt(rhos))
        # Third row (Slow- magnetoacoustic wave)
        left_eigenvectors[...,2,1+abscissa] = -Ns * Css
        left_eigenvectors[...,2,1+ordinate] = -Ns * Qff * beta_y
        left_eigenvectors[...,2,1+applicate] = -Ns * Qff * beta_z
        left_eigenvectors[...,2,4] = fv.divide(Ns * alpha_s, rhos)
        left_eigenvectors[...,2,5+ordinate] = -fv.divide(Ns * Af * beta_y, rhos)
        left_eigenvectors[...,2,5+applicate] = -fv.divide(Ns * Af * beta_z, rhos)
        # Fourth row (Entropy/contact wave)
        left_eigenvectors[...,3,0] = 1
        left_eigenvectors[...,3,4] = -2 * Nf
        # Fifth row (Divergence wave)
        left_eigenvectors[...,4,5+abscissa] = 1
        # Sixth row (Slow+ magnetoacoustic wave)
        left_eigenvectors[...,5,1+abscissa] = Ns * Css
        left_eigenvectors[...,5,1+ordinate] = Ns * Qff * beta_y
        left_eigenvectors[...,5,1+applicate] = Ns * Qff * beta_z
        left_eigenvectors[...,5,4] = fv.divide(Ns * alpha_s, rhos)
        left_eigenvectors[...,5,5+ordinate] = -fv.divide(Ns * Af * beta_y, rhos)
        left_eigenvectors[...,5,5+applicate] = -fv.divide(Ns * Af * beta_z, rhos)
        # Seventh row (Alfven+ wave)
        left_eigenvectors[...,6,1+ordinate] = beta_z/2
        left_eigenvectors[...,6,1+applicate] = -beta_y/2
        left_eigenvectors[...,6,5+ordinate] = -fv.divide(beta_z * S, 2 * np.sqrt(rhos))
        left_eigenvectors[...,6,5+applicate] = fv.divide(beta_y * S, 2 * np.sqrt(rhos))
        # Eighth row (Fast+ magnetoacoustic wave)
        left_eigenvectors[...,7,1+abscissa] = Nf * Cff
        left_eigenvectors[...,7,1+ordinate] = -Nf * Qss * beta_y
        left_eigenvectors[...,7,1+applicate] = -Nf * Qss * beta_z
        left_eigenvectors[...,7,4] = fv.divide(Nf * alpha_f, rhos)
        left_eigenvectors[...,7,5+ordinate] = fv.divide(Nf * As * beta_y, rhos)
        left_eigenvectors[...,7,5+applicate] = fv.divide(Nf * As * beta_z, rhos)

    # Generate the RIGHT eigenvectors
    if vectors.casefold().startswith(("b", "r")):
        # First column (Fast- magnetoacoustic wave)
        right_eigenvectors[...,0,0] = rhos * alpha_f
        right_eigenvectors[...,1+abscissa,0] = -Cff
        right_eigenvectors[...,1+ordinate,0] = Qss * beta_y
        right_eigenvectors[...,1+applicate,0] = Qss * beta_z
        right_eigenvectors[...,4,0] = rhos * alpha_f * cs**2
        right_eigenvectors[...,5+ordinate,0] = As * beta_y
        right_eigenvectors[...,5+applicate,0] = As * beta_z
        # Second column (Alfven- wave)
        right_eigenvectors[...,1+ordinate,1] = -beta_z
        right_eigenvectors[...,1+applicate,1] = beta_y
        right_eigenvectors[...,5+ordinate,1] = -beta_z * S * np.sqrt(rhos)
        right_eigenvectors[...,5+applicate,1] = beta_y * S * np.sqrt(rhos)
        # Third column (Slow- magnetoacoustic wave)
        right_eigenvectors[...,0,2] = rhos * alpha_s
        right_eigenvectors[...,1+abscissa,2] = -Css
        right_eigenvectors[...,1+ordinate,2] = -Qff * beta_y
        right_eigenvectors[...,1+applicate,2] = -Qff * beta_z
        right_eigenvectors[...,4,2] = rhos * alpha_s * cs**2
        right_eigenvectors[...,5+ordinate,2] = -Af * beta_y
        right_eigenvectors[...,5+applicate,2] = -Af * beta_z
        # Fourth column (Entropy/contact wave)
        right_eigenvectors[...,0,3] = 1
        # Fifth column (Divergence wave)
        right_eigenvectors[...,5+abscissa,4] = 1
        # Sixth column (Slow+ magnetoacoustic wave)
        right_eigenvectors[...,0,5] = rhos * alpha_s
        right_eigenvectors[...,1+abscissa,5] = Css
        right_eigenvectors[...,1+ordinate,5] = Qff * beta_y
        right_eigenvectors[...,1+applicate,5] = Qff * beta_z
        right_eigenvectors[...,4,5] = rhos * alpha_s * cs**2
        right_eigenvectors[...,ordinate+5,5] = -Af * beta_y
        right_eigenvectors[...,applicate+5,5] = -Af * beta_z
        # Seventh column (Alfven+ wave)
        right_eigenvectors[...,1+ordinate,6] = beta_z
        right_eigenvectors[...,1+applicate,6] = -beta_y
        right_eigenvectors[...,5+ordinate,6] = -beta_z * S * np.sqrt(rhos)
        right_eigenvectors[...,5+applicate,6] = beta_y * S * np.sqrt(rhos)
        # Eighth column (Fast+ magnetoacoustic wave)
        right_eigenvectors[...,0,7] = rhos * alpha_f
        right_eigenvectors[...,1+abscissa,7] = Cff
        right_eigenvectors[...,1+ordinate,7] = -Qss * beta_y
        right_eigenvectors[...,1+applicate,7] = -Qss * beta_z
        right_eigenvectors[...,4,7] = rhos * alpha_f * cs**2
        right_eigenvectors[...,5+ordinate,7] = As * beta_y
        right_eigenvectors[...,5+applicate,7] = As * beta_z

    if vectors.casefold().startswith("l"):
        return left_eigenvectors
    elif vectors.casefold().startswith("r"):
        return right_eigenvectors
    else:
        return left_eigenvectors, right_eigenvectors


# Make the right eigenvectors for adiabatic magnetohydrodynamics [Derigs et al., 2016]
def make_right_eigenvectors(grids, sim_variables, axis):
    abscissa, ordinate, applicate = (axis + np.array(range(3)))%3
    gamma = sim_variables.gamma

    rhos, vels, pressures, Bfields = grids[...,sim_variables.rho], grids[...,sim_variables.vels], grids[...,sim_variables.pressure], grids[...,sim_variables.Bfields]
    vx, vy, vz = vels[...,abscissa], vels[...,ordinate], vels[...,applicate]
    Bx, By, Bz = Bfields[...,abscissa], Bfields[...,ordinate], Bfields[...,applicate]

    # Define the right eigenvectors for each cell in each grid
    right_eigenvectors = np.repeat(np.zeros_like(grids)[...,None], grids.shape[-1], axis=-1)

    # Compute wavespeeds
    sound_speed, alfven_speed_x, fast_magnetosonic_wave, slow_magnetosonic_wave = make_wavespeeds(grids, sim_variables, axis)

    # Define frequently used components
    S = np.sign(Bx)
    #S[S == 0] = 1
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

    # Scale the right eigenvectors with a diagonal scaling matrix, so as to prevent degeneracies [Barth, 1999]
    # For adiabatic magnetohydrodynamics in entropy-stable flux (primitive variables)
    if sim_variables.solver.startswith('e'):
        diag_scaler = np.zeros_like(right_eigenvectors)
        diag_scaler[...,0,0] = 1/(2*gamma*rhos)
        diag_scaler[...,1+abscissa,1+abscissa] = fv.divide(pressures, 2*rhos**3)
        diag_scaler[...,1+ordinate,1+ordinate] = 1/(2*gamma*rhos)
        diag_scaler[...,1+applicate,1+applicate] = (rhos*(gamma-1))/gamma
        diag_scaler[...,4,4] = fv.divide(pressures, rhos)
        diag_scaler[...,5+abscissa,5+abscissa] = 1/(2*gamma*rhos)
        diag_scaler[...,5+ordinate,5+ordinate] = fv.divide(pressures, 2*rhos**3)
        diag_scaler[...,5+applicate,5+applicate] = 1/(2*gamma*rhos)
        right_eigenvectors = right_eigenvectors @ np.sqrt(diag_scaler)

    return right_eigenvectors