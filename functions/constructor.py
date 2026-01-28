import numpy as np

from functions import analytic, fv
from functions.generic import verbose_timer

##############################################################################
# Functions for constructing objects such as the grid, eigenvectors, Jacobian and flux terms
##############################################################################

# Create a physical grid for a single axis
def make_physical_grid(axis_coord, cells):
    start_pos, end_pos = axis_coord
    dh = np.abs(np.diff(axis_coord)[0])/cells
    half_cell = dh/2
    return np.linspace(start_pos-half_cell, end_pos+half_cell, cells+2)[1:-1]


# Initialise the discrete POINTWISE solution array with initial conditions and primitive variables w, and transform into discrete AVERAGES <w>
# For magnetohydrodynamics, this returns a staggered grid
@verbose_timer
def initialise(sim_variables):
    config, cells, gamma, dimensions, multidimensional, precision = sim_variables.config, sim_variables.cells, sim_variables.gamma, sim_variables.dimensions, sim_variables.multidimensional, sim_variables.precision
    rho, vx, vy, vz, pressure, Bx, By, Bz = sim_variables.rho, sim_variables.vx, sim_variables.vy, sim_variables.vz, sim_variables.pressure, sim_variables.Bx, sim_variables.By, sim_variables.Bz
    ds, axis_coord, shock_pos, params = sim_variables.ds, sim_variables.axis_coord, sim_variables.shock_pos, sim_variables.misc
    init_cond, ambient = sim_variables.init_cond, sim_variables.ambient
    axes = sim_variables.axes

    match = lambda match_type, substrings: match_type(substring in config for substring in substrings)


    computational_grid = np.zeros(list(cells)+[len(ambient),], dtype=precision, order='C')
    computational_grid[:] = ambient

    x_centre = np.average(axis_coord[0])
    physical_grid_x = make_physical_grid(axis_coord[0], cells[0])

    if multidimensional:
        y_centre = np.average(axis_coord[1])
        physical_grid_y = make_physical_grid(axis_coord[1], cells[1])

        if dimensions > 2:
            z_centre = np.average(axis_coord[2])
            physical_grid_z = make_physical_grid(axis_coord[2], cells[2])

            x, y, z = np.meshgrid(physical_grid_x, physical_grid_y, physical_grid_z, indexing='ij')
            r = np.sqrt((x-x_centre)**2 + (y-y_centre)**2 + (z-z_centre)**2)

            if match(any, ["sedov", "blast"]):
                mask = np.where((r <= (shock_pos-x_centre)) & (r <= (shock_pos-y_centre)) & (r <= (shock_pos-z_centre)))
                computational_grid[mask] = init_cond
                computational_grid = analytic.resample_blast(computational_grid, sim_variables)
                if config.startswith('m') or "mhd" in config:
                    computational_grid[...,5+axes] = params['ampl']

            elif config.startswith("sin"):
                computational_grid[...,rho] = analytic.sine_func(r, params)

            elif config.startswith("gauss"):
                computational_grid[...,rho] = analytic.gauss_func(r, params)

            elif match(any, ["orszag", "tang"]) or config == "ot":
                _x, _y, _z, ampl, eps = params['norm_factor']*x, params['norm_factor']*y, params['norm_factor']*z, params['ampl'], params['eps']

                computational_grid[...,vx] = -(1 + eps*np.sin(_z)) * np.sin(_y)
                computational_grid[...,vy] = (1 + eps*np.sin(_z)) * np.sin(_x)
                computational_grid[...,vz] = eps * np.sin(_z)
                computational_grid[...,Bx] = -ampl * np.sin(_y)
                computational_grid[...,By] = ampl * np.sin(2*_x)

            elif match(all, ["mhd", "vortex"]):
                factor = np.exp(params['q'] * (1 - r**2))
                computational_grid[...,vx] = 1 - (y-y_centre)*params['kappa']*factor
                computational_grid[...,vy] = 1 + (x-x_centre)*params['kappa']*factor
                computational_grid[...,pressure] = 1 + (1/(4*params['q'])) * ((1 - 2*params['q']*(r**2 - (z-z_centre)**2)) * params['mu']**2 - params['kappa']**2) * factor**2
                computational_grid[...,Bx] = -(y-y_centre)*params['mu']*factor
                computational_grid[...,By] = (x-x_centre)*params['mu']*factor

        else:
            x, y = np.meshgrid(physical_grid_x, physical_grid_y, indexing='ij')
            r = np.sqrt((x-x_centre)**2 + (y-y_centre)**2)

            if match(any, ["sedov", "blast"]):
                mask = np.where((r <= (shock_pos-x_centre)) & (r <= (shock_pos-y_centre)))
                computational_grid[mask] = init_cond
                computational_grid = analytic.resample_blast(computational_grid, sim_variables)
                if config.startswith('m') or "mhd" in config:
                    computational_grid[...,5+axes] = params['ampl']

            elif config.startswith("sin"):
                computational_grid[...,rho] = analytic.sine_func(r, params)

            elif config.startswith("gauss"):
                computational_grid[...,rho] = analytic.gauss_func(r, params)

            elif match(any, ["kelvin", "helmholtz", "khi"]):
                layer = np.where(np.abs(y) <= shock_pos)
                computational_grid[layer] = init_cond
                computational_grid[...,vy] = params['ampl'] * np.sin(params['freq']*np.pi*x/np.diff(axis_coord[0]))
                if params['perturb']:
                    perturbation = np.random.uniform(-params['perturb_ampl'], params['perturb_ampl'], size=(computational_grid[...,(vx,vy)][layer].shape))
                    computational_grid[...,(vx,vy)][layer] += perturbation
                if config.startswith('m') or "mhd" in config:
                    computational_grid[...,Bx] = params['Bx']

            elif match(any, ["ivc", "isentropic"]):
                b, freq = params['vortex_str'], params['freq']

                dv = lambda _array: (b*np.exp(.5*(1-r**2))*_array)/(np.sqrt(freq)*np.pi)
                computational_grid[...,vx] = 1 + dv(-(y-y_centre))
                computational_grid[...,vy] = 1 + dv(x-x_centre)

                db = lambda _array: (b*np.exp(.5*(1-r**2))*_array)/(freq*np.pi)
                computational_grid[...,Bx] = db(-(y-y_centre))
                computational_grid[...,By] = db(x-x_centre)

                dp = -(b**2 * (1+r**2) * np.exp(1-r**2))/(2 * (freq*np.pi)**2)
                computational_grid[...,pressure] += dp

            elif "gresho" in config:
                core, ring = np.where((0 <= r) & (r < .2)), np.where((.2 <= r) & (r < .4))
                rx, ry = -np.sin(np.arctan2(y-y_centre,x-x_centre)), np.cos(np.arctan2(y-y_centre,x-x_centre))
                p0 = init_cond[...,rho]/(gamma*params['mach']**2)

                computational_grid[...,pressure] = p0 - 2 + 4*np.log(2)

                v_phi = 5 * r
                computational_grid[...,vx][core] = (v_phi * rx)[core]
                computational_grid[...,vy][core] = (v_phi * ry)[core]
                computational_grid[...,pressure][core] = (p0 + (25/2)*r**2)[core]

                v_phi = 2 - 5*r
                computational_grid[...,vx][ring] = (v_phi * rx)[ring]
                computational_grid[...,vy][ring] = (v_phi * ry)[ring]
                computational_grid[...,pressure][ring] = (p0 + (25/2)*r**2 + 4*(1 - 5*r + np.log(5*r)))[ring]

            elif match(any, ["lax", "liu", "ll"]):
                computational_grid[np.where(x < shock_pos)] = init_cond
                computational_grid[np.where((x < shock_pos) & (y < shock_pos))] = params['bottom_left']
                computational_grid[np.where((x >= shock_pos) & (y < shock_pos))] = params['bottom_right']
                if config.startswith('m') or "mhd" in config:
                    computational_grid[...,Bx] = np.cos(y) * np.sin(x)
                    computational_grid[...,By] = -np.cos(x) * np.sin(y)

            elif match(any, ["orszag", "tang"]) or config == "ot":
                _x, _y, ampl = params['norm_factor']*x, params['norm_factor']*y, params['ampl']

                computational_grid[...,vx] = -np.sin(_y)
                computational_grid[...,vy] = np.sin(_x)
                computational_grid[...,Bx] = -ampl * np.sin(_y)
                computational_grid[...,By] = ampl * np.sin(2*_x)

            elif match(all, ["mhd", "vortex"]):
                computational_grid[...,vx] = 1 - (((y-y_centre)*params['kappa'])/(2*np.pi) * np.exp((1-r**2)/2))
                computational_grid[...,vy] = 1 + (((x-x_centre)*params['kappa'])/(2*np.pi) * np.exp((1-r**2)/2))
                computational_grid[...,pressure] = 1 + (((1-r**2)*params['kappa']**2 - params['mu']**2)/(8*np.pi**2) * np.exp(1-r**2))
                computational_grid[...,Bx] = (-(y-y_centre)*params['mu'])/(2*np.pi) * np.exp((1-r**2)/2)
                computational_grid[...,By] = ((x-x_centre)*params['mu'])/(2*np.pi) * np.exp((1-r**2)/2)

            elif "rotor" in config:
                ring_pos = shock_pos + params['ring_width']
                phi = (ring_pos - r)/(ring_pos - shock_pos)

                ring = np.where((r <= (ring_pos-x_centre)) & (r <= (ring_pos-y_centre)))
                computational_grid[...,rho][ring] = (1 + 9*phi)[ring]
                computational_grid[...,vx][ring] = ((-params['omega']*phi*(y-y_centre)*shock_pos)/r)[ring]
                computational_grid[...,vy][ring] = ((params['omega']*phi*(x-x_centre)*shock_pos)/r)[ring]

                core = np.where((r <= (shock_pos-x_centre)) & (r <= (shock_pos-y_centre)))
                computational_grid[core] = init_cond
                computational_grid[...,vx][core] = ((-params['omega']*(y-y_centre))/shock_pos)[core]
                computational_grid[...,vy][core] = ((params['omega']*(x-x_centre))/shock_pos)[core]

            elif match(any, ["current", "sheet"]):
                computational_grid[...,vx] = params['ampl'] * np.sin(2*np.pi*y)
                mask = np.where((-shock_pos < x) & (x < shock_pos))
                computational_grid[...,By][mask] = -computational_grid[...,By][mask]

            elif "noh" in config:
                mask = np.where(((x-axis_coord[0][0])**2 + (y-axis_coord[1][0])**2) > (shock_pos-axis_coord[0][0])**2)
                computational_grid[...,vx][mask] = -np.sin(x-shock_pos)[mask]
                computational_grid[...,vy][mask] = -np.cos(x-shock_pos)[mask]

            elif "cloud" in config:
                computational_grid[np.where(x < shock_pos)] = init_cond
                mask = np.where(((x-.8)**2 + (y-.5)**2) < .15**2)
                computational_grid[...,rho][mask] = params['cloud_mass']

            elif "jet" in config:
                nozzle = np.where((np.abs(x) < shock_pos) & (y <= (axis_coord[1][0] + ds[1])))
                sim_variables.mask = nozzle
                computational_grid[...,rho][nozzle] = gamma
                computational_grid[...,vy][nozzle] = params['velocity']
                computational_grid[...,By] *= np.sqrt(10)  # weak: 1, moderate:np.sqrt(10), strong:np.sqrt(1e2), extreme:np.sqrt(1e3)
                if params['perturb']:
                    perturbation = np.random.uniform(-10, 10, size=(computational_grid[...,(vx,vy)].shape))
                    computational_grid[...,(vx,vy)] += perturbation

            elif match(any, ["circular", "polarised", "alfven"]) or config == "cpaw":
                computational_grid[...,vx] = -params['A']/np.sqrt(2) * np.sin(2*np.pi*(x+y))
                computational_grid[...,vy] = params['A']/np.sqrt(2) * np.sin(2*np.pi*(x+y))
                computational_grid[...,vz] = params['A'] * np.cos(2*np.pi*(x+y))
                computational_grid[...,Bx] = params['ampl']/np.sqrt(2) + (params['A']/np.sqrt(2) * np.sin(2*np.pi*(x+y)))
                computational_grid[...,By] = params['ampl']/np.sqrt(2) - (params['A']/np.sqrt(2) * np.sin(2*np.pi*(x+y)))
                computational_grid[...,Bz] = -params['A'] * np.cos(2*np.pi*(x+y))

            else:
                computational_grid[np.where(x < shock_pos)] = init_cond

    else:
        x = physical_grid_x

        if match(any, ["sedov", "blast"]) or config.startswith('sq'):
            mask = np.where(np.abs(x) <= shock_pos)
            if match(any, ["sedov", "blast"]):
                computational_grid = analytic.resample_blast(computational_grid, sim_variables)
        else:
            mask = np.where(x <= shock_pos)
        computational_grid[mask] = init_cond

        if config.startswith('m') or "mhd" in config:
            computational_grid[...,5+axes] = params['ampl']

        if match(any, ["shu", "osher"]) or config == "so":
            computational_grid[np.where(x > shock_pos), rho] = analytic.sine_func(x[x > shock_pos], params)
        elif config.startswith("sin"):
            computational_grid[...,rho] = analytic.sine_func(x, params)
        elif config.startswith('gauss'):
            computational_grid[...,rho] = analytic.gauss_func(x-params['peak_pos'], params)

    sim_variables.magnetic = computational_grid[...,sim_variables.Bfields].any()

    return computational_grid


# Make flux as a function of cell-averaged (primitive) variables
def make_flux(grid, sim_variables, axis):
    abscissa, ordinate, applicate = (axis + np.array(range(3)))%3
    gamma, permeability = sim_variables.gamma, sim_variables.constants.mu_0

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
    gamma, permeability = sim_variables.gamma, sim_variables.constants.mu_0

    # In code units
    rhos, vels, pressures, Bfields = grid[...,sim_variables.rho], grid[...,sim_variables.vels], grid[...,sim_variables.pressure], grid[...,sim_variables.Bfields]

    # Create empty square arrays for each cell
    _arr = np.zeros_like(grid)
    arr = np.repeat(_arr[...,None], _arr.shape[-1], axis=-1)
    i, j = np.diag_indices(_arr.shape[-1])

    abscissa, ordinate, applicate = (axis + np.array(range(3)))%3
    Bx, By, Bz = Bfields[...,abscissa], Bfields[...,ordinate], Bfields[...,applicate]

    # Input matrix with values at position [row i, col j]. Positions refer to x-axis alignment; the coordinate rotation is already done by permuting the input variables
    # Hydrodynamic components
    arr[...,i,j] = vels[...,abscissa][...,None]  # diagonal elements
    arr[...,0,1] = rhos
    arr[...,1,4] = 1/rhos
    arr[...,4,1] = gamma * pressures

    # Magneto- components
    arr[...,2,6] = -fv.divide(Bx, permeability*rhos)
    arr[...,3,7] = -fv.divide(Bx, permeability*rhos)
    arr[...,1,6] = fv.divide(By, permeability*rhos)
    arr[...,1,7] = fv.divide(Bz, permeability*rhos)

    arr[...,6,2] = -Bx
    arr[...,7,3] = -Bx
    arr[...,6,1] = By
    arr[...,7,1] = Bz

    return arr


# Compute wavespeeds for a grid
def make_wavespeeds(grid, sim_variables, axis, waves='all'):
    gamma, permeability = sim_variables.gamma, sim_variables.constants.mu_0
    rho, pressure, Bfields = sim_variables.rho, sim_variables.pressure, sim_variables.Bfields

    waves = waves.lower()
    match = lambda substrings: any(wave in waves for wave in substrings)

    if match(['sound', 'fast', 'cff', 'slow', 'css', 'all']) or waves in ['cs', 'a']:
        sound_speed = np.sqrt(fv.divide(gamma * grid[...,pressure], grid[...,rho]))
        if 'sound' in waves or waves in ['cs', 'a']:
            return sound_speed
    if match(['alfven', 'ca', 'fast', 'cff', 'slow', 'css', 'all']):
        if match(['fast', 'slow', 'all']):
            alfven_speed_x = fv.divide(grid[...,5+axis], np.sqrt(grid[...,rho] * permeability))
            alfven_speed = fv.divide(fv.norm(grid[...,Bfields]), np.sqrt(grid[...,rho] * permeability))
        else:
            if waves.endswith(('x', 'y', 'z')):
                return fv.divide(grid[...,5+axis], np.sqrt(grid[...,rho] * permeability))
            else:
                return fv.divide(fv.norm(grid[...,Bfields]), np.sqrt(grid[...,rho] * permeability))
    if match(['fast', 'cff', 'all']):
        fast_magnetosonic_wave = np.sqrt(.5 * (sound_speed**2 + alfven_speed**2 + np.sqrt((sound_speed**2 + alfven_speed**2)**2 - (2 * sound_speed * alfven_speed_x)**2)))
        if waves != 'all':
            return fast_magnetosonic_wave
    if match(['slow', 'css', 'all']):
        slow_magnetosonic_wave = np.sqrt(.5 * (sound_speed**2 + alfven_speed**2 - np.sqrt((sound_speed**2 + alfven_speed**2)**2 - (2 * sound_speed * alfven_speed_x)**2)))
        if waves != 'all':
            return slow_magnetosonic_wave

    if waves == 'all':
        return sound_speed, alfven_speed, alfven_speed_x, fast_magnetosonic_wave, slow_magnetosonic_wave


# Characteristics (diagonalised eigenmatrix) [Stone et al., 2008]
def make_characteristics(grid, sim_variables, axis):
    uN = grid[...,1+axis]
    if sim_variables.magnetic:
        _, cA, _, cFF, cSS = make_wavespeeds(grid, sim_variables, axis=axis)
        characteristics = np.array([uN - cFF, uN - cA, uN - cSS, uN, uN + cSS, uN + cA, uN + cFF]).transpose(np.roll(np.arange(sim_variables.dimensions+1), -1))
    else:
        cs = make_wavespeeds(grid, sim_variables, axis=axis, waves='sound')
        characteristics = np.array([uN - cs, uN, uN, uN, uN + cs]).transpose(np.roll(np.arange(sim_variables.dimensions+1), -1))
    return characteristics


# Make the left & right eigenvectors for adiabatic magnetohydrodynamics [Roe & Balsara, 1996; Stone et al., 2008; Derigs et al., 2016]
# Here, Stone and Roe & Balsara only uses the 7-wave formulation due to constrained transport; the divergence wave is not needed
# Powell adds an 8th "divergence" wave to correct for the longitudinal magnetic field for divergence cleaning. Derigs modifies it further for entropy-stable formulation
def make_eigenvectors(grids, sim_variables, axis, vectors="both"):
    abscissa, ordinate, applicate = (axis + np.array(range(3)))%3

    # In code units
    rhos, vels, pressures, Bfields = grids[...,sim_variables.rho], grids[...,sim_variables.vels], grids[...,sim_variables.pressure], grids[...,sim_variables.Bfields]
    Bx, By, Bz = Bfields[...,abscissa], Bfields[...,ordinate], Bfields[...,applicate]
    vx, vy, vz = vels[...,abscissa], vels[...,ordinate], vels[...,applicate]

    if sim_variables.magnetic:
        # Compute wavespeeds
        cs, cA, cAx, cFF, cSS = make_wavespeeds(grids, sim_variables, axis=axis)

        # Define abbreviated terms
        alpha_f = np.sqrt(fv.divide(cs**2 - cSS**2, cFF**2 - cSS**2))
        alpha_s = np.sqrt(fv.divide(cFF**2 - cs**2, cFF**2 - cSS**2))

        # Magnetic field degenerate cases
        S = np.sign(Bx)
        transverse_field = np.sqrt(By**2 + Bz**2)
        non_degenerate_transverse_field = np.where(transverse_field != 0)
        beta_y = np.full_like(By, 1/np.sqrt(2))
        beta_z = np.full_like(Bz, 1/np.sqrt(2))
        beta_y[non_degenerate_transverse_field] = fv.divide(By, transverse_field)[non_degenerate_transverse_field]
        beta_z[non_degenerate_transverse_field] = fv.divide(Bz, transverse_field)[non_degenerate_transverse_field]

        # Handle degeneracy cases
        degenerate = np.where((cAx == cs) & (cA == cs))
        alpha_f[degenerate], alpha_s[degenerate] = 1, 0

        # Define frequently used terms
        Cff, Css = cFF * alpha_f, cSS * alpha_s
        Qff, Qss = Cff * S, Css * S
        Af, As = cs * alpha_f * np.sqrt(rhos), cs * alpha_s * np.sqrt(rhos)

        # Pop the longitudinal magnetic field
        _ = len(sim_variables.ambient) - 1

    else:
        # Compute sound speed
        cs = make_wavespeeds(grids, sim_variables, axis=axis, waves='sound')

        # Pop the magnetic field components
        _ = len(sim_variables.ambient) - 3


    # Compute characteristics and generate the RIGHT eigenvectors; the coordinate rotation is already done by permuting the input variables
    if vectors.casefold().startswith(("b", "r")):
        right_eigenvectors = np.zeros(sim_variables.cells + [_,_])

        if sim_variables.magnetic:
            ralphaf, ralphas = rhos * alpha_f, rhos * alpha_s
            r2alphaf, r2alphas = ralphaf * cs**2, ralphas * cs**2
            QssBy, QssBz = Qss * beta_y, Qss * beta_z
            QffBy, QffBz = Qff * beta_y, Qff * beta_z
            AsBy, AsBz = As * beta_y, As * beta_z
            AfBy, AfBz = Af * beta_y, Af * beta_z
            BySrho, BzSrho = beta_y * S * np.sqrt(rhos), beta_z * S * np.sqrt(rhos)

            # First column (Fast- magnetoacoustic wave)
            right_eigenvectors[...,0,0] = ralphaf
            right_eigenvectors[...,1,0] = -Cff
            right_eigenvectors[...,2,0] = QssBy
            right_eigenvectors[...,3,0] = QssBz
            right_eigenvectors[...,4,0] = r2alphaf
            right_eigenvectors[...,5,0] = AsBy
            right_eigenvectors[...,6,0] = AsBz
            # Second column (Alfven- wave)
            right_eigenvectors[...,2,1] = -beta_z
            right_eigenvectors[...,3,1] = beta_y
            right_eigenvectors[...,5,1] = -BzSrho
            right_eigenvectors[...,6,1] = BySrho
            # Third column (Slow- magnetoacoustic wave)
            right_eigenvectors[...,0,2] = ralphas
            right_eigenvectors[...,1,2] = -Css
            right_eigenvectors[...,2,2] = -QffBy
            right_eigenvectors[...,3,2] = -QffBz
            right_eigenvectors[...,4,2] = r2alphas
            right_eigenvectors[...,5,2] = -AfBy
            right_eigenvectors[...,6,2] = -AfBz
            # Fourth column (Entropy/contact wave)
            right_eigenvectors[...,0,3] = 1
            # Fifth column (Slow+ magnetoacoustic wave)
            right_eigenvectors[...,0,4] = ralphas
            right_eigenvectors[...,1,4] = Css
            right_eigenvectors[...,2,4] = QffBy
            right_eigenvectors[...,3,4] = QffBz
            right_eigenvectors[...,4,4] = r2alphas
            right_eigenvectors[...,5,4] = -AfBy
            right_eigenvectors[...,6,4] = -AfBz
            # Sixth column (Alfven+ wave)
            right_eigenvectors[...,2,5] = beta_z
            right_eigenvectors[...,3,5] = -beta_y
            right_eigenvectors[...,5,5] = -BzSrho
            right_eigenvectors[...,6,5] = BySrho
            # Seventh column (Fast+ magnetoacoustic wave)
            right_eigenvectors[...,0,6] = ralphaf
            right_eigenvectors[...,1,6] = Cff
            right_eigenvectors[...,2,6] = -QssBy
            right_eigenvectors[...,3,6] = -QssBz
            right_eigenvectors[...,4,6] = r2alphaf
            right_eigenvectors[...,5,6] = AsBy
            right_eigenvectors[...,6,6] = AsBz

        else:
            csrho = fv.divide(cs, rhos)
            cs2 = cs**2

            # First column
            right_eigenvectors[...,0,0] = 1
            right_eigenvectors[...,1,0] = -csrho
            right_eigenvectors[...,4,0] = cs2
            # Second column
            right_eigenvectors[...,0,1] = 1
            # Third column
            right_eigenvectors[...,2,2] = 1
            # Fourth column
            right_eigenvectors[...,3,3] = 1
            # Fifth column
            right_eigenvectors[...,0,4] = 1
            right_eigenvectors[...,1,4] = csrho
            right_eigenvectors[...,4,4] = cs2

    # Compute characteristics and generate the LEFT eigenvectors; the coordinate rotation is already done by permuting the input variables
    if vectors.casefold().startswith(("b", "l")):
        left_eigenvectors = np.zeros(sim_variables.cells + [_,_])

        if sim_variables.magnetic:
            Nf = Ns = 1/(2*cs**2)
            NfCff, NsCss = Nf * Cff, Ns * Css
            Nfalphaf, Nsalphas = fv.divide(Nf * alpha_f, rhos), fv.divide(Ns * alpha_s, rhos)
            NfQssBy, NfQssBz = Nf * Qss * beta_y, Nf * Qss * beta_z
            NsQffBy, NsQffBz = Ns * Qff * beta_y, Ns * Qff * beta_z
            NfAsBy, NfAsBz = fv.divide(Nf * As * beta_y, rhos), fv.divide(Nf * As * beta_z, rhos)
            NsAfBy, NsAfBz = fv.divide(Ns * Af * beta_y, rhos), fv.divide(Ns * Af * beta_z, rhos)
            ByS2rho, BzS2rho = fv.divide(beta_y * S, 2 * np.sqrt(rhos)), fv.divide(beta_z * S, 2 * np.sqrt(rhos))

            # First row (Fast- magnetoacoustic wave)
            left_eigenvectors[...,0,1] = -NfCff
            left_eigenvectors[...,0,2] = NfQssBy
            left_eigenvectors[...,0,3] = NfQssBz
            left_eigenvectors[...,0,4] = Nfalphaf
            left_eigenvectors[...,0,5] = NfAsBy
            left_eigenvectors[...,0,6] = NfAsBz
            # Second row (Alfven- wave)
            left_eigenvectors[...,1,2] = -beta_z/2
            left_eigenvectors[...,1,3] = beta_y/2
            left_eigenvectors[...,1,5] = -BzS2rho
            left_eigenvectors[...,1,6] = ByS2rho
            # Third row (Slow- magnetoacoustic wave)
            left_eigenvectors[...,2,1] = -NsCss
            left_eigenvectors[...,2,2] = -NsQffBy
            left_eigenvectors[...,2,3] = -NsQffBz
            left_eigenvectors[...,2,4] = Nsalphas
            left_eigenvectors[...,2,5] = -NsAfBy
            left_eigenvectors[...,2,6] = -NsAfBz
            # Fourth row (Entropy/contact wave)
            left_eigenvectors[...,3,0] = 1
            left_eigenvectors[...,3,4] = -2 * Nf
            # Fifth row (Slow+ magnetoacoustic wave)
            left_eigenvectors[...,4,1] = NsCss
            left_eigenvectors[...,4,2] = NsQffBy
            left_eigenvectors[...,4,3] = NsQffBz
            left_eigenvectors[...,4,4] = Nsalphas
            left_eigenvectors[...,4,5] = -NsAfBy
            left_eigenvectors[...,4,6] = -NsAfBz
            # Sixth row (Alfven+ wave)
            left_eigenvectors[...,5,2] = beta_z/2
            left_eigenvectors[...,5,3] = -beta_y/2
            left_eigenvectors[...,5,5] = -BzS2rho
            left_eigenvectors[...,5,6] = ByS2rho
            # Seventh row (Fast+ magnetoacoustic wave)
            left_eigenvectors[...,6,1] = NfCff
            left_eigenvectors[...,6,2] = -NfQssBy
            left_eigenvectors[...,6,3] = -NfQssBz
            left_eigenvectors[...,6,4] = Nfalphaf
            left_eigenvectors[...,6,5] = NfAsBy
            left_eigenvectors[...,6,6] = NfAsBz

        else:
            rho2cs = fv.divide(rhos, 2*cs)
            a2 = 1/(2 * cs**2)

            # First row
            left_eigenvectors[...,0,1] = -rho2cs
            left_eigenvectors[...,0,4] = a2
            # Second row
            left_eigenvectors[...,1,0] = 1
            left_eigenvectors[...,1,4] = -2 * a2
            # Third row
            left_eigenvectors[...,2,2] = 1
            # Fourth row
            left_eigenvectors[...,3,3] = 1
            # Fifth row
            left_eigenvectors[...,4,1] = rho2cs
            left_eigenvectors[...,4,4] = a2

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
    cs, _, _, cFF, cSS = make_wavespeeds(grids, sim_variables, axis)

    # Define frequently used components
    S = np.sign(Bx)
    alpha_f = np.sqrt(fv.divide(cs**2 - cSS**2, cFF**2 - cSS**2))
    alpha_s = np.sqrt(fv.divide(cFF**2 - cs**2, cFF**2 - cSS**2))
    b_perpend = np.sqrt(fv.divide(By**2 + Bz**2, rhos))
    beta2 = fv.divide(By, np.sqrt(By**2 + Bz**2))
    beta3 = fv.divide(Bz, np.sqrt(By**2 + Bz**2))

    psi_plus_slow = (
        .5 * alpha_s * rhos * fv.norm(vels)**2
        - cs * alpha_f * rhos * b_perpend
        + (alpha_s * rhos * cs**2)/(gamma - 1)
        + alpha_s * cSS * rhos * vx
        + alpha_f * cFF * rhos * S * (vy*beta2 + vz*beta3)
        )
    psi_minus_slow = (
        .5 * alpha_s * rhos * fv.norm(vels)**2
        - cs * alpha_f * rhos * b_perpend
        + (alpha_s * rhos * cs**2)/(gamma - 1)
        - alpha_s * cSS * rhos * vx
        - alpha_f * cFF * rhos * S * (vy*beta2 + vz*beta3)
        )
    psi_plus_fast = (
        .5 * alpha_f * rhos * fv.norm(vels)**2
        + cs * alpha_s * rhos * b_perpend
        + (alpha_f * rhos * cs**2)/(gamma - 1)
        + alpha_f * cFF * rhos * vx
        - alpha_s * cSS * rhos * S * (vy*beta2 + vz*beta3)
        )
    psi_minus_fast = (
        .5 * alpha_f * rhos * fv.norm(vels)**2
        + cs * alpha_s * rhos * b_perpend
        + (alpha_f * rhos * cs**2)/(gamma - 1)
        - alpha_f * cFF * rhos * vx
        + alpha_s * cSS * rhos * S * (vy*beta2 + vz*beta3)
        )

    # Generate the right eigenvectors
    # First column (Fast- magnetoacoustic wave)
    right_eigenvectors[...,0,0] = rhos * alpha_f
    right_eigenvectors[...,1,0] = rhos * alpha_f * (vx - cFF)
    right_eigenvectors[...,2,0] = rhos * (alpha_f*vy + alpha_s*cSS*beta2*S)
    right_eigenvectors[...,3,0] = rhos * (alpha_f*vz + alpha_s*cSS*beta3*S)
    right_eigenvectors[...,4,0] = psi_minus_fast
    right_eigenvectors[...,6,0] = alpha_s * cs * beta2 * np.sqrt(rhos)
    right_eigenvectors[...,7,0] = alpha_s * cs * beta3 * np.sqrt(rhos)
    # Second column (Alfven- wave)
    right_eigenvectors[...,2,1] = -beta3 * rhos**1.5
    right_eigenvectors[...,3,1] = beta2 * rhos**1.5
    right_eigenvectors[...,4,1] = (beta2*vz - beta3*vy) * rhos**1.5
    right_eigenvectors[...,6,1] = -rhos * beta3
    right_eigenvectors[...,7,1] = rhos * beta2
    # Third column (Slow- magnetoacoustic wave)
    right_eigenvectors[...,0,2] = rhos * alpha_s
    right_eigenvectors[...,1,2] = rhos * alpha_s * (vx - cSS)
    right_eigenvectors[...,2,2] = rhos * (alpha_s*vy - alpha_f*cFF*beta2*S)
    right_eigenvectors[...,3,2] = rhos * (alpha_s*vz - alpha_f*cFF*beta3*S)
    right_eigenvectors[...,4,2] = psi_minus_slow
    right_eigenvectors[...,6,2] = -alpha_f * cs * beta2 * np.sqrt(rhos)
    right_eigenvectors[...,7,2] = -alpha_f * cs * beta3 * np.sqrt(rhos)
    # Fourth column (Entropy wave)
    right_eigenvectors[...,0,3] = 1
    right_eigenvectors[...,1,3] = vx
    right_eigenvectors[...,2,3] = vy
    right_eigenvectors[...,3,3] = vz
    right_eigenvectors[...,4,3] = .5 * fv.norm(vels)**2
    # Fifth column (Divergence wave)
    right_eigenvectors[...,4,4] = Bx
    right_eigenvectors[...,6,4] = 1
    # Sixth column (Slow+ magnetoacoustic wave)
    right_eigenvectors[...,0,5] = rhos * alpha_s
    right_eigenvectors[...,1,5] = rhos * alpha_s * (vx + cSS)
    right_eigenvectors[...,2,5] = rhos * (alpha_s*vy + alpha_f*cFF*beta2*S)
    right_eigenvectors[...,3,5] = rhos * (alpha_s*vz + alpha_f*cFF*beta3*S)
    right_eigenvectors[...,4,5] = psi_plus_slow
    right_eigenvectors[...,6,5] = -alpha_f * cs * beta2 * np.sqrt(rhos)
    right_eigenvectors[...,7,5] = -alpha_f * cs * beta3 * np.sqrt(rhos)
    # Seventh column (Alfven+ wave)
    right_eigenvectors[...,2,6] = beta3 * rhos**1.5
    right_eigenvectors[...,3,6] = -beta2 * rhos**1.5
    right_eigenvectors[...,4,6] = (beta3*vy - beta2*vz) * rhos**1.5
    right_eigenvectors[...,6,6] = -rhos * beta3
    right_eigenvectors[...,7,6] = rhos * beta2
    # Eighth column (Fast+ magnetoacoustic wave)
    right_eigenvectors[...,0,7] = rhos * alpha_f
    right_eigenvectors[...,1,7] = rhos * alpha_f * (vx + cFF)
    right_eigenvectors[...,2,7] = rhos * (alpha_f*vy - alpha_s*cSS*beta2*S)
    right_eigenvectors[...,3,7] = rhos * (alpha_f*vz - alpha_s*cSS*beta3*S)
    right_eigenvectors[...,4,7] = psi_plus_fast
    right_eigenvectors[...,6,7] = alpha_s * cs * beta2 * np.sqrt(rhos)
    right_eigenvectors[...,7,7] = alpha_s * cs * beta3 * np.sqrt(rhos)

    # Scale the right eigenvectors with a diagonal scaling matrix, so as to prevent degeneracies [Barth, 1999]
    # For adiabatic magnetohydrodynamics in entropy-stable flux (primitive variables)
    if sim_variables.solver.startswith('e'):
        diag_scaler = np.zeros_like(right_eigenvectors)
        diag_scaler[...,0,0] = 1/(2*gamma*rhos)
        diag_scaler[...,1,1] = fv.divide(pressures, 2*rhos**3)
        diag_scaler[...,2,2] = 1/(2*gamma*rhos)
        diag_scaler[...,3,3] = (rhos*(gamma-1))/gamma
        diag_scaler[...,4,4] = fv.divide(pressures, rhos)
        diag_scaler[...,5,5] = 1/(2*gamma*rhos)
        diag_scaler[...,6,6] = fv.divide(pressures, 2*rhos**3)
        diag_scaler[...,7,7] = 1/(2*gamma*rhos)
        right_eigenvectors = right_eigenvectors @ np.sqrt(diag_scaler)

    return right_eigenvectors