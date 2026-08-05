import numpy as np

from functions import math as mfuncs
from functions import grid as gutils
from functions.generic import verbose_timer
from physics import turbulence

##############################################################################
# Grid initialisation function
##############################################################################

# Initialise the discrete POINTWISE solution array with initial conditions and primitive variables w, and transform into discrete AVERAGES <w>
# For magnetohydrodynamics, this returns a staggered grid
@verbose_timer
def initialise(sim_variables):
    config, cells, gamma, dimensions, multidimensional = sim_variables.config, sim_variables.cells, sim_variables.gamma, sim_variables.dimensions, sim_variables.multidimensional
    rho, vx, vy, vz, pressure, Bx, By, Bz = sim_variables.rho, sim_variables.vx, sim_variables.vy, sim_variables.vz, sim_variables.pressure, sim_variables.Bx, sim_variables.By, sim_variables.Bz
    ds, coordinates, shock_pos, test_specifics = sim_variables.ds, sim_variables.coordinates, sim_variables.shock_pos, sim_variables.test_specifics
    init_cond, ambient = sim_variables.init_cond, sim_variables.ambient
    axes = sim_variables.axes

    match = lambda match_type, substrings: match_type(substring in config for substring in substrings)


    computational_grid = np.zeros(list(cells)+[len(ambient),], dtype=float, order='C')
    computational_grid[:] = ambient

    x_centre, physical_grid_x = gutils.make_physical_grid(coordinates, cells, 0)

    if multidimensional:
        y_centre, physical_grid_y = gutils.make_physical_grid(coordinates, cells, 1)

        if dimensions > 2:
            ##############################
            #  3-dimensional cases
            ##############################
            z_centre, physical_grid_z = gutils.make_physical_grid(coordinates, cells, 2)

            x, y, z = np.meshgrid(physical_grid_x, physical_grid_y, physical_grid_z, indexing='ij')
            x0, y0, z0 = x - x_centre, y - y_centre, z - z_centre
            r = np.sqrt(x0**2 + y0**2 + z0**2)
            r0 = np.sqrt((shock_pos-x_centre)**2 + (shock_pos-y_centre)**2 + (shock_pos-z_centre)**2)

            if ("sedov" in config) or match(all, ["mhd", "blast"]) or (match(any, ["supernova", "tycho"]) or config == "sn"):
                if match(any, ["supernova", "tycho"]) or config == "sn":
                    if test_specifics['mode'].lower().startswith(('o','q')):
                        x_centre, y_centre, z_centre = (axis_coord[0] for axis_coord in coordinates.values())
                        r = np.sqrt((x-x_centre)**2 + (y-y_centre)**2 + (z-z_centre)**2)

                    E, M, t0 = test_specifics['E'], test_specifics['M'], test_specifics['t0']
                    shock_pos = 2 * t0 * np.sqrt(gamma * E/M)
                    sim_variables.shock_pos = shock_pos

                    r0 = np.sqrt((shock_pos-x_centre)**2 + (shock_pos-y_centre)**2 + (shock_pos-z_centre)**2)
                    rho0 = 25/(21*np.pi) * (E**2)/M * t0**4 * r0**-7

                    mask = np.where(r > r0)
                    computational_grid[...,rho][mask] += rho0 * (r[mask]/r0)**-7

                    core = np.where(r <= r0)
                    computational_grid[core] = init_cond
                    computational_grid[...,rho][core] += rho0

                    sigma, V = .75 * r0, 4/3 * np.pi * r0**3
                    e_tot = ambient[pressure]/(gamma-1) + mfuncs.smoothing_kernel(E/V, r, d=dimensions, sigma=sigma)
                    computational_grid[...,pressure] = (gamma - 1) * e_tot

                    if test_specifics['rotation']:
                        tau0, age = test_specifics['tau0'], test_specifics['age']
                        computational_grid[...,vx][core] = -tau0 * age**-.51 * y[core]
                        computational_grid[...,vy][core] = tau0 * age**-.51 * x[core]

                else:
                    mask = np.where(r <= r0)
                    computational_grid[mask] = init_cond

                    if "sedov" in config:
                        mu = np.sqrt(x_centre**2 + y_centre**2 + z_centre**2)
                        sigma = np.abs(r0 - mu)
                        computational_grid[...,pressure] = ambient[pressure] + mfuncs.smoothing_kernel(init_cond[pressure], r, d=dimensions, mu=mu, sigma=sigma)

                    if match(all, ["mhd", "blast"]):
                        computational_grid[...,5+axes] = test_specifics['ampl']

            elif config.startswith("sin"):
                computational_grid[...,rho] = mfuncs.sine_func(r, test_specifics)

            elif config.startswith("gauss"):
                computational_grid[...,rho] = mfuncs.gauss_func(r, test_specifics)

            elif match(any, ["manufacture", "euler"]):
                Lx, Ly, Lz = np.diff(coordinates[0]), np.diff(coordinates[1]), np.diff(coordinates[2])
                freq = test_specifics['freq']

                computational_grid[...,rho] = 1 + .35*np.sin(freq*x/Lx) + .24*np.cos(freq*y/Ly) + .1*np.sin(freq*z/Lz)
                computational_grid[...,pressure] = 1 + .23*np.sin(freq*x/Lx) + .19*np.cos(freq*y/Ly) + .2*np.cos(freq*z/Lz)

            elif match(any, ["turb", "blank"]):
                if "turb" in config:
                    if test_specifics['magnetic']:
                        computational_grid[...,Bx] = -test_specifics['mag_ampl'] * np.sin(2*np.pi*y)
                        computational_grid[...,By] = test_specifics['mag_ampl'] * np.sin(4*np.pi*x)
                else:
                    computational_grid[...,rho] += np.random.uniform(-test_specifics['perturb_ampl'], test_specifics['perturb_ampl'], size=(computational_grid.shape))[...,rho]

            elif match(any, ["orszag", "tang"]) or config == "ot":
                _x, _y, _z, ampl, eps = test_specifics['norm_factor']*x, test_specifics['norm_factor']*y, test_specifics['norm_factor']*z, test_specifics['ampl'], test_specifics['eps']

                computational_grid[...,vx] = -(1 + eps*np.sin(_z)) * np.sin(_y)
                computational_grid[...,vy] = (1 + eps*np.sin(_z)) * np.sin(_x)
                computational_grid[...,vz] = eps * np.sin(_z)
                computational_grid[...,Bx] = -ampl * np.sin(_y)
                computational_grid[...,By] = ampl * np.sin(2*_x)

            elif match(all, ["mhd", "vortex"]):
                factor = np.exp(test_specifics['q'] * (1 - r**2))
                computational_grid[...,vx] = 1 - y0*test_specifics['kappa']*factor
                computational_grid[...,vy] = 1 + x0*test_specifics['kappa']*factor
                computational_grid[...,pressure] = 1 + (1/(4*test_specifics['q'])) * ((1 - 2*test_specifics['q']*(r**2 - z0**2)) * test_specifics['mu']**2 - test_specifics['kappa']**2) * factor**2
                computational_grid[...,Bx] = -y0*test_specifics['mu']*factor
                computational_grid[...,By] = x0*test_specifics['mu']*factor

            elif "torus" in config:
                r = np.sqrt(x0**2 + y0**2)
                cA2 = lambda _r: 2 * (test_specifics['K']/test_specifics['beta0']) * (init_cond[rho] * _r**2)**(gamma-1)
                cs2 = np.sqrt(gamma * init_cond[pressure]/init_cond[rho])
                torus_phi = -test_specifics['GM']/test_specifics['r0'] + test_specifics['L']**2/(2*test_specifics['r0']**2) + (2*cs2 + gamma*cA2(test_specifics['r0']))/(2*(gamma-1))

                computational_grid[...,rho] = (
                    mfuncs.divide(
                        np.maximum(0, torus_phi + test_specifics['GM']/test_specifics['r0'] - test_specifics['L']**2/(2*r**2)),
                        test_specifics['K'] * (gamma/(gamma-1)) * (1 + (r**(2*(gamma-1)))/test_specifics['beta0'])
                    )
                )**(1/(gamma-1))
                computational_grid[...,vx] = -np.sqrt(test_specifics['GM']) * (y/r**1.5)
                computational_grid[...,vy] = np.sqrt(test_specifics['GM']) * (x/r**1.5)

                _r = np.sqrt((x-x_centre)**2 + (y-y_centre)**2)
                _P = test_specifics['K'] * init_cond[rho]**gamma
                cA2 = 2 * (test_specifics['K']/test_specifics['beta0']) * (init_cond[rho] * test_specifics['r0']**2)**(gamma-1)
                cs2 = np.sqrt(gamma * _P/init_cond[rho])
                torus_phi = -test_specifics['GM']/test_specifics['r0'] + test_specifics['L']**2/(2*test_specifics['r0']**2) + (2*cs2 + gamma*cA2)/(2*(gamma-1))

                computational_grid[...,rho] = (
                    mfuncs.divide(
                        np.maximum(0, torus_phi + test_specifics['GM']/r - test_specifics['L']**2/(2*_r**2)),
                        test_specifics['K'] * (gamma/(gamma-1)) * (1 + (_r**(2*(gamma-1)))/test_specifics['beta0'])
                    )
                )**(1/(gamma-1))
                computational_grid[...,pressure] = _P
                computational_grid[...,vx] = -np.sqrt(test_specifics['GM'] * test_specifics['L']**2) * (y/_r**2)
                computational_grid[...,vy] = np.sqrt(test_specifics['GM'] * test_specifics['L']**2) * (x/_r**2)
                computational_grid[...,Bx] = -test_specifics['B_phi'] * y/_r
                computational_grid[...,By] = test_specifics['B_phi'] * x/_r

            elif match(any, ["blob"]):
                omega, B_ampl, [theta, phi] = test_specifics['omega'], test_specifics['B_ampl'], test_specifics['rotation_axis']
                omega_hat = np.array([np.sin(theta*np.pi/180)*np.cos(phi*np.pi/180), np.sin(theta*np.pi/180)*np.sin(phi*np.pi/180), np.cos(theta*np.pi/180)])

                ndotr = np.dot(np.stack([x,y,z], axis=-1), omega_hat)
                R = np.sqrt(r**2 - ndotr**2)
                smoothing = lambda q: q * np.exp(-.5 * (R/r0)**2)

                computational_grid[...,rho] += smoothing(init_cond[rho])
                computational_grid[...,pressure] += smoothing(.25 * ambient[rho] * (omega*r0)**2)

                computational_grid[...,vx] = smoothing(omega) * (omega_hat[1]*z - omega_hat[2]*y)
                computational_grid[...,vy] = smoothing(omega) * -(omega_hat[0]*z - omega_hat[2]*x)
                computational_grid[...,vz] = smoothing(omega) * (omega_hat[0]*y - omega_hat[1]*x)

                if config.startswith('m'):
                    computational_grid[...,Bx] = -B_ampl * np.sin(y)
                    computational_grid[...,By] = B_ampl * np.sin(2*x)

            elif "dwarf" in config:
                a, b, Mgas = test_specifics['a'], test_specifics['b'], test_specifics['Mgas']

                rho0 = Mgas / (8 * np.pi * a**2 * b * mfuncs.catalan())
                rho_profile = rho0 * np.arccos(r/a) * np.arccos(z/b)**2
                computational_grid[...,rho] = np.maximum(rho_profile, 1)

        else:
            ##############################
            #  2-dimensional cases
            ##############################
            x, y = np.meshgrid(physical_grid_x, physical_grid_y, indexing='ij')
            x0, y0 = x - x_centre, y - y_centre
            r = np.sqrt(x0**2 + y0**2)
            r0 = np.sqrt((shock_pos-x_centre)**2 + (shock_pos-y_centre)**2)

            if ("sedov" in config) or match(all, ["mhd", "blast"]) or (match(any, ["supernova", "tycho"]) or config == "sn"):
                if match(any, ["supernova", "tycho"]) or config == "sn":
                    if test_specifics['mode'].lower().startswith(('o','q')):
                        x_centre, y_centre = (axis_coord[0] for axis_coord in coordinates.values())
                        r = np.sqrt((x-x_centre)**2 + (y-y_centre)**2)

                    E, M, t0 = test_specifics['E'], test_specifics['M'], test_specifics['t0']
                    shock_pos = 2 * t0 * np.sqrt(gamma * E/M)
                    sim_variables.shock_pos = shock_pos

                    r0 = np.sqrt((shock_pos-x_centre)**2 + (shock_pos-y_centre)**2)
                    rho0 = 25/(21*np.pi) * (E**2)/M * t0**4 * r0**-7

                    mask = np.where(r > r0)
                    computational_grid[...,rho][mask] += rho0 * (r[mask]/r0)**-7

                    core = np.where(r <= r0)
                    computational_grid[core] = init_cond
                    computational_grid[...,rho][core] += rho0

                    sigma, A = .75 * r0, np.pi * r0**2
                    e_tot = ambient[pressure]/(gamma-1) + mfuncs.smoothing_kernel(E/A, r, d=dimensions, sigma=sigma)
                    computational_grid[...,pressure] = (gamma - 1) * e_tot

                    if test_specifics['rotation']:
                        tau0, age = test_specifics['tau0'], test_specifics['age']
                        computational_grid[...,vx][core] = -tau0 * age**-.51 * y[core]
                        computational_grid[...,vy][core] = tau0 * age**-.51 * x[core]

                else:
                    mask = np.where(r <= r0)
                    computational_grid[mask] = init_cond

                    if "sedov" in config:
                        mu = np.sqrt(x_centre**2 + y_centre**2)
                        sigma = np.abs(r0 - mu)
                        computational_grid[...,pressure] = ambient[pressure] + mfuncs.smoothing_kernel(init_cond[pressure], r, d=dimensions, mu=mu, sigma=sigma)

                    if match(all, ["mhd", "blast"]):
                        computational_grid[...,5+axes] = test_specifics['ampl']

            elif config.startswith("sin"):
                computational_grid[...,rho] = mfuncs.sine_func(r, test_specifics)

            elif config.startswith("gauss"):
                computational_grid[...,rho] = mfuncs.gauss_func(r, test_specifics)

            elif match(any, ["manufacture", "euler"]):
                Lx, Ly = np.diff(coordinates[0]), np.diff(coordinates[1])
                freq = test_specifics['freq']

                computational_grid[...,vz] = 0
                computational_grid[...,rho] = 1 + .35*np.sin(freq*x/Lx) + .24*np.cos(freq*y/Ly)
                computational_grid[...,pressure] = 1 + .23*np.sin(freq*x/Lx) + .19*np.cos(freq*y/Ly)

            elif match(any, ["kelvin", "helmholtz", "khi"]):
                layer = np.where(np.abs(y) <= shock_pos)
                computational_grid[layer] = init_cond
                computational_grid[...,vy] = test_specifics['ampl'] * np.sin(test_specifics['freq']*np.pi*x/np.diff(coordinates[0]))
                if test_specifics['perturb']:
                    perturbations = turbulence.pertubations(computational_grid, test_specifics['ampl'])
                    computational_grid[...,(vx,vy)] += perturbations[...,(vx,vy)]
                if config.startswith('m') or "mhd" in config:
                    computational_grid[...,Bx] = test_specifics['Bx']

            elif match(any, ["rayleigh", "taylor", "rti"]):
                layer = np.where(y > shock_pos)
                computational_grid[layer] = init_cond
                computational_grid[...,pressure] = init_cond[pressure] - .1*computational_grid[...,rho]*y
                if test_specifics['perturb']:
                    perturbations = turbulence.pertubations(computational_grid, 2*test_specifics['ampl'])
                    computational_grid[...,vy] += perturbations[...,vy] * (1 + np.cos(8*np.pi*y/3))
                else:
                    computational_grid[...,vy] = test_specifics['ampl'] * (1 + np.cos(4*np.pi*x)) * (1 + np.cos(3*np.pi*y))
                if config.startswith('m') or "mhd" in config:
                    computational_grid[...,Bx] = test_specifics['Bx']

            elif match(any, ["ivc", "isentropic"]):
                b, freq = test_specifics['vortex_str'], test_specifics['freq']

                dv = lambda _array: (b*np.exp(.5*(1-r**2))*_array)/(np.sqrt(freq)*np.pi)
                computational_grid[...,vx] = 1 + dv(-y0)
                computational_grid[...,vy] = 1 + dv(x0)

                db = lambda _array: (b*np.exp(.5*(1-r**2))*_array)/(freq*np.pi)
                computational_grid[...,Bx] = db(-y0)
                computational_grid[...,By] = db(x0)

                dp = -(b**2 * (1+r**2) * np.exp(1-r**2))/(2 * (freq*np.pi)**2)
                computational_grid[...,pressure] += dp

            elif "gresho" in config:
                core, ring = np.where((0 <= r) & (r < .2)), np.where((.2 <= r) & (r < .4))
                p0 = init_cond[...,rho]/(gamma*test_specifics['mach']**2)

                computational_grid[...,pressure] = p0 - 2 + 4*np.log(2)

                computational_grid[...,vx][ring] = (5 - 2*r)[ring]
                computational_grid[...,vy][ring] = (2*r - 5)[ring]
                computational_grid[...,pressure][ring] = (p0 + (25/2)*r**2 + 4*(1 - 5*r + np.log(5*r)))[ring]

                computational_grid[...,vx][core] = -5
                computational_grid[...,vy][core] = 5
                computational_grid[...,pressure][core] = (p0 + (25/2)*r**2)[core]

            elif match(any, ["lax", "liu", "ll"]):
                computational_grid[np.where(x < shock_pos)] = init_cond
                computational_grid[np.where((x < shock_pos) & (y < shock_pos))] = test_specifics['bottom_left']
                computational_grid[np.where((x >= shock_pos) & (y < shock_pos))] = test_specifics['bottom_right']
                if config.startswith('m') or "mhd" in config:
                    computational_grid[...,Bx] = np.cos(y) * np.sin(x)
                    computational_grid[...,By] = -np.cos(x) * np.sin(y)

            elif match(any, ["yee", "sjögreen", "sjoegreen"]) or config == "ys":
                computational_grid[np.where(x < shock_pos)] = init_cond
                computational_grid[np.where((x < shock_pos) & (y < shock_pos))] = test_specifics['bottom_left']
                computational_grid[np.where((x >= shock_pos) & (y < shock_pos))] = test_specifics['bottom_right']

            elif match(any, ["liska", "wendroff", "implosion"]):
                xr = np.cos(test_specifics['angle'] * np.pi/180)*x + np.sin(test_specifics['angle'] * np.pi/180)*y
                yr = -np.sin(test_specifics['angle'] * np.pi/180)*x + np.cos(test_specifics['angle'] * np.pi/180)*y

                mask = (np.abs(xr) <= shock_pos) & (np.abs(yr) <= shock_pos)
                computational_grid[mask] = init_cond

            elif match(any, ["orszag", "tang"]) or config == "ot":
                _x, _y, ampl = test_specifics['norm_factor']*x, test_specifics['norm_factor']*y, test_specifics['ampl']

                computational_grid[...,vx] = -np.sin(_y)
                computational_grid[...,vy] = np.sin(_x)
                computational_grid[...,Bx] = -ampl * np.sin(_y)
                computational_grid[...,By] = ampl * np.sin(2*_x)

            elif match(all, ["mhd", "vortex"]):
                computational_grid[...,vx] = 1 - ((y0*test_specifics['kappa'])/(2*np.pi) * np.exp((1-r**2)/2))
                computational_grid[...,vy] = 1 + ((x0*test_specifics['kappa'])/(2*np.pi) * np.exp((1-r**2)/2))
                computational_grid[...,pressure] = 1 + (((1-r**2)*test_specifics['kappa']**2 - test_specifics['mu']**2)/(8*np.pi**2) * np.exp(1-r**2))
                computational_grid[...,Bx] = (-y0*test_specifics['mu'])/(2*np.pi) * np.exp((1-r**2)/2)
                computational_grid[...,By] = (x0*test_specifics['mu'])/(2*np.pi) * np.exp((1-r**2)/2)

            elif "torus" in config:
                _P = test_specifics['K'] * init_cond[rho]**gamma
                cA2 = 2 * (test_specifics['K']/test_specifics['beta0']) * (init_cond[rho] * test_specifics['r0']**2)**(gamma-1)
                cs2 = np.sqrt(gamma * _P/init_cond[rho])
                torus_phi = -test_specifics['GM']/test_specifics['r0'] + test_specifics['L']**2/(2*test_specifics['r0']**2) + (2*cs2 + gamma*cA2)/(2*(gamma-1))

                computational_grid[...,rho] = (
                    mfuncs.divide(
                        np.maximum(0, torus_phi + test_specifics['GM']/r - test_specifics['L']**2/(2*r**2)),
                        test_specifics['K'] * (gamma/(gamma-1)) * (1 + (r**(2*(gamma-1)))/test_specifics['beta0'])
                    )
                )**(1/(gamma-1))
                computational_grid[...,pressure] = _P
                computational_grid[...,vx] = -np.sqrt(test_specifics['GM'] * test_specifics['L']**2) * (y/r**2)
                computational_grid[...,vy] = np.sqrt(test_specifics['GM'] * test_specifics['L']**2) * (x/r**2)
                computational_grid[...,Bx] = -test_specifics['B_phi'] * y/r
                computational_grid[...,By] = test_specifics['B_phi'] * x/r

            elif match(any, ["rotor", "blob"]):
                if "blob" in config:
                    omega, B_ampl = test_specifics['omega'], test_specifics['B_ampl']
                    smoothing = lambda q: q * np.exp(-.5 * (r/r0)**2)

                    computational_grid[...,rho] += smoothing(init_cond[rho])
                    computational_grid[...,pressure] += smoothing(.25 * ambient[rho] * (omega*r0)**2)

                    computational_grid[...,vx] = smoothing(omega) * -y
                    computational_grid[...,vy] = smoothing(omega) * x

                    if config.startswith('m'):
                        computational_grid[...,Bx] = -B_ampl * np.sin(y)
                        computational_grid[...,By] = B_ampl * np.sin(2*x)

                else:
                    ring_pos = r0 + test_specifics['ring_width']
                    phi = (ring_pos - r)/(ring_pos - r0)
                    r_ring = np.sqrt((ring_pos-x_centre)**2 + (ring_pos-y_centre)**2)

                    ring = np.where(r <= r_ring)
                    computational_grid[...,rho][ring] = (1 + 9*phi)[ring]
                    computational_grid[...,vx][ring] = ((-test_specifics['omega']*phi*y0*shock_pos)/r)[ring]
                    computational_grid[...,vy][ring] = ((test_specifics['omega']*phi*x0*shock_pos)/r)[ring]

                    core = np.where(r <= r0)
                    computational_grid[core] = init_cond
                    computational_grid[...,vx][core] = ((-test_specifics['omega']*y0)/shock_pos)[core]
                    computational_grid[...,vy][core] = ((test_specifics['omega']*x0)/shock_pos)[core]

            elif match(any, ["turb", "blank"]):
                if "turb" in config:
                    if test_specifics['magnetic']:
                        computational_grid[...,Bx] = -test_specifics['mag_ampl'] * np.sin(2*np.pi*y)
                        computational_grid[...,By] = test_specifics['mag_ampl'] * np.sin(4*np.pi*x)
                else:
                    computational_grid[...,rho] += np.random.uniform(-test_specifics['perturb_ampl'], test_specifics['perturb_ampl'], size=(computational_grid.shape))[...,rho]

            elif match(any, ["current", "sheet"]):
                computational_grid[...,vx] = test_specifics['ampl'] * np.sin(np.pi*y)
                mask = np.where(abs(x) < shock_pos)
                computational_grid[...,By][mask] *= -1

            elif "noh" in config:
                mask = np.where(((x-coordinates[0][0])**2 + (y-coordinates[1][0])**2) > (shock_pos-coordinates[0][0])**2)
                computational_grid[...,vx][mask] = -np.sin(x-shock_pos)[mask]
                computational_grid[...,vy][mask] = -np.cos(x-shock_pos)[mask]

            elif "cloud" in config:
                computational_grid[np.where(x < shock_pos)] = init_cond

                (x0,y0), cloud_r = test_specifics['pos'], test_specifics['radius']
                r = np.sqrt((x-x0)**2 + (y-y0)**2)
                mask = np.where(r**2 < cloud_r**2)

                if test_specifics['smoothing']:
                    computational_grid[...,rho][mask] = ambient[rho] + .5*(test_specifics['mass']-ambient[rho])*(1-np.tanh((r[mask] - cloud_r)/.005))  # top-hat distribution
                else:
                    computational_grid[...,rho][mask] = test_specifics['mass']

            elif "jet" in config:
                nozzle = np.where((np.abs(x) < shock_pos) & (y <= (coordinates[1][0] + ds[1])))
                sim_variables.mask = nozzle
                computational_grid[...,rho][nozzle] = gamma
                computational_grid[...,vy][nozzle] = test_specifics['velocity']
                computational_grid[...,By] *= np.sqrt(10)  # weak: 1, moderate:np.sqrt(10), strong:np.sqrt(1e2), extreme:np.sqrt(1e3)
                if test_specifics['perturb']:
                    perturbations = turbulence.pertubations(computational_grid, test_specifics['velocity']/4)
                    computational_grid[...,(vx,vy)] += perturbations[...,(vx,vy)]

            elif match(any, ["circular", "polarised", "alfven"]) or config == "cpaw":
                alpha, ampl, wave = test_specifics['alpha'], test_specifics['ampl'], test_specifics['wave']
                s = x*np.cos(alpha) + y*np.sin(alpha)

                v_perp = B_perp = ampl*np.sin(2*np.pi*s)
                computational_grid[...,vz] = ampl*np.cos(2*np.pi*s)
                computational_grid[...,Bz] = ampl*np.cos(2*np.pi*s)

                # Ensure that v_parallel = 0, B_parallel = 1
                computational_grid[...,vx] = -v_perp*np.sin(alpha)
                computational_grid[...,vy] = v_perp*np.cos(alpha)
                computational_grid[...,Bx] = np.cos(alpha) - B_perp*np.sin(alpha)
                computational_grid[...,By] = np.sin(alpha) + B_perp*np.cos(alpha)

                # v_parallel = 1
                if wave == "standing":
                    computational_grid[...,vx] = np.cos(alpha) - v_perp*np.sin(alpha)
                    computational_grid[...,vy] = np.sin(alpha) + v_perp*np.cos(alpha)

            else:
                mask = np.where(r <= r0)
                computational_grid[mask] = init_cond

    else:
        ##############################
        #  1-dimensional cases
        ##############################
        x = physical_grid_x

        if ("sedov" in config) or match(all, ["mhd", "blast"]) or (match(any, ["supernova", "tycho"]) or config == "sn") or config.startswith('sq'):
            if match(any, ["supernova", "tycho"]) or config == "sn":
                if test_specifics['mode'].lower().startswith(('o','q')):
                    x_centre = coordinates[0][0]
                    x -= x_centre

                E, M, t0 = test_specifics['E'], test_specifics['M'], test_specifics['t0']
                shock_pos = 2 * t0 * np.sqrt(gamma * E/M)
                sim_variables.shock_pos = shock_pos

                r0 = shock_pos - x_centre
                rho0 = 25/(21*np.pi) * (E**2)/M * t0**4 * r0**-7

                mask = np.where(r > r0)
                computational_grid[...,rho][mask] += rho0 * (r[mask]/r0)**-7

                core = np.where(r <= r0)
                computational_grid[core] = init_cond
                computational_grid[...,rho][core] += rho0

                sigma, L = .75 * r0, r0
                e_tot = ambient[pressure]/(gamma-1) + mfuncs.smoothing_kernel(E/L, r, d=dimensions, sigma=sigma)
                computational_grid[...,pressure] = (gamma - 1) * e_tot

            else:
                mask = np.where(np.abs(x) <= shock_pos)
                computational_grid[mask] = init_cond

                if "sedov" in config:
                    sigma = np.abs(shock_pos - x_centre)
                    computational_grid[...,pressure] = ambient[pressure] + mfuncs.smoothing_kernel(init_cond[pressure], x, d=dimensions, mu=x_centre, sigma=sigma)

        else:
            mask = np.where(x <= shock_pos)
            computational_grid[mask] = init_cond

        if config.startswith('m') or "mhd" in config:
            computational_grid[...,5+axes] = test_specifics['ampl']

        if match(any, ["shu", "osher"]) or config == "so":
            computational_grid[np.where(x > shock_pos), rho] = mfuncs.sine_func(x[x > shock_pos], test_specifics)
        elif config.startswith("sin"):
            computational_grid[...,rho] = mfuncs.sine_func(x, test_specifics)
        elif config.startswith('gauss'):
            computational_grid[...,rho] = mfuncs.gauss_func(x-test_specifics['peak_pos'], test_specifics)
        elif match(any, ["manufacture", "euler"]):
                Lx = np.diff(coordinates[0])
                freq = test_specifics['freq']
                computational_grid[...,rho] = 1 + .35*np.sin(freq*x/Lx)
                computational_grid[...,vy] = computational_grid[...,vz] = 0
                computational_grid[...,pressure] = 1 + .23*np.sin(freq*x/Lx)
        elif match(any, ["turb", "blank"]):
            if "turb" in config:
                if test_specifics['magnetic']:
                    computational_grid[...,Bx] = -test_specifics['mag_ampl'] * np.sin(2*np.pi*y)
                    computational_grid[...,By] = test_specifics['mag_ampl'] * np.sin(4*np.pi*x)
            else:
                computational_grid[...,rho] += np.random.uniform(-test_specifics['perturb_ampl'], test_specifics['perturb_ampl'], size=(computational_grid.shape))[...,rho]

    sim_variables.magnetic = computational_grid[...,sim_variables.Bfields].any()

    if sim_variables.grid_interpolate:
        return gutils.method_convert_cell('point', computational_grid, sim_variables)
    else:
        return computational_grid