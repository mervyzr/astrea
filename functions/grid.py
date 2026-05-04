import concurrent.futures
from itertools import repeat

import numpy as np
from skimage.measure import block_reduce

from functions import generic
from functions import math as mfuncs
from functions.generic import verbose_timer

##############################################################################
# Grid functions used throughout the finite volume code
##############################################################################

# Create a physical grid for a single axis
def make_physical_grid(axis_coord, cells):
    start_pos, end_pos = axis_coord
    dh = np.abs(np.diff(axis_coord)[0])/cells
    half_cell = .5 * dh
    return np.linspace(start_pos-half_cell, end_pos+half_cell, cells+2)[1:-1]


# Initialise the discrete POINTWISE solution array with initial conditions and primitive variables w, and transform into discrete AVERAGES <w>
# For magnetohydrodynamics, this returns a staggered grid
@verbose_timer
def initialise(sim_variables):
    config, cells, gamma, dimensions, multidimensional = sim_variables.config, sim_variables.cells, sim_variables.gamma, sim_variables.dimensions, sim_variables.multidimensional
    rho, vx, vy, vz, pressure, Bx, By, Bz = sim_variables.rho, sim_variables.vx, sim_variables.vy, sim_variables.vz, sim_variables.pressure, sim_variables.Bx, sim_variables.By, sim_variables.Bz
    ds, coordinates, shock_pos, params = sim_variables.ds, sim_variables.coordinates, sim_variables.shock_pos, sim_variables.misc
    init_cond, ambient = sim_variables.init_cond, sim_variables.ambient
    axes = sim_variables.axes

    match = lambda match_type, substrings: match_type(substring in config for substring in substrings)


    computational_grid = np.zeros(list(cells)+[len(ambient),], dtype=float, order='C')
    computational_grid[:] = ambient

    x_centre = np.average(coordinates[0])
    physical_grid_x = make_physical_grid(coordinates[0], cells[0])

    if multidimensional:
        y_centre = np.average(coordinates[1])
        physical_grid_y = make_physical_grid(coordinates[1], cells[1])

        if dimensions > 2:
            ##############################
            #  3-dimensional cases
            ##############################
            z_centre = np.average(coordinates[2])
            physical_grid_z = make_physical_grid(coordinates[2], cells[2])

            x, y, z = np.meshgrid(physical_grid_x, physical_grid_y, physical_grid_z, indexing='ij')
            x0, y0, z0 = x - x_centre, y - y_centre, z - z_centre
            r = np.sqrt(x0**2 + y0**2 + z0**2)
            r0 = np.sqrt((shock_pos-x_centre)**2 + (shock_pos-y_centre)**2 + (shock_pos-z_centre)**2)

            if ("sedov" in config) or match(all, ["mhd", "blast"]) or (match(any, ["supernova", "tycho"]) or config == "sn"):
                if match(any, ["supernova", "tycho"]) or config == "sn":
                    if params['mode'].lower().startswith(('o','q')):
                        x_centre, y_centre, z_centre = (axis_coord[0] for axis_coord in coordinates.values())
                        r = np.sqrt((x-x_centre)**2 + (y-y_centre)**2 + (z-z_centre)**2)

                    E, M, t0 = params['E'], params['M'], params['t0']
                    shock_pos = t0 * np.sqrt(gamma * E/M)
                    sim_variables.shock_pos = shock_pos

                    r0 = np.sqrt((shock_pos-x_centre)**2 + (shock_pos-y_centre)**2 + (shock_pos-z_centre)**2)
                    rho0 = 25/(21*np.pi) * (E**2)/M * t0**4 * r0**-7

                    mask = np.where(r > r0)
                    computational_grid[...,rho][mask] += rho0 * (r[mask]/r0)**-7

                    core = np.where(r <= r0)
                    computational_grid[core] = init_cond
                    computational_grid[...,rho][core] += rho0

                    sigma = .75 * r0
                    e_tot = (ambient[...,pressure]/(gamma-1) + (E/(4/3 * np.pi * r0**3))/((2*np.pi*sigma**2)**(dimensions/2)) * np.exp(-.5 * r/sigma**2))
                    computational_grid[...,pressure] = (gamma - 1) * e_tot

                else:
                    mask = np.where(r <= r0)
                    computational_grid[mask] = init_cond

                    if match(all, ["mhd", "blast"]):
                        computational_grid[...,5+axes] = params['ampl']

                computational_grid = resample_blast(computational_grid, sim_variables)

            elif config.startswith("sin"):
                computational_grid[...,rho] = mfuncs.sine_func(r, params)

            elif config.startswith("gauss"):
                computational_grid[...,rho] = mfuncs.gauss_func(r, params)

            elif match(any, ["manufacture", "euler"]):
                Lx, Ly, Lz = np.diff(coordinates[0]), np.diff(coordinates[1]), np.diff(coordinates[2])
                freq = params['freq']

                computational_grid[...,rho] = 1 + .35*np.sin(freq*x/Lx) + .24*np.cos(freq*y/Ly) + .1*np.sin(freq*z/Lz)
                computational_grid[...,pressure] = 1 + .23*np.sin(freq*x/Lx) + .19*np.cos(freq*y/Ly) + .2*np.cos(freq*z/Lz)

            elif "blank" in config:
                computational_grid[...,rho] += np.random.uniform(-params['perturb_ampl'], params['perturb_ampl'], size=(computational_grid.shape))[...,rho]

            elif match(any, ["orszag", "tang"]) or config == "ot":
                _x, _y, _z, ampl, eps = params['norm_factor']*x, params['norm_factor']*y, params['norm_factor']*z, params['ampl'], params['eps']

                computational_grid[...,vx] = -(1 + eps*np.sin(_z)) * np.sin(_y)
                computational_grid[...,vy] = (1 + eps*np.sin(_z)) * np.sin(_x)
                computational_grid[...,vz] = eps * np.sin(_z)
                computational_grid[...,Bx] = -ampl * np.sin(_y)
                computational_grid[...,By] = ampl * np.sin(2*_x)

            elif match(all, ["mhd", "vortex"]):
                factor = np.exp(params['q'] * (1 - r**2))
                computational_grid[...,vx] = 1 - y0*params['kappa']*factor
                computational_grid[...,vy] = 1 + x0*params['kappa']*factor
                computational_grid[...,pressure] = 1 + (1/(4*params['q'])) * ((1 - 2*params['q']*(r**2 - z0**2)) * params['mu']**2 - params['kappa']**2) * factor**2
                computational_grid[...,Bx] = -y0*params['mu']*factor
                computational_grid[...,By] = x0*params['mu']*factor

            elif "torus" in config:
                r = np.sqrt(x0**2 + y0**2)
                cA2 = lambda _r: 2 * (params['polytropeK']/params['beta']) * (init_cond[rho] * _r**2)**(gamma-1)
                cs2 = np.sqrt(gamma * init_cond[pressure]/init_cond[rho])
                torus_phi = -params['GM']/params['r0'] + params['L']**2/(2*params['r0']**2) + (2*cs2 + gamma*cA2(params['r0']))/(2*(gamma-1))

                computational_grid[...,rho] = (
                    mfuncs.divide(
                        np.maximum(0, torus_phi + params['GM']/params['r0'] - params['L']**2/(2*r**2)),
                        params['polytropeK'] * (gamma/(gamma-1)) * (1 + (r**(2*(gamma-1)))/params['beta'])
                    )
                )**(1/(gamma-1))
                computational_grid[...,vx] = -np.sqrt(params['GM']) * (y/r**1.5)
                computational_grid[...,vy] = np.sqrt(params['GM']) * (x/r**1.5)

                _r = np.sqrt((x-x_centre)**2 + (y-y_centre)**2)
                _P = params['K'] * init_cond[rho]**gamma
                cA2 = 2 * (params['K']/params['beta0']) * (init_cond[rho] * params['r0']**2)**(gamma-1)
                cs2 = np.sqrt(gamma * _P/init_cond[rho])
                torus_phi = -params['GM']/params['r0'] + params['L']**2/(2*params['r0']**2) + (2*cs2 + gamma*cA2)/(2*(gamma-1))

                computational_grid[...,rho] = (
                    mfuncs.divide(
                        np.maximum(0, torus_phi + params['GM']/r - params['L']**2/(2*_r**2)),
                        params['K'] * (gamma/(gamma-1)) * (1 + (_r**(2*(gamma-1)))/params['beta0'])
                    )
                )**(1/(gamma-1))
                computational_grid[...,pressure] = _P
                computational_grid[...,vx] = -np.sqrt(params['GM'] * params['L']**2) * (y/_r**2)
                computational_grid[...,vy] = np.sqrt(params['GM'] * params['L']**2) * (x/_r**2)
                computational_grid[...,Bx] = -params['B_phi'] * y/_r
                computational_grid[...,By] = params['B_phi'] * x/_r
                if sim_variables.ext_gravity:
                    computational_grid[...,sim_variables.gx] = -params['GM']/r**3 * x
                    computational_grid[...,sim_variables.gy] = -params['GM']/r**3 * y
                    computational_grid[...,sim_variables.gz] = -params['GM']/r**3 * z

            elif match(any, ["blob"]):
                dr = np.sqrt(np.sum([dh**2 for dh in ds.values()]))
                smoothing = 1 - np.tanh((r - r0)/(5 * dr))
                computational_grid[...,rho] += (init_cond-ambient)[rho] * .5 * smoothing
                computational_grid[...,pressure] += (init_cond-ambient)[pressure] * .5 * smoothing

                omega = np.sqrt(init_cond[rho] * np.pi/shock_pos)
                computational_grid[...,vx] = -omega * y
                computational_grid[...,vy] = omega * x

                B_phi = np.sqrt((2*init_cond[pressure])/params['beta']) * smoothing * r/r0
                computational_grid[...,Bx] = -B_phi * (y/(r+params['eps']))
                computational_grid[...,By] = B_phi * (x/(r+params['eps']))

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
                    if params['mode'].lower().startswith(('o','q')):
                        x_centre, y_centre = (axis_coord[0] for axis_coord in coordinates.values())
                        r = np.sqrt((x-x_centre)**2 + (y-y_centre)**2)

                    E, M, t0 = params['E'], params['M'], params['t0']
                    shock_pos = t0 * np.sqrt(gamma * E/M)
                    sim_variables.shock_pos = shock_pos

                    r0 = np.sqrt((shock_pos-x_centre)**2 + (shock_pos-y_centre)**2)
                    rho0 = 25/(21*np.pi) * (E**2)/M * t0**4 * r0**-7

                    mask = np.where(r > r0)
                    computational_grid[...,rho][mask] += rho0 * (r[mask]/r0)**-7

                    core = np.where(r <= r0)
                    computational_grid[core] = init_cond
                    computational_grid[...,rho][core] += rho0

                    sigma = .75 * r0
                    e_tot = (ambient[...,pressure]/(gamma-1) + (E/(np.pi * r0**2))/((2*np.pi*sigma**2)**(dimensions/2)) * np.exp(-.5 * r/sigma**2))
                    computational_grid[...,pressure] = (gamma - 1) * e_tot

                else:
                    mask = np.where(r <= r0)
                    computational_grid[mask] = init_cond

                    if match(all, ["mhd", "blast"]):
                        computational_grid[...,5+axes] = params['ampl']

                computational_grid = resample_blast(computational_grid, sim_variables)

            elif config.startswith("sin"):
                computational_grid[...,rho] = mfuncs.sine_func(r, params)

            elif config.startswith("gauss"):
                computational_grid[...,rho] = mfuncs.gauss_func(r, params)

            elif match(any, ["manufacture", "euler"]):
                Lx, Ly = np.diff(coordinates[0]), np.diff(coordinates[1])
                freq = params['freq']

                computational_grid[...,vz] = 0
                computational_grid[...,rho] = 1 + .35*np.sin(freq*x/Lx) + .24*np.cos(freq*y/Ly)
                computational_grid[...,pressure] = 1 + .23*np.sin(freq*x/Lx) + .19*np.cos(freq*y/Ly)

            elif match(any, ["kelvin", "helmholtz", "khi"]):
                layer = np.where(np.abs(y) <= shock_pos)
                computational_grid[layer] = init_cond
                computational_grid[...,vy] = params['ampl'] * np.sin(params['freq']*np.pi*x/np.diff(coordinates[0]))
                if params['perturb']:
                    computational_grid[...,(vx,vy)] += np.random.uniform(-params['ampl']/2, params['ampl']/2, size=computational_grid.shape)[...,(vx,vy)]
                if config.startswith('m') or "mhd" in config:
                    computational_grid[...,Bx] = params['Bx']

            elif match(any, ["rayleigh", "taylor", "rti"]):
                layer = np.where(y > shock_pos)
                computational_grid[layer] = init_cond
                computational_grid[...,pressure] = init_cond[pressure] - .1*computational_grid[...,rho]*y
                if sim_variables.ext_gravity:
                    computational_grid[...,sim_variables.gy] = -params['grav_acc']
                if params['perturb']:
                    computational_grid[...,vy] += (.5 * np.random.uniform(-2*params['ampl'], 2*params['ampl'], size=computational_grid.shape))[...,vy] * (1 + np.cos(8*np.pi*y/3))
                else:
                    computational_grid[...,vy] = params['ampl'] * (1 + np.cos(4*np.pi*x)) * (1 + np.cos(3*np.pi*y))
                if config.startswith('m') or "mhd" in config:
                    computational_grid[...,Bx] = params['Bx']

            elif match(any, ["ivc", "isentropic"]):
                b, freq = params['vortex_str'], params['freq']

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
                p0 = init_cond[...,rho]/(gamma*params['mach']**2)

                computational_grid[...,pressure] = p0 - 2 + 4*np.log(2)

                computational_grid[...,vx][ring] = (5 - 2*r)[ring]
                computational_grid[...,vy][ring] = (2*r - 5)[ring]
                computational_grid[...,pressure][ring] = (p0 + (25/2)*r**2 + 4*(1 - 5*r + np.log(5*r)))[ring]

                computational_grid[...,vx][core] = -5
                computational_grid[...,vy][core] = 5
                computational_grid[...,pressure][core] = (p0 + (25/2)*r**2)[core]

            elif match(any, ["lax", "liu", "ll"]):
                computational_grid[np.where(x < shock_pos)] = init_cond
                computational_grid[np.where((x < shock_pos) & (y < shock_pos))] = params['bottom_left']
                computational_grid[np.where((x >= shock_pos) & (y < shock_pos))] = params['bottom_right']
                if config.startswith('m') or "mhd" in config:
                    computational_grid[...,Bx] = np.cos(y) * np.sin(x)
                    computational_grid[...,By] = -np.cos(x) * np.sin(y)

            elif match(any, ["yee", "sjögreen", "sjoegreen"]) or config == "ys":
                computational_grid[np.where(x < shock_pos)] = init_cond
                computational_grid[np.where((x < shock_pos) & (y < shock_pos))] = params['bottom_left']
                computational_grid[np.where((x >= shock_pos) & (y < shock_pos))] = params['bottom_right']

            elif match(any, ["orszag", "tang"]) or config == "ot":
                _x, _y, ampl = params['norm_factor']*x, params['norm_factor']*y, params['ampl']

                computational_grid[...,vx] = -np.sin(_y)
                computational_grid[...,vy] = np.sin(_x)
                computational_grid[...,Bx] = -ampl * np.sin(_y)
                computational_grid[...,By] = ampl * np.sin(2*_x)

            elif match(all, ["mhd", "vortex"]):
                computational_grid[...,vx] = 1 - ((y0*params['kappa'])/(2*np.pi) * np.exp((1-r**2)/2))
                computational_grid[...,vy] = 1 + ((x0*params['kappa'])/(2*np.pi) * np.exp((1-r**2)/2))
                computational_grid[...,pressure] = 1 + (((1-r**2)*params['kappa']**2 - params['mu']**2)/(8*np.pi**2) * np.exp(1-r**2))
                computational_grid[...,Bx] = (-y0*params['mu'])/(2*np.pi) * np.exp((1-r**2)/2)
                computational_grid[...,By] = (x0*params['mu'])/(2*np.pi) * np.exp((1-r**2)/2)

            elif "torus" in config:
                _P = params['K'] * init_cond[rho]**gamma
                cA2 = 2 * (params['K']/params['beta0']) * (init_cond[rho] * params['r0']**2)**(gamma-1)
                cs2 = np.sqrt(gamma * _P/init_cond[rho])
                torus_phi = -params['GM']/params['r0'] + params['L']**2/(2*params['r0']**2) + (2*cs2 + gamma*cA2)/(2*(gamma-1))

                computational_grid[...,rho] = (
                    mfuncs.divide(
                        np.maximum(0, torus_phi + params['GM']/r - params['L']**2/(2*r**2)),
                        params['K'] * (gamma/(gamma-1)) * (1 + (r**(2*(gamma-1)))/params['beta0'])
                    )
                )**(1/(gamma-1))
                computational_grid[...,pressure] = _P
                computational_grid[...,vx] = -np.sqrt(params['GM'] * params['L']**2) * (y/r**2)
                computational_grid[...,vy] = np.sqrt(params['GM'] * params['L']**2) * (x/r**2)
                computational_grid[...,Bx] = -params['B_phi'] * y/r
                computational_grid[...,By] = params['B_phi'] * x/r
                if sim_variables.ext_gravity:
                    computational_grid[...,sim_variables.gx] = -params['GM']/r**3 * x
                    computational_grid[...,sim_variables.gy] = -params['GM']/r**3 * y

            elif match(any, ["rotor", "blob"]):
                if "blob" in config:
                    dr = np.sqrt(np.sum([dh**2 for dh in ds.values()]))
                    smoothing = 1 - np.tanh((r - r0)/(5 * dr))
                    computational_grid[...,rho] += (init_cond-ambient)[rho] * .5 * smoothing
                    computational_grid[...,pressure] += (init_cond-ambient)[pressure] * .5 * smoothing

                    omega = np.sqrt(init_cond[rho] * np.pi/shock_pos)
                    computational_grid[...,vx] = -omega * y
                    computational_grid[...,vy] = omega * x

                    B_phi = np.sqrt((2*init_cond[pressure])/params['beta']) * smoothing * r/r0
                    computational_grid[...,Bx] = -B_phi * (y/(r+params['eps']))
                    computational_grid[...,By] = B_phi * (x/(r+params['eps']))

                else:
                    ring_pos = r0 + params['ring_width']
                    phi = (ring_pos - r)/(ring_pos - r0)
                    r_ring = np.sqrt((ring_pos-x_centre)**2 + (ring_pos-y_centre)**2)

                    ring = np.where(r <= r_ring)
                    computational_grid[...,rho][ring] = (1 + 9*phi)[ring]
                    computational_grid[...,vx][ring] = ((-params['omega']*phi*y0*shock_pos)/r)[ring]
                    computational_grid[...,vy][ring] = ((params['omega']*phi*x0*shock_pos)/r)[ring]

                    core = np.where(r <= r0)
                    computational_grid[core] = init_cond
                    computational_grid[...,vx][core] = ((-params['omega']*y0)/shock_pos)[core]
                    computational_grid[...,vy][core] = ((params['omega']*x0)/shock_pos)[core]

            elif "blank" in config:
                computational_grid[...,rho] += np.random.uniform(-params['perturb_ampl'], params['perturb_ampl'], size=(computational_grid.shape))[...,rho]

            elif match(any, ["current", "sheet"]):
                computational_grid[...,vx] = params['ampl'] * np.sin(np.pi*y)
                mask = np.where(abs(x) < shock_pos)
                computational_grid[...,By][mask] *= -1

            elif "noh" in config:
                mask = np.where(((x-coordinates[0][0])**2 + (y-coordinates[1][0])**2) > (shock_pos-coordinates[0][0])**2)
                computational_grid[...,vx][mask] = -np.sin(x-shock_pos)[mask]
                computational_grid[...,vy][mask] = -np.cos(x-shock_pos)[mask]

            elif "cloud" in config:
                computational_grid[np.where(x < shock_pos)] = init_cond
                mask = np.where(((x-.8)**2 + (y-.5)**2) < .15**2)
                computational_grid[...,rho][mask] = params['cloud_mass']

            elif "jet" in config:
                nozzle = np.where((np.abs(x) < shock_pos) & (y <= (coordinates[1][0] + ds[1])))
                sim_variables.mask = nozzle
                computational_grid[...,rho][nozzle] = gamma
                computational_grid[...,vy][nozzle] = params['velocity']
                computational_grid[...,By] *= np.sqrt(10)  # weak: 1, moderate:np.sqrt(10), strong:np.sqrt(1e2), extreme:np.sqrt(1e3)
                if params['perturb']:
                    computational_grid[...,(vx,vy)] += np.random.uniform(-params['velocity']/4, params['velocity']/4, size=(computational_grid.shape))[...,(vx,vy)]

            elif match(any, ["circular", "polarised", "alfven"]) or config == "cpaw":
                alpha, ampl, wave = params['alpha'], params['ampl'], params['wave']
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
                if params['mode'].lower().startswith(('o','q')):
                    x_centre = coordinates[0][0]
                    x -= x_centre

                E, M, t0 = params['E'], params['M'], params['t0']
                shock_pos = t0 * np.sqrt(gamma * E/M)
                sim_variables.shock_pos = shock_pos

                r0 = shock_pos - x_centre
                rho0 = 25/(21*np.pi) * (E**2)/M * t0**4 * r0**-7

                mask = np.where(r > r0)
                computational_grid[...,rho][mask] += rho0 * (r[mask]/r0)**-7

                core = np.where(r <= r0)
                computational_grid[core] = init_cond
                computational_grid[...,rho][core] += rho0

                sigma = .75 * r0
                e_tot = (ambient[...,pressure]/(gamma-1) + (E/r0)/((2*np.pi*sigma**2)**(dimensions/2)) * np.exp(-.5 * r/sigma**2))
                computational_grid[...,pressure] = (gamma - 1) * e_tot

            else:
                mask = np.where(r <= r0)
                computational_grid[mask] = init_cond

            if not config.startswith('sq'):
                computational_grid = resample_blast(computational_grid, sim_variables)

        else:
            mask = np.where(x <= shock_pos)
            computational_grid[mask] = init_cond

        if config.startswith('m') or "mhd" in config:
            computational_grid[...,5+axes] = params['ampl']

        if match(any, ["shu", "osher"]) or config == "so":
            computational_grid[np.where(x > shock_pos), rho] = mfuncs.sine_func(x[x > shock_pos], params)
        elif config.startswith("sin"):
            computational_grid[...,rho] = mfuncs.sine_func(x, params)
        elif config.startswith('gauss'):
            computational_grid[...,rho] = mfuncs.gauss_func(x-params['peak_pos'], params)
        elif match(any, ["manufacture", "euler"]):
                Lx = np.diff(coordinates[0])
                freq = params['freq']
                computational_grid[...,rho] = 1 + .35*np.sin(freq*x/Lx)
                computational_grid[...,vy] = computational_grid[...,vz] = 0
                computational_grid[...,pressure] = 1 + .23*np.sin(freq*x/Lx)
        elif "blank" in config:
            computational_grid[...,rho] += np.random.uniform(-params['perturb_ampl'], params['perturb_ampl'], size=(computational_grid.shape))[...,rho]

    sim_variables.magnetic = computational_grid[...,sim_variables.Bfields].any()

    if sim_variables.grid_interpolate:
        return method_convert_cell('point', computational_grid, sim_variables)
    else:
        return computational_grid


# Slice grid along axis
def slice_(grid, axis, start=0, end=None, step=1, *args):
    slc = [slice(None)] * grid.ndim

    if args and (2 <= len(args) <= 3):
        try:
            start, end, step = args
        except ValueError:
            start, end = args

    if end == None:
        end = grid.shape[axis]

    slc[axis] = slice(start, end, step)
    return grid[tuple(slc)]


# Finite difference derivative (second order) of a padded grid
# [ W(i+1) - W(i) ] - [ W(i) - W(i-1) ] = W(i+1) - 2W(i) + W(i-1)
def laplacian(grid, sim_variables, axis):
    padded_grid = add_boundary(grid, sim_variables, axis=axis)
    return 1/(sim_variables.ds[axis]**2) * (np.diff(slice_(padded_grid, axis, start=1), axis=axis) - np.diff(slice_(padded_grid, axis, end=-1), axis=axis))


# Add boundary conditions
def add_boundary(grid, sim_variables, stencil=1, axis=0):
    padding = [(0,0)] * grid.ndim
    padding[axis] = (stencil,stencil)
    return np.pad(grid, padding, mode=sim_variables.boundary)


# Convert between pressure P and total energy density e_tot; P is also related to the internal energy density e_int: P = (gamma-1) * e_int
# Do note that the energy densities e are related to the energies E: e_tot = rho * E_tot, e_int = rho * E_int
def convert_thermo_variable(variable, grid, sim_variables):
    rho, pressure, vels, Bfields = sim_variables.rho, sim_variables.pressure, sim_variables.vels, sim_variables.Bfields
    energy, momentums = pressure, vels
    gamma, permeability = sim_variables.gamma, sim_variables.constants.mu_0

    if variable.lower().startswith('p'):
        # pressure -> (total) energy density
        return (
            grid[...,pressure]/(gamma-1)
            + .5*(grid[...,rho]*mfuncs.norm(grid[...,vels])**2)
            + .5*(mfuncs.norm(grid[...,Bfields])**2)/permeability
        )
    elif variable.lower().startswith('e') or 'energy' in variable.lower():
        # (total) energy density -> pressure
        return (
            (gamma-1) * (
                grid[...,energy]
                - .5 * (grid[...,rho]*mfuncs.norm(mfuncs.divide(grid[...,momentums], grid[...,rho][...,None]))**2)
                - .5 * (mfuncs.norm(grid[...,Bfields])**2)/permeability
                )
        )


# Handler for conversion
def convert(variable_form, grid, sim_variables):
    converter = variable_convert if sim_variables.grid_interpolate else variable_point_convert
    return converter(variable_form, grid, sim_variables)


# Pointwise (exact) conversion of conservative variables q <-> primitive variables w (up to 2nd-order accurate)
def variable_point_convert(variable_form, grid, sim_variables):
    rho, pressure, energy, vels, momentums = sim_variables.rho, sim_variables.pressure, sim_variables.energy, sim_variables.vels, sim_variables.momentums
    arr = np.copy(grid)

    if variable_form.lower().startswith("p"):
        arr[...,energy] = convert_thermo_variable('pressure', grid, sim_variables)
        arr[...,momentums] = grid[...,vels] * grid[...,rho][...,None]
    elif variable_form.lower().startswith("c"):
        arr[...,pressure] = convert_thermo_variable('energy', grid, sim_variables)
        arr[...,vels] = mfuncs.divide(grid[...,momentums], grid[...,rho][...,None])
    return arr


# Variable inversion using the conversion of the grid (base) and the Taylor expansion terms (expansion) through a Laplacian (2nd-deriv, 2nd-order) approx. for each axis (up to 4th-order accurate)
def variable_inversion_per_axis(variable_form, grid, sim_variables, axis):
    original_expansion = (sim_variables.ds[axis]**2)/24 * laplacian(grid, sim_variables, axis)
    converted_avg = variable_point_convert(variable_form, grid, sim_variables)
    converted_expansion = (sim_variables.ds[axis]**2)/24 * laplacian(converted_avg, sim_variables, axis)
    return original_expansion, converted_expansion


# Converting cell-averaged conservative variables <q>_{i,j} <-> cell-averaged primitive variables <w>_{i,j} at higher-order accuracy
def variable_convert(variable_form, grid, sim_variables):
    base, expansion = np.copy(grid), np.zeros_like(grid)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        jobs = executor.map(variable_inversion_per_axis, repeat(variable_form), repeat(grid), repeat(sim_variables), sim_variables.axes)

        for original_expansion, converted_expansion in jobs:
            base -= original_expansion
            expansion += converted_expansion

    return variable_point_convert(variable_form, base, sim_variables) + expansion


# Converting face-averaged conservative variables <q>_{i+1/2,j} <-> face-averaged primitive variables <w>_{i+1/2,j}
def variable_convert_intf(variable_form, grid, sim_variables, axis):
    base, expansion = np.copy(grid), np.zeros_like(grid)

    if sim_variables.grid_interpolate and sim_variables.multidimensional:
        ortho_axes = sim_variables.axes[sim_variables.axes != axis]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            jobs = executor.map(variable_inversion_per_axis, repeat(variable_form), repeat(grid), repeat(sim_variables), ortho_axes)

            for original_expansion, converted_expansion in jobs:
                base -= original_expansion
                expansion += converted_expansion

    new_grid = variable_point_convert(variable_form, base, sim_variables) + expansion

    if sim_variables.magnetic:
        new_grid[...,5+sim_variables.axes] = grid[...,5+sim_variables.axes]

    return new_grid


# Method convert between point-representation (finite difference) and averaged-representation (finite volume) [ALL AXES]
# Converting cell-centred variables q_{i,j} <-> cell-averaged variables <q>_{i,j} through a Laplacian (2nd-deriv, 2nd-order) approx. for each axis (up to 4th-order accurate)
def method_convert_cell(grid_form, grid, sim_variables, axis=None):
    base = np.copy(grid)

    if sim_variables.grid_interpolate:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            jobs = executor.map(laplacian, repeat(grid), repeat(sim_variables), sim_variables.axes)

            for idx, expansion in enumerate(jobs):
                if grid_form.lower().startswith('a'):
                    # averaged -> point
                    base -= (sim_variables.ds[sim_variables.axes[idx]]**2)/24 * expansion
                elif grid_form.lower().startswith('p'):
                    # point -> averaged
                    base += (sim_variables.ds[sim_variables.axes[idx]]**2)/24 * expansion
    return base


# Method convert between point-representation (finite difference) and averaged-representation (finite volume) for interfaces [ORTHOGONAL AXES]
# Converting face-centred variables q_{i+1/2,j} <-> face-averaged variables <q>_{i+1/2,j} through a Laplacian (2nd-deriv, 2nd-order) approx. (up to 4th-order accurate)
def method_convert_intf(grid_form, grid, sim_variables, axis):
    base = np.copy(grid)

    if sim_variables.grid_interpolate and sim_variables.multidimensional:
        ortho_axes = sim_variables.axes[sim_variables.axes != axis]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            jobs = executor.map(laplacian, repeat(grid), repeat(sim_variables), ortho_axes)

            for idx, expansion in enumerate(jobs):
                if grid_form.lower().startswith('a'):
                    # averaged -> point
                    base -= (sim_variables.ds[ortho_axes[idx]]**2)/24 * expansion
                elif grid_form.lower().startswith('p'):
                    # point -> averaged
                    base += (sim_variables.ds[ortho_axes[idx]]**2)/24 * expansion
    return base


# Handler for converting (at higher-order) each +/- interface in each axis from averaged interfaces to point/centred interfaces in the multi-dimensional higher-order schemes
def approx_face_avg(interfaces, sim_variables, axis):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        return list(executor.map(method_convert_intf, repeat('avg'), interfaces, repeat(sim_variables), repeat(axis)))
    

# Compute the 4th-order interface-centred fluxes from the interface-averaged fluxes via higher order approximation for each orthogonal axis
def approx_flux_avg(cntrd_fluxes, avgd_fluxes, sim_variables, axis):
    ortho_axes = sim_variables.axes[sim_variables.axes != axis]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        jobs = executor.map(laplacian, repeat(avgd_fluxes), repeat(sim_variables), ortho_axes)
        for idx, job in enumerate(jobs):
            cntrd_fluxes -= (sim_variables.ds[ortho_axes[idx]]**2)/24 * job
    return cntrd_fluxes


# Re-align the interfaces so that cell wall is in between interfaces
def assign_interfaces(interfaces, grid, sim_variables, axis):
    wL, wR = interfaces
    return slice_(add_boundary(wL, sim_variables, axis=axis), axis, start=1), slice_(add_boundary(wR, sim_variables, axis=axis), axis, end=-1)


# Resample grid for circular blast injection to populate cell variables with a circle/sphere; value in grid cell is weighted by area/volume covered
def resample_blast(grid, sim_variables, subsample_limit=25600**2):
    print(f"{generic.BColours.WARNING}Blast config. used; supersampling initialised grid before starting simulation for better resolution..{generic.BColours.ENDC}")
    cells, dimensions, multidimensional, coordinates, shock_pos = sim_variables.cells, sim_variables.dimensions, sim_variables.multidimensional, sim_variables.coordinates, sim_variables.shock_pos

    # Dynamic sub-sampling (prevents clogging up computing resources with some outrageous grid size, e.g. (512 cells x 50 sub-sample)^3 is some crazy number)
    subsample_size = int(np.minimum(100, np.floor((subsample_limit / np.prod(cells)) ** (1/dimensions))))

    try:
        semi = sim_variables.misc['mode'].lower().startswith(('o','q'))
    except Exception:
        semi = False

    fine_grid = np.resize(np.zeros_like(grid), np.asarray(cells)*subsample_size)
    physical_grid = lambda axis: make_physical_grid(coordinates[axis], cells[axis]*subsample_size)

    if semi:
        x_centre = coordinates[0][0]
    else:
        x_centre = np.average(coordinates[0])
    fine_physical_grid_x = physical_grid(0)
    fine_x = fine_physical_grid_x - x_centre

    fine_y = fine_z = np.zeros_like(fine_x)
    y_centre = z_centre = 0

    if multidimensional:
        if semi:
            y_centre = coordinates[1][0]
        else:
            y_centre = np.average(coordinates[1])
        fine_physical_grid_y = physical_grid(1)
        fine_x, fine_y = np.meshgrid(fine_physical_grid_x, fine_physical_grid_y, indexing='ij')
        fine_z = np.zeros_like(fine_x)

        if dimensions == 3:
            if semi:
                z_centre = coordinates[2][0]
            else:
                z_centre = np.average(coordinates[2])
            fine_physical_grid_z = physical_grid(2)
            fine_x, fine_y, fine_z = np.meshgrid(fine_physical_grid_x, fine_physical_grid_y, fine_physical_grid_z, indexing='ij')

    fine_r = np.sqrt((fine_x-x_centre)**2 + (fine_y-y_centre)**2 + (fine_z-z_centre)**2)
    fine_r0 = np.sqrt((shock_pos-x_centre)**2 + (shock_pos-y_centre)**2 + (shock_pos-z_centre)**2)
    fine_mask = np.where(fine_r <= fine_r0)
    fine_grid[fine_mask] = 1

    remapped_grid = block_reduce(fine_grid, block_size=tuple([subsample_size,]*dimensions), func=np.sum)
    mask = np.where(remapped_grid > 0)

    # Remap density and pressure
    arr = np.copy(grid)
    arr[...,sim_variables.pressure][mask] *= (remapped_grid/np.max(remapped_grid))[mask]
    arr[...,sim_variables.rho][mask] *= (remapped_grid/np.max(remapped_grid))[mask]

    return arr