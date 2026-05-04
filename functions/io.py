import os
import random
import argparse

import yaml
import h5py
import numpy as np
from tinydb import TinyDB, Query

from functions import generic
from functions.generic import BColours
from static import tests
from physics import constants as const
from physics.krome import krome_funcs

##############################################################################
# I/O functions for simulation
##############################################################################

# Make simulation variables when testing functions in Python REPL; 
# most functions require sim_variables, so it might be useful to have a function auto-generate one as needed
def make_sim_variables():
    config_variables = {
        'home': '.',
        'db_path': "static/.db.json",
        'hdf5': ".astrea_hdf5_temp_-1",
        'chemistry': False,
        'tracers': False,
        'gravity': False,
        'init': False,
        'verbose': False,
        'write_chkpt': False,
        'test': False,
        'quiet': False,
    }
    with open('parameters.yml', "r") as _f:
        _config_variables = yaml.safe_load(_f)
        for parameters in _config_variables.values():
            for k,v in parameters.items():
                config_variables[k] = v
    config_variables = filter_variables(config_variables)
    test_variables = tests.generate_test_conditions(config_variables)
    sim_variables = SimulationVariables(-1, config_variables, test_variables)
    return sim_variables


# CLI arguments handler; updates the simulation variables (which is a dict) and checks for any invalid values
def handle_CLI(db_path):

    def bool_handler(value):
        return (value.lower() == 'true' or value.lower() == '1')

    db, params = TinyDB(db_path), Query()

    bool_choices = ['true','false','True','False',1,0]
    accepted_values = lambda _type: [value for category in db.search(params.type == _type) for value in category['accepted']]
    quotes = db.get(params.type == 'quotes')['name']

    parser = argparse.ArgumentParser(description='Astrea is a multi-dimensional magnetohydrodynamics simulation written in Python 3. Refer to the README for more information.', 
                                     epilog=f"--- {BColours.ITALIC}{quotes[random.randint(0,len(quotes)-1)]}{BColours.ENDC} ---", 
                                     formatter_class=argparse.RawTextHelpFormatter, 
                                     usage=argparse.SUPPRESS)

    parser.add_argument('-v', '--verbose', dest='verbose', help='switch on verbose description of simulation', action='store_true')
    parser.add_argument('-q', '--quiet', dest='quiet', help='switch off printing to screen', action='store_true')
    parser.add_argument('-w', '--write', dest='write_chkpt', help='switch on checkpoint file saving', action='store_true')
    parser.add_argument('-t', '--test', dest='test', help='run the tests for astrea (convergence, conservation, etc.)', action='store_true')

    parser.add_argument('--config', metavar='', type=str.lower, default=argparse.SUPPRESS, help='configuration to run in the simulation', choices=accepted_values('config'))
    parser.add_argument('--cells', '--grid', dest='cells', metavar='', default=argparse.SUPPRESS, help='number of cells in the grid')
    parser.add_argument('--cfl', metavar='', type=float, default=argparse.SUPPRESS, help='Courant number in the Courant-Friedrichs-Lewy stability condition')
    parser.add_argument('--gamma', metavar='', type=float, default=argparse.SUPPRESS, help='adiabatic index')
    parser.add_argument('--dimensions', type=int, metavar='', default=argparse.SUPPRESS, help='dimensionality of the simulation', choices=db.get(params.type == 'dimensions')['accepted'])
    parser.add_argument('--gravity', metavar='', type=str.lower, default=argparse.SUPPRESS, help='set gravity in the simulation', choices=db.get(params.type == 'gravity')['accepted'])
    parser.add_argument('--units', metavar='', type=str.lower, default=argparse.SUPPRESS, help='set units/scale of the simulation', choices=db.get(params.type == 'units')['accepted'])

    parser.add_argument('--subgrid', metavar='', type=str.lower, default=argparse.SUPPRESS, help='subgrid model used for reconstruction within grid cells', choices=accepted_values('subgrid'))
    parser.add_argument('--time_evo', metavar='', type=str.lower, default=argparse.SUPPRESS, help='time integration method used for temporal evolution', choices=accepted_values('time_evo'))
    parser.add_argument('--solver', metavar='', type=str.lower, default=argparse.SUPPRESS, help='solver method for the Riemann problem', choices=accepted_values('solver'))

    parser.add_argument('--checkpoints', metavar='', type=int, default=argparse.SUPPRESS, help='number of checkpoints in simulation')

    parser.add_argument('--live_plot', metavar='', type=bool_handler, default=argparse.SUPPRESS, help='toggle the live plotting function', choices=bool_choices)
    parser.add_argument('--save_snaps', metavar='', type=bool_handler, default=argparse.SUPPRESS, help='toggle saving snapshots of the simulation', choices=bool_choices)
    parser.add_argument('--save_plots', metavar='', type=bool_handler, default=argparse.SUPPRESS, help='toggle saving quantitative plots of the simulation', choices=bool_choices)
    parser.add_argument('--save_video', metavar='', type=bool_handler, default=argparse.SUPPRESS, help='toggle saving a video of the simulation', choices=bool_choices)
    parser.add_argument('--save_file', metavar='', type=bool_handler, default=argparse.SUPPRESS, help='toggle saving the simulation data file (.hdf5)', choices=bool_choices)
    parser.add_argument('--plot_style', metavar='', type=str.lower, default=argparse.SUPPRESS, help='plot styles (based on matplotlib style sheets)')
    parser.add_argument('--plot_options', metavar='', type=str.lower, default=argparse.SUPPRESS, help='simulation variables to plot')

    parser.add_argument('--file', dest='chkpt_file', metavar='', type=str.lower, default='', help='(absolute) path to astrea checkpoint file')
    parser.add_argument('--tracers', help='switch on tracer particles in the simulation', action='store_true')

    parser.add_argument('--chemistry', help='switch on chemical network in simulation', action='store_true')
    parser.add_argument('--network', metavar='', type=str.lower, default='', help='(absolute) path to chemical network file')
    parser.add_argument('--abundances', metavar='', type=str.lower, default='', help='(absolute) path to (.yml) file for initial abundances of chemical species')

    parser.add_argument('--init', default=argparse.SUPPRESS, help=argparse.SUPPRESS, action='store_true')

    args = parser.parse_args()

    return vars(args)


def filter_variables(config_variables):
    db, params = TinyDB(config_variables['db_path']), Query()
    eps = np.finfo(float).eps
    config_variables['eps'] = eps

    # Check validity of variables; revert to default values if not valid
    for k,v in config_variables.items():
        if k in ['live_plot', 'save_snaps', 'save_plots', 'save_video', 'save_file']:
            if not isinstance(v, bool):
                v = False
        elif k in ['checkpoints', 'dimensions']:
            if not isinstance(v, int):
                v = 1
            if k == 'dimensions' and not (1 <= v <= 3):
                v = 1
        elif k == "cells":
            if isinstance(v, (int, float)):
                v = [int(v)-int(v)%2,] * config_variables['dimensions']
            elif isinstance(v, str):
                try:
                    v = [int(n)-int(n)%2 for n in v.strip('()').replace(' ','').replace('x',',').split(',')]
                    if len(v) <= 1:
                        v *= config_variables['dimensions']
                except Exception:
                    v = [128,] * config_variables['dimensions']
                else:
                    if len(v) > config_variables['dimensions']:
                        v = v[:config_variables['dimensions']]
            elif isinstance(v, list):
                try:
                    v = [int(_)-int(_)%2 for _ in v]
                except Exception:
                    v = [128,] * config_variables['dimensions']
                else:
                    v = v[:config_variables['dimensions']]
            else:
                v = [128,] * config_variables['dimensions']
        elif k in ['gamma', 'cfl']:
            if not isinstance(v, (int, float)):
                if "/" in v:
                    num, dem = v.split('/')
                    v = float(num)/float(dem)
                else:
                    if k == "gamma":
                        v = 1.4
                    elif k == "cfl":
                        v = .5
                    else:
                        v = 1.
            if k == "gamma" and v == 1:
                v += eps
            if k == "cfl":
                if v <= 0:
                    v = eps
        elif k == "gravity":
            if isinstance(v, str):
                if v.lower() not in ['true', '1', 'self', 'ext', 'external']:
                    v = False
            else:
                if not isinstance(v, bool):
                    if isinstance(v, int):
                        if v not in (0,1):
                            v = False
                        else:
                            v = bool(v)
                    else:
                        v = False
        elif k == "plot_options":
            accepted_plot_options, valid, invalid = db.get(params.type == k)['accepted'], [], []
            try:
                if isinstance(v, str):
                    v = v.replace(' ','').replace('-',',').replace('/',',').replace('|',',').split(',')
                for option in v:
                    _option = option.replace(' ','').replace('-','').replace('_','')
                    if _option.lower() not in accepted_plot_options:
                        invalid.append(option)
                    else:
                        valid.append(option)
                v = [i.lower() for i in valid]
                _ = v[0]  # Check for empty list
            except (IndexError, TypeError):
                v = db.get(params.type == 'default')[k]
                print(f"{BColours.WARNING}No valid plot options; reverting to default values..{BColours.ENDC}")
            finally:
                if invalid != []:
                    print(f"{BColours.WARNING}Invalid plot options: {invalid}{BColours.ENDC}")
        else:
            if k in ['config', 'subgrid', 'time_evo', 'solver']:
                if isinstance(v, str):
                    v = v.lower()

                found = False
                for dct in db.search(params.type == k):
                    if v in dct['accepted']:
                        found = True
                        break

                if not found:
                    v = db.get(params.type == 'default')[k]
                    print(f"{BColours.WARNING}{k.upper()} value not valid; reverting back to default value: {v}..{BColours.ENDC}")

        config_variables[k] = v

    return config_variables


class Constants(object):
    def __init__(self, obj, units):
        try:
            for name, value in obj.__dict__.items():
                if not name.startswith("_"):
                    setattr(self, name, value)
        except Exception:
            for name, value in obj.items():
                setattr(self, name, value)

        # Set up scaling for physical units (CGS)
        if units != "code":
            if units == 'custom':
                L0 = 1
                rho0 = 1
                v0 = 1
                length_scale = 1
                length_label = " [pc]"
                time_scale = 1
                time_label = " yr"
            elif units == 'stellar':
                L0 = self.r_sun
                rho0 = self.m_sun/self.au**3
                v0 = self.kms
                length_scale = self.au
                length_label = " [au]"
                time_scale = self.sec_per_year
                time_label = " yr"
            elif units == 'cluster':
                L0 = self.pc
                rho0 = 10 * (self.m_sun/self.pc**3)
                v0 = self.kms
                length_scale = self.pc
                length_label = " [pc]"
                time_scale = self.Myr
                time_label = " Myr"
            elif units == 'galactic':
                L0 = 1e3 * self.pc
                rho0 = 1e11 * (self.m_sun/(1e4 * self.pc**3))
                v0 = 10 * self.kms
                length_scale = 1e3 * self.pc
                length_label = " [kpc]"
                time_scale = self.Myr
                time_label = " Myr"

            m0 = rho0 * L0**3
            if self.mu_0 != 1:
                B0 = v0 * np.sqrt(self.mu_0*rho0)
            else:
                B0 = np.sqrt(4*np.pi*rho0 * v0**2 * L0**3)

            # Scale quantities to plot units
            self.plot_scales = {
                "length":           L0 / length_scale,      # code -> cm -> au/pc/kpc (length_label)
                "time":             (L0/v0) / time_scale,   # code -> s -> s/yr/Myr (time_label)
                "density":          rho0,                   # code -> g/cm3 -> g/cm3
                "velocity":         v0 * 1e-5,              # code -> cm/s -> km/s
                "mass":             m0/self.m_sun,          # code -> g -> M_sun
                "momentum":         rho0 * v0,              # code -> g/(cm2 s) -> g/(cm2 s)
                "pressure":         10 * rho0 * v0**2,      # code -> dyn/cm3 -> Pa
                "energy":           rho0 * v0**2 * L0**3,   # code -> erg -> erg
                "energy density":   rho0 * v0**2,           # code -> erg/cm3 -> erg/cm3
                "Bfield":           1e6 * B0,               # code -> G -> uG
                "divergence":       1e6 * B0/L0,            # code -> G/cm -> uG/cm
                "Mach":             1,                      # unitless
            }

            # Set plot units
            self.scale_labels = {
                "length":           length_label,                                   # cm/au/pc/kpc
                "time":             time_label,                                     # s/yr/Myr
                "density":          r" [$\mathrm{g}/\mathrm{cm}^3$]",               # g/cm3
                "velocity":         r" [$\mathrm{km}/\mathrm{s}$]",                 # km/s
                "mass":             r" [$\mathrm{M}_\odot$]",                       # M_sun
                "momentum":         r" [$\mathrm{g}/(\mathrm{cm}^2 \mathrm{s})$]",  # g/(cm2 s)
                "pressure":         r" [$\mathrm{Pa}$]",                            # Pa
                "energy":           r" [$\mathrm{erg}$]",                           # erg
                "energy density":   r" [$\mathrm{erg}/\mathrm{cm}^3$]",             # erg/cm3
                "Bfield":           r" [$\mu\mathrm{G}$]",                          # uG
                "divergence":       r" [$\mu\mathrm{G}/\mathrm{cm}$]",              # uG/cm
                "Mach":             "",                                             # unitless
            }


class SimulationVariables(object):
    __slots__ = [
        '__dict__',
        'rho', 'vx', 'vy', 'vz', 'pressure', 'Bx', 'By', 'Bz', 'gx', 'gy', 'gz', 'energy', 'vels', 'Bfields', 'momentums', 'gs',
        'config', 'cells', 'cfl', 'gamma', 'gravity', 'self_gravity', 'ext_gravity', 'dimensions', 'subgrid', 'time_evo', 'solver',
        'coordinates', 'shock_pos', 't_end', 'boundary', 'misc', 'init_cond', 'ambient', 'ds',
        'checkpoints', 'live_plot', 'save_snaps', 'save_plots', 'save_video', 'save_file', 'plot_style', 'plot_options',
        'axes', 'magnetic', 'convert', 'roots', 'weights', 'ppm_dissipate', 'higher_order', 'grid_interpolate', 'multidimensional', 'config_category', 'subgrid_category', 'solver_category',
        'seed', 'now', 'elapsed', 'access_key', 'datetime', 'eps', 'home', 'save_path', 'db_path', 'hdf5', 'timesteps', 'print_status',
        'full_set_required', 'write_chkpt', 'chkpt_file', 'quiet', 'verbose', 'test',
        'units', 'constants', 'chemistry', 'network', 'pykrome', 'species', 'abundances', 'tracers', 'nvars',
    ]

    def __init__(self, seed, config_variables, test_variables):
        db, params = TinyDB(config_variables['db_path']), Query()

        # Declare physical variables and their index in the array: [density, vx/px, vy/py, vz/pz, pressure/energy, Bx, By, Bz, source terms]
        self.nvars = 8
        self.rho, self.vx, self.vy, self.vz, self.pressure, self.Bx, self.By, self.Bz = range(self.nvars)
        self.vels, self.Bfields = slice(1,4), slice(5,8)
        self.energy, self.momentums = self.pressure, self.vels

        # Parse configuration variables into the class
        for key in config_variables:
            setattr(self, key, config_variables[key])

        # Parse tests variables into the class
        for key in test_variables:
            setattr(self, key, test_variables[key])

        # Parse additional variables into the class
        self.seed = int(seed)
        self.now = None
        self.elapsed = None
        self.access_key = None
        self.timesteps = 0

        self.constants = Constants(const, self.units)

        # 5th-order Gauss-Legendre quadrature with interval [0,1] for OS solver
        roots, weights = np.array(list(np.polynomial.legendre.leggauss(5)))/2
        self.roots = roots + .5
        self.weights = weights

        self.config_category = db.get(params.accepted.any([self.config]))['category']
        self.subgrid_category = db.get(params.accepted.any([self.subgrid]))['category']
        self.solver_category = db.get(params.accepted.any([self.solver]))['category']

        # Higher-order method options
        self.higher_order = self.grid_interpolate = False
        if self.subgrid_category in ["ppm", "weno"]:
            self.higher_order = self.grid_interpolate = True

            # WENO-Z can use point representation
            if "z" in self.subgrid:
                self.grid_interpolate = False

            # PPM-specific options
            if self.subgrid_category == "ppm":
                self.ppm_author = os.getenv("PPM_AUTHOR", "MC:2011")  # [McCorquodale & Colella, 2011 (MC:2011); Colella et al., 2011 (C+:2011); Peterson & Hammett, 2008 (PH:2008)]
                self.ppm_dissipate = os.getenv("PPM_DISSIPATE", False)

        # CT-specific options
        self.ct_dissipative = os.getenv("CT_DISSIPATIVE", False)

        # Permutations for axes
        self.multidimensional = self.dimensions >= 2
        self.axes = np.array(range(self.dimensions))

        # Gravity set-up
        self.self_gravity = self.ext_gravity = False
        if self.gravity:
            if self.gravity == "self":
                self.self_gravity = True
            elif self.gravity in ("ext", "external"):
                self.ext_gravity = True
            else:
                self.self_gravity = self.ext_gravity = True
        self.gravity = True if (self.self_gravity or self.ext_gravity) else False

        if self.ext_gravity:
            self.nvars += 3
            self.gx, self.gy, self.gz = range(8,11)
            self.gs = slice(8,11)

        # Chemistry network set-up
        if self.chemistry:
            if not self.network:
                krome_path = os.path.join(self.home, 'physics', 'krome')
            else:
                try:
                    krome_path = [os.path.join(root, dirname) for root, dirs, _ in os.walk(self.home) for dirname in dirs if 'krome' in os.path.join(root, dirname)][0]
                except IndexError:
                    print(f"{BColours.WARNING}Chemistry switched on but krome folder cannot be found. Switching off chemistry..{BColours.ENDC}")
                    krome_path = None

            paths = [self.home, krome_path, self.network]
            options = [
                '-iRHS',
                '-noRecCheck',
                '-coolFile=data/coolZ.dat',
                '-cooling=ATOMIC,H2,DUST,Z,CI,OI,CII',
                '-heating=COMPRESS,PHOTO,CHEM,PHOTODUST'
            ]
            self.pykrome, self.species, self.useX = krome_funcs.build_krome(paths, options)

            if self.pykrome == None or self.species == None:
                print(f"{BColours.WARNING}krome built but cannot be accessed. Switching off chemistry..{BColours.ENDC}")
                self.chemistry = False

        # Printer functions
        if self.verbose:
            self.print_status = generic.print_verbose
        else:
            self.print_status = generic.print_simple

        # Media options
        if self.test:
            self.save_plots = True
            if (self.live_plot or self.save_snaps or self.save_video):
                self.live_plot = self.save_snaps = self.save_video = False

        if (self.live_plot or self.save_plots or self.save_video) and self.dimensions > 2:
            print(f"{BColours.WARNING}Unable to display 3d simulation results with astrea..{BColours.ENDC}")
            self.live_plot = self.save_plots = self.save_video = False

        if (self.save_snaps or self.save_plots or self.save_video) and self.live_plot:
            print(f"{BColours.WARNING}Live plot can only be switched on when NOT saving media files because live_plot interferes with matplotlib.savefig..{BColours.ENDC}")
            self.live_plot = False

        if self.save_snaps or self.save_plots or self.save_video or self.save_file:
            self.save_path = ''

        self.beautify_1d_plots = os.getenv("BEAUTIFY_1D_PLOTS", False)
        self.save_as_pdf = os.getenv("SAVE_AS_PDF", False)

        # Set up boxes for plotting
        self.box_volume = np.prod([np.diff(_) for _ in self.coordinates.values()])
        if self.units != "code":
            try:
                semi = self.misc['mode'].lower().startswith(('o','q'))
            except Exception:
                semi = False

            if semi:
                full_box = self.constants.plot_scales['length']
                self.box_lengths = {ax: [start_pos, full_box*end_pos] for ax, (start_pos, end_pos) in self.coordinates.items()}
            else:
                half_box = self.constants.plot_scales['length']/2
                centres = {ax: np.average(axis_coord) for ax, axis_coord in self.coordinates.items()}
                self.box_lengths = {ax: [half_box*(start_pos-centres[ax]), half_box*(end_pos-centres[ax])] for ax, (start_pos, end_pos) in self.coordinates.items()}
        else:
            self.box_lengths = self.coordinates

        self.full_set_required = True if (self.save_plots or self.save_video or self.save_file) else False


# Write grid to HDF5 checkpoint files
def write_chkpt_file(grid, t, idx, sim_variables):
    if sim_variables.test:
        file_name = f"astrea_hdf5_{sim_variables.cells}_chkpt_{sim_variables.timesteps:05}_{t:.6f}".replace('.','')
    else:
        file_name = f"astrea_hdf5_chkpt_{sim_variables.timesteps:05}_{t:.6f}".replace('.','')

    with h5py.File(f"{sim_variables.save_path}/{file_name}", "w") as f:
        f.attrs['datetime'] = sim_variables.access_key
        f.attrs['seed'] = sim_variables.seed
        f.attrs['code'] = 'astrea'
        f.attrs['time'] = float(t)
        f.attrs['idx'] = int(idx)

        f.attrs['config'] = sim_variables.config
        f.attrs['cells'] = sim_variables.cells
        f.attrs['cfl'] = sim_variables.cfl
        f.attrs['gamma'] = sim_variables.gamma
        f.attrs['dimensions'] = sim_variables.dimensions
        f.attrs['eps'] = sim_variables.eps
        f.attrs['subgrid'] = sim_variables.subgrid
        f.attrs['time_evo'] = sim_variables.time_evo
        f.attrs['solver'] = sim_variables.solver
        f.attrs['magnetic'] = sim_variables.magnetic
        f.attrs['units'] = sim_variables.units
        f.attrs['self_gravity'] = sim_variables.self_gravity
        f.attrs['ext_gravity'] = sim_variables.ext_gravity
        f.attrs['boundary'] = sim_variables.boundary
        f.attrs['aspect_ratio'] = sim_variables.aspect_ratio
        f.attrs['coordinates'] = tuple(sim_variables.coordinates.values())
        f.attrs['box_lengths'] = tuple(sim_variables.box_lengths.values())

        f.create_dataset('grid', data=grid, compression="gzip", compression_opts=9)


# Load HDF5 checkpoint files
def load_chkpt_file(config_variables, file):
    with h5py.File(file, "r") as f:
        try:
            code = f.attrs['code']
        except Exception as e:
            print(f"{BColours.WARNING}Checkpoint file not created by astrea..{BColours.ENDC}")
            return None
        else:
            if code != 'astrea':
                print(f"{BColours.WARNING}Checkpoint file not created by astrea..{BColours.ENDC}")
                return None
            else:
                seed = int(f.attrs['seed'])
                time = float(f.attrs['time'])
                idx = int(f.attrs['idx'])
                grid = f['grid'][:]

                config_variables['config'] = f.attrs['config']
                config_variables['cells'] = f.attrs['cells']
                config_variables['cfl'] = float(f.attrs['cfl'])
                config_variables['gamma'] = float(f.attrs['gamma'])
                config_variables['dimensions'] = int(f.attrs['dimensions'])
                config_variables['eps'] = f.attrs['eps']
                config_variables['subgrid'] = f.attrs['subgrid']
                config_variables['time_evo'] = f.attrs['time_evo']
                config_variables['solver'] = f.attrs['solver']
                config_variables['magnetic'] = f.attrs['magnetic']
                config_variables['units'] = f.attrs['units']
                config_variables['self_gravity'] = f.attrs['self_gravity']
                config_variables['ext_gravity'] = f.attrs['ext_gravity']
                config_variables['aspect_ratio'] = f.attrs['aspect_ratio']
                config_variables['boundary'] = f.attrs['boundary']
                config_variables['coordinates'] = {ax:axis_coord for ax, axis_coord in enumerate(f.attrs['coordinates'])}
                config_variables['box_lengths'] = {ax:start_end for ax, start_end in enumerate(f.attrs['box_lengths'])}

                return seed, config_variables, {'time':time, 'idx':idx, 'grid':grid}