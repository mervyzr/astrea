import random
import argparse

import yaml
import h5py
import numpy as np
from tinydb import TinyDB, Query

from functions import fv
from functions.generic import BColours
from static import tests

##############################################################################
# I/O functions for simulation
##############################################################################

# Make simulation variables; most functions accept sim_variables with all the options included,
# so it might be useful to have a function auto-generate it when needed
def make_sim_variables():
    with open('parameters.yml', "r") as _f:
        config_variables = yaml.safe_load(_f)
    config_variables = parse_cli_variables(config_variables, {})
    test_variables = tests.generate_test_conditions(config_variables['config'], config_variables['cells'])
    sim_variables = SimulationVariables(1, config_variables, test_variables)
    return sim_variables


# CLI arguments handler; updates the simulation variables (which is a dict) and checks for any invalid values
def handle_CLI(db_path):

    def bool_handler(value):
        return (value.lower() == 'true' or value.lower() == '1')

    db = TinyDB(db_path)
    params = Query()

    bool_choices = ['true','false','True','False',1,0]
    accepted_values = lambda _type: [value for category in db.search(params.type == _type) for value in category['accepted']]
    quotes = db.get(params.type == 'quotes')['name']

    parser = argparse.ArgumentParser(description='Run the astrea simulation.\n\nastrea is a 2D magnetohydrodynamics simulation written in Python 3. Refer to the README for more information.', 
                                     epilog=f"--- {BColours.ITALIC}{quotes[random.randint(0,len(quotes)-1)]}{BColours.ENDC} ---", 
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--config', metavar='', type=str.lower, default=argparse.SUPPRESS, help='configuration to run in the simulation', choices=accepted_values('config'))
    parser.add_argument('--grid', '--cells', '--cell', '--N', '--n', dest='cells', metavar='', default=argparse.SUPPRESS, help='number of cells in the grid')
    parser.add_argument('--cfl', metavar='', type=float, default=argparse.SUPPRESS, help='courant number in the Courant-Friedrichs-Lewy stability condition')
    parser.add_argument('--gamma', metavar='', type=float, default=argparse.SUPPRESS, help='adiabatic index')
    parser.add_argument('--permeability', metavar='', type=float, default=argparse.SUPPRESS, help='magnetic permeability')
    parser.add_argument('--dimension', '--dim', dest='dimension', type=int, metavar='', default=argparse.SUPPRESS, help='dimension of the simulation', choices=db.get(params.type == 'dimension')['accepted'])

    parser.add_argument('--subgrid', metavar='', type=str.lower, default=argparse.SUPPRESS, help='subgrid model used in the reconstruction of the grid', choices=accepted_values('subgrid'))
    parser.add_argument('--time_evo', '--time-evo', dest='time_evo', metavar='', type=str.lower, default=argparse.SUPPRESS, help='sime-stepping algorithm used in the update step of the simulation', choices=accepted_values('time_evo'))
    parser.add_argument('--solver', metavar='', type=str.lower, default=argparse.SUPPRESS, help='solver used for the Riemann problem', choices=accepted_values('solver'))

    parser.add_argument('--run_type', metavar='', type=str.lower, default=argparse.SUPPRESS, help='run a single run or multiple runs for each simulation', choices=db.get(params.type == 'run_type')['accepted'])
    parser.add_argument('--checkpoints', '--chkpts', dest='checkpoints', metavar='', type=int, default=argparse.SUPPRESS, help='number of checkpoints in simulation')

    parser.add_argument('--plot_options', '--plot-options', dest='plot_options', metavar='', type=str.lower, default=argparse.SUPPRESS, help='simulation variables to plot')
    parser.add_argument('--live_plot', '--live-plot', '--live', dest='live_plot', metavar='', type=bool_handler, default=argparse.SUPPRESS, help='toggle the live plotting function', choices=bool_choices)
    parser.add_argument('--save_plots', '--save-plots', dest='save_plots', metavar='', type=bool_handler, default=argparse.SUPPRESS, help='toggle saving snapshots of the simulation', choices=bool_choices)
    parser.add_argument('--full_plots', '--full-plots', dest='full_plots', metavar='', type=bool_handler, default=argparse.SUPPRESS, help='toggle saving full plots of the simulation, including quantities, conservation, total variation, etc.', choices=bool_choices)
    parser.add_argument('--save_video', '--save-video', dest='save_video', metavar='', type=bool_handler, default=argparse.SUPPRESS, help='toggle saving a video of the simulation', choices=bool_choices)
    parser.add_argument('--save_file', '--save-file', dest='save_file', metavar='', type=bool_handler, default=argparse.SUPPRESS, help='toggle saving the entire simulation data file (.hdf5)', choices=bool_choices)
    parser.add_argument('--write_chkpt', '--write-chkpt', '--write_checkpoint', '--write-checkpoint', dest='write_chkpt', metavar='', type=bool_handler, default=argparse.SUPPRESS, help='toggle saving checkpoint files', choices=bool_choices)

    parser.add_argument('--debug', '--DEBUG', dest='debug', help='toggle for more detailed description of errors/bugs', action='store_true')
    parser.add_argument('--quiet', '-q', dest='quiet', help='toggle printing to screen', action='store_true')
    parser.add_argument('--test', default=argparse.SUPPRESS, help=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--file', '--chkpt', '--checkpoint', dest='chkpt_file', metavar='', type=str.lower, default=argparse.SUPPRESS, help='load an astrea checkpoint file')

    args = parser.parse_args()

    return vars(args), args.debug


def parse_cli_variables(_config_variables, _cli_variables, _db_path):
    db = TinyDB(_db_path)
    params = Query()
    temp_dct = {}

    # Remove nested configuration dictionary
    for parameters in _config_variables.values():
        for k,v in parameters.items():
            temp_dct[k] = v

    # Replace the relevant configuration variables with the CLI variables
    for k,v in _cli_variables.items():
        if k in temp_dct:
            if k == 'plot_options':
                v = v.replace('-',' ').replace('/',',').replace('|',',')
            temp_dct[k] = v

    try:
        temp_dct['quiet'] = _cli_variables["quiet"]
    except KeyError:
        temp_dct['quiet'] = False

    # Check validity of variables; revert to default values if not valid
    config_variables = {}
    for k,v in temp_dct.items():
        if k in ['live_plot', 'save_plots', 'full_plots', 'save_video', 'save_file', 'write_chkpt']:
            if not isinstance(v, bool):
                v = False
        elif k in ['checkpoints', 'dimension']:
            if not isinstance(v, int):
                v = 1
            if k == 'dimension' and not (0 < v < 3):
                v = 1
        elif k == "cells":
            if isinstance(v, (int, float)):
                v = [int(v)-int(v)%2,] * temp_dct['dimension']
            elif isinstance(v, str):
                try:
                    v = [int(n)-int(n)%2 for n in v.strip('()').replace(' ','').replace('x',',').split(',')]
                    if len(v) < 2:
                        v *= temp_dct['dimension']
                except Exception:
                    v = [128,] * temp_dct['dimension']
                else:
                    if len(v) > temp_dct['dimension']:
                        v = v[:temp_dct['dimension']]
            elif isinstance(v, list):
                try:
                    v = [int(_)-int(_)%2 for _ in v]
                except Exception:
                    v = [128,] * temp_dct['dimension']
                else:
                    v = v[:temp_dct['dimension']]
            else:
                v = [128,] * temp_dct['dimension']
        elif k in ['gamma', 'cfl', 'permeability']:
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
                v += 1e-9
            if k == "cfl":
                if v <= 0:
                    v = 1e-9
                elif v > 1:
                    v = 1
            if k == "permeability":
                if v < 1:
                    v = 1.
        elif k == "plot_options":
            accepted_plot_options, invalid = db.get(params.type == k)['accepted'], []
            try:
                if isinstance(v, str):
                    v = v.replace(' ','').replace('-',',').replace('/',',').replace('|',',').split(',')
                for option in v:
                    option = option.replace(' ','').replace('-','')
                    if option.lower() not in accepted_plot_options:
                        invalid.append(option)
                        v.remove(option)
                v = [i.lower() for i in v]
                _ = v[0]
            except (IndexError, TypeError):
                v = db.get(params.type == 'default')[k]
                print(f"{BColours.WARNING}No valid plot options; reverting to default values..{BColours.ENDC}")
            finally:
                if invalid != []:
                    print(f"{BColours.WARNING}Invalid plot options: {invalid}{BColours.ENDC}")
        elif k == 'quiet':
            pass
        else:
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


class SimulationVariables(object):
    __slots__ = [
        '__dict__',
        'rho', 'vx', 'vy', 'vz', 'pressure', 'Bx', 'By', 'Bz', 'energy', 'vels', 'Bfields', 'momentums',
        'config', 'cells', 'cfl', 'gamma', 'permeability', 'dimension', 'precision', 'subgrid', 'time_evo', 'solver',
        'seed', 'now', 'elapsed', 'access_key', 'datetime', 'save_path', 'timesteps',
        'permeability', 'magnetic', 'roots', 'weights', 'axes', 'ppm_dissipate',
        'config_category', 'solver_category', 'convert', 'higher_order',
        'x_axis', 'y_axis', 'shock_pos', 't_end', 'boundary', 'misc', 'initial_left', 'initial_right', 'ds',
        'run_type', 'live_plot', 'save_plots', 'full_plots', 'save_video', 'save_file', 'plot_options', 'plot_style', 'beautify',
        'checkpoints', 'full_set_required', 'write_chkpt', 'quiet',
    ]

    def __init__(self, seed, config_variables, test_variables, db_path):
        db = TinyDB(db_path)
        params = Query()

        # Declare physical variables and their index in the array: [density, vx/px, vy/py, vz/pz, pressure/energy, Bx, By, Bz]
        self.rho, self.vx, self.vy, self.vz, self.pressure, self.Bx, self.By, self.Bz = range(8)
        self.vels, self.Bfields = slice(1,4), slice(5,8)
        self.energy, self.momentums = self.pressure, self.vels

        # Parse configuration variables into the class
        for key in config_variables:
            setattr(self, key, config_variables[key])

        # Parse test variables into the class
        for key in test_variables:
            setattr(self, key, test_variables[key])

        # Parse additional variables into the class
        # 5th-order Gauss-Legendre quadrature with interval [0,1] for OS solver
        roots, weights = np.array(list(np.polynomial.legendre.leggauss(5)))/2

        self.seed = int(seed)
        self.now = None
        self.elapsed = None
        self.access_key = None
        self.timesteps = 0

        self.plot_style = None
        self.permeability = 1.
        self.roots = roots + .5
        self.weights = weights

        self.config_category = db.get(params.accepted.any([self.config]))['category']
        self.solver_category = db.get(params.accepted.any([self.solver]))['category']
        self.magnetic = self.initial_left[self.Bfields].any() or self.initial_right[self.Bfields].any()

        self.convert = fv.point_convert
        self.higher_order = False
        self.ppm_dissipate = False

        # Higher-order conversion functions
        if self.subgrid.startswith("w") or self.subgrid in ["ppm", "parabolic", "p"]:
            self.convert = fv.high_order_convert
            self.higher_order = True

        # Permutations for axes
        if '2D' in self.config_category and self.dimension != 2:
            self.dimension = 2
            if len(self.cells) != 2:
                self.cells *= 2
        self.axes = tuple(range(self.dimension))

        # Exclusion cases
        if self.solver in db.get(params.type == 'solver' and params.category == 'hll')['accepted']:
            if (self.solver_category == "hll" and self.solver.endswith('c')) and self.config in db.get(params.type == 'config' and params.category == 'magnetic')['accepted']:
                print(f"{BColours.WARNING}HLLC solver does not work with magnetic fields present..{BColours.ENDC}")
                self.solver = db.get(params.type == 'default')['solver']

        # Media options
        if self.run_type.startswith('m'):
            if self.save_video:
                print(f"{BColours.WARNING}Videos can only be saved for single simulation runs..{BColours.ENDC}")
                self.save_video = False
            if self.live_plot:
                print(f"{BColours.WARNING}Live plots can only be switched on for single simulation runs..{BColours.ENDC}")
                self.live_plot = False
            if self.save_plots:
                print(f"{BColours.WARNING}Saving snapshots can only be switched on for single simulation runs..{BColours.ENDC}")
                self.save_plots = False
        else:
            if (self.save_plots or self.full_plots or self.save_video) and (self.live_plot):
                print(f"{BColours.WARNING}Live plot can only be switched on when NOT saving media files because live plot interferes with matplotlib.savefig..{BColours.ENDC}")
                self.live_plot = False
            if self.save_plots or self.full_plots or self.save_video or self.save_file:
                self.save_path = ''

        self.full_set_required = True if (self.full_plots or self.save_video or self.save_file) else False


# Write grid to HDF5 checkpoint files
def write_chkpt_file(grid, t, sim_variables):
    if sim_variables.run_type.startswith('m'):
        file_name = f"astrea_hdf5_{sim_variables.cells}_chk_{sim_variables.timesteps}"
    else:
        file_name = f"astrea_hdf5_chk_{sim_variables.timesteps:05}"

    with h5py.File(f"{sim_variables.save_path}/{file_name}", "w") as f:
        f.attrs['datetime'] = sim_variables.access_key
        f.attrs['seed'] = sim_variables.seed
        f.attrs['code'] = 'astrea'

        f.attrs['time'] = float(t)
        f.attrs['t_end'] = sim_variables.t_end
        f.attrs['config'] = sim_variables.config
        f.attrs['cells'] = sim_variables.cells
        f.attrs['cfl'] = sim_variables.cfl
        f.attrs['gamma'] = sim_variables.gamma
        f.attrs['precision'] = sim_variables.precision
        f.attrs['permeability'] = sim_variables.permeability
        f.attrs['dimension'] = sim_variables.dimension
        f.attrs['subgrid'] = sim_variables.subgrid
        f.attrs['time_evo'] = sim_variables.time_evo
        f.attrs['solver'] = sim_variables.solver

        f.create_dataset('grid', data=grid, compression="gzip", compression_opts=9)