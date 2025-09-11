import os
import random
import argparse
import subprocess

import yaml
import h5py
import numpy as np
from tinydb import TinyDB, Query

from functions import fv, generic
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
    test_variables = tests.generate_test_conditions(config_variables['config'], config_variables['cells'], config_variables['gamma'])
    sim_variables = SimulationVariables(1, config_variables, test_variables)
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

    parser.add_argument('-v', '--verbose', dest='verbose', default=argparse.SUPPRESS, help='verbose description of simulation', action='store_true')
    parser.add_argument('-q', '--quiet', dest='quiet', default=argparse.SUPPRESS, help='switch off printing to screen', action='store_true')
    parser.add_argument('-w', '--write', dest='write_chkpt', default=argparse.SUPPRESS, help='toggle saving checkpoint files', action='store_true')

    parser.add_argument('--config', metavar='', type=str.lower, default=argparse.SUPPRESS, help='configuration to run in the simulation', choices=accepted_values('config'))
    parser.add_argument('--grid', '--cells', dest='cells', metavar='', default=argparse.SUPPRESS, help='number of cells in the grid')
    parser.add_argument('--cfl', metavar='', type=float, default=argparse.SUPPRESS, help='Courant number in the Courant-Friedrichs-Lewy stability condition')
    parser.add_argument('--gamma', metavar='', type=float, default=argparse.SUPPRESS, help='adiabatic index')
    parser.add_argument('--permeability', metavar='', type=float, default=argparse.SUPPRESS, help='magnetic permeability')
    parser.add_argument('--dimension', '--dim', dest='dimension', type=int, metavar='', default=argparse.SUPPRESS, help='dimensionality of the simulation', choices=db.get(params.type == 'dimension')['accepted'])

    parser.add_argument('--subgrid', metavar='', type=str.lower, default=argparse.SUPPRESS, help='subgrid model used for reconstruction within grid cells', choices=accepted_values('subgrid'))
    parser.add_argument('--time_evo', metavar='', type=str.lower, default=argparse.SUPPRESS, help='time integration method used for temporal evolution', choices=accepted_values('time_evo'))
    parser.add_argument('--solver', metavar='', type=str.lower, default=argparse.SUPPRESS, help='solver method for the Riemann problem', choices=accepted_values('solver'))

    parser.add_argument('--run_type', metavar='', type=str.lower, default=argparse.SUPPRESS, help='run a single run or multiple runs for each simulation', choices=db.get(params.type == 'run_type')['accepted'])
    parser.add_argument('--checkpoints', metavar='', type=int, default=argparse.SUPPRESS, help='number of checkpoints in simulation')

    parser.add_argument('--live_plot', '--live', dest='live_plot', metavar='', type=bool_handler, default=argparse.SUPPRESS, help='toggle the live plotting function', choices=bool_choices)
    parser.add_argument('--save_snaps', metavar='', type=bool_handler, default=argparse.SUPPRESS, help='toggle saving snapshots of the simulation', choices=bool_choices)
    parser.add_argument('--save_plots', metavar='', type=bool_handler, default=argparse.SUPPRESS, help='toggle saving other plots of the simulation, including quantities, conservation, total variation, etc.', choices=bool_choices)
    parser.add_argument('--save_video', metavar='', type=bool_handler, default=argparse.SUPPRESS, help='toggle saving a video of the simulation', choices=bool_choices)
    parser.add_argument('--save_file', metavar='', type=bool_handler, default=argparse.SUPPRESS, help='toggle saving the entire simulation data file (.hdf5)', choices=bool_choices)
    parser.add_argument('--plot_style', metavar='', type=str.lower, default=argparse.SUPPRESS, help='plot styles (based on matplotlib style sheets)')
    parser.add_argument('--plot_options', metavar='', type=str.lower, default=argparse.SUPPRESS, help='simulation variables to plot')

    parser.add_argument('--file', dest='chkpt_file', metavar='', type=str.lower, default=argparse.SUPPRESS, help='load astrea checkpoint file')
    parser.add_argument('--chemistry', default=argparse.SUPPRESS, help='switch on chemistry for simulation', action='store_true')

    parser.add_argument('-t', '--test', dest='test', default=argparse.SUPPRESS, help=argparse.SUPPRESS, action='store_true')

    args = parser.parse_args()

    return vars(args)


def parse_cli_variables(config_variables, cli_variables):
    db, params = TinyDB(config_variables['db_path']), Query()

    skip_cases = ['home', 'db_path', 'plot_style']

    # Replace the relevant configuration variables with the CLI variables
    for k,v in cli_variables.items():
        if k in config_variables:
            if k == 'plot_options':
                v = v.replace('-',' ').replace('/',',').replace('|',',')
            config_variables[k] = v

    # Optional CLI cases; set to False if not given
    optional_cli = ['verbose', 'quiet', 'write_chkpt', 'chkpt_file', 'chemistry']
    for _var in optional_cli:
        try:
            config_variables[_var] = cli_variables[_var]
        except KeyError:
            config_variables[_var] = False

    # Check validity of variables; revert to default values if not valid
    for k,v in config_variables.items():
        if k in ['live_plot', 'save_snaps', 'save_plots', 'save_video', 'save_file']:
            if not isinstance(v, bool):
                v = False
        elif k in ['checkpoints', 'dimension']:
            if not isinstance(v, int):
                v = 1
            if k == 'dimension' and not (1 <= v <= 3):
                v = 1
        elif k == "cells":
            if isinstance(v, (int, float)):
                v = [int(v)-int(v)%2,] * config_variables['dimension']
            elif isinstance(v, str):
                try:
                    v = [int(n)-int(n)%2 for n in v.strip('()').replace(' ','').replace('x',',').split(',')]
                    if len(v) <= 1:
                        v *= config_variables['dimension']
                except Exception:
                    v = [128,] * config_variables['dimension']
                else:
                    if len(v) > config_variables['dimension']:
                        v = v[:config_variables['dimension']]
            elif isinstance(v, list):
                try:
                    v = [int(_)-int(_)%2 for _ in v]
                except Exception:
                    v = [128,] * config_variables['dimension']
                else:
                    v = v[:config_variables['dimension']]
            else:
                v = [128,] * config_variables['dimension']
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
                _ = v[0]  # Check for empty list
            except (IndexError, TypeError):
                v = db.get(params.type == 'default')[k]
                print(f"{BColours.WARNING}No valid plot options; reverting to default values..{BColours.ENDC}")
            finally:
                if invalid != []:
                    print(f"{BColours.WARNING}Invalid plot options: {invalid}{BColours.ENDC}")
        elif k in optional_cli or k in skip_cases:
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
        'axis_coord', 'shock_pos', 't_end', 'boundary', 'misc', 'initial_left', 'initial_right', 'ds',
        'run_type', 'checkpoints', 'live_plot', 'save_snaps', 'save_plots', 'save_video', 'save_file', 'plot_style', 'plot_options',
        'axes', 'magnetic', 'convert', 'roots', 'weights', 'ppm_dissipate', 'higher_order', 'multidimensional', 'config_category', 'subgrid_category', 'solver_category',
        'seed', 'now', 'elapsed', 'access_key', 'datetime', 'home', 'save_path', 'db_path', 'timesteps', 'print_status',
        'full_set_required', 'write_chkpt', 'chkpt_file', 'quiet', 'verbose', 'chemistry', 'test',
    ]

    def __init__(self, seed, config_variables, test_variables):
        db, params = TinyDB(config_variables['db_path']), Query()

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
        self.seed = int(seed)
        self.now = None
        self.elapsed = None
        self.access_key = None
        self.timesteps = 0

        # 5th-order Gauss-Legendre quadrature with interval [0,1] for OS solver
        roots, weights = np.array(list(np.polynomial.legendre.leggauss(5)))/2
        self.roots = roots + .5
        self.weights = weights

        self.config_category = db.get(params.accepted.any([self.config]))['category']
        self.subgrid_category = db.get(params.accepted.any([self.subgrid]))['category']
        self.solver_category = db.get(params.accepted.any([self.solver]))['category']

        self.magnetic = self.initial_left[self.Bfields].any() or self.initial_right[self.Bfields].any()
        self.convert = fv.point_convert
        self.higher_order = False
        self.ppm_dissipate = False

        # Higher-order conversion functions
        if self.subgrid_category in ["weno", "ppm"]:
            self.convert = fv.high_order_convert
            self.higher_order = True

        # Permutations for axes
        self.multidimensional = self.dimension >= 2
        self.axes = np.array(range(self.dimension))

        if self.verbose:
            self.print_status = generic.print_verbose
        else:
            self.print_status = generic.print_simple

        # Exclusion cases
        if self.solver in db.get(params.type == 'solver' and params.category == 'hll')['accepted']:
            if (self.solver_category == "hll" and self.solver.endswith('c')) and self.magnetic:
                print(f"{BColours.WARNING}HLLC solver does not work with magnetic fields present..{BColours.ENDC}")
                self.solver = db.get(params.type == 'default')['solver']

        if self.chemistry:
            krome_path = [os.path.join(root, dirname) for root, dirs, _ in os.walk(self.home) for dirname in dirs if 'krome' in os.path.join(root, dirname)][0]
            if not krome_path:
                print(f"{BColours.WARNING}Chemistry switched on but krome folder cannot be found. Switching off chemistry..{BColours.ENDC}")
                self.chemistry = False
            else:
                print(f"Chemistry switched on. Follow the prompts in krome for setup :\n")
                os.chdir(krome_path)
                #subprocess.run(["./krome", "-test=hello"])
                os.chdir(self.home)

        # Media options
        if (self.live_plot or self.save_plots or self.save_video) and self.dimension > 2:
            print(f"{BColours.WARNING}Unable to display 3d simulation results with astrea..{BColours.ENDC}")
            self.live_plot = self.save_plots = self.save_video = False

        if self.run_type.startswith('m'):
            if self.save_video:
                print(f"{BColours.WARNING}Videos can only be saved for single simulation runs..{BColours.ENDC}")
                self.save_video = False
            if self.live_plot:
                print(f"{BColours.WARNING}Live plots can only be switched on for single simulation runs..{BColours.ENDC}")
                self.live_plot = False
            if self.save_snaps:
                print(f"{BColours.WARNING}Saving snapshots can only be switched on for single simulation runs..{BColours.ENDC}")
                self.save_snaps = False
        else:
            if (self.save_snaps or self.save_plots or self.save_video) and self.live_plot:
                print(f"{BColours.WARNING}Live plot can only be switched on when NOT saving media files because live plot interferes with matplotlib.savefig..{BColours.ENDC}")
                self.live_plot = False
            if self.save_snaps or self.save_plots or self.save_video or self.save_file:
                self.save_path = ''

        self.full_set_required = True if (self.save_plots or self.save_video or self.save_file) else False


# Write grid to HDF5 checkpoint files
def write_chkpt_file(grid, t, idx, sim_variables):
    if sim_variables.run_type.startswith('m'):
        file_name = f"astrea_hdf5_{sim_variables.cells}_chk_{sim_variables.timesteps:05}"
    else:
        file_name = f"astrea_hdf5_chk_{sim_variables.timesteps:05}"

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
        f.attrs['permeability'] = sim_variables.permeability
        f.attrs['dimension'] = sim_variables.dimension
        f.attrs['precision'] = sim_variables.precision
        f.attrs['subgrid'] = sim_variables.subgrid
        f.attrs['time_evo'] = sim_variables.time_evo
        f.attrs['solver'] = sim_variables.solver
        f.attrs['axis_coord'] = sim_variables.axis_coord

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
                config_variables['permeability'] = float(f.attrs['permeability'])
                config_variables['dimension'] = int(f.attrs['dimension'])
                config_variables['precision'] = f.attrs['precision']
                config_variables['subgrid'] = f.attrs['subgrid']
                config_variables['time_evo'] = f.attrs['time_evo']
                config_variables['solver'] = f.attrs['solver']
                config_variables['run_type'] = f.attrs['single']

                return seed, config_variables, {'time':time, 'idx':idx, 'grid':grid}