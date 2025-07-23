import os
import random
import argparse
import itertools
from datetime import timedelta

import yaml
import numpy as np
from tinydb import TinyDB, Query

from functions import fv
from static import tests

##############################################################################
# Generic functions not specific to the finite volume method
##############################################################################

CURRENTDIR = os.getcwd()
DB = TinyDB(f"{CURRENTDIR}/static/.db.json")
PARAMS, ACCEPTED = Query(), Query()


# Colours for printing to terminal
class BColours:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'


# Make simulation variables; most functions accept sim_variables with all the options included,
# so it might be useful to have a function auto-generate it when needed
def make_sim_variables():
    with open('parameters.yml', "r") as _f:
        config_variables = yaml.safe_load(_f)
    config_variables = parse_cli_variables(config_variables, {})
    test_variables = tests.generate_test_conditions(config_variables['config'], config_variables['cells'])
    sim_variables = SimulationVariables(1, config_variables, test_variables)
    return sim_variables


# Print progress status to Terminal
def print_progress(t, sim_variables):
    _seed = f"{BColours.OKBLUE}{sim_variables.seed}{BColours.ENDC}"
    _config = f"{BColours.OKCYAN}{sim_variables.config.upper()}{BColours.ENDC}"
    _subgrid = f"{BColours.OKCYAN}{sim_variables.subgrid.upper()}{BColours.ENDC}"
    _timestep = f"{BColours.OKCYAN}{sim_variables.timestep.upper()}{BColours.ENDC}"
    _solver = f"{BColours.OKCYAN}{sim_variables.solver.upper()}{BColours.ENDC}"
    _cfl = f"{BColours.OKCYAN}{sim_variables.cfl}{BColours.ENDC}"
    _dimension = f"{BColours.OKCYAN}{BColours.BOLD}({sim_variables.dimension}D){BColours.ENDC}"
    _instance = f"{BColours.WARNING}{'%.6f'%t} / {'%.2f'%sim_variables.t_end}{BColours.ENDC}"

    if sim_variables.dimension != 1:
        _cells = f"{BColours.OKCYAN}{sim_variables.cells}^{sim_variables.dimension}{BColours.ENDC}"
    else:
        _cells = f"{BColours.OKCYAN}{sim_variables.cells}{BColours.ENDC}"

    print(f"[{sim_variables.now.strftime('%Y-%m-%d %H:%M:%S')} | {_seed}] {_dimension} CONFIG={_config}, CELLS={_cells}, CFL={_cfl}, SUBGRID={_subgrid}, SOLVER={_solver}, TIMESTEP={_timestep} || {_instance}", end='\r')
    pass


# Print final status to Terminal
def print_final(sim_variables, timestep_count):
    _seed = f"{BColours.OKBLUE}{sim_variables.seed}{BColours.ENDC}"
    _config = f"{BColours.OKCYAN}{sim_variables.config.upper()}{BColours.ENDC}"
    _subgrid = f"{BColours.OKCYAN}{sim_variables.subgrid.upper()}{BColours.ENDC}"
    _timestep = f"{BColours.OKCYAN}{sim_variables.timestep.upper()}{BColours.ENDC}"
    _solver = f"{BColours.OKCYAN}{sim_variables.solver.upper()}{BColours.ENDC}"
    _cfl = f"{BColours.OKCYAN}{sim_variables.cfl}{BColours.ENDC}"
    _dimension = f"{BColours.OKCYAN}{BColours.BOLD}({sim_variables.dimension}D){BColours.ENDC}"
    #_performance = f"{BColours.OKGREEN}{round(kwargs['elapsed']*1e6/(sim_variables.cells*run_length), 3)} \u03BCs/(dt*N){BColours.ENDC}"

    if sim_variables.dimension != 1:
        _cells = f"{BColours.OKCYAN}{sim_variables.cells}^{sim_variables.dimension}{BColours.ENDC}"
    else:
        _cells = f"{BColours.OKCYAN}{sim_variables.cells}{BColours.ENDC}"

    if sim_variables.elapsed >= 60*60:
        _elapsed = f"{BColours.FAIL}{str(timedelta(seconds=sim_variables.elapsed))}s{BColours.ENDC}"
    elif 60*60 > sim_variables.elapsed >= 30*60:
        _elapsed = f"{BColours.WARNING}{str(timedelta(seconds=sim_variables.elapsed))}s{BColours.ENDC}"
    else:
        _elapsed = f"{BColours.OKGREEN}{str(timedelta(seconds=sim_variables.elapsed))}s{BColours.ENDC}"

    print(f"[{sim_variables.now.strftime('%Y-%m-%d %H:%M:%S')} | {_seed}] {_dimension} CONFIG={_config}, CELLS={_cells}, CFL={_cfl}, SUBGRID={_subgrid}, SOLVER={_solver}, TIMESTEP={_timestep} || Elapsed: {_elapsed} ({timestep_count})", flush=True)
    pass


# CLI arguments handler; updates the simulation variables (which is a dict) and checks for any invalid values
def handle_CLI():

    def bool_handler(value):
        return (value.lower() == 'true' or value.lower() == '1')

    bool_choices = ['true','false','True','False',1,0]
    accepted_values = lambda _type: [value for category in DB.search(PARAMS.type == _type) for value in category['accepted']]
    quotes = DB.get(PARAMS.type == 'quotes')['name']

    parser = argparse.ArgumentParser(description='Run the astrea simulation.\n\nastrea is a 1D or 2D (magneto-)hydrodynamics finite volume simulation written in Python3. Refer to the README for more information.', 
                                     epilog=f"--- {BColours.ITALIC}{quotes[random.randint(0,len(quotes)-1)]}{BColours.ENDC} ---", 
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--config', metavar='', type=str.lower, default=argparse.SUPPRESS, help='configuration to run in the simulation', choices=accepted_values('config'))
    parser.add_argument('--cells', '--N', '--n', dest='cells', metavar='', type=int, default=argparse.SUPPRESS, help='number of cells in the grid')
    parser.add_argument('--cfl', metavar='', type=float, default=argparse.SUPPRESS, help='courant number in the Courant-Friedrichs-Lewy stability condition')
    parser.add_argument('--gamma', metavar='', type=float, default=argparse.SUPPRESS, help='adiabatic index')
    parser.add_argument('--dimension', '--dim', dest='dimension', type=int, metavar='', default=argparse.SUPPRESS, help='dimension of the simulation', choices=DB.get(PARAMS.type == 'dimension')['accepted'])
    parser.add_argument('--subgrid', metavar='', type=str.lower, default=argparse.SUPPRESS, help='subgrid model used in the reconstruction of the grid', choices=accepted_values('subgrid'))
    parser.add_argument('--timestep', metavar='', type=str.lower, default=argparse.SUPPRESS, help='sime-stepping algorithm used in the update step of the simulation', choices=accepted_values('timestep'))
    parser.add_argument('--solver', metavar='', type=str.lower, default=argparse.SUPPRESS, help='solver used for the Riemann problem', choices=accepted_values('solver'))

    parser.add_argument('--run_type', metavar='', type=str.lower, default=argparse.SUPPRESS, help='run a single run or multiple runs for each simulation', choices=DB.get(PARAMS.type == 'run_type')['accepted'])
    parser.add_argument('--checkpoints', '--chkpts', dest='checkpoints', metavar='', type=int, default=argparse.SUPPRESS, help='number of checkpoints in simulation')

    parser.add_argument('--plot_options', '--plot-options', dest='plot_options', metavar='', type=str.lower, default=argparse.SUPPRESS, help='simulation variables to plot')
    parser.add_argument('--live_plot', '--live-plot', '--live', dest='live_plot', metavar='', type=bool_handler, default=argparse.SUPPRESS, help='toggle the live plotting function', choices=bool_choices)
    parser.add_argument('--take_snaps', '--take-snaps', dest='take_snaps', metavar='', type=bool_handler, default=argparse.SUPPRESS, help='toggle saving snapshots of the simulation', choices=bool_choices)
    parser.add_argument('--save_plots', '--save-plots', dest='save_plots', metavar='', type=bool_handler, default=argparse.SUPPRESS, help='toggle saving final plots of the simulation', choices=bool_choices)
    parser.add_argument('--save_video', '--save-video', dest='save_video', metavar='', type=bool_handler, default=argparse.SUPPRESS, help='toggle saving a video of the simulation', choices=bool_choices)
    parser.add_argument('--save_file', '--save-file', dest='save_file', metavar='', type=bool_handler, default=argparse.SUPPRESS, help='toggle saving the simulation data file (.hdf5)', choices=bool_choices)

    parser.add_argument('--debug', '--DEBUG', dest='debug', help='toggle for more detailed description of errors/bugs', action='store_true')
    parser.add_argument('--quiet', '-q', dest='quiet', help='toggle printing to screen', action='store_true')
    parser.add_argument('--test', '--TEST', dest='test', default=argparse.SUPPRESS, help=argparse.SUPPRESS, action='store_true')

    args = parser.parse_args()

    return vars(args), args.debug


def parse_cli_variables(_config_variables, _cli_variables):
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
        if k in ['live_plot', 'take_snaps', 'save_video', 'save_plots', 'save_file']:
            if not isinstance(v, bool):
                v = False
        elif k in ['checkpoints', 'dimension']:
            if not isinstance(v, int):
                v = 1
        elif k == "cells":
            if isinstance(v, (int, float)):
                v = int(v) - int(v)%2
            else:
                v = 128
        elif k in ['gamma', 'cfl']:
            if not isinstance(v, (int, float)):
                if "/" in v:
                    num, dem = v.split('/')
                    v = float(num)/float(dem)
                else:
                    if k == "gamma":
                        v = 1.4
                    else:
                        v = .5
            if k == "gamma" and v == 1:
                v += np.finfo(_config_variables['precision']).eps
            if k == "cfl":
                if v <= 0:
                    v = np.finfo(_config_variables['precision']).eps
                elif v > 1:
                    v = 1
        elif k == "plot_options":
            accepted_plot_options, invalid = DB.get(PARAMS.type == k)['accepted'], []
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
                v = DB.get(PARAMS.type == 'default')[k]
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
            for dct in DB.search(PARAMS.type == k):
                if v in dct['accepted']:
                    found = True
                    break

            if not found:
                v = DB.get(PARAMS.type == 'default')[k]
                print(f"{BColours.WARNING}{k.upper()} value not valid; reverting back to default value: {v}..{BColours.ENDC}")

        config_variables[k] = v

    return config_variables


class SimulationVariables(object):
    __slots__ = [
        '__dict__',
        'config', 'cells', 'cfl', 'gamma', 'dimension', 'precision', 'subgrid', 'timestep', 'solver',
        'seed', 'now', 'elapsed', 'access_key', 'datetime',
        'permeability', 'magnetic', 'roots', 'weights', 'axes', 'ortho_axis', 'permutations',
        'config_category', 'solver_category', 'convert_primitive', 'convert_conservative', 'higher_order',
        'start_pos', 'end_pos', 'shock_pos', 't_end', 'boundary', 'misc', 'initial_left', 'initial_right', 'dx', 'dy', 'dz',
        'run_type', 'checkpoints', 'live_plot', 'take_snaps', 'save_plots', 'save_video', 'save_file', 'plot_options', 'plot_style',
        'debug', 'quiet', 'test'
    ]

    def __init__(self, seed, config_variables, test_variables):
        for key in config_variables:
            setattr(self, key, config_variables[key])

        for key in test_variables:
            setattr(self, key, test_variables[key])

        # 5th-order Gauss-Legendre quadrature with interval [0,1] for OS solver
        roots, weights = np.array(list(np.polynomial.legendre.leggauss(5)))/2

        self.seed = int(seed)
        self.now = None
        self.elapsed = None
        self.access_key = None

        self.plot_style = None
        self.permeability = 1.
        self.roots = roots + .5
        self.weights = weights

        self.config_category = DB.get(PARAMS.accepted.any([self.config]))['category']
        self.solver_category = DB.get(PARAMS.accepted.any([self.solver]))['category']
        self.magnetic = 'magnetic' in DB.get(PARAMS.accepted.any([self.config]))['category']

        self.convert_primitive = fv.point_convert_primitive
        self.convert_conservative = fv.point_convert_conservative
        self.higher_order = False

        # Higher-order conversion functions
        if self.subgrid.startswith("w") or self.subgrid in ["ppm", "parabolic", "p"]:
            self.convert_primitive = fv.high_order_convert_primitive
            self.convert_conservative = fv.high_order_convert_conservative
            self.higher_order = True

        # Permutations for axes
        if '2D' in self.config_category and self.dimension != 2:
            self.dimension = 2
        self.axes = tuple(range(self.dimension))

        # Exclusion cases
        if self.solver in DB.get(PARAMS.type == 'solver' and PARAMS.category == 'hll')['accepted']:
            if (self.solver_category == "hll" and self.solver.endswith('c')) and self.config in DB.get(PARAMS.type == 'config' and PARAMS.category == 'magnetic')['accepted']:
                print(f"{BColours.WARNING}HLLC solver does not work with magnetic fields present..{BColours.ENDC}")
                self.solver = DB.get(PARAMS.type == 'default')['solver']

        if self.run_type.startswith('m'):
            if self.save_video:
                print(f"{BColours.WARNING}Videos can only be saved for single simulation runs..{BColours.ENDC}")
                self.save_video = False
            if self.live_plot:
                print(f"{BColours.WARNING}Live plots can only be switched on for single simulation runs..{BColours.ENDC}")
                self.live_plot = False
            if self.take_snaps:
                print(f"{BColours.WARNING}Saving snapshots can only be switched on for single simulation runs..{BColours.ENDC}")
                self.take_snaps = False
        else:
            if (self.take_snaps or self.save_plots or self.save_video) and (self.live_plot):
                print(f"{BColours.WARNING}Live plot can only be switched on when NOT saving media files because live plot interferes with matplotlib.savefig..{BColours.ENDC}")
                self.live_plot = False
            if self.take_snaps or self.save_plots or self.save_video or self.save_file:
                self.save_path = ''