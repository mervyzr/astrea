import os
import random
import argparse
import itertools
from datetime import timedelta
from tinydb import TinyDB, Query

from numpy.polynomial import legendre

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


# Simple name space for recursive dict
class RecursiveNamespace:

    @staticmethod
    def map_entry(entry):
        if isinstance(entry, dict):
            return RecursiveNamespace(**entry)
        return entry

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            if type(val) == dict:
                setattr(self, key, RecursiveNamespace(**val))
            elif type(val) == list:
                setattr(self, key, list(map(self.map_entry, val)))
            else:
                setattr(self, key, val)


# Print status to Terminal
def print_output(instance_time, seed, sim_variables, **kwargs):
    _seed = f"{BColours.OKBLUE}{seed}{BColours.ENDC}"
    _config = f"{BColours.OKCYAN}{sim_variables.config.upper()}{BColours.ENDC}"
    _subgrid = f"{BColours.OKCYAN}{sim_variables.subgrid.upper()}{BColours.ENDC}"
    _timestep = f"{BColours.OKCYAN}{sim_variables.timestep.upper()}{BColours.ENDC}"
    _scheme = f"{BColours.OKCYAN}{sim_variables.scheme.upper()}{BColours.ENDC}"
    _cfl = f"{BColours.OKCYAN}{sim_variables.cfl}{BColours.ENDC}"
    _dimension = f"{BColours.OKCYAN}{BColours.BOLD}({sim_variables.dimension}D){BColours.ENDC}"

    if sim_variables.dimension != 1:
        _cells = f"{BColours.OKCYAN}{sim_variables.cells}^{sim_variables.dimension}{BColours.ENDC}"
    else:
        _cells = f"{BColours.OKCYAN}{sim_variables.cells}{BColours.ENDC}"

    if kwargs:
        if kwargs['elapsed'] >= 3600:
            _elapsed = f"{BColours.FAIL}{str(timedelta(seconds=kwargs['elapsed']))}s{BColours.ENDC}"
        elif 3600 > kwargs['elapsed'] >= 1800:
            _elapsed = f"{BColours.WARNING}{str(timedelta(seconds=kwargs['elapsed']))}s{BColours.ENDC}"
        else:
            _elapsed = f"{BColours.OKGREEN}{str(timedelta(seconds=kwargs['elapsed']))}s{BColours.ENDC}"
        #_performance = f"{BColours.OKGREEN}{round(kwargs['elapsed']*1e6/(sim_variables.cells*run_length), 3)} \u03BCs/(dt*N){BColours.ENDC}"
        print(f"[{instance_time} | {_seed}] {_dimension} CONFIG={_config}, CELLS={_cells}, CFL={_cfl}, SUBGRID={_subgrid}, SCHEME={_scheme}, TIMESTEP={_timestep} || Elapsed: {_elapsed} ({kwargs['run_length']})", flush=True)
        pass
    else:
        print(f"[{instance_time} | {_seed}] {_dimension} CONFIG={_config}, CELLS={_cells}, CFL={_cfl}, SUBGRID={_subgrid}, SCHEME={_scheme}, TIMESTEP={_timestep} || {BColours.WARNING}RUNNING SIMULATION..{BColours.ENDC}", end='\r')
        pass


# CLI arguments handler; updates the simulation variables (which is a dict) and checks for any invalid values
def handle_CLI():

    def bool_handler(value):
        return (value.lower() == 'true' or value.lower() == '1')

    bool_choices = ['true','false','True','False',1,0]
    accepted_values = lambda _type: [value for category in DB.search(PARAMS.type == _type) for value in category['accepted']]
    quotes = DB.get(PARAMS.type == 'quotes')['name']

    parser = argparse.ArgumentParser(description='Run the mHydyS simulation.\n\nmHydyS is a 1D or 2D (magneto-)hydrodynamics finite volume simulation written in Python3. Refer to the README for more information.', 
                                     epilog=f"--- {BColours.ITALIC}{quotes[random.randint(0,len(quotes)-1)]}{BColours.ENDC} ---", 
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--config', metavar='', type=str.lower, default=argparse.SUPPRESS, help='configuration to run in the simulation', choices=accepted_values('config'))
    parser.add_argument('--cells', '--N', '--n', dest='cells', metavar='', type=int, default=argparse.SUPPRESS, help='number of cells in the grid')
    parser.add_argument('--cfl', metavar='', type=float, default=argparse.SUPPRESS, help='courant number in the Courant-Friedrichs-Lewy stability condition')
    parser.add_argument('--gamma', metavar='', type=float, default=argparse.SUPPRESS, help='adiabatic index')
    parser.add_argument('--dimension', '--dim', dest='dimension', type=int, metavar='', default=argparse.SUPPRESS, help='dimension of the simulation', choices=DB.get(PARAMS.type == 'dimension')['accepted'])
    parser.add_argument('--subgrid', metavar='', type=str.lower, default=argparse.SUPPRESS, help='subgrid model used in the reconstruction of the grid', choices=accepted_values('subgrid'))
    parser.add_argument('--timestep', metavar='', type=str.lower, default=argparse.SUPPRESS, help='sime-stepping algorithm used in the update step of the simulation', choices=accepted_values('timestep'))
    parser.add_argument('--scheme', metavar='', type=str.lower, default=argparse.SUPPRESS, help='scheme of solver for the Riemann problem', choices=accepted_values('scheme'))
    parser.add_argument('--run_type', metavar='', type=str.lower, default=argparse.SUPPRESS, help='run a single run or multiple runs for each simulation', choices=DB.get(PARAMS.type == 'run_type')['accepted'])

    parser.add_argument('--snapshots', metavar='', type=int, default=argparse.SUPPRESS, help='number of snapshots to save')
    parser.add_argument('--live_plot', '--live-plot', '--live', dest='live_plot', type=bool_handler, default=argparse.SUPPRESS, help='toggle the live plotting function', choices=bool_choices)
    parser.add_argument('--take_snaps', '--take-snaps', dest='take_snaps', type=bool_handler, default=argparse.SUPPRESS, help='toggle saving snapshots of the simulation', choices=bool_choices)
    parser.add_argument('--save_plots', '--save-plots', dest='save_plots', type=bool_handler, default=argparse.SUPPRESS, help='toggle saving final plots of the simulation', choices=bool_choices)
    parser.add_argument('--save_video', '--save-video', dest='save_video', type=bool_handler, default=argparse.SUPPRESS, help='toggle saving a video of the simulation', choices=bool_choices)
    parser.add_argument('--save_file', '--save-file', dest='save_file', type=bool_handler, default=argparse.SUPPRESS, help='toggle saving the simulation data file (.hdf5)', choices=bool_choices)

    parser.add_argument('--debug', '--DEBUG', dest='debug', help='toggle for more detailed description of errors/bugs', action='store_true')
    parser.add_argument('-q', '--quiet', '--noprint', '--NOPRINT', '--no-print', dest='noprint', help='toggle printing to screen', action='store_true')
    parser.add_argument('--test', '--TEST', dest='test', default=argparse.SUPPRESS, help=argparse.SUPPRESS, action='store_true')

    args = parser.parse_args()

    return vars(args), args.debug, args.noprint


# Variables handler; handles all variables from CLI & settings file and revert to default values for the simulation variables (dict) if unknown
def handle_variables(seed: float, config_variables: dict, cli_variables: dict):
    # Remove nested configuration dictionary
    _config_variables = {}
    for parameters in config_variables.values():
        for k,v in parameters.items():
            _config_variables[k] = v

    # Replace the relevant configuration variables with the CLI variables
    if bool(cli_variables):
        for k,v in cli_variables.items():
            if k in _config_variables:
                _config_variables[k] = v

    # Check validity of variables
    final_dict = {}
    for k,v in _config_variables.items():
        if k in ['live_plot', 'take_snaps', 'save_video', 'save_plots', 'save_file']:
            if not isinstance(v, bool):
                v = False
        elif k in ['snapshots', 'dimension']:
            if not isinstance(v, int):
                v = 1
        elif k == "cells":
            try:
                v = int(v) - int(v)%2
            except Exception as e:
                v = 128
        elif k in ['gamma', 'cfl']:
            if not isinstance(v, float):
                if k == "gamma":
                    v = 1.4
                else:
                    v = .5
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

        final_dict[k] = v

    # Add relevant key-pairs to the dictionary
    final_dict['seed'] = int(seed)
    final_dict['permutations'] = [axes for axes in list(itertools.permutations(list(range(final_dict['dimension']+1)))) if axes[-1] == final_dict['dimension']]
    final_dict['config_category'] = DB.get(PARAMS.accepted.any([final_dict['config']]))['category']
    final_dict['timestep_category'] = DB.get(PARAMS.accepted.any([final_dict['timestep']]))['category']
    final_dict['scheme_category'] = DB.get(PARAMS.accepted.any([final_dict['scheme']]))['category']

    if final_dict['scheme'] in DB.get(PARAMS.type == 'scheme' and PARAMS.category == 'complete')['accepted']:
        _roots, _weights = legendre.leggauss(3)  # 3rd-order Gauss-Legendre quadrature with interval [-1,1]
        final_dict['roots'] = .5*_roots + .5  # Gauss-Legendre quadrature with interval [0,1]
        final_dict['weights'] = _weights/2  # Gauss-Legendre quadrature with interval [0,1]

    # Exclusion cases
    if final_dict['scheme'] in DB.get(PARAMS.type == 'scheme' and PARAMS.category == 'hll')['accepted']:
        if (final_dict['scheme_category'] == "hll" and final_dict['scheme'].endswith('c')) and final_dict['config'] in DB.get(PARAMS.type == 'config' and PARAMS.category == 'magnetic')['accepted']:
            print(f"{BColours.WARNING}HLLC scheme does not work with magnetic fields present..{BColours.ENDC}")
            final_dict['scheme'] = DB.get(PARAMS.type == 'default')['scheme']

    if final_dict['run_type'].startswith('m'):
        if final_dict['save_video']:
            print(f"{BColours.WARNING}Videos can only be saved for single simulation runs..{BColours.ENDC}")
            final_dict['save_video'] = False
        if final_dict['live_plot']:
            print(f"{BColours.WARNING}Live plots can only be switched on for single simulation runs..{BColours.ENDC}")
            final_dict['live_plot'] = False
        if final_dict['take_snaps']:
            print(f"{BColours.WARNING}Saving snapshots can only be switched on for single simulation runs..{BColours.ENDC}")
            final_dict['take_snaps'] = False
    else:
        if (final_dict['take_snaps'] or final_dict['save_video'] or final_dict['save_plots']) and (final_dict['live_plot']):
            print(f"{BColours.WARNING}Live plot can only be switched on when NOT saving media files because live plot interferes with matplotlib.savefig..{BColours.ENDC}")
            final_dict['live_plot'] = False

    return final_dict