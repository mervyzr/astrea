import os
import math
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
PARAMS = Query()
ACCEPTED_VALUES = {
    "config": [
        "sod", 
        "sin", "sin-wave", 
        "sedov", 
        "shu-osher", "shu", "osher", 
        "slow", "slow-moving shock", "slow shock", 
        "double rarefaction", "rarefaction", "double", 
        "gaussian", "gauss", 
        "sq", "square", "square-wave", 
        "ryu-jones", "ryu", "jones", "rj", 
        "brio-wu", "brio", "wu", "bw", 
        "khi", "kelvin", "helmholtz", "kelvin-helmholtz", 
        "ivc", "vortex", "isentropic vortex", 
        "toro1", "toro2", "toro3", "toro4", "toro5", 
        "ll3", "ll4", "ll6", "ll11", "ll12", "ll15", "lax-liu3", "lax-liu4", "lax-liu6", "lax-liu11", "lax-liu12", "lax-liu15"
        ],
    "dimension": [1, 1.5, 2],
    "subgrid": [
        "pcm", "constant", "c", 
        "plm", "linear", "l", 
        "ppm", "parabolic", "p", 
        "weno", "weno3", "weno-3", "weno5", "weno-5", "weno7", "weno-7", "w"
        ],
    "timestep": [
        "euler", 
        "rk4", 
        "ssprk(2,2)", "(2,2)", 
        "ssprk(3,3)", "(3,3)",
        "ssprk(4,3)", "(4,3)",
        "ssprk(5,3)", "(5,3)",
        "ssprk(5,4)", "(5,4)",
        "ssprk(10,4)", "(10,4)"],
    "scheme": [
        "lf", "llf", "lax","friedrich", "lax-friedrich", 
        "lw", "lax-wendroff", "wendroff", 
        "hllc", "c", 
        "osher-solomon", "osher", "solomon", "os", 
        "entropy", "stable", "entropy-stable", "es"],
    "run_type": ["s", "single", "m", "multiple", "multi", "many"]
    }

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
    _cells = f"{BColours.OKCYAN}{sim_variables.cells}{BColours.ENDC}"
    _subgrid = f"{BColours.OKCYAN}{sim_variables.subgrid.upper()}{BColours.ENDC}"
    _timestep = f"{BColours.OKCYAN}{sim_variables.timestep.upper()}{BColours.ENDC}"
    _scheme = f"{BColours.OKCYAN}{sim_variables.scheme.upper()}{BColours.ENDC}"
    _cfl = f"{BColours.OKCYAN}{sim_variables.cfl}{BColours.ENDC}"

    if sim_variables.dimension%1 != 0:
        _dimension = f"{BColours.OKCYAN}{BColours.BOLD}({sim_variables.dimension}D){BColours.ENDC}"
    else:
        _dimension = f"{BColours.OKCYAN}{BColours.BOLD}({int(sim_variables.dimension)}D){BColours.ENDC}"

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


# Function for tidying simulation variables
def handle_config(_dct):
    dct = {}
    for parameters in _dct.values():
        for k,v in parameters.items():
            if k == "cells":
                try:
                    v -= v%2
                except Exception as e:
                    v = 128
            if k == "precision":
                precision_list = {1:"float16", 2:"float32", 4:"float64", 8:"float128"}
                try:
                    int(v)
                except Exception as e:
                    try:
                        v.lower()
                    except Exception as e:
                        v = "float64"
                    else:
                        if "bit" in v:
                            bit = int(v.replace("-","").replace(" ","").split("bit")[0])
                        elif "float" in v:
                            bit = int(v.split("float")[1])
                        v = precision_list[bit//16]
                else:
                    v = precision_list[v//16]
            if k == "scheme" and v in ["osher-solomon", "osher", "solomon", "os"]:
                _roots, _weights = legendre.leggauss(3)  # 3rd-order Gauss-Legendre quadrature with interval [-1,1]
                dct['roots'] = .5*_roots + .5  # Gauss-Legendre quadrature with interval [0,1]
                dct['weights'] = _weights/2  # Gauss-Legendre quadrature with interval [0,1]
            if k == "dimension":
                dct['permutations'] = [axes for axes in list(itertools.permutations(list(range(math.ceil(v+1))))) if axes[-1] == math.ceil(v)]
            if isinstance(v, str):
                v = v.lower()

            dct[k] = v
    return dct


# CLI arguments handler; updates the simulation variables (which is a dict) and checks for any invalid values
def handle_CLI():
    quotes = DB.get(PARAMS.type == 'quotes')['name']

    parser = argparse.ArgumentParser(description='Run the mHydyS simulation.\n\nmHydyS is a 1D or 2D (magneto-)hydrodynamics finite volume simulation written in Python3. Refer to the README for more information.', 
                                     epilog=f"Fun quote: {quotes[random.randint(0,len(quotes)-1)]}", 
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--config', metavar='', type=str.lower, default=argparse.SUPPRESS, help='Configuration to run in the simulation', choices=ACCEPTED_VALUES['config'])
    parser.add_argument('--cells', '--N', '--n', dest='cells', metavar='', type=int, default=argparse.SUPPRESS, help='Number of cells in the grid', choices=range(2,16385,2))
    parser.add_argument('--cfl', metavar='', type=float, default=argparse.SUPPRESS, help='Courant number in the Courant-Friedrichs-Lewy stability condition')
    parser.add_argument('--gamma', metavar='', type=float, default=argparse.SUPPRESS, help='Adiabatic index')
    parser.add_argument('--dimension', '--dim', dest='dimension', type=float, metavar='', default=argparse.SUPPRESS, help='Dimension of the simulation', choices=ACCEPTED_VALUES['dimension'])
    parser.add_argument('--subgrid', metavar='', type=str.lower, default=argparse.SUPPRESS, help='Subgrid model used in the reconstruction of the grid', choices=ACCEPTED_VALUES['subgrid'])
    parser.add_argument('--timestep', metavar='', type=str.lower, default=argparse.SUPPRESS, help='Time-stepping algorithm used in the update step of the simulation', choices=ACCEPTED_VALUES['timestep'])
    parser.add_argument('--scheme', metavar='', type=str.lower, default=argparse.SUPPRESS, help='Scheme of solver for the Riemann problem', choices=ACCEPTED_VALUES['scheme'])
    parser.add_argument('--run_type', metavar='', type=str.lower, default=argparse.SUPPRESS, help='Number of runs in a complete simulation', choices=ACCEPTED_VALUES['run_type'])
    parser.add_argument('--live_plot', metavar='', type=bool, default=argparse.SUPPRESS, help='Toggle the live plot', choices=[True, False])
    parser.add_argument('--save_plots', metavar='', type=bool, default=argparse.SUPPRESS, help='Save plots to file', choices=[True, False])
    parser.add_argument('--snapshots', metavar='', type=int, default=argparse.SUPPRESS, help='Number of snapshots of the simulation to save to file')
    parser.add_argument('--save_video', metavar='', type=bool, default=argparse.SUPPRESS, help='Save a video of the entire simulation', choices=[True, False])
    parser.add_argument('--save_file', metavar='', type=bool, default=argparse.SUPPRESS, help='Save the simulation data file (.hdf5)', choices=[True, False])
    parser.add_argument('--debug', '--DEBUG', dest='debug', help='Toggle for more detailed description of errors/bugs', action='store_true')
    parser.add_argument('-q', '--quiet', '--noprint', '--NOPRINT', dest='noprint', help='Toggle printing to screen', action='store_true')
    parser.add_argument('--test', '--TEST', dest='test', default=argparse.SUPPRESS, help=argparse.SUPPRESS, action='store_true')

    args = parser.parse_args()

    return vars(args), args.debug, args.noprint


# Variables handler; handles all variables from CLI & settings file and revert to default values for the simulation variables (dict) if unknown
def handle_variables(dct):
    dct['dx'] = abs(dct['end_pos']-dct['start_pos'])/dct['cells']

    default_values = {
        "config": ["sod", "Test unknown; reverting to Sod shock tube test.."],
        "dimension": [1, "Invalid value for dimension; reverting to 1D.."],
        "subgrid": ["pcm", "Subgrid option unknown; reverting to piecewise constant method.."],
        "timestep": ["euler", "Timestep unknown; reverting to Forward Euler timestep.."],
        "scheme": ["lf", "Scheme unknown; reverting to Lax-Friedrich scheme.."],
        "run_type": ["single", "Run type unknown; reverting to run_type='single' simulation.."]
        }

    for k, lst in ACCEPTED_VALUES.items():
        if dct[k] not in lst:
            print(f"{BColours.WARNING}{default_values[k][1]}{BColours.ENDC}")
            dct[k] = default_values[k][0]

    if dct['scheme'] in ["hllc", "c"] and (dct['initial_left'][-3:].any() or dct['initial_right'][-3:].any()):
        print(f"{BColours.WARNING}HLLC scheme does not work with magnetic fields present..{BColours.ENDC}")
        dct['scheme'] = default_values['scheme'][0]

    if dct['run_type'] not in ["s", "single"] and dct['run_type'].startswith('m'):
        if dct['save_video']:
            print(f"{BColours.WARNING}Videos can only be saved with run_type='single'..{BColours.ENDC}")
            dct['save_video'] = False
        if dct['live_plot']:
            print(f"{BColours.WARNING}Live plots can only be switched on for single simulation runs..{BColours.ENDC}")
            dct['live_plot'] = False

    if dct['run_type'] in ["s", "single", "1", 1] and (dct['save_plots'] or dct['save_video']) and (dct['live_plot']):
        print(f"{BColours.WARNING}Switching off live plot when saving media because live plot interferes with matplotlib.savefig..{BColours.ENDC}")
        dct['live_plot'] = False

    return dct




# Variables handler; handles all variables from CLI & settings file and revert to default values for the simulation variables (dict) if unknown
def handle_variables(config_variables: dict, cli_variables: dict):
    # Remove nested configuration dictionary
    _config_variables = {}
    for parameters in config_variables.values():
        for k,v in parameters.items():
            _config_variables[k] = v

    # Replace the relevant configuration variables with the CLI variables
    for k,v in cli_variables.items():
        if k in _config_variables:
            _config_variables[k] = v

    final_dict = {}
    # Check validity of variables
    for k,v in _config_variables.items():
        if k in ['live_plot', 'save_plots', 'save_video', 'save_file']:
            if not isinstance(v, bool):
                v = False
        elif k == "snapshots":
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
                print(f"{BColours.WARNING}{k.upper()} variable not valid; reverting back to default value: {v}..{BColours.ENDC}")

            if k == "scheme" and v in DB.get(PARAMS.type == k and PARAMS.category == 'full')['accepted']:
                _roots, _weights = legendre.leggauss(3)  # 3rd-order Gauss-Legendre quadrature with interval [-1,1]
                final_dict['roots'] = .5*_roots + .5  # Gauss-Legendre quadrature with interval [0,1]
                final_dict['weights'] = _weights/2  # Gauss-Legendre quadrature with interval [0,1]

            if k == "dimension":
                final_dict['permutations'] = [axes for axes in list(itertools.permutations(list(range(math.ceil(v+1))))) if axes[-1] == math.ceil(v)]



        

        

            


  config: Sod
  dimension: 1
  precision: float64
  subgrid: weno
  timestep: SSPRK(5,4)
  scheme: LF
  run_type: single


 
    #######################################################
    dct = {}
    for parameters in _dct.values():
        for k,v in parameters.items():
            if k == "cells":
                try:
                    v -= v%2
                except Exception as e:
                    v = 128
            if k == "precision":
                precision_list = {1:"float16", 2:"float32", 4:"float64", 8:"float128"}
                try:
                    int(v)
                except Exception as e:
                    try:
                        v.lower()
                    except Exception as e:
                        v = "float64"
                    else:
                        if "bit" in v:
                            bit = int(v.replace("-","").replace(" ","").split("bit")[0])
                        elif "float" in v:
                            bit = int(v.split("float")[1])
                        v = precision_list[bit//16]
                else:
                    v = precision_list[v//16]
            if k == "scheme" and v in ["osher-solomon", "osher", "solomon", "os"]:
                _roots, _weights = legendre.leggauss(3)  # 3rd-order Gauss-Legendre quadrature with interval [-1,1]
                dct['roots'] = .5*_roots + .5  # Gauss-Legendre quadrature with interval [0,1]
                dct['weights'] = _weights/2  # Gauss-Legendre quadrature with interval [0,1]
            if k == "dimension":
                dct['permutations'] = [axes for axes in list(itertools.permutations(list(range(math.ceil(v+1))))) if axes[-1] == math.ceil(v)]
            if isinstance(v, str):
                v = v.lower()

            dct[k] = v
    
    #######################################################

    default_values = {
        "config": ["sod", "Test unknown; reverting to Sod shock tube test.."],
        "dimension": [1, "Invalid value for dimension; reverting to 1D.."],
        "subgrid": ["pcm", "Subgrid option unknown; reverting to piecewise constant method.."],
        "timestep": ["euler", "Timestep unknown; reverting to Forward Euler timestep.."],
        "scheme": ["lf", "Scheme unknown; reverting to Lax-Friedrich scheme.."],
        "run_type": ["single", "Run type unknown; reverting to run_type='single' simulation.."]
        }

    for k, lst in ACCEPTED_VALUES.items():
        if dct[k] not in lst:
            print(f"{BColours.WARNING}{default_values[k][1]}{BColours.ENDC}")
            dct[k] = default_values[k][0]

    if dct['scheme'] in ["hllc", "c"] and (dct['initial_left'][-3:].any() or dct['initial_right'][-3:].any()):
        print(f"{BColours.WARNING}HLLC scheme does not work with magnetic fields present..{BColours.ENDC}")
        dct['scheme'] = default_values['scheme'][0]

    if dct['run_type'] not in ["s", "single"] and dct['run_type'].startswith('m'):
        if dct['save_video']:
            print(f"{BColours.WARNING}Videos can only be saved with run_type='single'..{BColours.ENDC}")
            dct['save_video'] = False
        if dct['live_plot']:
            print(f"{BColours.WARNING}Live plots can only be switched on for single simulation runs..{BColours.ENDC}")
            dct['live_plot'] = False

    if dct['run_type'] in ["s", "single", "1", 1] and (dct['save_plots'] or dct['save_video']) and (dct['live_plot']):
        print(f"{BColours.WARNING}Switching off live plot when saving media because live plot interferes with matplotlib.savefig..{BColours.ENDC}")
        dct['live_plot'] = False

    return dct





# Generate the simulation variables from settings (dict)
    with open(f"{CURRENT_DIR}/settings.yml", "r") as settings_file:
        config_dict = yaml.safe_load(settings_file)
        config_variables = generic.handle_config(config_dict)

    # Check CLI arguments
    if len(sys.argv) > 1:
        config_variables, noprint, debug = generic.handle_CLI(config_variables)
    else:
        noprint, debug = False, False
    
    if not debug:
        np.seterr(all='ignore')

    #config_variables = generic.handle_variables(config_variables, cli_variables)

    # Generate test configuration
    test_variables = tests.generate_test_conditions(config_variables['config'])
    sim_variables = config_variables | test_variables

    # Variables handler; filter erroneous entries and default values
    sim_variables = generic.handle_variables(sim_variables)



"""import os
from tinydb import TinyDB, Query

db = TinyDB(f'{os.getcwd()}/db.json')

item = {'type':'config', 'category':'shock', 'accepted': ['sod', 'sod shock', 'sod-shock', 'sedov', 'sedov blast', 'sedov wave', 'sedov-blast', 'sedov-wave']}
item = {'type':'config', 'category':'special', 'accepted': ['shu-osher', 'shu osher', 'shu', 'osher', 'so', 'toro1', 'toro2', 'toro3', 'toro4', 'toro5', 'toro-1', 'toro-2', 'toro-3', 'toro-4', 'toro-5', 'toro 1', 'toro 2', 'toro 3', 'toro 4', 'toro 5']}
item = {'type':'config', 'category':'interacting', 'accepted': ['slow', 'slow-moving', 'slow-moving shock', 'slow shock', 'slow-shock', 'double', 'double-rarefaction', 'double rarefaction', 'rarefaction', 'double-rarefaction wave']}
item = {'type':'config', 'category':'smooth', 'accepted': ['sin', 'sin wave', 'sin-wave', 'gaussian', 'gauss']}
item = {'type':'config', 'category':'stationary', 'accepted': ['square', 'sq', 'square-wave', 'square wave']}
item = {'type':'config', 'category':'magnetic', 'accepted': ['ryu-jones', 'rj', 'ryu', 'jones', 'brio-wu', 'bw', 'brio', 'wu']}
item = {'type':'config', 'category':'2D-only', 'accepted': ['kelvin-helmholtz', 'kelvin', 'helmholtz', 'kelvin helmholtz', 'kelvin-helmholtz instability', 'kelvin helmholtz instability', 'khi', 'vortex', 'isentropic', 'isentropic vortex', 'ivc', 'll3', 'll4', 'll6', 'll11', 'll12', 'll15', 'lax-liu3', 'lax-liu4', 'lax-liu6', 'lax-liu11', 'lax-liu12', 'lax-liu15', 'lax-liu 3', 'lax-liu 4', 'lax-liu 6', 'lax-liu 11', 'lax-liu 12', 'lax-liu 15']}


item = {'type':'dimension', 'accepted': [1, 1.5, 2]}


item = {'type':'precision', 'accepted': ['float', 'float16', 'float32', 'float64', 'float128', 16, 32, 64, 128]}


item = {'type':'subgrid', 'category':'pcm', 'accepted': ['pcm', 'constant', 'c']}
item = {'type':'subgrid', 'category':'plm', 'accepted': ['plm', 'linear', 'l']}
item = {'type':'subgrid', 'category':'ppm', 'accepted': ['ppm', 'parabolic', 'polynomial', 'p']}
item = {'type':'subgrid', 'category':'weno', 'accepted': ['weno', 'weno3', 'weno-3', 'weno 3', 'weno5', 'weno-5', 'weno 5', 'weno7', 'weno-7', 'weno 7']}


item = {'type':'timestep', 'category':'euler', 'accepted': ['euler', 'forward', 'forward-euler', 'forward euler']}
item = {'type':'timestep', 'category':'runge-kutta', 'accepted': ['rk4', 'runge-kutta', 'runge', 'kutta', 'runge-kutta 4', 'runge-kutta4']}
item = {'type':'timestep', 'category':'ssprk', 'accepted': ['ssprk(2,2)', '(2,2)', 'ssprk22', 'ssprk(3,3)', '(3,3)', 'ssprk33', 'ssprk(4,3)', '(4,3)', 'ssprk43', 'ssprk(5,3)', '(5,3)', 'ssprk53', 'ssprk(5,4)', '(5,4)', 'ssprk54', 'ssprk(10,4)', '(10,4)', 'ssprk104']}


item = {'type':'scheme', 'category':'lax', 'accepted': ['lf', 'friedrich', 'lax-friedrich', 'llf', 'local lax-friedrich', "lw", "lax-wendroff", "wendroff"]}
item = {'type':'scheme', 'category':'hll', 'accepted': ['hllc', 'c']}
item = {'type':'scheme', 'category':'full', 'accepted': ['os', 'osher', 'solomon', 'osher-solomon', 'osher solomon', 'es', 'entropy', 'entropy-stable']}


item = {'type':'run_type', 'accepted': ["s", "single", "m", "multiple", "multi", "many"]}


item = {'type':'default', 'config':'sod', 'dimension':1, 'precision':'float64', 'subgrid':'pcm', 'timestep':'euler', 'scheme':'lf', 'run_type':'single'}


item = {'type':'quotes', 'name': ["It's not a bug; it's an undocumented feature",\
            "Experience is the name everyone gives to their mistakes",\
            "Confusion is part of programming",\
            "Light attracts bugs",\
            "Programmer: A machine that turns coffee into code",\
            "When I wrote this code, only God and I understood what I did. Now only God knows",\
            "If, at first you do not succeed, call it version 1.0",\
            "Keep It Simple, Stupid",\
            "If you torture the data long enough, it will confess",\
            "To steal ideas from one person is plagiarism; to steal from many is research",\
            "Never forget the greatest researcher of our time: et al.",\
            "The difference between screwing around and science is writing it down",\
            "Computer science is no more about computers than astronomy is about telescopes"]}"""