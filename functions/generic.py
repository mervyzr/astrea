from datetime import timedelta

##############################################################################
# Generic functions not specific to finite volume
##############################################################################

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
class Namespace:

    @staticmethod
    def map_entry(entry):
        if isinstance(entry, dict):
            return Namespace(**entry)
        return entry

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            if type(val) == dict:
                setattr(self, key, Namespace(**val))
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
    _dimension = f"{BColours.OKCYAN}{sim_variables.dimension}{BColours.ENDC}"
    if kwargs:
        if kwargs['elapsed'] >= 3600:
            _elapsed = f"{BColours.FAIL}{str(timedelta(seconds=kwargs['elapsed']))}s{BColours.ENDC}"
        elif 3600 > kwargs['elapsed'] >= 1800:
            _elapsed = f"{BColours.WARNING}{str(timedelta(seconds=kwargs['elapsed']))}s{BColours.ENDC}"
        else:
            _elapsed = f"{BColours.OKGREEN}{str(timedelta(seconds=kwargs['elapsed']))}s{BColours.ENDC}"
        #_performance = f"{BColours.OKGREEN}{round(kwargs['elapsed']*1e6/(sim_variables.cells*run_length), 3)} \u03BCs/(dt*N){BColours.ENDC}"
        print(f"[{instance_time} | {_seed}] TEST={_config}, CELLS={_cells}, CFL={_cfl}, SUBGRID={_subgrid}, SCHEME={_scheme}, TIMESTEP={_timestep}, DIM={_dimension} || Elapsed: {_elapsed} ({kwargs['run_length']})")
        pass
    else:
        print(f"[{instance_time} | {_seed}] TEST={_config}, CELLS={_cells}, CFL={_cfl}, SUBGRID={_subgrid}, SCHEME={_scheme}, TIMESTEP={_timestep}, DIM={_dimension} || {BColours.WARNING}RUNNING SIMULATION..{BColours.ENDC}", end='\r')
        pass


# Function for tidying dictionary
def tidy_dict(_dct):
    dct = {}
    for k, v in _dct.items():
        if isinstance(v, int):
            if k == "cells":
                dct[k] = int(v) - int(v)%2
            else:
                dct[k] = int(v)
        elif isinstance(v, str):
            dct[k] = v.lower()
        else:
            dct[k] = v
    return dct


# Error condition(s) handler; revert to default values for the simulation variables (dict) if unknown
def handle_errors(dct):
    accepted_values = {
        "config": ["sod", "sin", "sin-wave", "sinc", "sinc-wave", "sedov", "shu-osher", "shu", "osher", "gaussian", "gauss", "sq", "square", "square-wave", "toro1", "toro2", "toro3", "toro4", "toro5", "ryu-jones", "ryu", "jones", "rj", "brio-wu", "brio", "wu", "bw"],
        "dimension": [1, 1.5, 2],
        "subgrid": ["pcm", "constant", "c", "plm", "linear", "l", "ppm", "parabolic", "p", "weno", "w"],
        "timestep": ["euler", "rk4", "ssprk(2,2)","ssprk(3,3)", "ssprk(4,3)", "ssprk(5,3)", "ssprk(5,4)", "(2,2)", "(3,3)", "(4,3)", "(5,3)", "(5,4)"],
        "scheme": ["lf", "llf", "lax","friedrich", "lax-friedrich", "lw", "lax-wendroff", "wendroff", "hllc", "c", "osher-solomon", "osher", "solomon", "os", "entropy", "stable", "entropy-stable", "es"],
        "run_type": ["s", "single", "m", "multiple", "multi", "many", 1, "1"]
        }
    default_values = {
        "config": ["sod", "Test unknown; reverting to Sod shock tube test.."],
        "dimension": [1, "Invalid value for dimension; reverting to 1D.."],
        "subgrid": ["pcm", "Subgrid option unknown; reverting to piecewise constant method.."],
        "timestep": ["euler", "Timestep unknown; reverting to Forward Euler timestep.."],
        "scheme": ["lf", "Scheme unknown; reverting to Lax-Friedrich scheme.."],
        "run_type": ["single", "Run type unknown; reverting to run_type='single' simulation.."]
        }

    for k, lst in accepted_values.items():
        if dct[k] not in lst:
            print(f"{BColours.WARNING}{default_values[k][1]}{BColours.ENDC}")
            dct[k] = default_values[k][0]

    if dct['run_type'] not in ["s", "single", "1", 1] and dct['run_type'].startswith('m'):
        if dct['save_video']:
            print(f"{BColours.WARNING}Videos can only be saved with run_type='single'..{BColours.ENDC}")
            dct['save_video'] = False
        if dct['live_plot']:
            print(f"{BColours.WARNING}Live plots can only be switched on for single simulation runs..{BColours.ENDC}")
            dct['live_plot'] = False

    if dct['run_type'] in ["s", "single", "1", 1] and (dct['save_plots'] or dct['save_video']) and (dct['live_plot']):
        print(f"{BColours.WARNING}Switching off live plot when saving media because live plot interferes with matplotlib.savefig..{BColours.ENDC}")
        dct['live_plot'] = False

    if (dct['dimension'] < 1 or dct['dimension'] > 2) and (dct['live_plot'] or dct['save_plots'] or dct['save_video']):
        print(f"{BColours.WARNING}Saving media currently not supported for 3D..{BColours.ENDC}")
        dct['live_plot'] = False
        dct['save_plots'] = False
        dct['save_video'] = False

    return dct


quotes = ["It's not a bug; it's an undocumented feature",\
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
        "Computer science is no more about computers than astronomy is about telescopes"]