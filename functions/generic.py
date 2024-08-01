from datetime import timedelta

##############################################################################

# Colours for printing to terminal
class bcolours:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# Customised rounding function
def roundOff(value):
    if value%int(value) >= .5:
        return int(value) + 1
    else:
        return int(value)


# Print status to Terminal
def printOutput(instanceTime, seed, simVariables, **kwargs):
    _seed = f"{bcolours.OKBLUE}{seed}{bcolours.ENDC}"
    _config = f"{bcolours.OKCYAN}{simVariables.config.upper()}{bcolours.ENDC}"
    _cells = f"{bcolours.OKCYAN}{simVariables.cells}{bcolours.ENDC}"
    _subgrid = f"{bcolours.OKCYAN}{simVariables.subgrid.upper()}{bcolours.ENDC}"
    _timestep = f"{bcolours.OKCYAN}{simVariables.timestep.upper()}{bcolours.ENDC}"
    _scheme = f"{bcolours.OKCYAN}{simVariables.scheme.upper()}{bcolours.ENDC}"
    _cfl = f"{bcolours.OKCYAN}{simVariables.cfl}{bcolours.ENDC}"
    if kwargs:
        if kwargs['elapsed'] >= 3600:
            _elapsed = f"{bcolours.FAIL}{str(timedelta(seconds=kwargs['elapsed']))}s{bcolours.ENDC}"
        elif 3600 > kwargs['elapsed'] >= 1800:
            _elapsed = f"{bcolours.WARNING}{str(timedelta(seconds=kwargs['elapsed']))}s{bcolours.ENDC}"
        else:
            _elapsed = f"{bcolours.OKGREEN}{str(timedelta(seconds=kwargs['elapsed']))}s{bcolours.ENDC}"
        #_performance = f"{bcolours.OKGREEN}{round(kwargs['elapsed']*1e6/(simVariables.cells*runLength), 3)} \u03BCs/(dt*N){bcolours.ENDC}"
        print(f"[{instanceTime} | {_seed}] TEST={_config}, CELLS={_cells}, CFL={_cfl}, SUBGRID={_subgrid}, SCHEME={_scheme}, TIMESTEP={_timestep} || Elapsed: {_elapsed} ({kwargs['runLength']})")
        pass
    else:
        print(f"[{instanceTime} | {_seed}] TEST={_config}, CELLS={_cells}, CFL={_cfl}, SUBGRID={_subgrid}, SCHEME={_scheme}, TIMESTEP={_timestep} || {bcolours.WARNING}RUNNING SIMULATION..{bcolours.ENDC}", end='\r', flush=True)
        pass


# Function for tidying dictionary
def tidyDict(_dct):
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
def handleErrors(dct):
    acceptedValues = {
        "config": ["sod", "sin", "sin-wave", "sinc", "sinc-wave", "sedov", "shu-osher", "shu", "osher", "gaussian", "gauss", "sq", "square", "square-wave", "toro1", "toro2", "toro3", "toro4", "toro5", "ryu-jones", "ryu", "jones", "rj"],
        "dim": [1, 2, 3],
        "subgrid": ["pcm", "constant", "c", "plm", "linear", "l", "ppm", "parabolic", "p", "weno", "w"],
        "timestep": ["euler", "rk4", "ssprk(2,2)","ssprk(3,3)", "ssprk(4,3)", "ssprk(5,3)", "ssprk(5,4)", "(2,2)", "(3,3)", "(4,3)", "(5,3)", "(5,4)"],
        "scheme": ["lf", "llf", "lax","friedrich", "lax-friedrich", "lw", "lax-wendroff", "wendroff", "hllc", "c", "osher-solomon", "osher", "solomon", "os", "entropy", "stable", "entropy-stable", "es"],
        "runType": ["s", "single", "m", "multiple", "multi", "many", 1, "1"]
        }
    defaultValues = {
        "config": ["sod", "Test unknown; reverting to Sod shock tube test.."],
        "dim": [1, "Invalid value for dimensions; reverting to 1D.."],
        "subgrid": ["pcm", "Subgrid option unknown; reverting to piecewise constant method.."],
        "timestep": ["euler", "Timestep unknown; reverting to Forward Euler timestep.."],
        "scheme": ["lf", "Scheme unknown; reverting to Lax-Friedrich scheme.."],
        "runType": ["single", "Run type unknown; reverting to runType='single' simulation.."]
        }

    for k, lst in acceptedValues.items():
        if dct[k] not in lst:
            print(f"{bcolours.WARNING}{defaultValues[k][1]}{bcolours.ENDC}")
            dct[k] = defaultValues[k][0]

    if dct['runType'] not in ["s", "single", "1", 1] and dct['runType'].startswith('m'):
        if dct['saveVideo']:
            print(f"{bcolours.WARNING}Videos can only be saved with runType='single'..{bcolours.ENDC}")
            dct['saveVideo'] = False
        if dct['livePlot']:
            print(f"{bcolours.WARNING}Live plots can only be switched on for single simulation runs..{bcolours.ENDC}")
            dct['livePlot'] = False

    if dct['runType'] in ["s", "single", "1", 1] and (dct['savePlots'] or dct['saveVideo']) and (dct['livePlot']):
        print(f"{bcolours.WARNING}Switching off live plot when saving media because live plot interferes with matplotlib.savefig..{bcolours.ENDC}")
        dct['livePlot'] = False

    if (dct['dim'] < 1 or dct['dim'] > 2) and (dct['livePlot'] or dct['savePlots'] or dct['saveVideo']):
        print(f"{bcolours.WARNING}Saving media currently not supported for 3D..{bcolours.ENDC}")
        dct['livePlot'] = False
        dct['savePlots'] = False
        dct['saveVideo'] = False

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