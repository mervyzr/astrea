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
        print(f"[{instanceTime} | {_seed}] TEST={_config}, CELLS={_cells}, CFL={_cfl}, SUBGRID={_subgrid}, SCHEME={_scheme}, TIMESTEP={_timestep} || {bcolours.WARNING}RUNNING SIMULATION..{bcolours.ENDC}", end='\r')
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