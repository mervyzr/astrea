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
def printOutput(instanceTime, seed, cfg, **kwargs):
    _seed = f"{bcolours.OKBLUE}{seed}{bcolours.ENDC}"
    _config = f"{bcolours.OKCYAN}{cfg['config'].upper()}{bcolours.ENDC}"
    _cells = f"{bcolours.OKCYAN}{cfg['cells']}{bcolours.ENDC}"
    _solver = f"{bcolours.OKCYAN}{cfg['solver'].upper()}{bcolours.ENDC}"
    _timestep = f"{bcolours.OKCYAN}{cfg['timestep'].upper()}{bcolours.ENDC}"
    _cfl = f"{bcolours.OKCYAN}{cfg['cfl']}{bcolours.ENDC}"
    if kwargs:
        if kwargs['elapsed'] >= 3600:
            _elapsed = f"{bcolours.FAIL}{str(timedelta(seconds=kwargs['elapsed']))}s{bcolours.ENDC}"
        elif 3600 > kwargs['elapsed'] >= 1800:
            _elapsed = f"{bcolours.WARNING}{str(timedelta(seconds=kwargs['elapsed']))}s{bcolours.ENDC}"
        else:
            _elapsed = f"{bcolours.OKGREEN}{str(timedelta(seconds=kwargs['elapsed']))}s{bcolours.ENDC}"
        #_performance = f"{bcolours.OKGREEN}{round(kwargs['elapsed']*1e6/(cfg['cells']*runLength), 3)} \u03BCs/(dt*N){bcolours.ENDC}"
        print(f"[{instanceTime} | {_seed}] TEST={_config}, CELLS={_cells}, CFL={_cfl}, RECONSTRUCT={_solver}, ITERATE={_timestep} || Elapsed: {_elapsed}  ({kwargs['runLength']})")
        pass
    else:
        print(f"[{instanceTime} | {_seed}] TEST={_config}, CELLS={_cells}, CFL={_cfl}, RECONSTRUCT={_solver}, ITERATE={_timestep} || {bcolours.WARNING}RUNNING SIMULATION..{bcolours.ENDC}", end='\r')
        pass