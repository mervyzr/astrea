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
def printOutput(instanceTime, cfg, elapsed, runLength):
    _config = f"{bcolours.OKGREEN}{cfg['config'].upper()}{bcolours.ENDC}"
    _cells = f"{bcolours.OKGREEN}{cfg['cells']}{bcolours.ENDC}"
    _solver = f"{bcolours.OKGREEN}{cfg['solver'].upper()}{bcolours.ENDC}"
    _timestep = f"{bcolours.OKGREEN}{cfg['timestep'].upper()}{bcolours.ENDC}"
    _cfl = f"{bcolours.OKGREEN}{cfg['cfl']}{bcolours.ENDC}"
    if elapsed >= 3600:
        _elapsed = f"{bcolours.FAIL}{str(timedelta(seconds=elapsed))}s{bcolours.ENDC}"
    elif 3600 > elapsed >= 1800:
        _elapsed = f"{bcolours.WARNING}{str(timedelta(seconds=elapsed))}s{bcolours.ENDC}"
    else:
        _elapsed = f"{bcolours.OKCYAN}{str(timedelta(seconds=elapsed))}s{bcolours.ENDC}"
    #_performance = f"{bcolours.OKGREEN}{round(elapsed*1e6/(cfg['cells']*runLength), 3)} \u03BCs/(dt*N){bcolours.ENDC}"
    print(f"[{instanceTime}] TEST={_config}, CELLS={_cells}, CFL={_cfl}, RECONSTRUCT={_solver}, TIMESTEP={_timestep} || Elapsed: {_elapsed} ({runLength})")
    pass