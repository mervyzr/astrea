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


# Lower the element in a list if string
def lowerList(lst):
    arr = []
    for i in lst:
        if isinstance(i, str):
            arr.append(i.lower())
        else:
            arr.append(i)
    return arr


# Print status to Terminal
def printOutput(instanceTime, config, cells, cfl, solver, timestep, elapsed, runLength):
    currentTime = f"{bcolours.BOLD}{instanceTime}{bcolours.ENDC}"
    _config = f"{bcolours.OKGREEN}{config.upper()}{bcolours.ENDC}"
    _cells = f"{bcolours.OKGREEN}{cells}{bcolours.ENDC}"
    _solver = f"{bcolours.OKGREEN}{solver.upper()}{bcolours.ENDC}"
    _timestep = f"{bcolours.OKGREEN}{timestep.upper()}{bcolours.ENDC}"
    _cfl = f"{bcolours.OKGREEN}{cfl}{bcolours.ENDC}"
    _elapsed = f"{bcolours.OKGREEN}{str(timedelta(seconds=elapsed))}s{bcolours.ENDC}"
    _performance = f"{bcolours.OKGREEN}{round(elapsed*1e6/(cells*runLength), 3)} \u03BCs/time step per cell{bcolours.ENDC}"
    print(f"[{currentTime}] TEST={_config}, CELLS={_cells}, RECONSTRUCT={_solver}, TIMESTEP={_timestep}, CFL={_cfl} || Elapsed: {_elapsed} || Performance: {_performance}  ({runLength})")
    pass