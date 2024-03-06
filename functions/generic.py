
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
def printOutput(instanceTime, config, cells, solver, timestep, cfl, elapsed, runLength):
    print(f"[{bcolours.BOLD}{instanceTime}{bcolours.ENDC}] TEST={bcolours.OKGREEN}{config.upper()}{bcolours.ENDC}, CELLS={bcolours.OKGREEN}{cells}{bcolours.ENDC}, RECONSTRUCT={bcolours.OKGREEN}{solver.upper()}{bcolours.ENDC}, TIMESTEP={bcolours.OKGREEN}{timestep.upper()}{bcolours.ENDC}, CFL={bcolours.OKGREEN}{cfl}{bcolours.ENDC} || Elapsed: {bcolours.OKGREEN}{elapsed}s{bcolours.ENDC}  ({runLength})")
    pass