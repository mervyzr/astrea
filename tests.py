import numpy as np

import settings as cfg
from functions import generic, fv

##############################################################################

if cfg.config.lower() == "sod":
    startPos = 0
    endPos = 1
    shockPos = .5
    tEnd = .2
    boundary = "edge"  # outflow
    initialLeft = np.array([1,0,0,0,1])  # primitive variables [rho, vx, vy, vz, P]
    initialRight = np.array([.125,0,0,0,.1])  # primitive variables [rho, vx, vy, vz, P]

elif cfg.config.lower() == "sin":
    startPos = 0
    endPos = 1
    shockPos = 1
    tEnd = 2
    boundary = "wrap"  # periodic
    initialLeft = np.array([0,1,1,1,1])  # primitive variables [rho, vx, vy, vz, P]
    initialRight = np.array([0,0,0,0,0])  # primitive variables [rho, vx, vy, vz, P]

elif cfg.config.lower() == "sedov":
    startPos = -10
    endPos = 10
    shockPos = .1  # blast boundary
    tEnd = .6
    boundary = "edge"  # outflow
    initialLeft = np.array([1,0,0,0,100])  # primitive variables [rho, vx, vy, vz, P]
    initialRight = np.array([1,0,0,0,1])  # primitive variables [rho, vx, vy, vz, P]

elif cfg.config.lower() == "shu-osher":
    startPos = -1
    endPos = 1
    shockPos = -.8
    tEnd = .47
    boundary = "edge"  # outflow
    freq = 5
    initialLeft = np.array([3.857143,2.629369,0,0,10.3333])  # primitive variables [rho, vx, vy, vz, P]
    initialRight = np.array([0,0,0,0,1])                    # primitive variables [rho, vx, vy, vz, P]

elif "sq" in cfg.config.lower():
    startPos = -1
    endPos = 1
    shockPos = 1/3
    tEnd = .05
    boundary = "wrap"  # periodic
    initialLeft = np.array([1,1,0,0,1])  # primitive variables [rho, vx, vy, vz, P]
    initialRight = np.array([.01,1,0,0,1])  # primitive variables [rho, vx, vy, vz, P]
    
elif "toro" in cfg.config.lower():
    startPos = 0
    endPos = 1
    boundary = "edge"  # outflow

    if "2" in cfg.config.lower():
        shockPos = .5
        tEnd = .14
        initialLeft = np.array([1,-2,0,0,.4])  # primitive variables [rho, vx, vy, vz, P]
        initialRight = np.array([1,2,0,0,.4])  # primitive variables [rho, vx, vy, vz, P]

    elif "3" in cfg.config.lower():
        shockPos = .5
        tEnd = .012
        initialLeft = np.array([1,0,0,0,1000])  # primitive variables [rho, vx, vy, vz, P]
        initialRight = np.array([1,0,0,0,.01])  # primitive variables [rho, vx, vy, vz, P]

    elif "4" in cfg.config.lower():
        shockPos = .3
        tEnd = .05
        initialLeft = np.array([5.99924,19.5975,0,0,460.894])  # primitive variables [rho, vx, vy, vz, P]
        initialRight = np.array([5.99242,-6.19633,0,0,46.095])  # primitive variables [rho, vx, vy, vz, P]

    elif "5" in cfg.config.lower():
        shockPos = .8
        tEnd = .012
        initialLeft = np.array([1,-19.59745,0,0,1000])  # primitive variables [rho, vx, vy, vz, P]
        initialRight = np.array([1,-19.59745,0,0,.01])  # primitive variables [rho, vx, vy, vz, P]

    else:
        shockPos = .3
        tEnd = .2
        initialLeft = np.array([1,.75,0,0,1])  # primitive variables [rho, vx, vy, vz, P]
        initialRight = np.array([.125,0,0,0,.1])  # primitive variables [rho, vx, vy, vz, P]

else:
    print(f"{generic.bcolours.WARNING}Test unknown; reverting to Sod shock tube test..{generic.bcolours.ENDC}")
    startPos = 0
    endPos = 1
    shockPos = .5
    tEnd = .2
    boundary = "edge"  # outflow
    initialLeft = np.array([1,0,0,0,1])  # primitive variables
    initialRight = np.array([.125,0,0,0,.1])  # primitive variables

variables = [startPos, endPos, shockPos, tEnd, boundary, initialLeft, initialRight]


# Initialise the solution array with initial conditions and primitive variables w, and return array with conserved variables
def initialise(config, N, g, start, end, shock, wL, wR):
    arr = np.zeros((N, len(wL)), dtype=np.float64)
    arr[:] = wR

    if config == "sedov" or "sq" in config:
        midpoint = (end+start)/2
        half_width = int(N/2 * ((shock-midpoint)/(end-midpoint)))
        arr[int(N/2-half_width):int(N/2+half_width)] = wL
    else:
        split_point = int(N * ((shock-start)/(end-start)))
        arr[:split_point] = wL
    
    if config == "sin":
        xi = np.linspace(start, end, N)
        arr[:, 0] = 1 + (.1 * np.sin(2*np.pi*xi))
    elif config == "shu-osher":
        xi = np.linspace(shock, end, N-split_point)
        arr[split_point:, 0] = 1 + (.2 * np.sin(freq*np.pi*xi))
    
    return fv.pointConvertPrimitive(arr, g)  # convert domain to conservative variables q