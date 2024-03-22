import numpy as np

import settings as cfg
from functions import generic

##############################################################################

config = cfg.config.lower()

if config == "sod":
    startPos = 0
    endPos = 1
    shockPos = .5
    tEnd = .2
    boundary = "edge"  # outflow
    freq = None
    initialLeft = np.array([1,0,0,0,1])  # primitive variables [rho, vx, vy, vz, P]
    initialRight = np.array([.125,0,0,0,.1])  # primitive variables [rho, vx, vy, vz, P]

elif config.startswith('sin'):
    startPos = 0
    endPos = 1
    shockPos = 1
    tEnd = 2
    boundary = "wrap"  # periodic
    freq = 2
    initialLeft = np.array([0,1,1,1,1])  # primitive variables [rho, vx, vy, vz, P]
    initialRight = np.array([0,0,0,0,0])  # primitive variables [rho, vx, vy, vz, P]

elif config == "sedov":
    startPos = -10
    endPos = 10
    shockPos = .1  # blast boundary
    tEnd = .6
    boundary = "edge"  # outflow
    freq = None
    initialLeft = np.array([1,0,0,0,100])  # primitive variables [rho, vx, vy, vz, P]
    initialRight = np.array([1,0,0,0,1])  # primitive variables [rho, vx, vy, vz, P]

elif "shu" in config or "osher" in config:
    startPos = -1
    endPos = 1
    shockPos = -.8
    tEnd = .47
    boundary = "edge"  # outflow
    freq = 5
    initialLeft = np.array([3.857143,2.629369,0,0,10.3333])  # primitive variables [rho, vx, vy, vz, P]
    initialRight = np.array([0,0,0,0,1])                    # primitive variables [rho, vx, vy, vz, P]

elif config.startswith('sq'):
    startPos = -1
    endPos = 1
    shockPos = 1/3
    tEnd = .05
    boundary = "wrap"  # periodic
    freq = None
    initialLeft = np.array([1,1,0,0,1])  # primitive variables [rho, vx, vy, vz, P]
    initialRight = np.array([.01,1,0,0,1])  # primitive variables [rho, vx, vy, vz, P]
    
elif "toro" in config:
    startPos = 0
    endPos = 1
    boundary = "edge"  # outflow
    freq = None

    if "2" in config:
        shockPos = .5
        tEnd = .14
        initialLeft = np.array([1,-2,0,0,.4])  # primitive variables [rho, vx, vy, vz, P]
        initialRight = np.array([1,2,0,0,.4])  # primitive variables [rho, vx, vy, vz, P]

    elif "3" in config:
        shockPos = .5
        tEnd = .012
        initialLeft = np.array([1,0,0,0,1000])  # primitive variables [rho, vx, vy, vz, P]
        initialRight = np.array([1,0,0,0,.01])  # primitive variables [rho, vx, vy, vz, P]

    elif "4" in config:
        shockPos = .3
        tEnd = .05
        initialLeft = np.array([5.99924,19.5975,0,0,460.894])  # primitive variables [rho, vx, vy, vz, P]
        initialRight = np.array([5.99242,-6.19633,0,0,46.095])  # primitive variables [rho, vx, vy, vz, P]

    elif "5" in config:
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
    freq = None
    initialLeft = np.array([1,0,0,0,1])  # primitive variables
    initialRight = np.array([.125,0,0,0,.1])  # primitive variables



variables = {
    'startPos': startPos,
    'endPos': endPos,
    'shockPos': shockPos,
    'tEnd': tEnd,
    'boundary': boundary.lower(),
    'freq': freq,
    'initialLeft': initialLeft,
    'initialRight': initialRight
}