import numpy as np

import functions as fn
import settings as cfg

##############################################################################

if cfg.config == "sod":
    startPos = 0
    endPos = 1
    shockPos = .5
    tEnd = .2
    boundary = "outflow"

    initialLeft = np.array([1,0,0,0,1])  # primitive variables
    initialRight = np.array([.125,0,0,0,.1])  # primitive variables
elif cfg.config == "sin":
    startPos = 0
    endPos = 1
    shockPos = 1
    tEnd = 2
    boundary = "periodic"

    initialLeft = np.array([0,1,1,1,1])  # primitive variables
    initialRight = np.array([0,0,0,0,0])  # primitive variables
elif cfg.config == "sedov":
    startPos = -10
    endPos = 10
    shockPos = 1
    tEnd = .6
    boundary = "outflow"

    initialLeft = np.array([1,0,0,0,100])  # primitive variables
    initialRight = np.array([1,0,0,0,1])  # primitive variables
elif cfg.config == "shu-osher":
    startPos = 0
    endPos = 1
    shockPos = .125
    tEnd = .18
    boundary = "outflow"
    freq = 8

    initialLeft = np.array([3.857143,2.629369,0,0,10.3333])  # primitive variables
    initialRight = np.array([0,0,0,0,1])  # primitive variables
elif "toro" in cfg.config:
    startPos = 0
    endPos = 1
    boundary = "outflow"

    if "2" in cfg.config:
        shockPos = .5
        tEnd = .14

        initialLeft = np.array([1,-2,0,0,.4])  # primitive variables
        initialRight = np.array([1,2,0,0,.4])  # primitive variables
    elif "3" in cfg.config:
        shockPos = .5
        tEnd = .012

        initialLeft = np.array([1,0,0,0,1000])  # primitive variables
        initialRight = np.array([1,0,0,0,.01])  # primitive variables
    elif "4" in cfg.config:
        shockPos = .3
        tEnd = .05

        initialLeft = np.array([5.99924,19.5975,0,0,460.894])  # primitive variables
        initialRight = np.array([5.99242,-6.19633,0,0,46.095])  # primitive variables
    elif "5" in cfg.config:
        shockPos = .8
        tEnd = .012

        initialLeft = np.array([1,-19.59745,0,0,1000])  # primitive variables
        initialRight = np.array([1,-19.59745,0,0,.01])  # primitive variables
    else:
        shockPos = .3
        tEnd = .2

        initialLeft = np.array([1,.75,0,0,1])  # primitive variables
        initialRight = np.array([.125,0,0,0,.1])  # primitive variables
else:
    startPos = 0
    endPos = 1
    shockPos = .5
    tEnd = .2
    boundary = "outflow"

    initialLeft = np.array([1,0,0,0,1])  # primitive variables
    initialRight = np.array([.125,0,0,0,.1])  # primitive variables

variables = [startPos, endPos, shockPos, tEnd, boundary, initialLeft, initialRight]


# Initialise the solution array with initial conditions and primitive variables w, and return array with conserved variables
def initialise(config, N, g, start, end, shock, wL, wR):
    if config == "sod":
        cellsLeft = int(shock/(end-start) * N)
        arrL, arrR = np.tile(wL, (cellsLeft, 1)), np.tile(wR, (N - cellsLeft, 1))
        arr = np.concatenate((arrL, arrR)).astype(float)
        return fn.pointConvertPrimitive(arr, g)  # convert domain to conservative variables q
    
    elif config == "sin":
        cellsLeft = int(shock/(end-start) * N)
        arrL, arrR = np.tile(wL, (cellsLeft, 1)), np.tile(wR, (N - cellsLeft, 1))
        arr = np.concatenate((arrL, arrR)).astype(float)
        xi = np.linspace(start, end, N)
        arr[:, 0] = 1 + (.1 * np.sin(2*np.pi*xi))
        return fn.pointConvertPrimitive(arr, g)  # convert domain to conservative variables q
    
    elif config == "sedov":
        cellsLeft = int(shock/(end-0) * N/2)
        arrL, arrR = np.tile(wL, (cellsLeft, 1)), np.tile(wR, (int(N/2 - cellsLeft), 1))
        arr = np.concatenate((arrR, arrL, arrL, arrR)).astype(float)
        return fn.pointConvertPrimitive(arr, g)  # convert domain to conservative variables q
    
    elif config == "shu-osher":
        cellsLeft = int(shock/(end-start) * N)
        arrL, arrR = np.tile(wL, (cellsLeft, 1)), np.tile(wR, (N - cellsLeft, 1))
        arr = np.concatenate((arrL, arrR)).astype(float)
        xi = np.linspace(shock, end, N - cellsLeft)
        arr[cellsLeft:, 0] = 1 + (.2 * np.sin(freq*np.pi*xi))
        return fn.pointConvertPrimitive(arr, g)  # convert domain to conservative variables q
    
    elif "toro" in config:
        cellsLeft = int(shock/(end-start) * N)
        arrL, arrR = np.tile(wL, (cellsLeft, 1)), np.tile(wR, (N - cellsLeft, 1))
        arr = np.concatenate((arrL, arrR)).astype(float)
        return fn.pointConvertPrimitive(arr, g)  # convert domain to conservative variables q
    
    else:
        cellsLeft = int(shock/(end-start) * N)
        arrL, arrR = np.tile(wL, (cellsLeft, 1)), np.tile(wR, (N - cellsLeft, 1))
        arr = np.concatenate((arrL, arrR)).astype(float)
        return fn.pointConvertPrimitive(arr, g)  # convert domain to conservative variables q