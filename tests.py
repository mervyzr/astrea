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
    
    else:
        cellsLeft = int(shock/(end-start) * N)
        arrL, arrR = np.tile(wL, (cellsLeft, 1)), np.tile(wR, (N - cellsLeft, 1))
        arr = np.concatenate((arrL, arrR)).astype(float)
        return fn.pointConvertPrimitive(arr, g)  # convert domain to conservative variables q