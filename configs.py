import numpy as np

import functions as fn


# Initialise the solution array with initial conditions and primitive variables
def initialise(N, config, g, start, end, shock):
    if config == "sod":
        initialLeft = np.array([1,0,0,0,1])  # primitive variables
        initialRight = np.array([.125,0,0,0,.1])  # primitive variables

        cellsLeft = int(shock/(end-start) * N)
        arrL, arrR = np.tile(initialLeft, (cellsLeft, 1)), np.tile(initialRight, (N - cellsLeft, 1))  # Initialise 1D grid with initial conditions
        arr = np.concatenate((arrL, arrR)).astype(float)
        return fn.convertPrimitive(arr, g)  # convert domain to conservative variables q
    
    elif config == "sin":
        initialLeft = np.array([0,1,1,1,1])  # primitive variables
        initialRight = np.array([0,0,0,0,0])  # primitive variables

        cellsLeft = int(shock/(end-start) * N)
        arrL, arrR = np.tile(initialLeft, (cellsLeft, 1)), np.tile(initialRight, (N - cellsLeft, 1))  # Initialise 1D grid with initial conditions
        arr = np.concatenate((arrL, arrR)).astype(float)
        xi = np.linspace(start, end, N)
        arr[:, 0] = 1 + (.1 * np.sin(2*np.pi*xi))
        return fn.convertPrimitive(arr, g)  # convert domain to conservative variables q
    
    elif config == "sedov":
        initialLeft = np.array([1,0,0,0,100])  # primitive variables
        initialRight = np.array([1,0,0,0,1])  # primitive variables

        cellsLeft = int(shock/(end-0) * N/2)
        arrL, arrR = np.tile(initialLeft, (cellsLeft, 1)).astype(float), np.tile(initialRight, (int(N/2 - cellsLeft), 1)).astype(float)  # Initialise 1D grid with initial conditions
        arr = np.concatenate((arrR, arrL, arrL, arrR))
        return fn.convertPrimitive(arr, g)  # convert domain to conservative variables q
    
    else:
        initialLeft = np.array([1,0,0,0,1])  # primitive variables
        initialRight = np.array([.125,0,0,0,.1])  # primitive variables

        cellsLeft = int(shock/(end-start) * N)
        arrL, arrR = np.tile(initialLeft, (cellsLeft, 1)), np.tile(initialRight, (N - cellsLeft, 1))  # Initialise 1D grid with initial conditions
        arr = np.concatenate((arrL, arrR)).astype(float)
        return fn.convertPrimitive(arr, g)  # convert domain to conservative variables q