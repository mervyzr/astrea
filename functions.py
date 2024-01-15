import numpy as np
import scipy as sp


##############################################################################

# Converting primitive variables w to conservative variables q
def convertPrimitive(tube, g):
    rhos, vecs, pressures = tube[:,0], tube[:,1:4], tube[:,4]
    energies = (pressures/(g-1)) + (.5*rhos*np.linalg.norm(vecs, axis=1)**2)
    return np.c_[rhos, np.multiply(vecs, rhos[:, np.newaxis]), energies]


# Converting conservative variables q to primitive variables w
def convertConservative(tube, g):
    rhos, vecs, energies = tube[:,0], np.divide(tube[:,1:4], tube[:,0][:, np.newaxis]), tube[:,4]
    pressures = (g-1) * (energies - (.5*rhos*np.linalg.norm(vecs, axis=1)**2))
    return np.c_[rhos, vecs, pressures]


# Jacobian matrix using primitive variables
def makeJacobian(tube, g):
    rho, vx, pressure = tube[:,0], tube[:,1], tube[:,4]
    gridLength, variables = len(tube), len(tube[0])
    arr = np.zeros((gridLength, variables, variables))  # create empty square arrays for each cell
    i,j = np.diag_indices(variables)
    arr[:,i,j], arr[:,0,1], arr[:,1,4], arr[:,4,1] = vx[:,None], rho, 1/rho, g*pressure  # replace matrix with values
    return arr


# Make f_i based on initial conditions and primitive variables
def makeFlux(tube, g):
    rhos, vecs, pressures = tube[:,0], tube[:,1:4], tube[:,4]
    return np.c_[rhos*vecs[:,0], rhos*(vecs[:,0]**2) + pressures, rhos*vecs[:,0]*vecs[:,1], rhos*vecs[:,0]*vecs[:,2],\
                    vecs[:,0] * ((.5*rhos*np.linalg.norm(vecs, axis=1)**2) + ((g*pressures)/(g-1)))]


# Function for solution error calculation for all variables
def calculateSolutionError(simulation, start, end):
    dx = abs(end-start)/len(simulation[0])
    return dx * np.sum(np.abs(simulation[0] - simulation[list(simulation.keys())[-1]]), axis=0)


# Initialise the solution array with initial conditions and primitive variables w, and return array with conserved variables
def initialise(N, config, g, wL, wR, start, end, shock):
    if config == "sod":
        cellsLeft = int(shock/(end-start) * N)
        arrL, arrR = np.tile(wL, (cellsLeft, 1)), np.tile(wR, (N - cellsLeft, 1))
        arr = np.concatenate((arrL, arrR)).astype(float)
        return convertPrimitive(arr, g)  # convert domain to conservative variables q
    
    elif config == "sin":
        cellsLeft = int(shock/(end-start) * N)
        arrL, arrR = np.tile(wL, (cellsLeft, 1)), np.tile(wR, (N - cellsLeft, 1))
        arr = np.concatenate((arrL, arrR)).astype(float)
        xi = np.linspace(start, end, N)
        arr[:, 0] = 1 + (.1 * np.sin(2*np.pi*xi))
        return convertPrimitive(arr, g)  # convert domain to conservative variables q
    
    elif config == "sedov":
        cellsLeft = int(shock/(end-0) * N/2)
        arrL, arrR = np.tile(wL, (cellsLeft, 1)), np.tile(wR, (int(N/2 - cellsLeft), 1))
        arr = np.concatenate((arrR, arrL, arrL, arrR)).astype(float)
        return convertPrimitive(arr, g)  # convert domain to conservative variables q
    
    else:
        cellsLeft = int(shock/(end-start) * N)
        arrL, arrR = np.tile(wL, (cellsLeft, 1)), np.tile(wR, (N - cellsLeft, 1))
        arr = np.concatenate((arrL, arrR)).astype(float)
        return convertPrimitive(arr, g)  # convert domain to conservative variables q










# Determine the analytical solution for a Sod shock test
def analyticalSod(tube,t,  gamma, wL, wR, start, end, tolerance=1e-3):
    cs1, cs2 = np.sqrt(gamma*(wL[4]/wL[0])), np.sqrt(gamma*(wR[4]/wR[0]))
    Gamma, beta = (gamma-1)/(gamma+1), (gamma-1)/(2*gamma)
    arr = np.zeros((len(tube), len(tube[0])))

    # locate the regions of the tube based on density
    differences = np.abs(np.diff(tube[:,0]))
    peaks = sp.signal.find_peaks(differences, height=tolerance)[0]+1

    # Region 1

    # Region 2
    u2 = 2/(gamma+1) * (cs1 + 1/t)
    
    if len(peaks) <= 4:
        arr[:peaks[0]] = wL
        arr[peaks[3]:] = wR
    else:
        pass