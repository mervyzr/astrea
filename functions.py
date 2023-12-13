import numpy as np


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


# Function for solution error calculation for all variables
def calculateSolutionError(simulation, start, end):
    dx = abs(end-start)/len(simulation[0])
    return dx * np.sum(np.abs(simulation[0] - simulation[list(simulation.keys())[-1]]), axis=0)