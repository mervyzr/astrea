import numpy as np
import scipy as sp


##############################################################################

# Make boundary conditions
def makeBoundary(tube, boundary):
    if boundary == "periodic":
        # Use periodic boundary for ghost boxes
        return np.concatenate(([tube[-1]],tube)), np.concatenate((tube,[tube[0]]))
    else:
        # Use outflow boundary for ghost boxes
        return np.concatenate(([tube[0]],tube)), np.concatenate((tube,[tube[-1]]))


# Point-converting primitive variables w to conservative variables q
def pointConvertPrimitive(tube, g):
    rhos, vecs, pressures = tube[:,0], tube[:,1:4], tube[:,4]
    energies = (pressures/(g-1)) + (.5*rhos*np.linalg.norm(vecs, axis=1)**2)
    return np.c_[rhos, np.multiply(vecs, rhos[:, np.newaxis]), energies]


# Point-converting conservative variables q to primitive variables w
def pointConvertConservative(tube, g):
    rhos, vecs, energies = tube[:,0], np.divide(tube[:,1:4], tube[:,0][:, np.newaxis]), tube[:,4]
    pressures = (g-1) * (energies - (.5*rhos*np.linalg.norm(vecs, axis=1)**2))
    return np.c_[rhos, vecs, pressures]


# Converting primitive variables w to conservative variables q through a higher-order approx.
def convertPrimitive(tube, g, boundary):
    wLs, wRs = makeBoundary(tube, boundary)
    wLs, wRs = wLs[:-1], wRs[1:]

    q = pointConvertPrimitive(tube, g)
    qLs, qRs = makeBoundary(q, boundary)
    qLs, qRs = qLs[:-1], qRs[1:]

    w = tube - ((wLs - (2*tube) + wRs) / 24)  # 2nd-order Taylor expansion (Laplacian)
    return pointConvertPrimitive(w, g) + ((qLs - (2*q) + qRs) / 24)
    

# Converting conservative variables q to primitive variables w through a higher-order approx.
def convertConservative(tube, g, boundary):
    qLs, qRs = makeBoundary(tube, boundary)
    qLs, qRs = qLs[:-1], qRs[1:]

    w = pointConvertConservative(tube, g)
    wLs, wRs = makeBoundary(w, boundary)
    wLs, wRs = wLs[:-1], wRs[1:]

    q = tube - ((qLs - (2*tube) + qRs) / 24)  # 2nd-order Taylor expansion (Laplacian)
    return pointConvertConservative(q, g) + ((wLs - (2*w) + wRs) / 24)


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










# Determine the analytical solution for a Sod shock test
def calculateSodAnalytical(tube, t, gamma, wL, wR, start, end, tolerance=1e-3):
    wL, wR = tube[0], tube[-1]
    cs1, cs5 = np.sqrt(gamma * (wL[4]/wL[0])), np.sqrt(gamma * (wR[4]/wR[0]))
    mu2 = (gamma - 1)/(gamma + 1)
    
    
    

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