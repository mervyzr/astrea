import numpy as np

##############################################################################

# Make boundary conditions
def makeBoundary(tube, boundary, size=1):
    arr = np.copy(tube)
    return np.pad(arr, [(size,size), (0,0)], mode=boundary)


# Point-converting primitive variables w to conservative variables q
def pointConvertPrimitive(tube, g):
    arr = np.copy(tube)
    rhos, vecs, pressures = tube[:,0], tube[:,1:4], tube[:,4]
    arr[:,4] = (pressures/(g-1)) + (.5*rhos*np.linalg.norm(vecs, axis=1)**2)
    arr[:,1:4] = (vecs.T * rhos).T
    return arr


# Point-converting conservative variables q to primitive variables w
def pointConvertConservative(tube, g):
    arr = np.copy(tube)
    rhos, vecs, energies = tube[:,0], (tube[:,1:4].T / tube[:,0]).T, tube[:,4]
    arr[:,4] = (g-1) * (energies - (.5*rhos*np.linalg.norm(vecs, axis=1)**2))
    arr[:,1:4] = vecs
    return arr


# Converting cell-averaged primitive variables w to cell-averaged conservative variables q through a higher-order approx.
def convertPrimitive(tube, g, boundary, denominator=24):
    arr = makeBoundary(tube, boundary)
    w = tube - (np.diff(arr[1:], axis=0) - np.diff(arr[:-1], axis=0))/denominator  # 2nd-order Taylor expansion (Laplacian)
    q = pointConvertPrimitive(arr, g)
    return pointConvertPrimitive(w, g) + (np.diff(q[1:], axis=0) - np.diff(q[:-1], axis=0))/denominator
    

# Converting cell-averaged conservative variables q to cell-averaged primitive variables w through a higher-order approx.
def convertConservative(tube, g, boundary, denominator=24):
    arr = makeBoundary(tube, boundary)
    q = tube - (np.diff(arr[1:], axis=0) - np.diff(arr[:-1], axis=0))/denominator  # 2nd-order Taylor expansion (Laplacian)
    w = pointConvertConservative(arr, g)
    return pointConvertConservative(q, g) + (np.diff(w[1:], axis=0) - np.diff(w[:-1], axis=0))/denominator


# Make f_i based on initial conditions and primitive variables
def makeFlux(tube, g):
    rhos, vecs, pressures = tube[:,0], tube[:,1:4], tube[:,4]
    arr = np.zeros(tube.shape)

    arr[:,0] = rhos*vecs[:,0]
    arr[:,1] = rhos*(vecs[:,0]**2) + pressures
    arr[:,2] = rhos*vecs[:,0]*vecs[:,1]
    arr[:,3] = rhos*vecs[:,0]*vecs[:,2]
    arr[:,4] = vecs[:,0] * ((.5*rhos*np.linalg.norm(vecs, axis=1)**2) + ((g*pressures)/(g-1)))
    return arr


# Jacobian matrix using primitive variables
def makeJacobian(tube, g):
    rho, vx, pressure = tube[:,0], tube[:,1], tube[:,4]
    gridLength, variables = len(tube), len(tube[0])
    arr = np.zeros((gridLength, variables, variables))  # create empty square arrays for each cell
    i,j = np.diag_indices(variables)
    arr[:,i,j], arr[:,0,1], arr[:,1,4], arr[:,4,1] = vx[:,None], rho, 1/rho, g*pressure  # replace matrix with values
    return arr