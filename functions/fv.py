import numpy as np

##############################################################################

# Evolve the system in space by a standardised workflow
def evolveSpace(shockTube, tube):
    reconstructedValues = shockTube.interpolate(tube)
    solutionLefts, solutionRights = shockTube.applyLimiter(reconstructedValues, tube)
    return shockTube.calculateRiemannFlux(solutionLefts, solutionRights)


# Operator L as a function of the reconstruction values: [F(i+1/2) - F(i-1/2)]/dx
def getL(fluxes, dx):
    return -np.diff(fluxes, axis=0)/dx


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


# Converting cell-averaged primitive variables w to cell-averaged conservative variables q through a higher-order approx.
def convertPrimitive(tube, g, boundary, dem=24):
    wLs, wRs = makeBoundary(tube, boundary)
    w = tube - ((np.diff(wRs, axis=0) - np.diff(wLs, axis=0)) / dem)  # 2nd-order Taylor expansion (Laplacian)
    qLs, qRs = pointConvertPrimitive(wLs, g), pointConvertPrimitive(wRs, g)
    return pointConvertPrimitive(w, g) + ((np.diff(qRs, axis=0) - np.diff(qLs, axis=0)) / dem)
    

# Converting cell-averaged conservative variables q to cell-averaged primitive variables w through a higher-order approx.
def convertConservative(tube, g, boundary, dem=24):
    qLs, qRs = makeBoundary(tube, boundary)
    q = tube - ((np.diff(qRs, axis=0) - np.diff(qLs, axis=0)) / dem)  # 2nd-order Taylor expansion (Laplacian)
    wLs, wRs = pointConvertConservative(qLs, g), pointConvertConservative(qRs, g)
    return pointConvertConservative(q, g) + ((np.diff(wRs, axis=0) - np.diff(wLs, axis=0)) / dem)


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
    return np.c_[rhos*vecs[:,0], rhos*(vecs[:,0]**2) + pressures, rhos*vecs[:,0]*vecs[:,1], rhos*vecs[:,0]*vecs[:,2], vecs[:,0] * ((.5*rhos*np.linalg.norm(vecs, axis=1)**2) + ((g*pressures)/(g-1)))]


# Function that returns the coefficient of the slope flattener
def getSlopeCoeff(tube, boundary, g, slope_determinants=[.75, .85, .33]):
    z0, z1, delta = slope_determinants
    domain = pointConvertConservative(tube, g)
    arr, chi = np.ones(len(domain)), np.ones(len(domain))

    wLs, wRs = makeBoundary(domain, boundary)
    if boundary == "periodic":
        wL2s, wR2s = np.concatenate(([wLs[-2]],wLs)), np.concatenate((wRs,[wRs[1]]))
    else:
        wL2s, wR2s = np.concatenate(([wLs[0]],wLs)), np.concatenate((wRs,[wRs[-1]]))

    z = np.abs((wRs[:,4][1:] - wLs[:,4][:-1]) / (wR2s[:,4][2:] - wL2s[:,4][:-2]))  # define the linear function
    eta = np.minimum(np.ones(len(z)), np.maximum(np.zeros(len(z)), 1 - ((z-z0)/(z1-z0))))  # limit the range between 0 and 1
    criteria = (wLs[:,1][:-1] - wRs[:,1][1:] > 0) & (np.abs(wRs[:,4][1:] - wLs[:,4][:-1])/np.minimum(wRs[:,4][1:], wLs[:,4][:-1]) > delta)
    
    chi[criteria] = eta[criteria]
    chiL, chiR = makeBoundary(chi, boundary)

    signage = np.sign(wRs[:,4][1:] - wLs[:,4][:-1])
    arr[signage < 0] = np.minimum(chi, chiR[1:])[signage < 0]
    arr[signage > 0] = np.minimum(chi, chiL[:-1])[signage > 0]

    return np.tile(np.reshape(arr, (len(arr),1)), (1,5))