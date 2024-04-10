import numpy as np

##############################################################################

# Initialise the discrete solution array with initial conditions and primitive variables w
# Returns the solution array in conserved variables q
def initialise(cfg, tst):
    N = cfg['cells']
    start, end, shock = tst['startPos'], tst['endPos'], tst['shockPos']

    arr = np.zeros((N, len(tst['initialRight'])), dtype=np.float64)
    arr[:] = tst['initialRight']

    midpoint = (end+start)/2
    if cfg['config'] == "sedov" or cfg['config'].startswith('sq'):
        half_width = int(N/2 * ((shock-midpoint)/(end-midpoint)))
        left_edge, right_edge = int(N/2-half_width), int(N/2+half_width)
        arr[left_edge:right_edge] = tst['initialLeft']
    else:
        split_point = int(N * ((shock-start)/(end-start)))
        arr[:split_point] = tst['initialLeft']

    if cfg['config'].startswith('sin'):
        xi = np.linspace(start, end, N)
        arr[:,0] = 1 + (.1 * np.sin(tst['freq']*np.pi*xi))
    elif cfg['config'].startswith('gauss'):
        xi = np.linspace(start, end, N)
        arr[:,0] = 1e-3 + (1-1e-3) * np.exp(-(xi-midpoint)**2/.01)
    elif "shu" in cfg['config'] or "osher" in cfg['config']:
        xi = np.linspace(shock, end, N-split_point)
        arr[split_point:,0] = 1 + (.2 * np.sin(tst['freq']*np.pi*xi))

    return pointConvertPrimitive(arr, cfg['gamma'])


# Make boundary conditions
def makeBoundary(tube, boundary, stencil=1):
    arr = np.copy(tube)
    return np.pad(arr, [(stencil,stencil), (0,0)], mode=boundary)


# Pointwise (exact) conversion of primitive variables w to conservative variables q (up to 2nd-order accurate)
def pointConvertPrimitive(tube, g):
    arr = np.copy(tube)
    rhos, vecs, pressures, Bfield = tube[:,0], tube[:,1:4], tube[:,4], tube[:,5:8]
    arr[:,4] = (pressures/(g-1)) + (.5*rhos*np.linalg.norm(vecs, axis=1)**2) + (.5*np.linalg.norm(Bfield, axis=1)**2)
    arr[:,1:4] = (vecs.T * rhos).T
    arr[:,5:8] = Bfield
    return arr


# Pointwise (exact) conversion of conservative variables q to primitive variables w (up to 2nd-order accurate)
def pointConvertConservative(tube, g):
    arr = np.copy(tube)
    rhos, vecs, energies, Bfield = tube[:,0], (tube[:,1:4].T / tube[:,0]).T, tube[:,4], tube[:,5:8]
    arr[:,4] = (g-1) * (energies - (.5*rhos*np.linalg.norm(vecs, axis=1)**2) - (.5*np.linalg.norm(Bfield, axis=1)**2))
    arr[:,1:4] = vecs
    arr[:,5:8] = Bfield
    return arr


# Converting (cell-/face-averaged) primitive variables w to conservative variables q through a higher-order approx.
def convertPrimitive(tube, g, solver, boundary):
    if solver in ["ppm", "parabolic", "p"]:
        arr = makeBoundary(tube, boundary)
        w = tube - (np.diff(arr[1:], axis=0) - np.diff(arr[:-1], axis=0))/24  # 2nd-order Taylor expansion (Laplacian)
        q = pointConvertPrimitive(arr, g)
        return pointConvertPrimitive(w, g) + (np.diff(q[1:], axis=0) - np.diff(q[:-1], axis=0))/24
    else:
        return pointConvertPrimitive(tube, g)


# Converting (cell-/face-averaged) conservative variables q to primitive variables w through a higher-order approx.
def convertConservative(tube, g, solver, boundary):
    if solver in ["ppm", "parabolic", "p"]:
        arr = makeBoundary(tube, boundary)
        q = tube - (np.diff(arr[1:], axis=0) - np.diff(arr[:-1], axis=0))/24  # 2nd-order Taylor expansion (Laplacian)
        w = pointConvertConservative(arr, g)
        return pointConvertConservative(q, g) + (np.diff(w[1:], axis=0) - np.diff(w[:-1], axis=0))/24
    else:
        return pointConvertConservative(tube, g)


# Make flux based on cell-averaged (primitive) variables
def makeFlux(tube, g):
    rhos, vecs, pressures, Bfield = tube[:,0], tube[:,1:4], tube[:,4], tube[:,5:8]
    arr = np.zeros(tube.shape)

    arr[:,0] = rhos*vecs[:,0]
    arr[:,1] = rhos*(vecs[:,0]**2) + pressures + (.5*np.linalg.norm(Bfield, axis=1)**2) - Bfield[:,0]**2
    arr[:,2] = rhos*vecs[:,0]*vecs[:,1] - Bfield[:,0]*Bfield[:,1]
    arr[:,3] = rhos*vecs[:,0]*vecs[:,2] - Bfield[:,0]*Bfield[:,2]
    arr[:,4] = (vecs[:,0]*((.5*rhos*np.linalg.norm(vecs, axis=1)**2) + ((g*pressures)/(g-1)))) + (vecs[:,0]*np.linalg.norm(Bfield, axis=1)**2) - (Bfield[:,0]*np.sum(Bfield*vecs, axis=1))
    arr[:,6] = Bfield[:,1]*vecs[:,0] - Bfield[:,0]*vecs[:,1]
    arr[:,7] = Bfield[:,2]*vecs[:,0] - Bfield[:,0]*vecs[:,2]
    return arr


# Jacobian matrix based on primitive variables
def makeJacobian(tube, g):
    rho, vx, pressure, Bfield = tube[:,0], tube[:,1], tube[:,4], tube[:,5:8]
    gridLength, variables = len(tube), len(tube[0])

    # Create empty square arrays for each cell
    arr = np.zeros((gridLength, variables, variables))
    i,j = np.diag_indices(variables)

    # Replace matrix with values
    arr[:,i,j] = vx[:,None]  # diagonal elements
    arr[:,0,1] = rho
    arr[:,1,4] = 1/rho
    arr[:,4,1] = g*pressure

    arr[:,1,6] = Bfield[:,1]/rho
    arr[:,1,7] = Bfield[:,2]/rho
    arr[:,2,6] = -Bfield[:,0]/rho
    arr[:,3,7] = -Bfield[:,0]/rho
    arr[:,6,1] = Bfield[:,1]
    arr[:,6,2] = -Bfield[:,0]
    arr[:,7,1] = Bfield[:,2]
    arr[:,7,3] = -Bfield[:,0]
    return arr