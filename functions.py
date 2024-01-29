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
def calculateSodAnalytical(tube, t, gamma, start, end, shock):
    # Define array to be updated and returned
    arr = np.zeros((len(tube), len(tube[0])))

    # Get variables of the leftmost and rightmost states, which should be initial conditions
    rho5, vx5, vy5, vz5, P5 = tube[0]
    rho1, vx1, vy1, vz1, P1 = tube[-1]

    # Define parameters needed for computation
    cs5, cs1 = np.sqrt(gamma * P5/rho5), np.sqrt(gamma * P1/rho1)
    mu = (gamma - 1)/(gamma + 1)

    # Root-finding value for pressure in region 2 (post-shock)
    f = lambda x: (((x/P1) - 1) * np.sqrt((1 - mu)/(gamma*(mu + (x/P1))))) - ((2/(gamma-1)) * (cs5/cs1) * (1-((x/P5)**((gamma-1)/(2*gamma)))))
    P2 = P3 = sp.optimize.fsolve(f, (P5-P1)/2)[0]

    # Define variables in other regions
    rho2, rho3 = rho1 * ((P2 + (mu*P1))/(P1 + (mu*P2))), rho5 * (P2/P5)**(1/gamma)
    vx2 = vx3 = ((2*cs5)/(gamma-1)) * (1-((P2/P5)**((gamma-1)/(2*gamma))))

    # Get shock wave speed and rarefaction tail speed
    v_t = cs5 - (vx2/(1-mu))
    v_s = vx2/(1-(rho1/rho2))

    # Define array of x-values for rarefaction wave
    rarefaction = np.linspace(shock-(cs5*t), shock-(v_t*t), int(((cs5*t-v_t*t)/(end-start)) * len(tube)))

    boundary54 = int(((shock-(cs5*t)-start)/(end-start)) * len(tube))
    boundary43 = int(((shock-(v_t*t)-start)/(end-start)) * len(tube))
    boundary32 = int(((shock+(vx2*t)-start)/(end-start)) * len(tube))
    boundary21 = int(((shock+(v_s*t)-start)/(end-start)) * len(tube))

    # Update array for regions 1 and 5 (initial conditions)
    arr[:boundary54] = [rho5, vx5, 0, 0, P5]
    arr[boundary21:] = [rho1, vx1, 0, 0, P1]
    
    # Update array for regions 2 and 3 (post-shock and discontinuities)
    arr[boundary43:boundary21] = [0, vx2, 0, 0, P2]
    arr[boundary43:boundary32, 0] = rho3
    arr[boundary32:boundary21, 0] = rho2

    # Update variables for region 4 (rarefaction wave)
    arr[boundary54:boundary43, 0] = rho5 * (((1-mu) - (mu*(rarefaction/(cs5*t))))**(2/(gamma-1)))
    arr[boundary54:boundary43, 4] = P5 * (((1-mu) - (mu*(rarefaction/(cs5*t))))**((2*gamma)/(gamma-1)))
    arr[boundary54:boundary43, 1] = (1-mu) * (cs5+(rarefaction/t))

    return arr


"""
# Determine the analytical solution for a Sod shock test
def calculateSodAnalytical(tube, t, gamma, start, end, shock):
    # Define array to be updated and returned
    arr = np.zeros((len(tube), len(tube[0])))

    # Get variables of the leftmost and rightmost states, which should be initial conditions
    rho1, vx1, vy1, vz1, P1 = tube[0]
    rho5, vx5, vy5, vz5, P5 = tube[-1]

    # Define parameters needed for computation
    cs1, cs5 = np.sqrt(gamma * P1/rho1), np.sqrt(gamma * P5/rho5)
    Gamma, beta = (gamma-1)/(gamma+1), (gamma-1)/(2*gamma)

    # Root-finding value for pressure in region 3/4 (post-shock)
    f = lambda x: ((P1**beta - x**beta) * np.sqrt(((1-Gamma**2) * P1**(1/gamma))/(rho1 * Gamma**2))) - ((x - P5) * np.sqrt((1-Gamma)/(rho5*(x + (Gamma*P5)))))
    P3 = P4 = sp.optimize.fsolve(f, (P5-P1)/2)[0]

    # Define variables in other regions
    rho3, rho4 = rho1 * ((P3/P1)**(1/gamma)), rho5 * ((P4 + (Gamma*P5))/(P5 + (Gamma*P4)))
    vx3 = vx4 = vx5 + ((P3-P5)/np.sqrt((rho5/2) * (P3*(gamma+1) + P5*(gamma-1))))

    # Define array of x-values for the rarefaction wave; variables dependent on x in this region
    rarefaction_region = np.linspace(shock-(cs1*t), shock, int((shock-(cs1*t)) * len(tube)))

    # Define variables in region 2 (rarefaction wave)
    vx2 = (2/(gamma+1)) * (cs1 + (rarefaction_region-shock/t))
    rho2 = rho1 * ((1 - ((vx2/cs1)*((gamma-1)/2)))**(2/(gamma-1)))
    P2 = P1 * ((1 - ((vx2/cs1)*((gamma-1)/2)))**(1/beta))

    region1_cells = ((cs1*t - start) * len(tube))
    region2_cells = (shock - (cs1*t)) * len(tube) + region1_cells
    region3_cells = 
    






    # Update array for regions 1 and 5 (initial conditions)
    arr[:cs5_bound+1] = [rho5, vx5, 0, 0, P5]
    arr[vs_bound+1:] = [rho1, vx1, 0, 0, P1]
    
    # Update array for regions 2 and 3 (post-shock and discontinuities)
    arr[vt_bound+1:vs_bound+1] = [0, vx2, 0, 0, P2]
    arr[vt_bound+1:v2_bound+1, 0] = rho3
    arr[v2_bound+1:vs_bound+1, 0] = rho2

    # Update array for region 4 (rarefaction)
    arr[cs5_bound+1:vt_bound+1, 0] = rho5 * ((-mu*(rarefaction_region/(cs5*t))) + (1-mu))**(2/(gamma-1))
    arr[cs5_bound+1:vt_bound+1, 4] = P5 * ((-mu*(rarefaction_region/(cs5*t))) + (1-mu))**((2*gamma)/(gamma-1))
    arr[cs5_bound+1:vt_bound+1, 1] = (1-mu) * (cs5 + (rarefaction_region/t))

    return arr
"""