import numpy as np
import scipy as sp

##############################################################################

# Colours for printing to terminal
class bcolours:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# Customised rounding function
def roundOff(value):
    if value%int(value) >= .5:
        return int(value) + 1
    else:
        return int(value)


# Lower the element in a list if string
def lowerList(lst):
    arr = []
    for i in lst:
        if isinstance(i, str):
            arr.append(i.lower())
        else:
            arr.append(i)
    return arr


# Print to Terminal
def printOutput(instanceTime, config, cells, solver, timestep, elapsed, runLength):
    print(f"[{bcolours.BOLD}{instanceTime}{bcolours.ENDC} | TEST={bcolours.OKGREEN}{config.upper()}{bcolours.ENDC}, CELLS={bcolours.OKGREEN}{cells}{bcolours.ENDC}, RECONSTRUCT={bcolours.OKGREEN}{solver.upper()}{bcolours.ENDC}, TIMESTEP={bcolours.OKGREEN}{timestep.upper()}{bcolours.ENDC}]  Elapsed: {bcolours.OKGREEN}{elapsed}s{bcolours.ENDC}  ({runLength})")
    pass


# Define the operator L as a function of the reconstruction values based on interpolation and limiters
def getL(fluxes, dx):
    return np.diff(fluxes, axis=0)/dx


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
    q_initial, q_final = simulation[0], simulation[max(list(simulation.keys()))]
    thermal_initial, thermal_final = q_initial[:,4]/q_initial[:,0], q_final[:,4]/q_final[:,0]
    q_initial, q_final = np.c_[q_initial, thermal_initial], np.c_[q_final, thermal_final]
    dx = abs(end-start)/len(q_initial)
    return dx * np.sum(np.abs(q_initial - q_final), axis=0)


# Determine the analytical solution for a Sod shock test
def calculateSodAnalytical(tube, t, gamma, start, end, shock):
    # Define array to be updated and returned
    x_arr = np.linspace(-5, 5, len(tube))
    arr = np.zeros((len(tube), len(tube[0])))

    # Get variables of the leftmost and rightmost states, which should be initial conditions
    rho5, vx5, vy5, vz5, P5 = tube[0]
    rho1, vx1, vy1, vz1, P1 = tube[-1]

    # Define parameters needed for computation
    cs5, cs1 = np.sqrt(gamma * P5/rho5), np.sqrt(gamma * P1/rho1)
    mu, beta = (gamma-1)/(gamma+1), 2/(gamma-1)

    # Root-finding value for pressure in region 2 (post-shock)
    f = lambda x: (((x/P1) - 1) * np.sqrt((1 - mu)/(gamma*(mu + (x/P1))))) - (beta * (cs5/cs1) * (1-((x/P5)**(1/(gamma*beta)))))
    P2 = P3 = sp.optimize.fsolve(f, (P5-P1)/2)[0]

    # Define variables in other regions
    rho2, rho3 = rho1 * ((P2 + (mu*P1))/(P1 + (mu*P2))), rho5 * (P2/P5)**(1/gamma)
    vx2 = vx3 = (beta*cs5) * (1-(P2/P5)**(1/(gamma*beta)))

    # Get shock wave speed and rarefaction tail speed
    v_t = cs5 - (vx2/(1-mu))
    v_s = vx2/(1-(rho1/rho2))

    # Define boundary regions and number of cells within each region
    boundary54 = roundOff(((shock-(cs5*t)-start)/(end-start)) * len(tube))
    boundary43 = roundOff(((shock-(v_t*t)-start)/(end-start)) * len(tube))
    boundary32 = roundOff(((shock+(vx2*t)-start)/(end-start)) * len(tube))
    boundary21 = roundOff(((shock+(v_s*t)-start)/(end-start)) * len(tube))
    
    # Define number of cells in the rarefaction wave
    rarefaction_cells = roundOff(((cs5*t-v_t*t)/(end-start)) * len(tube))
    if rarefaction_cells - (boundary43-boundary54) < 0:
        rarefaction_cells += 1
    elif rarefaction_cells - (boundary43-boundary54) > 0:
        rarefaction_cells -= 1
    rarefaction = np.linspace(shock-(cs5*t), shock-(v_t*t), rarefaction_cells) - shock

    # Update array for regions 1 and 5 (initial conditions)
    arr[:boundary54] = [rho5, vx5, 0, 0, P5]
    arr[boundary21:] = [rho1, vx1, 0, 0, P1]
    
    # Update array for regions 2 and 3 (post-shock and discontinuities)
    arr[boundary43:boundary21] = [0, vx2, 0, 0, P2]
    arr[boundary43:boundary32, 0] = rho3
    arr[boundary32:boundary21, 0] = rho2

    # Update array for region 4 (rarefaction wave)
    arr[boundary54:boundary43, 0] = rho5 * ((1 - mu) - mu*rarefaction/(cs5*t))**beta
    arr[boundary54:boundary43, 4] = P5 * ((1 - mu) - mu*rarefaction/(cs5*t))**(gamma*beta)
    arr[boundary54:boundary43, 1] = (1-mu) * (cs5+(rarefaction/t))

    return arr