import numpy as np
import scipy as sp
from scipy.integrate import quad, simpson

from functions import fv

##############################################################################

# Customised rounding function
def roundOff(value):
    if value%int(value) >= .5:
        return int(value) + 1
    else:
        return int(value)


# Calculate scaled entropy density for an array [Derigs et al., 2015]
def calculateEntropyDensity(tube, gamma):
    return (tube[...,0] * np.log(tube[...,4]*tube[...,0]**-gamma))/(gamma-1)


# Function for solution error calculation of sin-wave, sinc-wave and Gaussian tests
def calculateSolutionError(simulation, simVariables, norm):
    timeKeys = [float(t) for t in simulation.keys()]
    w_num = simulation[str(max(timeKeys))]  # Get last array with (typically largest) time key

    xi = np.linspace(simVariables.startPos, simVariables.endPos, len(w_num))
    w_theo = np.copy(w_num)
    w_theo[:] = simVariables.initialLeft

    if simVariables.config.startswith("gauss"):
        w_theo[...,0] = fv.gauss_func(xi, simVariables.misc)
    else:
        if simVariables.config == "sinc":
            w_theo[...,0] = fv.sinc_func(xi, simVariables.misc)
        else:
            w_theo[...,0] = fv.sin_func(xi, simVariables.misc)

    thermal_num, thermal_theo = fv.divide(w_num[...,4], w_num[...,0]), fv.divide(w_theo[...,4], w_theo[...,0])
    w_num, w_theo = np.concatenate((w_num, thermal_num[...,None]), axis=-1), np.concatenate((w_theo, thermal_theo[...,None]), axis=-1)

    if norm > 10:
        return np.max(np.abs(w_num-w_theo), axis=tuple(range(simVariables.dim)))
    elif norm <= 0:
        return np.sum(np.abs(w_num-w_theo), axis=tuple(range(simVariables.dim)))/(simVariables.cells**simVariables.dim)
    else:
        return (np.sum(np.abs(w_num-w_theo)**norm, axis=tuple(range(simVariables.dim)))/(simVariables.cells**simVariables.dim))**(1/norm)


# Function for calculation of total variation (TVD scheme if TV(t+1) < TV(t)); total variation tests for oscillations
def calculateTV(simulation, simVariables):
    tv = {}
    for t in list(simulation.keys()):
        domain = simulation[t]
        thermal = fv.divide(domain[...,4], domain[...,0])
        for i in range(simVariables.dim):
            domain = np.diff(domain, axis=i)
            thermal = np.diff(thermal, axis=i)
        tv[float(t)] = np.sum(np.abs(domain), axis=tuple(range(simVariables.dim)))
        tv[float(t)] = np.append(tv[float(t)], np.sum(np.abs(thermal)))
    return tv


# Function for checking the conservation equations; works with primitive variables but needs to be converted
def calculateConservation(simulation, simVariables):
    N, gamma, dim, startPos, endPos = simVariables.cells, simVariables.gamma, simVariables.dim, simVariables.startPos, simVariables.endPos
    eq = {}

    for t in list(simulation.keys()):
        domain = fv.pointConvertPrimitive(simulation[t], gamma)
        for i in reversed(range(dim)):
            domain = simpson(domain, dx=(endPos-startPos)/N, axis=i) * (endPos-startPos)
        eq[float(t)] = domain
    return eq


# Function for checking the conservation equations at specific intervals; works with primitive variables but needs to be converted
# The reason is because at the boundaries, some values are lost to the ghost cells and not counted into the conservation plots
# This is the reason why there is a dip at exactly the halfway mark of the periodic smooth tests
def calculateConservationAtInterval(simulation, simVariables, interval=10):
    N, gamma, dim, startPos, endPos, tEnd = simVariables.cells, simVariables.gamma, simVariables.dim, simVariables.startPos, simVariables.endPos, simVariables.tEnd
    eq = {}

    intervals = np.array([], dtype=float)
    periods = np.linspace(0, tEnd, interval)
    timings = np.asarray(list(simulation.keys()), dtype=float)
    for period in periods:
        intervals = np.append(intervals, timings[np.argmin(abs(timings-period))])

    for t in intervals:
        domain = fv.pointConvertPrimitive(simulation[str(t)], gamma)
        for i in reversed(range(dim)):
            domain = simpson(domain, dx=(endPos-startPos)/N, axis=i) * (endPos-startPos)
        eq[t] = domain
    return eq


# Determine the analytical solution for a Sod shock test, in 1D
def calculateSodAnalytical(tube, t, simVariables):
    gamma, startPos, endPos, shockPos = simVariables.gamma, simVariables.startPos, simVariables.endPos, simVariables.shockPos

    # Define array to be updated and returned
    arr = np.zeros_like(tube)

    # Get variables of the leftmost and rightmost states, which should be initial conditions
    rho5, vx5, vy5, vz5, P5, Bx5, By5, Bz5 = tube[0]
    rho1, vx1, vy1, vz1, P1, Bx1, By1, Bz1 = tube[-1]

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
    boundary54 = roundOff(((shockPos-(cs5*t)-startPos)/(endPos-startPos)) * len(tube))
    boundary43 = roundOff(((shockPos-(v_t*t)-startPos)/(endPos-startPos)) * len(tube))
    boundary32 = roundOff(((shockPos+(vx2*t)-startPos)/(endPos-startPos)) * len(tube))
    boundary21 = roundOff(((shockPos+(v_s*t)-startPos)/(endPos-startPos)) * len(tube))

    # Define number of cells in the rarefaction wave
    rarefaction_cells = roundOff(((cs5*t-v_t*t)/(endPos-startPos)) * len(tube))
    if rarefaction_cells - (boundary43-boundary54) < 0:
        rarefaction_cells += 1
    elif rarefaction_cells - (boundary43-boundary54) > 0:
        rarefaction_cells -= 1
    rarefaction = np.linspace(shockPos-(cs5*t), shockPos-(v_t*t), rarefaction_cells) - shockPos

    # Update array for regions 1 and 5 (initial conditions)
    arr[:boundary54] = tube[0]
    arr[boundary21:] = tube[-1]

    # Update array for regions 2 and 3 (post-shock and discontinuities)
    arr[boundary43:boundary21, 1] = vx2
    arr[boundary43:boundary21, 4] = P2
    arr[boundary43:boundary32, 0] = rho3
    arr[boundary32:boundary21, 0] = rho2

    # Update array for region 4 (rarefaction wave)
    arr[boundary54:boundary43, 0] = rho5 * ((1 - mu) - mu*rarefaction/(cs5*t))**beta
    arr[boundary54:boundary43, 4] = P5 * ((1 - mu) - mu*rarefaction/(cs5*t))**(gamma*beta)
    arr[boundary54:boundary43, 1] = (1-mu) * (cs5+(rarefaction/t))

    return arr