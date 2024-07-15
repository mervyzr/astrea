import numpy as np
import scipy as sp
from scipy.integrate import odeint, quad, solve_ivp, simpson

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
    return (tube[:,0] * np.log(tube[:,4]*tube[:,0]**-gamma))/(gamma-1)


# Function for solution error calculation of sin-wave, sinc-wave and Gaussian tests
def calculateSolutionError(simulation, simVariables, norm):
    config, startPos, endPos, params = simVariables.config, simVariables.startPos, simVariables.endPos, simVariables.misc

    timeKeys = [float(t) for t in simulation.keys()]
    q_num = simulation[str(max(timeKeys))]  # Get last array with (typically largest) time key

    xi = np.linspace(startPos, endPos, len(q_num))
    q_theo = np.copy(q_num)
    q_theo[:] = simVariables.initialLeft

    if config.startswith("gauss"):
        q_theo[:,0] = fv.gauss_func(xi, params)
    else:
        if config == "sinc":
            q_theo[:,0] = fv.sinc_func(xi, params)
        else:
            q_theo[:,0] = fv.sin_func(xi, params)

    thermal_num, thermal_theo = q_num[:,4]/q_num[:,0], q_theo[:,4]/q_theo[:,0]
    q_num, q_theo = np.c_[q_num, thermal_num], np.c_[q_theo, thermal_theo]

    if norm > 10:
        return np.max(np.abs(q_num-q_theo), axis=0)
    elif norm <= 0:
        return np.sum(np.abs(q_num-q_theo), axis=0)/len(q_num)
    else:
        return (np.sum(np.abs(q_num-q_theo)**norm, axis=0)/len(q_num))**(1/norm)


# Function for calculation of total variation (TVD scheme if TV(t+1) < TV(t)); total variation tests for oscillations
def calculateTV(simulation):
    tv = {}
    for t in list(simulation.keys()):
        domain = simulation[t]
        tv[float(t)] = np.sum(np.abs(np.diff(domain, axis=0)), axis=0)
        tv[float(t)] = np.append(tv[float(t)], np.sum(np.abs(np.diff(domain[:, 4]/domain[:, 0]))))
    return tv


# Function for checking the conservation equations; works with primitive variables
def calculateConservation(simulation, simVariables):
    gamma, startPos, endPos = simVariables.gamma, simVariables.startPos, simVariables.endPos
    eq = {}

    for t in list(simulation.keys()):
        domain = fv.pointConvertPrimitive(simulation[t], gamma)
        eq[float(t)] = simpson(domain, dx=(endPos-startPos)/len(domain), axis=0) * (endPos-startPos)
    return eq


# Function for checking the conservation equations at specific intervals; works with primitive variables
# The reason is because at the boundaries, some values are lost to the ghost cells and not counted into the conservation plots
# This is the reason why there is a dip at exactly the halfway mark of the periodic smooth tests
def calculateConservationAtInterval(simulation, simVariables):
    gamma, startPos, endPos = simVariables.gamma, simVariables.startPos, simVariables.endPos
    eq = {}

    intervals = np.array([], dtype=float)
    periods = np.arange(11)
    timings = np.asarray(list(simulation.keys()), dtype=float)
    for period in periods:
        intervals = np.append(intervals, timings[np.argmin(abs(timings-period))])

    for t in intervals:
        domain = fv.pointConvertPrimitive(simulation[str(t)], gamma)
        eq[t] = simpson(domain, dx=(endPos-startPos)/len(domain), axis=0) * (endPos-startPos)
    return eq


# Determine the analytical solution for a Sod shock test
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