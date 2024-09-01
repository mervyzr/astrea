import math

import numpy as np
import scipy as sp
from scipy.integrate import quad, simpson

from functions import fv, constructors

##############################################################################
# Functions for analytic solutions
##############################################################################

# Customised rounding function
def round_off(value):
    if value%int(value) >= .5:
        return int(value) + 1
    else:
        return int(value)


# Calculate scaled entropy density for an array [Derigs et al., 2015]
def calculate_entropy_density(grid, gamma):
    return (grid[...,0] * np.log(grid[...,4]*grid[...,0]**-gamma))/(gamma-1)


# Function for solution error calculation of sin-wave, sinc-wave and Gaussian tests
def calculate_solution_error(simulation, sim_variables, norm):
    dimension = math.ceil(sim_variables.dimension)

    time_keys = [float(t) for t in simulation.keys()]
    w_num = simulation[str(max(time_keys))]  # Get last instance of the grid with largest time key

    #w_theo = constructors.initialise(sim_variables, convert=False, N=len(w_num))

    xi = np.linspace(sim_variables.start_pos, sim_variables.end_pos, len(w_num))
    w_theo = np.copy(w_num)
    w_theo[:] = sim_variables.initial_left

    if sim_variables.config.startswith("gauss"):
        w_theo[...,0] = fv.gauss_func(xi, sim_variables.misc)
    else:
        if sim_variables.config == "sinc":
            w_theo[...,0] = fv.sinc_func(xi, sim_variables.misc)
        else:
            w_theo[...,0] = fv.sin_func(xi, sim_variables.misc)

    thermal_num, thermal_theo = fv.divide(w_num[...,4], w_num[...,0]), fv.divide(w_theo[...,4], w_theo[...,0])
    w_num, w_theo = np.concatenate((w_num, thermal_num[...,None]), axis=-1), np.concatenate((w_theo, thermal_theo[...,None]), axis=-1)

    if norm > 10:
        return np.max(np.abs(w_num-w_theo), axis=tuple(range(dimension)))
    elif norm <= 0:
        return np.sum(np.abs(w_num-w_theo), axis=tuple(range(dimension)))/len(w_num)
    else:
        return (np.sum(np.abs(w_num-w_theo)**norm, axis=tuple(range(dimension)))/len(w_num))**(1/norm)


# Function for calculation of total variation (TVD scheme if TV(t+1) < TV(t)); total variation tests for oscillations
def calculate_tv(simulation, sim_variables):
    dimension, tv = math.ceil(sim_variables.dimension), {}
    for t in list(simulation.keys()):
        grid = simulation[t]
        thermal = fv.divide(grid[...,4], grid[...,0])
        for i in range(dimension):
            grid = np.diff(grid, axis=i)
            thermal = np.diff(thermal, axis=i)
        tv[float(t)] = np.sum(np.abs(grid), axis=tuple(range(dimension)))
        tv[float(t)] = np.append(tv[float(t)], np.sum(np.abs(thermal)))
    return tv


# Function for checking the conservation equations; works with primitive variables but needs to be converted
def calculate_conservation(simulation, sim_variables):
    N, gamma, start_pos, end_pos = sim_variables.cells, sim_variables.gamma, sim_variables.start_pos, sim_variables.end_pos
    dimension, eq = math.ceil(sim_variables.dimension), {}

    for t in list(simulation.keys()):
        grid = fv.point_convert_primitive(simulation[t], sim_variables)
        for i in range(dimension)[::-1]:
            grid = simpson(grid, dx=(end_pos-start_pos)/N, axis=i) * (end_pos-start_pos)
        eq[float(t)] = grid
    return eq


# Function for checking the conservation equations at specific intervals; works with primitive variables but needs to be converted
# The reason is because at the boundaries, some values are lost to the ghost cells and not counted into the conservation plots
# This is the reason why there is a dip at exactly the halfway mark of the periodic smooth tests
def calculate_conservation_at_interval(simulation, sim_variables, interval=10):
    N, gamma, start_pos, end_pos, t_end = sim_variables.cells, sim_variables.gamma, sim_variables.start_pos, sim_variables.end_pos, sim_variables.t_end
    dimension, eq = math.ceil(sim_variables.dimension), {}

    intervals = np.array([], dtype=float)
    periods = np.linspace(0, t_end, interval)
    timings = np.asarray(list(simulation.keys()), dtype=float)
    for period in periods:
        intervals = np.append(intervals, timings[np.argmin(abs(timings-period))])

    for t in intervals:
        grid = fv.point_convert_primitive(simulation[str(t)], sim_variables)
        for i in range(dimension)[::-1]:
            grid = simpson(grid, dx=(end_pos-start_pos)/N, axis=i) * (end_pos-start_pos)
        eq[t] = grid
    return eq


# Determine the analytical solution for a Sod shock test, in 1D
def calculate_Sod_analytical(grid, t, sim_variables):
    gamma, start_pos, end_pos, shock_pos = sim_variables.gamma, sim_variables.start_pos, sim_variables.end_pos, sim_variables.shock_pos

    # Define array to be updated and returned
    arr = np.zeros_like(grid)

    # Get variables of the leftmost and rightmost states, which should be initial conditions
    rho5, vx5, vy5, vz5, P5, Bx5, By5, Bz5 = grid[0]
    rho1, vx1, vy1, vz1, P1, Bx1, By1, Bz1 = grid[-1]

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
    boundary_54 = round_off(((shock_pos-(cs5*t)-start_pos)/(end_pos-start_pos)) * len(grid))
    boundary_43 = round_off(((shock_pos-(v_t*t)-start_pos)/(end_pos-start_pos)) * len(grid))
    boundary_32 = round_off(((shock_pos+(vx2*t)-start_pos)/(end_pos-start_pos)) * len(grid))
    boundary_21 = round_off(((shock_pos+(v_s*t)-start_pos)/(end_pos-start_pos)) * len(grid))

    # Define number of cells in the rarefaction wave
    rarefaction_cells = round_off(((cs5*t-v_t*t)/(end_pos-start_pos)) * len(grid))
    if rarefaction_cells - (boundary_43-boundary_54) < 0:
        rarefaction_cells += 1
    elif rarefaction_cells - (boundary_43-boundary_54) > 0:
        rarefaction_cells -= 1
    rarefaction = np.linspace(shock_pos-(cs5*t), shock_pos-(v_t*t), rarefaction_cells) - shock_pos

    # Update array for regions 1 and 5 (initial conditions)
    arr[:boundary_54] = grid[0]
    arr[boundary_21:] = grid[-1]

    # Update array for regions 2 and 3 (post-shock and discontinuities)
    arr[boundary_43:boundary_21, 1] = vx2
    arr[boundary_43:boundary_21, 4] = P2
    arr[boundary_43:boundary_32, 0] = rho3
    arr[boundary_32:boundary_21, 0] = rho2

    # Update array for region 4 (rarefaction wave)
    arr[boundary_54:boundary_43, 0] = rho5 * ((1 - mu) - mu*rarefaction/(cs5*t))**beta
    arr[boundary_54:boundary_43, 4] = P5 * ((1 - mu) - mu*rarefaction/(cs5*t))**(gamma*beta)
    arr[boundary_54:boundary_43, 1] = (1-mu) * (cs5+(rarefaction/t))

    return arr