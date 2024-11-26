import scipy
import numpy as np
import scipy.integrate
import scipy.optimize

from functions import constructor, fv

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


# Function for solution error calculation of sine-wave and Gaussian tests
def calculate_solution_error(simulation, sim_variables, norm):
    dimension = sim_variables.dimension

    time_keys = [float(t) for t in simulation.keys()]
    w_num = simulation[str(max(time_keys))]  # Get last instance of the grid with largest time key

    # Create theoretical array
    normalising_factor = 1/(len(w_num) ** dimension)
    sim_variables = sim_variables._replace(cells=len(w_num))
    w_theo = constructor.initialise(sim_variables)

    thermal_num, thermal_theo = fv.divide(w_num[...,4], w_num[...,0]), fv.divide(w_theo[...,4], w_theo[...,0])
    w_num, w_theo = np.concatenate((w_num, thermal_num[...,None]), axis=-1), np.concatenate((w_theo, thermal_theo[...,None]), axis=-1)

    if norm > 10:
        return np.max(np.abs(w_num-w_theo), axis=tuple(range(dimension)))
    elif norm <= 0:
        return normalising_factor * np.sum(np.abs(w_num-w_theo), axis=tuple(range(dimension)))
    else:
        return (normalising_factor * np.sum(np.abs(w_num-w_theo)**norm, axis=tuple(range(dimension))))**(1/norm)


# Function for calculation of total variation (TVD scheme if TV(t+1) < TV(t)); total variation tests for oscillations
def calculate_tv(simulation, sim_variables):
    dimension, tv = sim_variables.dimension, {}

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
    N, subgrid, start_pos, end_pos = sim_variables.cells, sim_variables.subgrid, sim_variables.start_pos, sim_variables.end_pos
    dimension, eq = sim_variables.dimension, {}

    if subgrid.startswith("w") or subgrid in ["ppm", "parabolic", "p"]:
        convert = fv.high_order_convert_primitive
    else:
        convert = fv.point_convert_primitive

    for t in list(simulation.keys()):
        grid = convert(simulation[t][:], sim_variables)
        for i in range(dimension)[::-1]:
            grid = scipy.integrate.simpson(grid, dx=(end_pos-start_pos)/N, axis=i) * (end_pos-start_pos)
        eq[float(t)] = grid
    return eq


# Function for checking the conservation equations at specific intervals; works with primitive variables but needs to be converted
# The reason is because at the boundaries, some values are lost to the ghost cells and not counted into the conservation plots
# This is the reason why there is a dip at exactly the halfway mark of the periodic smooth tests
def calculate_conservation_at_interval(simulation, sim_variables, interval=10):
    N, subgrid, start_pos, end_pos, t_end = sim_variables.cells, sim_variables.subgrid, sim_variables.start_pos, sim_variables.end_pos, sim_variables.t_end
    dimension, eq = sim_variables.dimension, {}

    if subgrid.startswith("w") or subgrid in ["ppm", "parabolic", "p"]:
        convert = fv.high_order_convert_primitive
    else:
        convert = fv.point_convert_primitive

    intervals = np.array([], dtype=float)
    periods = np.linspace(0, t_end, interval)
    timings = np.asarray(list(simulation.keys()), dtype=float)
    for period in periods:
        intervals = np.append(intervals, timings[np.argmin(abs(timings-period))])

    for t in intervals:
        grid = convert(simulation[str(t)][:], sim_variables)
        for i in range(dimension)[::-1]:
            grid = scipy.integrate.simpson(grid, dx=(end_pos-start_pos)/N, axis=i) * (end_pos-start_pos)
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
    P2 = P3 = scipy.optimize.fsolve(f, (P5-P1)/2)[0]

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


# Determine the analytical solution for a Sedov blast wave, in 1D [Kamm & Timmes, 2000]
def calculate_Sedov_analytical(grid, t, sim_variables, omg=0):
    gamma, j = sim_variables.gamma, sim_variables.dimension
    rho0, v0, P0, B0 = sim_variables.initial_left[0], sim_variables.initial_left[1:4], sim_variables.initial_left[4], sim_variables.initial_left[5:8]
    E_blast = (P0/(gamma-1)) + .5*(rho0*fv.norm(v0)**2 + fv.norm(B0)**2)
    eps = 1e-4
    ex = j + 2 - omg

    arr = np.zeros_like(grid)

    # Determine family type
    V2 = 4/ex
    Vstar = 2/(j*(gamma-1)+2)

    # Note the singularities
    omg2 = (2*(gamma-1) + j)/gamma
    omg3 = j * (2-gamma)

    # Define post-shock values from centre of symmetry
    r2 = lambda _alpha: ((E_blast*t**2)/(_alpha*rho0))**(1/ex)
    u2 = lambda _alpha: ((4*r2(_alpha))/(ex*t))/(gamma+1)
    rho2 = rho0 * (gamma+1)/(gamma-1)
    P2 = lambda _alpha: (2*rho0*((2*r2(_alpha))/(ex*t))**2)/(gamma+1)

    # Form the exponents and frequently used variables
    alp0 = 2/ex
    alp2 = -(gamma-1)/(gamma*(omg2-omg))
    alp1 = ((ex*gamma)/(2+j*(gamma-1))) * ((2*(j*(2-gamma)-omg))/(gamma*ex**2) - alp2)
    alp3 = (j-omg)/(gamma*(omg2-omg))
    alp4 = alp1 * ((ex*(j-omg))/(omg3-omg))
    alp5 = (omg*(1+gamma)-2*j)/(omg3-omg)

    a = .25 * ex * (gamma+1)
    b = (gamma+1)/(gamma-1)
    c = .5 * gamma * ex
    d = ((gamma+1)*ex)/((gamma+1)*ex - 2*(2+j*(gamma-1)))
    e = .5 * (2 + j*(gamma-1))

    # Define the auxiliary functions
    x1 = lambda V: a*V
    x2 = lambda V: b * (c*V - 1)
    x3 = lambda V: d * (1 - e*V)
    x4 = lambda V: b * (1 - (c*V)/gamma)

    if abs(V2-Vstar) <= eps:
        _lambda = lambda r: r/r2
        _dlambda = 0
        _f = lambda r: _lambda(r)
        _g = lambda r: _lambda(r)**(j-2)
        _h = lambda r: _lambda(r)**j

        J2 = (gamma+1)/(j*(j*(gamma-1)+2)**2)
        J1 = (2*J2)/(gamma-1)
        alpha = J2 * np.pi * 2**(j-1)
    else:
        if abs(omg-omg2) <= eps:
            factor = lambda V: (1-x1(V))/(x1(V)-(gamma+1)/(2*gamma))
            _lambda = lambda V: x1(V)**-alp0 * x2(V)**((gamma-1)/(2*e)) * np.exp(factor * (gamma+1)/(2*e))
            _dlambda = lambda V: -_lambda(V) * (a*alp0/x1(V) + b*c*(gamma-1)/(2*e*x2(V)) - (a*(gamma+1)/(2*e))*(factor/(1-x1(V)))*(1+factor))
            _f = lambda V: x1(V) * _lambda(V)
            _g = lambda V: x1(V)**(alp0*omg) * x2(V)**(4-j-(2*gamma)/(2*e)) * x4(V)**alp5 * np.exp(factor * (gamma+1)/e)
            _h = lambda V: x1(V)**(alp0*omg) * x3(V)**(-j*gamma/(2*e)) * x4(V)**(1+alp5)
        elif abs(omg-omg3) <= eps:
            factor = lambda V: np.exp(-(j*gamma*(gamma+1)*(1-x1(V)))/(2*e*(.5*(gamma+1)-x1(V))))
            _lambda = lambda V: x1(V)**-alp0 * x2(V)**-alp2 * x4(V)**-alp1
            _dlambda = lambda V: -_lambda(V) * (a*alp0/x1(V) + b*c*alp2/x2(V) - b*c*alp1/(gamma*x4(V)))
            _f = lambda V: x1(V) * _lambda(V)
            _g = lambda V: x1(V)**(alp0*omg) * x2(V)**(alp3+alp2*omg) * x4(V)**(1-2/e) * factor
            _h = lambda V: x1(V)**(alp0*omg) * x4(V)**((j*(gamma-1)-gamma)/e) * factor
        else:
            _lambda = lambda V: x1(V)**-alp0 * x2(V)**-alp2 * x3(V)**-alp1
            _dlambda = lambda V: -_lambda(V) * (a*alp0/x1(V) + b*c*alp2/x2(V) - d*e*alp1/x3(V))
            _f = lambda V: x1(V) * _lambda(V)
            _g = lambda V: x1(V)**(alp0*omg) * x2(V)**(alp3+alp2*omg) * x3(V)**(alp4+alp1*omg) * x4(V)**alp5
            _h = lambda V: x1(V)**(alp0*omg) * x3(V)**(alp4+alp1*(omg-2)) * x4(V)**(1+alp5)

        if V2 < Vstar-eps:
            Vmin = 2/(gamma*ex)
        else:
            Vmin = 2/ex

        J1 = lambda V: scipy.integrate.quad(((gamma+1)/(gamma-1)) * _lambda(V)**(j+1) * _g(V) * V**2 * _dlambda(V), Vmin, V2)
        J2 = lambda V: scipy.integrate.quad((8 * _h(V) * _dlambda(V) * _lambda(V)**(j+1))/((gamma+1) * ex**2), Vmin, V2)

        if j == 1:
            alpha = lambda V: .5 * J1(V) + J2(V)/(gamma-1)
        else:
            alpha = lambda V: (j-1) * np.pi * (J1(V) + (2*J2(V))/(gamma-1))

    # Determine the shock front position
    pressures = grid[...,4]
    physical_grid = np.linspace(sim_variables.start_pos-sim_variables.dx/2, sim_variables.end_pos+sim_variables.dx/2, sim_variables.cells+2)[1:-1]
    left_P, right_P = pressures[:int(sim_variables.cells/2)], pressures[int(sim_variables.cells/2):]
    if np.max(np.diff(left_P)) > np.max(np.diff(right_P)):
        index = np.argmax(np.diff(left_P))
    else:
        index = np.argmax(np.diff(right_P))
    r_want = abs(physical_grid[index])

    # Root-finding for similarity variable V*
    f = lambda V: r2(alpha(V)) * _lambda(V) - r_want
    V_star = scipy.optimize.fsolve(f, 1)[0]

    # Update the analytical solution
    mask = np.abs(physical_grid) > r2
    arr[...,0][mask] = rho0
    arr[...,1:4][mask] = v0
    arr[...,4][mask] = P0
    arr[...,5:8][mask] = B0

    mask = np.abs(physical_grid) <= r2
    arr[...,0][mask] = (rho2*_g(V_star))[mask]
    arr[...,1][mask] = (v0[...,0]*_f(V_star))[mask]
    arr[...,4][mask] = (P2*_h(V_star))[mask]

    return arr


"""def calculate_Sedov_analytical(gamma):
    dA = lambda eta, A, B, C: ((gamma+1)*(3*B + 5*A*C**2) + 2*C*(3*gamma*B - 13*A*C**2) + (gamma + 5 - 12*C)*((A*C*(gamma*(2*C-1)-1))/(2*(gamma-1))) - (12*A*(gamma+1)*C**2)/eta) * (2*A*(gamma-1))/((eta*(2*C-(gamma+1))*(gamma*(2*C-1)-1) - 10*eta*(gamma-1)*C**2 + 4*C*(gamma+1)*(gamma-1))*(2*C-(gamma+1))*A - 4*eta*gamma*(gamma-1)*(2*C-(gamma+1))*B)
    dB = lambda eta, A, B, C: (dA(eta,A,B,C)*(2*C-(gamma+1))**2)/(2*(gamma-1)) - A*C*(gamma+5-12*C)/(2*eta*(gamma-1)) - 2*B/eta
    dC = lambda eta, A, B, C: -(2*C-(gamma+1))*dA(eta,A,B,C)/(2*A) - 3*C/eta
    pass
"""