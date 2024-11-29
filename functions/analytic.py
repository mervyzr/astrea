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
    gamma, dimension = sim_variables.gamma, sim_variables.dimension

    time_keys = [float(t) for t in simulation.keys()]
    w_num = simulation[str(max(time_keys))]  # Get last instance of the grid with largest time key

    # Create theoretical array
    normalising_factor = 1/(len(w_num) ** dimension)
    sim_variables = sim_variables._replace(cells=len(w_num))
    w_theo = constructor.initialise(sim_variables)

    thermal_num, thermal_theo = fv.divide(w_num[...,4], w_num[...,0]*(gamma-1)), fv.divide(w_theo[...,4], w_theo[...,0]*(gamma-1))
    w_num, w_theo = np.concatenate((w_num, thermal_num[...,None]), axis=-1), np.concatenate((w_theo, thermal_theo[...,None]), axis=-1)

    if norm > 10:
        return np.max(np.abs(w_num-w_theo), axis=tuple(range(dimension)))
    elif norm <= 0:
        return normalising_factor * np.sum(np.abs(w_num-w_theo), axis=tuple(range(dimension)))
    else:
        return (normalising_factor * np.sum(np.abs(w_num-w_theo)**norm, axis=tuple(range(dimension))))**(1/norm)


# Function for calculation of total variation (TVD scheme if TV(t+1) < TV(t)); total variation tests for oscillations
def calculate_tv(simulation, sim_variables):
    gamma, dimension, tv = sim_variables.gamma, sim_variables.dimension, {}

    for t in list(simulation.keys()):
        grid = simulation[t]
        thermal = fv.divide(grid[...,4], grid[...,0]*(gamma-1))
        for i in range(dimension):
            grid = np.diff(grid, axis=i)
            thermal = np.diff(thermal, axis=i)
        tv[float(t)] = np.sum(np.abs(grid), axis=tuple(range(dimension)))
        tv[float(t)] = np.append(tv[float(t)], np.sum(np.abs(thermal)))
    return tv


# Function for checking the conservation equations; works with primitive variables but needs to be converted
def calculate_conservation(simulation, sim_variables):
    N, start_pos, end_pos = sim_variables.cells, sim_variables.start_pos, sim_variables.end_pos
    dimension, eq = sim_variables.dimension, {}

    for t in list(simulation.keys()):
        grid = sim_variables.convert_primitive(simulation[t][:], sim_variables)
        for i in range(dimension)[::-1]:
            grid = scipy.integrate.simpson(grid, dx=(end_pos-start_pos)/N, axis=i) * (end_pos-start_pos)
        eq[float(t)] = grid
    return eq


# Function for checking the conservation equations at specific intervals; works with primitive variables but needs to be converted
# The reason is because at the boundaries, some values are lost to the ghost cells and not counted into the conservation plots
# This is the reason why there is a dip at exactly the halfway mark of the periodic smooth tests
def calculate_conservation_at_interval(simulation, sim_variables, interval=10):
    N, start_pos, end_pos, t_end = sim_variables.cells, sim_variables.start_pos, sim_variables.end_pos, sim_variables.t_end
    dimension, eq = sim_variables.dimension, {}

    intervals = np.array([], dtype=float)
    periods = np.linspace(0, t_end, interval)
    timings = np.asarray(list(simulation.keys()), dtype=float)
    for period in periods:
        intervals = np.append(intervals, timings[np.argmin(abs(timings-period))])

    for t in intervals:
        grid = sim_variables.convert_primitive(simulation[str(t)][:], sim_variables)
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
def calculate_Sedov_analytical(grid, t, sim_variables, w=0):
    # Initialise initial conditions and variables
    gamma, j = sim_variables.gamma, sim_variables.dimension
    rho0, vx0, vy0, vz0, P0, Bx0, By0, Bz0 = sim_variables.initial_right
    eps = 1e-4
    E_blast = sim_variables.initial_left[4]/(rho0 *(gamma-1))

    _exp = j + 2 - w

    # Determine family type
    V2 = 4/_exp
    Vstar = 2/(j*(gamma-1)+2)

    # Note the singularities
    w2 = (2*(gamma-1) + j)/gamma
    w3 = j * (2-gamma)
    if abs(w-w2) <= eps:
        w2 = 1e-8
    elif abs(w-w3) <= eps:
        w3 = 1e-8

    # Form the exponents
    alpha0 = 2/_exp
    alpha2 = -(gamma-1)/(gamma*(w2-w))
    alpha1 = ((_exp*gamma)/(2+j*(gamma-1))) * ((2*(j*(2-gamma)-w))/(gamma*_exp**2) - alpha2)
    alpha3 = (j-w)/(gamma*(w2-w))
    alpha4 = alpha1 * ((_exp*(j-w))/(w3-w))
    alpha5 = (w*(1+gamma)-2*j)/(w3-w)

    # Form frequently used variables
    a = .25 * _exp * (gamma+1)
    b = (gamma+1)/(gamma-1)
    c = .5 * gamma * _exp
    d = ((gamma+1)*_exp)/((gamma+1)*_exp - 2*(2+j*(gamma-1)))
    e = .5 * (2 + j*(gamma-1))

    # Define the auxiliary functions and their derivatives
    x1 = lambda V: a * V
    x2 = lambda V: b * max(1e-30, c*V - 1)
    x3 = lambda V: d * (1 - e*V)
    x4 = lambda V: max(1e-12, b * (1 - (c*V)/gamma))
    dx1 = a
    dx2 = b * c
    dx3 = -d * e
    dx4 = -b * c / gamma

    # Singular
    if abs(V2-Vstar) <= eps:
        # Calculate the energy integrals (trivial)
        J2 = (gamma+1)/(j*(j*(gamma-1)+2)**2)
        J1 = (2*J2)/(gamma-1)
        alpha = J2 * np.pi * 2**(j-1)

        # Define the shock position
        r2 = (E_blast*t**2/(alpha*rho0))**(1/(_exp))

        # Compute the Sedov functions
        _lambda = lambda V: V/r2
        _dlambda = 0
        _f = lambda V: _lambda(V)
        _g = lambda V: _lambda(V)**(j-2)
        _h = lambda V: _lambda(V)**j

    else:
        # Compute the Sedov functions
        # Vacuum
        if V2 > Vstar + eps:
            _lambda = _dlambda =_f = _g = _h = 0

        else:
            # Singularity w2
            if abs(w-w2) <= eps:
                factor = lambda V: (1-x1(V))/(x1(V)-(gamma+1)/(2*gamma))
                _lambda = lambda V: x1(V)**-alpha0 * x2(V)**((gamma-1)/(2*e)) * np.exp(factor * (gamma+1)/(2*e))
                _dlambda = lambda V: -_lambda(V) * (dx1*alpha0/x1(V) + dx2*(gamma-1)/(2*e*x2(V)) - dx1*((gamma+1)/(2*e))*(factor/(1-x1(V)))*(1+factor))
                _f = lambda V: x1(V) * _lambda(V)
                _g = lambda V: x1(V)**(alpha0*w) * x2(V)**(4-j-(2*gamma)/(2*e)) * x4(V)**alpha5 * np.exp(factor * (gamma+1)/e)
                _h = lambda V: x1(V)**(alpha0*w) * x3(V)**(-j*gamma/(2*e)) * x4(V)**(1+alpha5)

            # Singularity w3
            elif abs(w-w3) <= eps:
                factor = lambda V: np.exp(-(j*gamma*(gamma+1)*(1-x1(V)))/(2*e*(.5*(gamma+1)-x1(V))))
                _lambda = lambda V: x1(V)**-alpha0 * x2(V)**-alpha2 * x4(V)**-alpha1
                _dlambda = lambda V: -_lambda(V) * (dx1*alpha0/x1(V) + dx2*alpha2/x2(V) + dx4*alpha1/x4(V))
                _f = lambda V: x1(V) * _lambda(V)
                _g = lambda V: x1(V)**(alpha0*w) * x2(V)**(alpha3+alpha2*w) * x4(V)**(1-2/e) * factor
                _h = lambda V: x1(V)**(alpha0*w) * x4(V)**((j*(gamma-1)-gamma)/e) * factor

            # Standard
            else:
                _lambda = lambda V: x1(V)**-alpha0 * x2(V)**-alpha2 * x3(V)**-alpha1
                _dlambda = lambda V: -_lambda(V) * (dx1*alpha0/x1(V) + dx2*alpha2/x2(V) + dx3*alpha1/x3(V))
                _f = lambda V: x1(V) * _lambda(V)
                _g = lambda V: x1(V)**(alpha0*w) * x2(V)**(alpha3+alpha2*w) * x3(V)**(alpha4+alpha1*w) * x4(V)**alpha5
                _h = lambda V: x1(V)**(alpha0*w) * x3(V)**(alpha4+alpha1*(w-2)) * x4(V)**(1+alpha5)

        # Evaluate the energy integrals
        rvv = 0
        # Standard
        if V2 < Vstar - eps:
            Vmin = 2/(_exp*gamma)
        # Vacuum
        else:
            Vmin = 2/_exp

        # Compute the energy integrals
        J1 = scipy.integrate.quad(lambda V: ((gamma+1)/(gamma-1)) * _lambda(V)**(j+1) * _g(V) * V**2 * _dlambda(V), Vmin, V2, epsabs=1e-12)[0]
        J2 = scipy.integrate.quad(lambda V: 8/((gamma+1)*_exp**2) * _lambda(V)**(j+1) * _h(V) * _dlambda(V), Vmin, V2, epsabs=1e-12)[0]

        # Compute alpha with the integrated energies
        if j == 1:
            alpha = .5 * J1 + J2/(gamma-1)
        else:
            alpha = (j-1) * np.pi * (J1 + 2*J2/(gamma-1))

        # Define the shock position
        r2 = (E_blast*t**2/(alpha*rho0))**(1/(_exp))
    
    # Define the post-shock values
    vx2 = ((4*r2)/(_exp*t))/(gamma+1)
    rho2 = rho0 * (gamma+1)/(gamma-1)
    P2 = 2*rho0*((2*r2)/(_exp*t))**2/(gamma+1)

    arr = np.zeros_like(grid)

    # Generate the array of radii
    centre = (sim_variables.end_pos + sim_variables.start_pos)/2
    physical_grid = np.linspace(sim_variables.start_pos-sim_variables.dx/2, sim_variables.end_pos+sim_variables/2, sim_variables.cells+2)[1:-1]
    radii = physical_grid[(centre <= physical_grid) & (physical_grid <= r2)]

    density = np.zeros_like(radii)
    pressure = np.zeros_like(radii)
    vx = np.zeros_like(radii)

    for index, r in enumerate(radii):
        f = lambda V: r2*_lambda(V) - r
        _V = scipy.optimize.fsolve(f, 1)[0]

        density[index] = rho2 * _g(_V)
        pressure[index] = P2 * _h(_V)
        vx[index] = vx2 * _f(_V)

    arr[...,0][physical_grid <= r2] = density
    arr[...,4][physical_grid <= r2] = pressure
    arr[...,1][physical_grid <= r2] = vx
    arr[...,0][physical_grid > r2] = rho0
    arr[...,4][physical_grid > r2] = P0
    arr[...,1][physical_grid > r2] = vx0

    return arr