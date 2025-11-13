import scipy as sp
import numpy as np
from skimage.measure import block_reduce

from functions import constructor, fv, generic

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
    density, pressure = 0, 4
    return (grid[...,density] * np.log(grid[...,pressure]*grid[...,density]**-gamma))/(gamma-1)


# Function for solution error calculation of sine-wave and Gaussian tests
def calculate_solution_error(grid, sim_variables, norm):
    gamma, axes = sim_variables.gamma, sim_variables.axes
    rho, pressure = sim_variables.rho, sim_variables.pressure
    w_num = np.copy(grid)
    grid_shape = w_num.shape[:-1]

    # Create theoretical array
    normalising_factor = 1/(np.prod(grid_shape))
    sim_variables.cells = list(grid_shape)
    w_theo = constructor.initialise(sim_variables)

    E_tot_num, E_tot_theo = fv.divide(fv.convert_variable('pressure', w_num, sim_variables), w_num[...,rho]), fv.divide(fv.convert_variable('pressure', w_theo, sim_variables), w_theo[...,rho])
    E_int_num, E_int_theo = fv.divide(w_num[...,pressure], w_num[...,rho]*(gamma-1)), fv.divide(w_theo[...,pressure], w_theo[...,rho]*(gamma-1))

    w_num, w_theo = np.concatenate((w_num, E_tot_num[...,None]), axis=-1), np.concatenate((w_theo, E_tot_theo[...,None]), axis=-1)
    w_num, w_theo = np.concatenate((w_num, E_int_num[...,None]), axis=-1), np.concatenate((w_theo, E_int_theo[...,None]), axis=-1)

    if norm > 10:
        return np.max(np.abs(w_num-w_theo), axis=tuple(axes))
    elif norm <= 0:
        return normalising_factor * np.sum(np.abs(w_num-w_theo), axis=tuple(axes))
    else:
        return (normalising_factor * np.sum(np.abs(w_num-w_theo)**norm, axis=tuple(axes)))**(1/norm)


# Function for calculation of total variation (TVD scheme if TV(t+1) < TV(t)); total variation tests for oscillations
def calculate_TV(simulation, sim_variables):
    gamma, dimensions, axes, tot_vary = sim_variables.gamma, sim_variables.dimensions, sim_variables.axes, {}
    rho, pressure = sim_variables.rho, sim_variables.pressure

    for t in list(simulation.keys()):
        grid = simulation[t]
        E_tot = fv.divide(fv.convert_variable('pressure', grid, sim_variables), grid[...,rho])
        E_int = fv.divide(grid[...,pressure], grid[...,rho]*(gamma-1))
        for i in range(dimensions):
            grid = np.diff(grid, axis=i)
            E_tot = np.diff(E_tot, axis=i)
            E_int = np.diff(E_int, axis=i)
        tot_vary[float(t)] = np.sum(np.abs(grid), axis=tuple(axes))
        tot_vary[float(t)] = np.append(tot_vary[float(t)], np.sum(np.abs(E_tot)))
        tot_vary[float(t)] = np.append(tot_vary[float(t)], np.sum(np.abs(E_int)))
    return tot_vary


# Function for checking the conservation equations; works with primitive variables but needs to be converted
def calculate_conservation(simulation, sim_variables):
    axes, axis_coord, conservation = sim_variables.axes, sim_variables.axis_coord, {}

    dV = np.prod(np.diff(list(axis_coord.values()), axis=1))
    for t in list(simulation.keys()):
        _grid = simulation[t][:]  # Needs the '[:]' to access the array
        grid = sim_variables.convert("primitive", _grid, sim_variables)
        grid = np.sum(grid, axis=tuple(axes))
        conservation[float(t)] = grid * dV
    return conservation


# Function for checking the conservation equations at specific intervals; works with primitive variables but needs to be converted
# The reason is because at the boundaries, some values are lost to the ghost cells and not counted into the conservation plots
# This is the reason why there is a dip at exactly the halfway mark of the periodic smooth tests
def calculate_conservation_at_interval(simulation, sim_variables, interval=10):
    axes, axis_coord, conservation = sim_variables.axes, sim_variables.axis_coord, {}

    dV = np.prod(np.diff(list(axis_coord.values()), axis=1))
    simulation_timings = list(simulation.keys())
    simulation_timings.sort()
    intervals = [timing[-1] for timing in np.array_split(simulation_timings, abs(interval))]

    for t in intervals:
        _grid = simulation[t][:]  # Needs the '[:]' to access the array
        grid = sim_variables.convert("primitive", _grid, sim_variables)
        grid = np.sum(grid, axis=tuple(axes))
        conservation[t] = grid * dV
    return conservation


# Determine the analytical solution for a Sod shock test (only in 1d)
def calculate_Sod_analytical(grid, t, sim_variables):
    gamma, axis_coord, shock_pos = sim_variables.gamma, sim_variables.axis_coord[0], sim_variables.shock_pos
    start_pos, end_pos = axis_coord

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
    boundary_54 = round_off(((shock_pos-(cs5*t)-start_pos)/np.diff(axis_coord)) * len(grid))
    boundary_43 = round_off(((shock_pos-(v_t*t)-start_pos)/np.diff(axis_coord)) * len(grid))
    boundary_32 = round_off(((shock_pos+(vx2*t)-start_pos)/np.diff(axis_coord)) * len(grid))
    boundary_21 = round_off(((shock_pos+(v_s*t)-start_pos)/np.diff(axis_coord)) * len(grid))

    # Define number of cells in the rarefaction wave
    rarefaction_cells = round_off(((cs5*t-v_t*t)/np.diff(axis_coord)) * len(grid))
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


# Resample grid for circular blast injection to populate cell variables with a circle/sphere; value in grid cell is weighted by area/volume covered
def resample_blast(grid, sim_variables, resample_size=50):
    print(f"{generic.BColours.WARNING}Blast config. used; supersampling initialised grid before starting simulation for better resolution..{generic.BColours.ENDC}")
    cells, dimensions, multidimensional, axis_coord, shock_pos = sim_variables.cells, sim_variables.dimensions, sim_variables.multidimensional, sim_variables.axis_coord, sim_variables.shock_pos

    fine_grid = np.resize(np.zeros_like(grid), np.asarray(cells)*resample_size)
    physical_grid = lambda axis: constructor.make_physical_grid(axis_coord[axis], cells[axis]*resample_size)

    x_centre = np.average(axis_coord[0])
    fine_physical_grid_x = physical_grid(0)
    fine_x = fine_physical_grid_x - x_centre

    fine_y = fine_z = np.zeros_like(fine_x)
    y_centre = z_centre = 0

    if multidimensional:
        y_centre = np.average(axis_coord[1])
        fine_physical_grid_y = physical_grid(1)
        fine_x, fine_y = np.meshgrid(fine_physical_grid_x, fine_physical_grid_y, indexing='ij')
        fine_z = np.zeros_like(fine_x)

        if dimensions == 3:
            z_centre = np.average(axis_coord[2])
            fine_physical_grid_z = physical_grid(2)
            fine_x, fine_y, fine_z = np.meshgrid(fine_physical_grid_x, fine_physical_grid_y, fine_physical_grid_z, indexing='ij')

    fine_r = np.sqrt((fine_x-x_centre)**2 + (fine_y-y_centre)**2 + (fine_z-z_centre)**2)
    fine_mask = np.where(fine_r**2 <= (shock_pos-x_centre)**2)
    fine_grid[fine_mask] = 1

    remapped_grid = block_reduce(fine_grid, block_size=tuple([resample_size,]*dimensions), func=np.sum)
    mask = np.where(remapped_grid > 0)

    _grid = np.copy(grid)
    _grid[mask][...,sim_variables.pressure] *= (remapped_grid/np.max(remapped_grid))[mask]

    return _grid


# Determine the analytical solution for a Sedov blast wave [Dullemond, Numerical Methods, Chpt. 10]
def calculate_Sedov_analytical(grid, t, sim_variables):
    # Initialise initial conditions and variables
    cells, gamma, dimensions, multidimensional, axis_coord = sim_variables.cells, sim_variables.gamma, sim_variables.dimensions, sim_variables.multidimensional, sim_variables.axis_coord
    rho0, vx0, vy0, vz0, P0, Bx0, By0, Bz0 = sim_variables.initial_left
    shock_pos = sim_variables.shock_pos

    # Create a physical half-grid for a single axis
    def make_half_grid(_axis, _cells):
        dh = np.abs(np.diff(_axis)[0])/_cells
        half_cell = dh/2
        return np.linspace(_axis[0]-half_cell, _axis[1]+half_cell, _cells+2)[1+int(_cells/2):-1]

    x_centre = np.average(axis_coord[0])
    physical_halfgrid_x = make_half_grid(axis_coord[0], cells[0])
    X, Y, Z = np.array(physical_halfgrid_x), np.zeros_like(physical_halfgrid_x), np.zeros_like(physical_halfgrid_x)

    if multidimensional:
        y_centre = np.average(axis_coord[1])
        physical_halfgrid_y = make_half_grid(axis_coord[1], cells[1])
        X, Y = np.meshgrid(physical_halfgrid_x, physical_halfgrid_y, indexing='ij')
        Z = np.zeros_like(X)

        if dimensions == 3:
            z_centre = np.average(axis_coord[2])
            physical_halfgrid_z = make_half_grid(axis_coord[2], cells[2])
            X, Y, Z = np.meshgrid(physical_halfgrid_x, physical_halfgrid_y, physical_halfgrid_z, indexing='ij')

    rx, ry, rz = X - x_centre, Y - y_centre, Z - z_centre
    r = np.sqrt(rx**2 + ry**2 + rz**2)
    E_blast = 4/3 * np.pi * (P0*(shock_pos-x_centre)**3)/(gamma-1)

    # ----------------------------------------------------
    # Self-similar ODEs for A(η), B(η), C(η)
    # ----------------------------------------------------
    def sedov_derivs(eta, eta_funcs):
        A, B, C = eta_funcs
        if eta <= 0:
            return [0.0, 0.0, 0.0]

        # Equation coefficients [eq. 10.30-10.32]
        coeff_dA_1 = eta * (2*C - (gamma+1))/(gamma+1)
        coeff_dB_1 = 0
        coeff_dC_1 = (2 * eta * A) / (gamma+1)
        const1 = -(6*A*C) / (gamma+1)

        coeff_dA_2 = 0
        coeff_dB_2 = eta * (gamma-1)/A
        coeff_dC_2 = eta * (2*C - (gamma+1))
        const2 = 2.5 * (gamma+1) * C - 2 * C**2 - 2 * (gamma-1) * B/A

        coeff_dA_3 = C**2 * eta * (gamma + 1 - 2*C)
        coeff_dB_3 = eta * (gamma + 1 - 2*gamma*C)
        coeff_dC_3 = 2 * eta * (A*C*(gamma+1) + gamma*B - A*C**2)
        const3 = 10 * C * (gamma*B + A*C**2) - 5 * (gamma+1) * (B + A*C**2)

        coeffs = np.array([
            [coeff_dA_1, coeff_dB_1, coeff_dC_1],
            [coeff_dA_2, coeff_dB_2, coeff_dC_2],
            [coeff_dA_3, coeff_dB_3, coeff_dC_3]
        ])
        consts = np.array([const1, const2, const3])

        try:
            dA, dB, dC = np.linalg.solve(coeffs, consts)
        except np.linalg.LinAlgError:
            coeffs += np.eye(3) * np.finfo(sim_variables.precision).eps
            dA, dB, dC = np.linalg.solve(coeffs, consts)

        return [dA, dB, dC]

    # ----------------------------------------------------
    # Integrate similarity ODEs inward from shock
    # ----------------------------------------------------
    def integrate_profiles(eta_s, eta_min=1e-6, npts=2000):
        eta_start = eta_s * (1 - 1e-8)
        solution = sp.integrate.solve_ivp(
            lambda _eta, _eta_funcs: sedov_derivs(_eta, _eta_funcs),
            t_span=(eta_start, eta_min),
            y0=[1, 1, 1],
            t_eval=np.linspace(eta_start, eta_min, npts),
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )

        if not solution.success:
            raise RuntimeError(solution.message)
        eta_arr = solution.t[::-1]
        A, B, C = solution.y[:, ::-1]
        return eta_arr, A, B, C

    # ----------------------------------------------------
    # Energy integral condition (eq. 10.34)
    # ----------------------------------------------------
    def energy_integral(eta, A, B, C):
        integrand = (B + A*C**2) * eta**4
        coeff = 32*np.pi / (25*(sim_variables.gamma**2 - 1))
        return coeff * np.trapezoid(integrand, eta)

    # ----------------------------------------------------
    # Find ηs by shooting
    # ----------------------------------------------------
    def find_eta_s():
        def residual(eta_s):
            eta, A, B, C = integrate_profiles(eta_s)
            return energy_integral(eta, A, B, C) - 1.0

        # Secant iteration
        g1, g2 = 1.0, 1.5
        r1, r2 = residual(g1), residual(g2)
        for _ in range(20):
            if abs(r2 - r1) < 1e-12:
                break
            g3 = g2 - r2 * (g2 - g1) / (r2 - r1)
            r3 = residual(g3)
            if abs(r3) < 1e-6:
                g2, r2 = g3, r3
                break
            g1, r1, g2, r2 = g2, r2, g3, r3
        eta_s = g2
        eta, A, B, C = integrate_profiles(eta_s, npts=3000)
        return eta_s, eta, A, B, C


    eta_s, eta, A, B, C = find_eta_s()

    # similarity length scale and shock radius
    length_scale = (E_blast * t**2 / rho0)**0.2
    shock_radius = eta_s * length_scale
    shock_vel = (2/5) * shock_radius / t

    # interpolate A,B,C
    Aint = sp.interpolate.interp1d(eta, A, kind="cubic", fill_value=(A[0], 1.0), bounds_error=False)
    Bint = sp.interpolate.interp1d(eta, B, kind="cubic", fill_value=(B[0], 1.0), bounds_error=False)
    Cint = sp.interpolate.interp1d(eta, C, kind="cubic", fill_value=(C[0], 1.0), bounds_error=False)

    eta_grid = r / length_scale
    inside_shock = (eta_grid <= eta_s) & (r > 0)

    # strong shock jump conditions
    rho2 = (gamma + 1)/(gamma - 1) * rho0
    P2 = 2 * rho0 * shock_vel**2 / (gamma + 1)
    v2 = 2 * shock_vel / (gamma + 1)

    density = np.full_like(r, rho0)
    pressure = np.zeros_like(r)
    vx = np.zeros_like(r)
    vy = np.zeros_like(r)
    vz = np.zeros_like(r)

    # interior values
    zeta = eta_grid[inside_shock]
    A_in, B_in, C_in = Aint(zeta), Bint(zeta), Cint(zeta)
    density[inside_shock] = rho2 * A_in
    pressure[inside_shock] = P2 * (zeta / eta_s)**2 * B_in
    vmag = v2 * (zeta / eta_s) * C_in
    vx[inside_shock] = vmag * (rx[inside_shock] / r[inside_shock])
    vy[inside_shock] = vmag * (ry[inside_shock] / r[inside_shock])
    vz[inside_shock] = vmag * (rz[inside_shock] / r[inside_shock])

    # handle center
    density[(r - x_centre) < 1e-6] = rho2 * A[0]
    pressure[(r - x_centre) < 1e-6] = P2 * (eta[0]/eta_s)**2 * B[0]

    # populate arr
    arr = np.zeros_like(grid)
    midpoint_x = int(cells[0]/2)

    if multidimensional:
        if dimensions == 2:
            midpoint_y = int(cells[1]/2)

            def rotate(key, quantity):
                temp_arr = np.zeros_like(arr[...,key])
                temp_arr[:midpoint_x, midpoint_y:] = quantity
                temp_arr[:midpoint_x, :midpoint_y] = np.rot90(quantity, k=1)
                temp_arr[midpoint_x:, :midpoint_y] = np.rot90(quantity, k=2)
                temp_arr[midpoint_x:, midpoint_y:] = np.rot90(quantity, k=3)
                return temp_arr

            arr[...,sim_variables.rho] = rotate(sim_variables.rho, density)
            arr[...,sim_variables.pressure] = rotate(sim_variables.pressure, pressure)
            arr[...,sim_variables.vx] = rotate(sim_variables.vx, vx)
            arr[...,sim_variables.vy] = rotate(sim_variables.vy, vy)
            arr[...,sim_variables.vz] = rotate(sim_variables.vz, vz)

        else:
            midpoint_z = int(cells[2]/2)

            def rotate(key, quantity):
                temp_arr = np.zeros_like(arr[...,key])
                temp_arr[:midpoint_x, :midpoint_y, :midpoint_z] = quantity
                temp_arr[midpoint_x:, :midpoint_y, :midpoint_z] = np.flip(quantity, axis=0)
                temp_arr[:midpoint_x, midpoint_y:, :midpoint_z] = np.flip(quantity, axis=1)
                temp_arr[midpoint_x:, midpoint_y:, :midpoint_z] = np.flip(quantity, axis=(0,1))
                temp_arr[:midpoint_x, :midpoint_y, midpoint_z:] = np.flip(quantity, axis=2)
                temp_arr[midpoint_x:, :midpoint_y, midpoint_z:] = np.flip(quantity, axis=(0,2))
                temp_arr[:midpoint_x, midpoint_y:, midpoint_z:] = np.flip(quantity, axis=(1,2))
                temp_arr[midpoint_x:, midpoint_y:, midpoint_z:] = np.flip(quantity, axis=(0,1,2))
                return temp_arr

            arr[...,sim_variables.rho] = rotate(sim_variables.rho, density)
            arr[...,sim_variables.pressure] = rotate(sim_variables.pressure, pressure)
            arr[...,sim_variables.vx] = rotate(sim_variables.vx, vx)
            arr[...,sim_variables.vy] = rotate(sim_variables.vy, vy)
            arr[...,sim_variables.vz] = rotate(sim_variables.vz, vz)

    else:
        arr[...,sim_variables.rho] = np.concatenate((np.flip(density), density))
        arr[...,sim_variables.pressure] = np.concatenate((np.flip(pressure), pressure))
        arr[...,sim_variables.vx] = np.concatenate((np.flip(vx), vx))
        arr[...,sim_variables.vy] = np.concatenate((np.flip(vy), vy))
        arr[...,sim_variables.vz] = np.concatenate((np.flip(vz), vz))

    return arr


"""# Determine the analytical solution for a Sedov blast wave (only in 1d, doesn't work currently) [Kamm & Timmes, 2000]
def calculate_Sedov_analytical(grid, t, sim_variables, w=0):

    # Create a physical grid for a single axis
    def make_physical_grid(_axis, _cells):
        dh = np.abs(np.diff(_axis)[0])/_cells
        half_cell = dh/2
        return np.linspace(_axis[0]-half_cell, _axis[1]+half_cell, _cells+2)[1:-1]

    # Initialise initial conditions and variables
    cells, gamma, j, axis_coord = sim_variables.cells, sim_variables.gamma, sim_variables.dimensions, sim_variables.axis_coord
    rho0, vx0, vy0, vz0, P0, Bx0, By0, Bz0 = sim_variables.initial_right
    rho, vx, pressure = sim_variables.rho, sim_variables.vx, sim_variables.pressure
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
        J1 = sp.integrate.quad(lambda V: ((gamma+1)/(gamma-1)) * _lambda(V)**(j+1) * _g(V) * V**2 * _dlambda(V), Vmin, V2, epsabs=1e-12)[0]
        J2 = sp.integrate.quad(lambda V: 8/((gamma+1)*_exp**2) * _lambda(V)**(j+1) * _h(V) * _dlambda(V), Vmin, V2, epsabs=1e-12)[0]

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
    x_centre = np.average(axis_coord)
    physical_grid_x = make_physical_grid(axis_coord, cells[0])
    radii = physical_grid_x[(x_centre <= physical_grid_x) & (physical_grid_x <= r2)]

    density = np.zeros_like(radii)
    pressure = np.zeros_like(radii)
    velx = np.zeros_like(radii)

    for index, r in enumerate(radii):
        f = lambda V: r2*_lambda(V) - r
        _V = sp.optimize.fsolve(f, 1)[0]

        density[index] = rho2 * _g(_V)
        pressure[index] = P2 * _h(_V)
        velx[index] = vx2 * _f(_V)

    arr[...,rho][physical_grid_x <= r2] = density
    arr[...,pressure][physical_grid_x <= r2] = pressure
    arr[...,vx][physical_grid_x <= r2] = velx
    arr[...,rho][physical_grid_x > r2] = rho0
    arr[...,pressure][physical_grid_x > r2] = P0
    arr[...,vx][physical_grid_x > r2] = vx0

    return arr"""