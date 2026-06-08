import scipy as sp
import numpy as np
from scipy.special import gamma as gamma_func

from functions import analytic
from functions import grid as gutils
from functions import math as mfuncs

##############################################################################
# Functions for analytic solutions
##############################################################################

# Calculate scaled entropy density for an array [Derigs et al., 2015]
def calculate_entropy_density(grid, gamma):
    density, pressure = 0, 4
    return (grid[...,density] * np.log(grid[...,pressure]*grid[...,density]**-gamma))/(gamma-1)


# Function for solution error calculation of sine-wave and Gaussian tests
def calculate_solution_error(grid, sim_variables, norm):
    axes = sim_variables.axes
    rho, vels, pressure, Bfields = sim_variables.rho, sim_variables.vels, sim_variables.pressure, sim_variables.Bfields
    vx, vy, vz = sim_variables.vx, sim_variables.vy, sim_variables.vz
    Etot = sim_variables.Bz+1
    momentums = slice(Etot, Etot+3)
    w_num = np.copy(grid)

    # Create theoretical array
    if "manufacture" in sim_variables.config or "euler" in sim_variables.config:
        w_theo = analytic.calculate_Euler_analytical(grid, sim_variables)
    else:
        w_theo = gutils.initialise(sim_variables)

    # Energy terms
    E_tot_num, E_tot_theo = mfuncs.divide(gutils.convert_thermo_variable('pressure', w_num, sim_variables), w_num[...,rho]), mfuncs.divide(gutils.convert_thermo_variable('pressure', w_theo, sim_variables), w_theo[...,rho])

    # Momentum terms
    momx_num, momx_theo = w_num[...,rho] * w_num[...,vx], w_theo[...,rho] * w_theo[...,vx]
    momy_num, momy_theo = w_num[...,rho] * w_num[...,vy], w_theo[...,rho] * w_theo[...,vy]
    momz_num, momz_theo = w_num[...,rho] * w_num[...,vz], w_theo[...,rho] * w_theo[...,vz]

    # Tag on at the end of the original arrays
    w_num = np.concatenate((
        w_num, E_tot_num[...,None],
        momx_num[...,None], momy_num[...,None], momz_num[...,None]
    ), axis=-1)
    w_theo = np.concatenate((
        w_theo, E_tot_theo[...,None],
        momx_theo[...,None], momy_theo[...,None], momz_theo[...,None]
    ), axis=-1)

    if norm > 5:
        solution_errors = np.max(np.abs(w_num-w_theo), axis=tuple(axes))
    else:
        normalising_factor = np.prod(list(sim_variables.ds.values()))
        if norm <= 0:
            solution_errors = normalising_factor * np.sum(np.abs(w_num-w_theo), axis=tuple(axes))
        else:
            solution_errors = normalising_factor * (np.sum(np.abs(w_num-w_theo)**norm, axis=tuple(axes)))**(1/norm)

    return {
        'density': solution_errors[...,rho],
        'vels': solution_errors[...,vels],
        'pressure': solution_errors[...,pressure],
        'Bfields': solution_errors[...,Bfields],
        'Etot': solution_errors[...,Etot],
        'momentums': solution_errors[...,momentums],
    }


# Function for calculation of total variation (TVD scheme if TV(t+1) < TV(t)); total variation tests for oscillations
def calculate_TV(simulation, sim_variables):
    dimensions, axes = sim_variables.dimensions, sim_variables.axes
    total_variation = {}

    for t in list(simulation.keys()):
        grid = simulation[t]

        E_tot = mfuncs.divide(gutils.convert_thermo_variable('pressure', grid, sim_variables), grid[...,sim_variables.rho])

        for i in range(dimensions):
            grid = np.diff(grid, axis=i)
            E_tot = np.diff(E_tot, axis=i)

        total_variation[float(t)] = np.sum(np.abs(grid), axis=tuple(axes))
        total_variation[float(t)] = np.append(total_variation[float(t)], np.sum(E_tot))
    return total_variation


# Function for checking the conservation equations; works with primitive variables but needs to be converted
def calculate_conservation(simulation, sim_variables):
    axes, coordinates, conservation = sim_variables.axes, sim_variables.coordinates, {}

    dV = np.prod(np.diff(list(coordinates.values()), axis=1))
    for t in list(simulation.keys()):
        _grid = simulation[t][:]  # Needs the '[:]' to access the array
        grid = gutils.convert("primitive", _grid, sim_variables)
        grid = np.sum(grid, axis=tuple(axes))
        conservation[float(t)] = grid * dV
    return conservation


# Function for checking the conservation equations at specific intervals; works with primitive variables but needs to be converted
# The reason is because at the boundaries, some values are lost to the ghost cells and not counted into the conservation plots
# This is the reason why there is a dip at exactly the halfway mark of the periodic smooth tests
def calculate_conservation_at_interval(simulation, sim_variables, interval=10):
    axes, coordinates, conservation = sim_variables.axes, sim_variables.coordinates, {}

    dV = np.prod(np.diff(list(coordinates.values()), axis=1))
    simulation_timings = list(simulation.keys())
    simulation_timings.sort()
    intervals = [timing[-1] for timing in np.array_split(simulation_timings, abs(interval))]

    for t in intervals:
        _grid = simulation[t][:]  # Needs the '[:]' to access the array
        grid = gutils.convert("primitive", _grid, sim_variables)
        grid = np.sum(grid, axis=tuple(axes))
        conservation[t] = grid * dV
    return conservation


# Determine the analytical solution for a Sod shock test (only in 1d)
def calculate_Sod_analytical(grid, t, sim_variables):
    gamma, axis_coord, shock_pos = sim_variables.gamma, sim_variables.coordinates[0], sim_variables.shock_pos
    start_pos, end_pos = axis_coord
    box_length = np.diff(axis_coord)[0]

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
    boundary_54 = mfuncs.round_off(((shock_pos-(cs5*t)-start_pos)/box_length) * len(grid))
    boundary_43 = mfuncs.round_off(((shock_pos-(v_t*t)-start_pos)/box_length) * len(grid))
    boundary_32 = mfuncs.round_off(((shock_pos+(vx2*t)-start_pos)/box_length) * len(grid))
    boundary_21 = mfuncs.round_off(((shock_pos+(v_s*t)-start_pos)/box_length) * len(grid))

    # Define number of cells in the rarefaction wave
    rarefaction_cells = mfuncs.round_off(((cs5*t-v_t*t)/box_length) * len(grid))
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


# Determine the analytical solution for a Sedov blast wave [Dullemond, Numerical Methods, Chpt. 10]
def calculate_Sedov_analytical(grid, t, sim_variables):
    cells, gamma, dimensions, multidimensional, coordinates = sim_variables.cells, sim_variables.gamma, sim_variables.dimensions, sim_variables.multidimensional, sim_variables.coordinates
    rho0, vx0, vy0, vz0, P0, Bx0, By0, Bz0 = sim_variables.ambient
    P_inj = sim_variables.init_cond[sim_variables.pressure]

    # Create a physical half-grid for a single axis
    def make_half_grid(axis_coord, _cells):
        dh = np.abs(np.diff(axis_coord)[0])/_cells
        half_cell = .5 * dh
        return np.linspace(axis_coord[0]-half_cell, axis_coord[1]+half_cell, _cells+2)[1+int(_cells/2):-1]

    x_centre = np.average(coordinates[0])
    physical_halfgrid_x = make_half_grid(coordinates[0], cells[0])

    if multidimensional:
        y_centre = np.average(coordinates[1])
        physical_halfgrid_y = make_half_grid(coordinates[1], cells[1])

        if dimensions > 2:
            z_centre = np.average(coordinates[2])
            physical_halfgrid_z = make_half_grid(coordinates[2], cells[2])

            x, y, z = np.meshgrid(physical_halfgrid_x, physical_halfgrid_y, physical_halfgrid_z, indexing='ij')
            x0, y0, z0 = x - x_centre, y - y_centre, z - z_centre
            r = np.sqrt(x0**2 + y0**2 + z0**2)

        else:
            x, y = np.meshgrid(physical_halfgrid_x, physical_halfgrid_y, indexing='ij')
            x0, y0 = x - x_centre, y - y_centre
            r = np.sqrt(x0**2 + y0**2)

    else:
        x = physical_halfgrid_x
        x0 = r = np.abs(x - x_centre)

    # Initialise initial conditions and variables (assume ideal gas)
    E_blast = P_inj/(gamma-1)
    Eps = dimensions + 2

    # ----------------------------------------------------
    # Self-similar ODEs for A(η), B(η), C(η)
    # ----------------------------------------------------
    def sedov_derivs(eta, eta_funcs, dim=3):
        A, B, C = eta_funcs
        if eta <= 0:
            return [0.0, 0.0, 0.0]

        # Equation coefficients; M·y' = R , y = (A,B,C) [eq. 10.30-10.32]
        kappa = (dim - 1)/(gamma + 1)
        # ----------- matrix M (dim=3, spherical symmetry) -----------------
        m00 = eta * (kappa*C - 1)
        m01 = 0
        m02 = kappa * eta * A

        m10 = 0
        m11 = (gamma-1)/(gamma+1) * eta/A
        m12 = eta * (kappa*C - 1)

        m20 = (kappa*C - 1) * C**2
        m21 = gamma*kappa*C - 1
        m22 = kappa*(gamma*B + 3*A*C**2) - 2*A*C

        M = np.array([[m00, m01, m02],
                      [m10, m11, m12],
                      [m20, m21, m22]], dtype=float)

        # ----------- RHS vector R (solve for dimension) -------------------
        R0 = -A * C * (2*Eps - 4)/(gamma+1)
        R1 = .5*Eps*C - kappa*C**2 - (2 * B/A * (gamma-1)/(gamma+1))
        R2 = Eps/eta * ((1 - gamma*kappa*C)*B + (1 - kappa*C)*A*C**2)

        R = np.array([R0, R1, R2], dtype=float)

        return np.linalg.solve(M, R)

    # ----------------------------------------------------
    # Integrate similarity ODEs inward from shock
    # ----------------------------------------------------
    def integrate_profiles(eta_s, eta_min=1e-6, npts=3000):
        eta_start = (1 - 1e-8) * eta_s
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
        integrand = (B + A*C**2) * eta**(Eps-1)
        numer = 8 * ((2 * np.pi**(Eps/2 - 1)) / gamma_func(Eps/2 - 1))
        denom = Eps**2 * (gamma**2 - 1)
        return numer/denom * np.trapezoid(integrand, eta)

    # ----------------------------------------------------
    # Find ηs by shooting
    # ----------------------------------------------------
    def find_eta_s(g1=0.85, g2=1.35, itr=30):
        def residual(eta_s):
            profiles = integrate_profiles(eta_s)
            return energy_integral(*profiles) - 1

        # Secant iteration
        r1, r2 = residual(g1), residual(g2)
        for _ in range(itr):
            if abs(r2 - r1) < 1e-12:
                break
            g3 = g2 - r2 * (g2 - g1) / (r2 - r1)
            r3 = residual(g3)
            if abs(r3) < 1e-6:
                g2, r2 = g3, r3
                break
            g1, r1, g2, r2 = g2, r2, g3, r3
        eta_s = g2
        profiles = integrate_profiles(eta_s)
        return eta_s, *profiles

    # Determine the analytical position eta_s, together with functions A, B, C
    eta_s, eta, A, B, C = find_eta_s()

    # Similarity length scale and shock radius
    length_scale = (E_blast/rho0 * t**2)**(1/Eps)
    shock_vel = (2 * eta_s * length_scale) / (Eps * t)

    # Interpolate A,B,C
    interp = lambda func, z: np.where(
        z <= eta[0], 
        func[0], 
        np.where(
            z > eta[-1], 
            1., 
            sp.interpolate.PchipInterpolator(eta, func, extrapolate=False)(z)
        )
    )

    # Create analytical grid
    scaled_grid = r / length_scale
    density = np.full_like(r, rho0)
    pressure = np.full_like(r, P0)
    vx = np.full_like(r, vx0)

    # Post-shock jump conditions
    rho2 = (gamma + 1)/(gamma - 1) * rho0
    P2 = 2 * rho0 * shock_vel**2 / (gamma + 1)
    v2 = 2 * shock_vel / (gamma + 1)

    # Assign post-shock values
    post_shock = scaled_grid <= eta_s
    zeta = scaled_grid[post_shock]
    A_in, B_in, C_in = interp(A, zeta), interp(B, zeta), interp(C, zeta)

    density[post_shock] = rho2 * A_in
    pressure[post_shock] = P2 * (zeta/eta_s)**2 * B_in
    vx[post_shock] = v2 * (zeta/eta_s) * C_in

    if multidimensional:
        vy = np.full_like(r, vy0)
        vy[post_shock] = v2 * (zeta/eta_s) * C_in
        if dimensions > 2:
            vz = np.full_like(r, vz0)
            vz[post_shock] = v2 * (zeta/eta_s) * C_in

    # Populate solution into final array
    def mirror_even(q):
        for axis in range(q.ndim):
            q = np.concatenate((np.flip(q, axis=axis), q), axis=axis)
        return q

    def mirror_odd(q, odd_axes=()):
        for axis in range(q.ndim):
            reflected = np.flip(q, axis=axis)
            if axis in odd_axes:
                reflected = -reflected
            q = np.concatenate((reflected, q), axis=axis)
        return q

    arr = np.zeros_like(grid)
    arr[...,sim_variables.rho] = mirror_even(density)
    arr[...,sim_variables.pressure] = mirror_even(pressure)
    arr[...,sim_variables.vx] = mirror_odd(vx, odd_axes=(0,))

    if multidimensional:
        arr[...,sim_variables.vy] = mirror_odd(vy, odd_axes=(1,))
        if dimensions > 2:
            arr[...,sim_variables.vz] = mirror_odd(vz, odd_axes=(2,))

    return arr, eta_s * length_scale


# Determine the analytical solution for a manufactured Euler solution [Roy et al., 2004]
def calculate_Euler_analytical(grid, sim_variables):
    cells, t_end, multidimensional, dimensions, coordinates = sim_variables.cells, sim_variables.t_end, sim_variables.multidimensional, sim_variables.dimensions, sim_variables.coordinates
    rho, vx, vy, vz, pressure = sim_variables.rho, sim_variables.vx, sim_variables.vy, sim_variables.vz, sim_variables.pressure
    freq = sim_variables.test_specifics['freq']

    # Define array to be updated and returned
    arr = np.zeros_like(grid)
    arr[...,vx] = .1
    arr[...,vy] = .2
    arr[...,vz] = .3

    Lx, physical_grid_x = gutils.make_physical_grid(coordinates, cells, 0)

    if multidimensional:
        Ly, physical_grid_y = gutils.make_physical_grid(coordinates, cells, 1)

        if dimensions > 2:
            Lz, physical_grid_z = gutils.make_physical_grid(coordinates, cells, 2)

            x, y, z = np.meshgrid(physical_grid_x, physical_grid_y, physical_grid_z, indexing='ij')

            arr[...,rho] = 1 + .35*np.sin(freq*(x-t_end)/Lx) + .24*np.cos(freq*(y-t_end)/Ly) + .1*np.sin(freq*(z-t_end)/Lz)
            arr[...,pressure] = 1 + .23*np.sin(freq*(x-t_end)/Lx) + .19*np.cos(freq*(y-t_end)/Ly) + .2*np.cos(freq*(z-t_end)/Lz)

        else:
            x, y = np.meshgrid(physical_grid_x, physical_grid_y, indexing='ij')

            arr[...,rho] = 1 + .35*np.sin(freq*(x-t_end)/Lx) + .24*np.cos(freq*(y-t_end)/Ly)
            arr[...,vz] = 0
            arr[...,pressure] = 1 + .23*np.sin(freq*(x-t_end)/Lx) + .19*np.cos(freq*(y-t_end)/Ly)
    else:
        x = physical_grid_x

        arr[...,rho] = 1 + .35*np.sin(freq*(x-t_end)/Lx)
        arr[...,vy] = arr[...,vz] = 0
        arr[...,pressure] = 1 + .23*np.sin(freq*(x-t_end)/Lx)

    return arr