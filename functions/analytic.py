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


"""
# Determine the analytical solution for a Sedov blast wave
def calculateSedovAnalytical(tube, t, gamma, start, end, shock, beta=1):
    N = len(tube)/2
    rho0, vx0, vy0, vz0, P0 = tube[-1]
    E_inject = P0/(rho0 * (gamma-1))
    rho, vx, P = tube[:,0], tube[:,1], tube[:,4]

    #x_arr = np.linspace(shock, end, int(N * ((end-shock)/(end-start))))
    x_arr = np.linspace(start, end, int(N))

    # Define array to be updated and returned
    arr = np.zeros((len(tube), len(tube[0])))

    R = beta * ((E_inject*t**2)/(rho0))**.2  # shock location
    D = .4 * R/t  # propagation velocity

    # Immediate post-shock values
    vS = (2 * D)/(gamma + 1)
    PS = (2 * rho0 * D**2)/(gamma + 1)
    rhoS = rho0 * (gamma + 1)/(gamma - 1)
    ES = PS/(rhoS * (gamma-1))
    csS = np.sqrt(gamma * PS/rhoS)

    _lambda=1
    f = lambda x: R*_lambda - x
    V_star = sp.optimize.fsolve(f, 1)[0]

    # Define scaled variables
    r, f, g, h = x_arr/R, vx/vS, rho/rhoS, P/PS


    rho_theo = rhoS * g
    vx_theo = D * f
    P_theo = PS * h

    return arr"""


"""# Determine the analytical solution for a Sedov blast wave
def calculateSedovAnalytical(tube, t, gamma, start, end, shock):

    abserr = 1e-8
    relerr = 1e-6

    # Solving the 1st-order coupled differential equations to determine the scaling of the post-shock variables
    # wrt to the immediate post-shock variables
    def equations(w, t, p):
        A, B, C = w
        ee, g = p
        
        x = (g+1)/(g+1-2*g*C)
        y = A/(2*(g-1))

        denom = y * (2*(g+1) - 4*C) - 2 * x * (g*B + 4*C**2 - A*C*(1-((2*C)/(g+1))))
        numer = (1/ee) * (5 * x * (2*C*(g*B + A*C**2) - B - A*C**2) - y * (C*(5*g + 5 - 4*C) - ((4*B*(g-1))/(A))))

        dA = (numer/denom - ((3*g - 1)/(ee*(g+1)))) / ((g-1-2*C)/(2*A) + (x*(1 - (2*C/(g+1)))*C**2)/(denom))
        dC = dA * ((g-1-2*C)/(2*A)) + C * ((3*g-1)/(ee*(g+1)))
        dB = x * ((2*(g*B + A*C**2)*(5*C + ee*dC) - 5*(B + A*C**2))/ee - ((dA * C**2 + 2*A*C*dC)*(1 - (2*C/(g+1)))))

        return [dA, dB, dC]
    
    # Determine the convergence of the values for A, B and C to 1
    def integral(ee, A, B, C, g):
        return ((32*np.pi)/(25*(g**2-1))) * (B + A*C**2) * ee**4

    rho0, vx0, vy0, vz0, P0 = tube[-1]
    E_inject = P0/(rho0 * (gamma-1))

    eta = ((E_inject*t**2)/(rho0))**-.2  # dimensionless scaling factor
    radii = np.linspace(shock, end, int(len(tube)/2 * ((end-shock)/(end-start))))

    w0, p0 = [1, 1, 1], [eta, gamma]
    wsol = odeint(equations, w0, radii, args=(p0,), atol=abserr, rtol=relerr)

    ns = 0
    for i in range(1000):
        A, B, C = wsol[-1]
        I = quad(integral, 0, ns, args=(A, B, C, gamma))
        if abs(I[0]-1) > relerr:
            ns += .001
        else:
            break

    R = ns * ((E_inject*t**2)/(rho0))**.2  # shock location
    vs = .4 * R/t  # propagation velocity

    # Immediate post-shock values
    vxS = (2 * vs)/(gamma + 1)
    PS = (2 * rho0 * vs**2)/(gamma + 1)
    rhoS = rho0 * (gamma + 1)/(gamma - 1)
    ES = PS/(rhoS * (gamma-1))
    csS = np.sqrt(gamma * PS/rhoS)

    # Define array to be updated and returned
    arr = np.zeros((len(tube), len(tube[0])))

    return arr"""



"""# Determine the analytical solution for a Sedov blast wave [Dullemond & Springel, 2012]
def calculateSedovAnalytical(tube, t, gamma, start, end, shock, steps=100):

    # Solving the 1st-order coupled differential equations to determine the scaling of the post-shock variables
    # wrt to the immediate post-shock variables
    def equations(eta, v):
        A, B, C = v

        alpha = 5*(gamma+1) - 4*C

        dB = (2*alpha*A*C**2 + B*(alpha - 2*gamma*(eta+3*C))) / (eta * (2*C*(2*gamma-1) - (gamma+1)))
        dC = (2*C**2 - ((5*C*(gamma+1))/2) + (((gamma-1)*(2*B + eta*dB))/A)) / (eta * (gamma+1-2*C))
        dA = -(2*A*(dC + (3*C/eta))) / (2*C - (gamma+1))

        return [dA, dB, dC]

    # Determine the convergence of the values for A, B and C to 1
    def integral(eta, v, g):
        A, B, C = v
        return np.sum(((32*np.pi)/(25*(g**2-1))) * (B + A*C**2) * eta**4)

    rho0, vx0, vy0, vz0, P0 = tube[-1]
    E_inject = P0/(rho0 * (gamma-1))

    for i in range(steps):
        etaS = i/steps
        wsol = solve_ivp(equations, (etaS,0), [1,1,1])

        I = integral(wsol.t, wsol.y, gamma)
        if abs(I - 1) <= 1e-6:
            break

    return None"""


# Determine the analytical solution for a Sedov blast wave (n = 1, 2, 3 for 1D, 2D, 3D respectively) [Timmes et. al., 2005]
def calculateSedovAnalytical(simInstance, t, simVariables, n=1):
    rho0, vx0, vy0, vz0, P0, Bx0, By0, Bz0 = simVariables.initialRight
    startPos, endPos, N, gamma = simVariables.startPos, simVariables.endPos, simVariables.cells, simVariables.gamma

    rho, vx, P = simInstance[int(len(simInstance)/2):,0], simInstance[int(len(simInstance)/2):,1], simInstance[int(len(simInstance)/2):,4]
    r = np.linspace((startPos+endPos)/2, endPos, int(N/2))
    E_blast = simVariables.initialLeft[4]/(simVariables.initialLeft[0]*(gamma-1))

    # Define the exponents
    a0, a2, a3, a5 = 2/(n-2), (1-gamma)/(n+2*(gamma-1)), n/(2*gamma-1+n), 2/(gamma-2)
    a1 = (((n+2)*gamma)/(2+n*(gamma-1))) * (((2*n*(2-gamma))/(gamma*(n+2)**2))-a2)
    a4 = (a1*(n+2))/(2-gamma)

    # Define frequently used components
    a = .25 * (n+2) * (gamma+1)
    b = (gamma+1)/(gamma-1)
    c = .5 * gamma * (n+2)
    d = ((n+2)*(gamma+1))/((n+2)*(gamma+1)-2*(2+n*(gamma-1)))
    e = .5 * (2 + n*(gamma-1))

    # Define the dimensionless shock speed and post-shock state
    v0, vs = 2/(gamma*(n+2)), 4/((n+2)*(gamma+1))

    # Define the energy integrals
    J1 = quad(
        lambda v: (
            ((gamma+1)/(1-gamma))
            * (v**2)
            * (a0/v + a2*c/(c*v-1) - a1*e/(1-e*v))
            * ((((a*v)**a0) * ((b*(c*v-1))**a2) * ((d*(1-e*v))**a1))**(-(n+2)))
            * ((b*(c*v-1))**a3)
            * ((d*(1-e*v))**a4)
            * ((b*(1-c*v/gamma))**a5)
        ), v0, vs
    )[0]
    J2 = quad(
        lambda v: (
            (-(gamma+1)/(2*gamma))
            * ((c*v-gamma)/(1-c*v))
            * (v**2)
            * (a0/v + a2*c/(c*v-1) - a1*e/(1-e*v))
            * ((((a*v)**a0) * ((b*(c*v-1))**a2) * ((d*(1-e*v))**a1))**(-(n+2)))
            * ((b*(c*v-1))**a3)
            * ((d*(1-e*v))**a4)
            * ((b*(1-c*v/gamma))**a5)
        ), v0, vs
    )[0]

    # Define the dimensionless energy of the shock
    if n == 1:
        alpha = .5*J1 + J2/(gamma-1)
    else:
        alpha = np.pi * (n-1) * (J1 + 2*J2/(gamma-1))
    E_dim = E_blast/alpha

    # Define post-shock variables
    r2 = ((E_dim/rho0)**(1/(n+2))) * (t**(2/(n+2)))
    us = (2/(n+2)) * (r2/t)
    u2 = 2*us/(gamma+1)
    rho2 = b * rho0
    P2 = (2*rho0*us**2)/(gamma+1)

    # Root-finding for similarity value
    fV = lambda Vs: r2*((a*Vs)**-a0)*((b*(c*Vs-1))**-a2)*((d*(1-e*Vs))**-a1) - r
    initial = np.ones_like(r)/10
    V_star = sp.optimize.fsolve(fV, initial)[0]

    # Define the Sedov functions
    _lambda = ((a*V_star)**-a0)*((b*(c*V_star-1))**-a2)*((d*(1-e*V_star))**-a1)
    f = a * vx * _lambda
    g = ((b*(c*vx-1))**a3) * ((d*(1-e*vx))**a4) * ((b*(1-.5*vx*(n+2)))**a5)
    h = ((a*vx)**(n*a0)) * ((d*(1-e*vx))**(a4-2*a1)) * ((b*(1-.5*vx*(n+2)))**(1+a5))

    # Define the solution
    arr = np.copy(simInstance[int(len(simInstance)/2):])
    arr[:] = simVariables.initialRight

    arr[:,0][r < r2] = (rho2*g)[r < r2]
    arr[:,1][r < r2] = (u2*f)[r < r2]
    arr[:,4][r < r2] = (P2*h)[r < r2]

    return np.concatenate((np.flip(arr, axis=0), arr))