import numpy as np
import scipy as sp
from scipy.integrate import odeint, quad, solve_ivp

##############################################################################

# Customised rounding function
def roundOff(value):
    if value%int(value) >= .5:
        return int(value) + 1
    else:
        return int(value)


# Function for solution error calculation of sin-wave and Gaussian tests (L1 error norm)
def calculateSolutionError(simulation, startPos, endPos, config):
    timeKeys = [float(t) for t in simulation.keys()]
    q_num = simulation[str(max(timeKeys))]  # Get last array with (typically largest) time key

    xi = np.linspace(startPos, endPos, len(q_num))
    q_theo = np.copy(q_num)
    if config.startswith("gauss"):
        q_theo[:] = np.array([0,1,1,1,1e-6,0,0,0])
        q_theo[:,0] = 1e-3 + (1-1e-3) * np.exp(-(xi-((endPos+startPos)/2))**2/.01)
    else:
        q_theo[:] = np.array([0,1,1,1,1,0,0,0])
        q_theo[:,0] = 1 + (.1 * np.sin(2*np.pi*xi))

    thermal_num, thermal_theo = q_num[:,4]/q_num[:,0], q_theo[:,4]/q_theo[:,0]
    q_num, q_theo = np.c_[q_num, thermal_num], np.c_[q_theo, thermal_theo]

    return np.sum(np.abs(q_num-q_theo), axis=0)/len(q_num)


# Determine the analytical solution for a Sod shock test
def calculateSodAnalytical(tube, t, gamma, start, end, shock):
    # Define array to be updated and returned
    arr = np.zeros(tube.shape)

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



# Determine the analytical solution for a Sedov blast wave
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

    return None