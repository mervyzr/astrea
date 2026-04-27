import numpy as np

##############################################################################
# (Strong stability-preserving) Runge-Kutta 5 time integration (5th-order)
##############################################################################

# Effective SSP coeff = 1.78
def run(sevolve_func, grid, fluxes, dt, sim_variables, **kwargs):
    # Coefficients for SSP formulation
    coeffs = [84449/(3**12), 313328/(5*3**11), 9344/(3**10), 137216/(3**12), (2**13)/(3**11), 0, (2**16)/(5*3**12)]

    # Computation of i-th registers (i = 1,2,3,4,5,6)
    k, expansion = np.copy(grid), 0
    for _ in range(1,7):
        k += 9/16*dt*fluxes
        expansion += coeffs[_] * k
        if _ < 6:
            fluxes = sevolve_func(k, sim_variables)

    return coeffs[0]*grid + expansion