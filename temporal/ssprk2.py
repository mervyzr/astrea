
##############################################################################
# (Strong stability-preserving) Runge-Kutta 2 time integration (2nd-order)
##############################################################################

# Effective SSP coeff = 0.5 [Gottlieb et al., 2008]
def run(sevolve_func, grid, fluxes, dt, sim_variables, **kwargs):
    # Computation of 1st register
    k1 = grid + dt*fluxes
    fluxes1 = sevolve_func(k1, sim_variables)

    # Computation of 2nd register
    return .5*(grid + k1 + dt*fluxes1)