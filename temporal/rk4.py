
##############################################################################
# (Standard) Runge-Kutta 4 time integration (4th-order)
##############################################################################

# Effective SSP coeff = 0.25
def run(sevolve_func, grid, fluxes, dt, sim_variables, **kwargs):
    # Computation of 1st register
    k1 = grid + .5*dt*fluxes
    fluxes1 = sevolve_func(k1, sim_variables)

    # Computation of 2nd register
    k2 = grid + .5*dt*fluxes1
    fluxes2 = sevolve_func(k2, sim_variables)

    # Computation of 3rd register
    k3 = grid + dt*fluxes2
    fluxes3 = sevolve_func(k3, sim_variables)

    # Computation of the final update
    return grid + 1/6 * (dt * (fluxes + 2*fluxes1 + 2*fluxes2 + fluxes3))