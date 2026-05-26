##############################################################################
# Forward (explicit) Euler time integration (1st-order)
##############################################################################

def run(sevolve_func, grid, fluxes, dt, sim_variables, **kwargs):
    return grid + dt*fluxes