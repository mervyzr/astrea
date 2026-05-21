from functions.generic import verbose_timer
from temporal import euler, rk4, ssprk2, ssprk3, ssprk4, ssprk5

##############################################################################
# Collates and controls time evolution
##############################################################################

@verbose_timer
def evolve(sevolve_func, grid, fluxes, dt, sim_variables):
    # Methods for linear and non-linear systems [Shu & Osher, 1988]
    # dt ≤ C_{SSP} * dt_{Euler} => CFL_{SSP-RK} ≤ C_{SSP} * CFL_{Euler}
    if sim_variables.time_evo.startswith("ssprk"):
        time_evo = sim_variables.time_evo.replace(',','').replace('(','').replace(')','').replace('ssprk','')
        register, order = int(time_evo[:-1]), int(time_evo[-1])

        if order == 5:
            update = lambda args: ssprk5.run(*args)
        elif order == 4:
            update = lambda args: ssprk4.run(*args, register=register)
        elif order == 3:
            update = lambda args: ssprk3.run(*args, register=register)
        else:
            update = lambda args: ssprk2.run(*args)

    elif sim_variables.time_evo.startswith("r"):
        update = lambda args: rk4.run(*args)

    else:
        update = lambda args: euler.run(*args)

    return update((sevolve_func, grid, fluxes, dt, sim_variables))