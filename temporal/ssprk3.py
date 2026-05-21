
##############################################################################
# (Strong stability-preserving) Runge-Kutta 3 time integration (3rd-order)
##############################################################################

def run(sevolve_func, grid, fluxes, dt, sim_variables, **kwargs):
    try:
        register = kwargs['register']
    except KeyError:
        register = 3

    if register == 5:
        # Evolve system by SSP-RK (5,3) method (3rd-order) [Spiteri & Ruuth, 2002; Gottlieb et al., 2008]
        # SSP coeff = 2.65, effective SSP coeff = 0.53
        # Computation of 1st register
        k1 = grid + .3772689151171*dt*fluxes
        fluxes1 = sevolve_func(k1, sim_variables)

        # Computation of 2nd register
        k2 = k1 + .3772689151171*dt*fluxes1
        fluxes2 = sevolve_func(k2, sim_variables)

        # Computation of 3rd register
        k3 = .56656131914033*grid + .43343868085967*k2 + .16352294089771*dt*fluxes2
        fluxes3 = sevolve_func(k3, sim_variables)

        # Computation of 4th register
        k4 = .09299483444413*grid + .0000209036962*k1 + .90698426185967*k3 + .00071997378654*dt*fluxes + .34217696850008*dt*fluxes3
        fluxes4 = sevolve_func(k4, sim_variables)

        # Computation of 5th register
        return .0073613226092*grid + .20127980325145*k1 + .00182955389682*k2 + .78952932024253*k4 + dt*(
            .0027771981946*fluxes + .00001567934613*fluxes1 + .29786487010104*fluxes4)

    elif register == 4:
        # Evolve system by SSP-RK (4,3) method (3rd-order) [Spiteri & Ruuth, 2002; Gottlieb et al., 2008]
        # SSP coeff = 2, effective SSP coeff = 0.5
        # Computation of 1st register
        k1 = grid + .5*dt*fluxes
        fluxes1 = sevolve_func(k1, sim_variables)

        # Computation of 2nd register
        k2 = k1 + .5*dt*fluxes1
        fluxes2 = sevolve_func(k2, sim_variables)

        # Computation of 3rd register
        k3 = 1/6 * (4*grid + 2*k2 + dt*fluxes2)
        fluxes3 = sevolve_func(k3, sim_variables)

        # Computation of 4th register
        return k3 + .5*dt*fluxes3

    else:
        # Evolve system by SSP-RK (3,3) method (3rd-order) [Shu & Osher, 1988; Gottlieb et al., 2008]
        # SSP coeff = 1, effective SSP coeff = 0.333
        # Computation of 1st register
        k1 = grid + dt*fluxes
        fluxes1 = sevolve_func(k1, sim_variables)

        # Computation of 2nd register
        k2 = .25 * (3*grid + k1 + dt*fluxes1)
        fluxes2 = sevolve_func(k2, sim_variables)

        # Computation of the 3rd register
        return 1/3 * (grid + 2*k2 + 2*dt*fluxes2)