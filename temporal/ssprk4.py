import numpy as np

##############################################################################
# (Strong stability-preserving) Runge-Kutta 4 time integration (4th-order)
##############################################################################

def run(sevolve_func, grid, fluxes, dt, sim_variables, **kwargs):
    try:
        register = kwargs['register']
    except KeyError:
        register = 5

    if register == 10:
        # Evolve system by SSP-RK (10,4) method (4th-order) [Ketcheson, 2008]
        # SSP coeff = 6, effective SSP coeff = 0.6
        # Computation of i-th registers (i = 1,2,3,4)
        k = np.copy(grid)
        for _ in range(5):
            k += 1/6*dt*fluxes
            fluxes = sevolve_func(k, sim_variables)

        # Computation of 5th register
        k5 = 3/5*grid + 6/15*k + 1/15*dt*fluxes
        fluxes = sevolve_func(k5, sim_variables)

        # Computation of i-th registers (i = 6,7,8,9)
        _k = np.copy(k5)
        for _ in range(4):
            _k += 1/6*dt*fluxes
            fluxes = sevolve_func(_k, sim_variables)

        # Computation of 10th register
        return -11/35*grid + 5/7*k5 + 3/5*_k + 1/10*dt*fluxes

    else:
        # Evolve system by SSP-RK (5,4) method (4th-order) [Kraaijevanger, 1991; Spiteri & Ruuth, 2002; Gottlieb et al., 2008]
        # SSP coeff = 1.508, effective SSP coeff = 0.302
        # Computation of 1st register
        k1 = grid + .39175222657189*dt*fluxes
        fluxes1 = sevolve_func(k1, sim_variables)

        # Computation of 2nd register
        k2 = .444370493651235*grid + .555629506348765*k1 + .368410593050371*dt*fluxes1
        fluxes2 = sevolve_func(k2, sim_variables)

        # Computation of 3rd register
        k3 = .620101851488403*grid + .379898148511597*k2 + .251891774271694*dt*fluxes2
        fluxes3 = sevolve_func(k3, sim_variables)

        # Computation of 4th register
        k4 = .178079954393132*grid + .821920045606868*k3 + .544974750228521*dt*fluxes3
        fluxes4 = sevolve_func(k4, sim_variables)

        # Computation of 5th register
        return grid + dt * (.14681187608478657*fluxes + .24848290944497617*fluxes1 + .10425883033198098*fluxes2 + .2744389009013507*fluxes3 + .226007483236906*fluxes4)
        #return .517231671970585*k2 + .096059710526147*k3 + .06369246866629*dt*fluxes3 + .386708617503269*k4 + .226007483236906*dt*fluxes4