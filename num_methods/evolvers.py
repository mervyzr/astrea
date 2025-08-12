import concurrent.futures as cfutures
from itertools import repeat

import numpy as np

from num_methods import ct
from schemes import pcm, plm, ppm, weno

##############################################################################
# Collates and controls space and time evolution
##############################################################################

# Evolve the system in space by a standardised workflow
def evolve_space(grid, sim_variables, first_stage=False):
    dimension, subgrid, axes, magnetic = sim_variables.dimension, sim_variables.subgrid, sim_variables.axes, sim_variables.magnetic
    pressure, dissipate = sim_variables.pressure, sim_variables.ppm_dissipate
    data = {}

    # Convert to primitive variables
    primitive = sim_variables.convert("conservative", grid, sim_variables, staggered=magnetic)


    # Compute additional dissipation for PPM, if active
    if dissipate and subgrid in ["ppm", "parabolic", "p"]:
        eta = np.ones_like(grid[...,pressure])
        with cfutures.ThreadPoolExecutor() as executor:
            for flattening_coeff in executor.map(ppm.get_flattening_coeff, repeat(primitive), repeat(sim_variables), axes):
                eta = np.minimum(eta, flattening_coeff)


    # Hydrodynamics computation (with fluxes and eigmax)
    with cfutures.ThreadPoolExecutor() as executor:
        if subgrid.startswith("w"):
            jobs = executor.map(weno.run, repeat(primitive), repeat(sim_variables), axes)

        elif subgrid in ["ppm", "parabolic", "p"]:            
            if dissipate:
                jobs = executor.map(ppm.run, repeat(primitive), repeat(sim_variables), axes, repeat(eta))
            else:
                jobs = executor.map(ppm.run, repeat(primitive), repeat(sim_variables), axes)

        elif subgrid in ["plm", "linear", "l"]:
            jobs = executor.map(plm.run, repeat(primitive), repeat(sim_variables), axes)

        else:
            jobs = executor.map(pcm.run, repeat(primitive), repeat(sim_variables), axes)

        for idx, result in enumerate(jobs):
            data[axes[idx]] = result


    get_flux = lambda dct: [axis_dict['fluxes'] for axis_dict in list(dct.values())]

    # Compute the maximum eigenvalues for determining the full time step
    eigmax = np.min([axis_dict['eigmax'] for axis_dict in list(data.values())])


    # Magnetohydrodynamics computation
    if magnetic and dimension == 2:
        e3U = ct.compute_corner(data, sim_variables)

        with cfutures.ThreadPoolExecutor() as executor:
            jobs = executor.map(ct.compute_ct_flux, repeat(e3U), get_flux(data), repeat(sim_variables), axes)
            for idx, result in enumerate(jobs):
                data[axes[idx]]['fluxes'] = result

    # Calculate the total fluxes through all upwind surfaces [F(i+1/2,j) - F(i-1/2,j)]/dx, [G(i,j+1/2) - G(i,j-1/2)]/dy
    fluxes = -np.sum(get_flux(data), axis=0)

    if first_stage:
        return fluxes, eigmax
    else:
        return fluxes


# Evolve the system in time by a standardised workflow
def evolve_time(grid, fluxes, dt, sim_variables):

    # Methods for linear and non-linear systems [Shu & Osher, 1988]
    if sim_variables.timestep.startswith("ssprk"):
        timestep = sim_variables.timestep.replace(',','').replace('(','').replace(')','').replace('ssprk','')
        register, order = int(timestep[:-1]), int(timestep[-1])

        if order == 4:
            if register == 10:
                # Evolve system by SSP-RK (10,4) method (4th-order); effective SSP coeff = 0.6 [Ketcheson, 2008]
                # Computation of i-th registers (i = 1,2,3,4)
                k = np.copy(grid)
                for _ in range(5):
                    k += 1/6*dt*fluxes
                    fluxes = evolve_space(k, sim_variables)

                # Computation of 5th register
                k5 = 3/5*grid + 6/15*k + 1/15*dt*fluxes
                fluxes = evolve_space(k5, sim_variables)

                # Computation of i-th registers (i = 6,7,8,9)
                _k = np.copy(k5)
                for _ in range(4):
                    _k += 1/6*dt*fluxes
                    fluxes = evolve_space(_k, sim_variables)

                # Computation of 10th register
                return -11/35*grid + 5/7*k5 + 3/5*_k + 1/10*dt*fluxes

            else:
                # Evolve system by SSP-RK (5,4) method (4th-order); effective SSP coeff = 0.302 [Kraaijevanger, 1991; Spiteri & Ruuth, 2002; Gottlieb et al., 2008]
                # Computation of 1st register
                k1 = grid + .39175222657189*dt*fluxes
                fluxes1 = evolve_space(k1, sim_variables)

                # Computation of 2nd register
                k2 = .444370493651235*grid \
                    + .555629506348765*k1 \
                    + .368410593050371*dt*fluxes1
                fluxes2 = evolve_space(k2, sim_variables)

                # Computation of 3rd register
                k3 = .620101851488403*grid \
                    + .379898148511597*k2 \
                    + .251891774271694*dt*fluxes2
                fluxes3 = evolve_space(k3, sim_variables)

                # Computation of 4th register
                k4 = .178079954393132*grid \
                    + .821920045606868*k3 \
                    + .544974750228521*dt*fluxes3
                fluxes4 = evolve_space(k4, sim_variables)

                # Computation of 5th register
                return .517231671970585*k2 \
                    + .096059710526147*k3 \
                    + .06369246866629*dt*fluxes3 \
                    + .386708617503269*k4 \
                    + .226007483236906*dt*fluxes4

        elif order == 3:
            if register == 5:
                # Evolve system by SSP-RK (5,3) method (3rd-order); effective SSP coeff = 0.53 [Spiteri & Ruuth, 2002; Gottlieb et al., 2008]
                # Computation of 1st register
                k1 = grid + .3772689151171*dt*fluxes
                fluxes1 = evolve_space(k1, sim_variables)

                # Computation of 2nd register
                k2 = k1 + .3772689151171*dt*fluxes1
                fluxes2 = evolve_space(k2, sim_variables)

                # Computation of 3rd register
                k3 = .56656131914033*grid \
                    + .43343868085967*k2 \
                    + .16352294089771*dt*fluxes2
                fluxes3 = evolve_space(k3, sim_variables)

                # Computation of 4th register
                k4 = .09299483444413*grid \
                    + .0000209036962*k1 \
                    + .90698426185967*k3 \
                    + .00071997378654*dt*fluxes \
                    + .34217696850008*dt*fluxes3
                fluxes4 = evolve_space(k4, sim_variables)

                # Computation of 5th register
                return .0073613226092*grid \
                    + .20127980325145*k1 \
                    + .00182955389682*k2 \
                    + .78952932024253*k4 \
                    + (dt * (
                        .0027771981946*fluxes \
                        + .00001567934613*fluxes1 \
                        + .29786487010104*fluxes4
                        ))

            elif register == 4:
                # Evolve system by SSP-RK (4,3) method (3rd-order); effective SSP coeff = 0.5 [Spiteri & Ruuth, 2002; Gottlieb et al., 2008]
                # Computation of 1st register
                k1 = grid + .5*dt*fluxes
                fluxes1 = evolve_space(k1, sim_variables)

                # Computation of 2nd register
                k2 = k1 + .5*dt*fluxes1
                fluxes2 = evolve_space(k2, sim_variables)

                # Computation of 3rd register
                k3 = 1/6 * (4*grid + 2*k2 + dt*fluxes2)
                fluxes3 = evolve_space(k3, sim_variables)

                # Computation of 4th register
                return k3 + .5*dt*fluxes3

            else:
                # Evolve system by SSP-RK (3,3) method (3rd-order); effective SSP coeff = 0.333 [Shu & Osher, 1988; Gottlieb et al., 2008]
                # Computation of 1st register
                k1 = grid + dt*fluxes
                fluxes1 = evolve_space(k1, sim_variables)

                # Computation of 2nd register
                k2 = .25 * (3*grid + k1 + dt*fluxes1)
                fluxes2 = evolve_space(k2, sim_variables)

                # Computation of the 3rd register
                return 1/3 * (grid + 2*k2 + 2*dt*fluxes2)

        else:
            # Evolve system by SSP-RK (2,2) method (2nd-order); effective SSP coeff = 0.5 [Gottlieb et al., 2008]
            # Computation of 1st register
            k1 = grid + dt*fluxes
            fluxes1 = evolve_space(k1, sim_variables)

            # Computation of 2nd register
            return .5*(grid + k1 + dt*fluxes1)

    elif sim_variables.timestep.startswith("r"):
        # Evolve the system by RK4 method (4th-order); effective SSP coeff = 0.25
        # Computation of 1st register
        k1 = grid + .5*dt*fluxes
        fluxes1 = evolve_space(k1, sim_variables)

        # Computation of 2nd register
        k2 = grid + .5*dt*fluxes1
        fluxes2 = evolve_space(k2, sim_variables)

        # Computation of 3rd register
        k3 = grid + dt*fluxes2
        fluxes3 = evolve_space(k3, sim_variables)

        # Computation of the final update
        return grid + 1/6 * (dt * (fluxes + 2*fluxes1 + 2*fluxes2 + fluxes3))

    else:
        # Evolve system by a full timestep (1st-order)
        return grid + dt*fluxes