import concurrent.futures
from itertools import repeat

import numpy as np

from num_methods import ct
from schemes import pcm, plm, ppm, weno, cweno
from functions.generic import verbose_timer

##############################################################################
# Collates and controls space and time evolution
##############################################################################

# Evolve the system in space by a standardised workflow
@verbose_timer
def evolve_space(grid, sim_variables, first_stage=False):
    multidimensional, subgrid_category, axes, magnetic = sim_variables.multidimensional, sim_variables.subgrid_category, sim_variables.axes, sim_variables.magnetic

    # Convert to primitive variables
    centred_grid = ct.inverse_reconstruct(grid, sim_variables) if magnetic else grid
    primitive = sim_variables.convert("conservative", centred_grid, sim_variables)
    primitive[...,5+axes] = grid[...,5+axes]


    # Hydrodynamics computation (with fluxes and eigmax)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        if subgrid_category == "cweno":
            jobs = executor.map(cweno.run, repeat(primitive), repeat(sim_variables), axes)

        elif subgrid_category == "weno":
            jobs = executor.map(weno.run, repeat(primitive), repeat(sim_variables), axes)

        elif subgrid_category == "ppm":
            # Compute additional dissipation for PPM, if active
            if sim_variables.ppm_dissipate:
                eta = np.ones_like(grid[...,sim_variables.pressure])
                flattening_coeffs = executor.map(ppm.get_flattening_coeff, repeat(primitive), repeat(sim_variables), axes)
                eta = np.minimum(eta, np.min([coeff for coeff in flattening_coeffs], axis=0))

                jobs = executor.map(ppm.run, repeat(primitive), repeat(sim_variables), axes, repeat(eta))
            else:
                jobs = executor.map(ppm.run, repeat(primitive), repeat(sim_variables), axes)

        elif subgrid_category == "plm":
            jobs = executor.map(plm.run, repeat(primitive), repeat(sim_variables), axes)

        else:
            jobs = executor.map(pcm.run, repeat(primitive), repeat(sim_variables), axes)

        data = dict(zip(axes, jobs))


    # Extract specific values needed from data dict
    extract = lambda variable: {axis: data[axis][variable] for axis in axes}
    fluxes = extract('fluxes')

    # Magnetohydrodynamics computation
    if magnetic and multidimensional:
        alphas = extract('alphas')

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Magnetic transverse interfaces reconstructed along orthogonal axis/axes; use the averaged (+) & (-) values
            reconstruct_transverse = ct.reconstruct_transverse
            if subgrid_category == "ppm" and sim_variables.ppm_dissipate:
                reconstruct_transverse = lambda _data, _sim_variables, _axis: ct.reconstruct_transverse(_data, _sim_variables, _axis, eta=eta)
            ortho_interfaces = dict(enumerate(executor.map(reconstruct_transverse, repeat(data), repeat(sim_variables), range(3))))

            # The proper assignment of the corners is important for directional updates, so the dict keys are used for this assignment
            compute_emf = ct.compute_emf
            if sim_variables.higher_order:
                compute_emf = lambda _ortho_interfaces, _alphas, _axis: ct.compute_emf(_ortho_interfaces, _alphas, _axis, dissipative=True)
            emfs = dict(enumerate(executor.map(compute_emf, repeat(ortho_interfaces), repeat(alphas), range(3))))

            # Update fluxes with CT implementation
            ct_fluxes = executor.map(ct.compute_ct_flux, map(fluxes.get, axes), repeat(emfs), repeat(sim_variables), axes)
            fluxes.update(dict(zip(axes, ct_fluxes)))

    # Calculate the total fluxes through all upwind surfaces [F(i+1/2,j,k) - F(i-1/2,j,k)]/dx, [G(i,j+1/2,k) - G(i,j-1/2,k)]/dy, [H(i,j,k+1/2) - H(i,j,k-1/2)]/dz
    total_flux = -np.sum(list(fluxes.values()), axis=0)

    if first_stage:
        # Compute the maximum eigenvalues from each axis for determining the full time step
        eigmaxes = extract('eigmax')

        return total_flux, np.min(list(eigmaxes.values()))
    else:
        return total_flux


# Evolve the system in time by a standardised workflow
@verbose_timer
def evolve_time(grid, fluxes, dt, sim_variables):

    # Methods for linear and non-linear systems [Shu & Osher, 1988]
    if sim_variables.time_evo.startswith("ssprk"):
        time_evo = sim_variables.time_evo.replace(',','').replace('(','').replace(')','').replace('ssprk','')
        register, order = int(time_evo[:-1]), int(time_evo[-1])

        if order == 5:
            # Evolve system by SSP-RK (6,5) method (5th-order); effective SSP coeff = 1.78

            # Coefficients for SSP formulation
            coeffs = [84449/(3**12), 313328/(5*3**11), 9344/(3**10), 137216/(3**12), (2**13)/(3**11), 0, (2**16)/(5*3**12)]

            # Computation of i-th registers (i = 1,2,3,4,5,6)
            k, expansion = np.copy(grid), 0
            for _ in range(1,7):
                k += 9/16*dt*fluxes
                expansion += coeffs[_] * k
                if _ < 6:
                    fluxes = evolve_space(k, sim_variables)

            return coeffs[0]*grid + expansion

        elif order == 4:
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

    elif sim_variables.time_evo.startswith("r"):
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
        # Evolve system by a full time-step (1st-order)
        return grid + dt*fluxes