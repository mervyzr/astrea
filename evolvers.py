import numpy as np

from schemes import pcm, plm, ppm, weno

##############################################################################
# Collates and controls space and time evolution
##############################################################################

# Operator H as a function of the reconstruction values; calculate the flux through the surface [F(i+1/2) - F(i-1/2)]/dx
def compute_H(interface_fluxes, sim_variables):
    total_flux = 0
    for axes in sim_variables.permutations:
        Riemann_flux = interface_fluxes[axes].flux
        flux_diff = np.diff(Riemann_flux, axis=0)/sim_variables.dx
        total_flux += flux_diff.transpose(np.argsort(axes))
    return -total_flux


# Evolve the system in space by a standardised workflow
def evolve_space(grid, sim_variables):
    if "weno" in sim_variables.subgrid or sim_variables.subgrid == "w":
        return weno.run(grid, sim_variables)
    elif sim_variables.subgrid in ["ppm", "parabolic", "p"]:
        return ppm.run_modified(grid, sim_variables, dissipate=False)
        #return ppm.run(grid, sim_variables)
    elif sim_variables.subgrid in ["plm", "linear", "l"]:
        return plm.run(grid, sim_variables)
    else:
        return pcm.run(grid, sim_variables)


# Evolve the system in time by a standardised workflow
def evolve_time(grid, interface_fluxes, dt, sim_variables):
    h_zero = compute_H(interface_fluxes, sim_variables)

    if sim_variables.timestep in ["ssprk(5,4)", "(5,4)"]:
        # Evolve system by SSP-RK (5,4) method (4th-order); effective SSP coeff = 0.302
        # Computation of 1st register
        k1 = grid + .39175222657189*dt*h_zero

        # Computation of 2nd register
        interface_fluxes1 = evolve_space(k1, sim_variables)
        k2 = .444370493651235*grid + .555629506348765*k1 + .368410593050371*dt*compute_H(interface_fluxes1, sim_variables)

        # Computation of 3rd register
        interface_fluxes2 = evolve_space(k2, sim_variables)
        k3 = .620101851488403*grid + .379898148511597*k2 + .251891774271694*dt*compute_H(interface_fluxes2, sim_variables)

        # Computation of 4th register
        interface_fluxes3 = evolve_space(k3, sim_variables)
        k4 = .178079954393132*grid + .821920045606868*k3 + .544974750228521*dt*compute_H(interface_fluxes3, sim_variables)

        # Computation of the final update
        interface_fluxes4 = evolve_space(k4, sim_variables)
        return .517231671970585*k2 + .096059710526147*k3 + .06369246866629*dt*compute_H(interface_fluxes3, sim_variables) + .386708617503269*k4 + .226007483236906*dt*compute_H(interface_fluxes4, sim_variables)

    elif sim_variables.timestep in ["ssprk(5,3)", "(5,3)"]:
        # Evolve system by SSP-RK (5,3) method (3rd-order); effective SSP coeff = 0.53
        # Computation of 1st register
        k1 = grid + .3772689151171*dt*h_zero

        # Computation of 2nd register
        interface_fluxes1 = evolve_space(k1, sim_variables)
        k2 = k1 + .3772689151171*dt*compute_H(interface_fluxes1, sim_variables)

        # Computation of 3rd register
        interface_fluxes2 = evolve_space(k2, sim_variables)
        k3 = .56656131914033*grid + .43343868085967*k2 + .16352294089771*dt*compute_H(interface_fluxes2, sim_variables)

        # Computation of 4th register
        interface_fluxes3 = evolve_space(k3, sim_variables)
        k4 = .09299483444413*grid + .0000209036962*k1 + .90698426185967*k3 + .00071997378654*dt*h_zero + .34217696850008*dt*compute_H(interface_fluxes3, sim_variables)

        # Computation of the final update
        interface_fluxes4 = evolve_space(k4, sim_variables)
        return .0073613226092*grid + .20127980325145*k1 + .00182955389682*k2 + .78952932024253*k4 + (dt * (.0027771981946*h_zero + .00001567934613*compute_H(interface_fluxes1, sim_variables) + .29786487010104*compute_H(interface_fluxes4, sim_variables)))

    elif sim_variables.timestep in ["ssprk(4,3)", "(4,3)"]:
        # Evolve system by SSP-RK (4,3) method (3rd-order); effective SSP coeff = 0.5
        # Computation of 1st register
        k1 = grid + .5*dt*h_zero

        # Computation of 2nd register
        interface_fluxes1 = evolve_space(k1, sim_variables)
        k2 = k1 + .5*dt*compute_H(interface_fluxes1, sim_variables)

        # Computation of 3rd register
        interface_fluxes2 = evolve_space(k2, sim_variables)
        k3 = 1/6 * (4*grid + 2*k2 + dt*compute_H(interface_fluxes2, sim_variables))

        # Computation of the final update
        interface_fluxes3 = evolve_space(k3, sim_variables)
        return k3 + .5*dt*compute_H(interface_fluxes3, sim_variables)

    elif sim_variables.timestep in ["ssprk(3,3)", "(3,3)"]:
        # Evolve system by SSP-RK (3,3) method (3rd-order); effective SSP coeff = 0.333
        # Computation of 1st register
        k1 = grid + dt*h_zero

        # Computation of 2nd register
        interface_fluxes1 = evolve_space(k1, sim_variables)
        k2 = .25 * (3*grid + k1 + dt*compute_H(interface_fluxes1, sim_variables))

        # Computation of the final update
        interface_fluxes2 = evolve_space(k2, sim_variables)
        return 1/3 * (grid + 2*k2 + 2*dt*compute_H(interface_fluxes2, sim_variables))

    elif sim_variables.timestep in ["ssprk(2,2)", "(2,2)"]:
        # Evolve system by SSP-RK (2,2) method (2nd-order); effective SSP coeff = 0.5
        # Computation of 1st register
        k1 = grid + .5*dt*h_zero

        # Computation of 2nd register
        interface_fluxes1 = evolve_space(k1, sim_variables)
        return .5*(grid + k1 + dt*compute_H(interface_fluxes1, sim_variables))

    elif sim_variables.timestep == "rk4":
        # Evolve the system by RK4 method (4th-order); effective SSP coeff = 0.25
        # Computation of 1st register
        k1 = grid + .5*dt*h_zero

        # Computation of 2nd register
        interface_fluxes1 = evolve_space(k1, sim_variables)
        k2 = grid + .5*dt*compute_H(interface_fluxes1, sim_variables)

        # Computation of 3rd register
        interface_fluxes2 = evolve_space(k2, sim_variables)
        k3 = grid + dt*compute_H(interface_fluxes2, sim_variables)

        # Computation of the final update
        interface_fluxes3 = evolve_space(k3, sim_variables)
        return grid + (dt * (h_zero + 2*compute_H(interface_fluxes1, sim_variables) + 2*compute_H(interface_fluxes2, sim_variables) + compute_H(interface_fluxes3, sim_variables)))/6

    else:
        # Evolve system by a full timestep (1st-order)
        return grid + dt*h_zero