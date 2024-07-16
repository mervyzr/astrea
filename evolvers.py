import numpy as np

from schemes import pcm, plm, ppm, weno

##############################################################################

# Operator L as a function of the reconstruction values; calculate the flux through the surface [F(i+1/2) - F(i-1/2)]/dx
def getL(fluxes, dx):
    return -np.diff(fluxes, axis=0)/dx


# Evolve the system in space by a standardised workflow
def evolveSpace(tube, simVariables):
    if simVariables.subgrid in ["weno", "w"]:
        return weno.run(tube, simVariables)
    elif simVariables.subgrid in ["ppm", "parabolic", "p"]:
        #return ppm.runModified(tube, simVariables, dissipate=False)
        return ppm.run(tube, simVariables)
    elif simVariables.subgrid in ["plm", "linear", "l"]:
        return plm.run(tube, simVariables)
    else:
        return pcm.run(tube, simVariables)


# Evolve the system in time by a standardised workflow
def evolveTime(domain, fluxes, dt, simVariables):
    dx, timestep = simVariables.dx, simVariables.timestep
    Lq0 = getL(fluxes, dx)

    if timestep == "ssprk(5,4)":
        # Evolve system by SSP-RK (5,4) method (4th-order); effective SSP coeff = 0.302
        # Computation of 1st register
        k1 = domain + .39175222657189*dt*Lq0

        # Computation of 2nd register
        data1 = evolveSpace(k1, simVariables)
        k2 = .444370493651235*domain + .555629506348765*k1 + .368410593050371*dt*getL(data1.flux, dx)

        # Computation of 3rd register
        data2 = evolveSpace(k2, simVariables)
        k3 = .620101851488403*domain + .379898148511597*k2 + .251891774271694*dt*getL(data2.flux, dx)

        # Computation of 4th register
        data3 = evolveSpace(k3, simVariables)
        k4 = .178079954393132*domain + .821920045606868*k3 + .544974750228521*dt*getL(data3.flux, dx)

        # Computation of the final update
        data4 = evolveSpace(k4, simVariables)
        return .517231671970585*k2 + .096059710526147*k3 + .06369246866629*dt*getL(data3.flux, dx) + .386708617503269*k4 + .226007483236906*dt*getL(data4.flux, dx)

    elif timestep == "ssprk(5,3)":
        # Evolve system by SSP-RK (5,3) method (3rd-order); effective SSP coeff = 0.53
        # Computation of 1st register
        k1 = domain + .3772689151171*dt*Lq0

        # Computation of 2nd register
        data1 = evolveSpace(k1, simVariables)
        k2 = k1 + .3772689151171*dt*getL(data1.flux, dx)

        # Computation of 3rd register
        data2 = evolveSpace(k2, simVariables)
        k3 = .56656131914033*domain + .43343868085967*k2 + .16352294089771*dt*getL(data2.flux, dx)

        # Computation of 4th register
        data3 = evolveSpace(k3, simVariables)
        k4 = .09299483444413*domain + .0000209036962*k1 + .90698426185967*k3 + .00071997378654*dt*Lq0 + .34217696850008*dt*getL(data3.flux, dx)

        # Computation of the final update
        data4 = evolveSpace(k4, simVariables)
        return .0073613226092*domain + .20127980325145*k1 + .00182955389682*k2 + .78952932024253*k4 + (dt * (.0027771981946*Lq0 + .00001567934613*getL(data1.flux, dx) + .29786487010104*getL(data4.flux, dx)))

    elif timestep == "ssprk(4,3)":
        # Evolve system by SSP-RK (4,3) method (3rd-order); effective SSP coeff = 0.5
        # Computation of 1st register
        k1 = domain + .5*dt*Lq0

        # Computation of 2nd register
        data1 = evolveSpace(k1, simVariables)
        k2 = k1 + .5*dt*getL(data1.flux, dx)

        # Computation of 3rd register
        data2 = evolveSpace(k2, simVariables)
        k3 = 1/6 * (4*domain + 2*k2 + dt*getL(data2.flux, dx))

        # Computation of the final update
        data3 = evolveSpace(k3, simVariables)
        return k3 + .5*dt*getL(data3.flux, dx)

    elif timestep == "ssprk(3,3)":
        # Evolve system by SSP-RK (3,3) method (3rd-order); effective SSP coeff = 0.333
        # Computation of 1st register
        k1 = domain + dt*Lq0

        # Computation of 2nd register
        data1 = evolveSpace(k1, simVariables)
        k2 = .25 * (3*domain + k1 + dt*getL(data1.flux, dx))

        # Computation of the final update
        data2 = evolveSpace(k2, simVariables)
        return 1/3 * (domain + 2*k2 + 2*dt*getL(data2.flux, dx))

    elif timestep == "ssprk(2,2)":
        # Evolve system by SSP-RK (2,2) method (2nd-order); effective SSP coeff = 0.5
        # Computation of 1st register
        k1 = domain + .5*dt*Lq0

        # Computation of 2nd register
        data1 = evolveSpace(k1, simVariables)
        return .5*(domain + k1 + dt*getL(data1.flux, dx))

    elif timestep == "rk4":
        # Evolve the system by RK4 method (4th-order); effective SSP coeff = 0.25
        # Computation of 1st register
        k1 = domain + .5*dt*Lq0

        # Computation of 2nd register
        data1 = evolveSpace(k1, simVariables)
        k2 = domain + .5*dt*getL(data1.flux, dx)

        # Computation of 3rd register
        data2 = evolveSpace(k2, simVariables)
        k3 = domain + dt*getL(data2.flux, dx)

        # Computation of the final update
        data3 = evolveSpace(k3, simVariables)
        return domain + (dt * (Lq0 + 2*getL(data1.flux, dx) + 2*getL(data2.flux, dx) + getL(data3.flux, dx)))/6

    else:
        # Evolve system by a full timestep (1st-order)
        return domain + dt*Lq0