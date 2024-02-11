import sys

import functions as fn

##############################################################################

def evolveTime(shockTube, dt, fluxes, stepper):
    dx = shockTube.dx
    Lq0 = fn.getL(fluxes, dx)

    if stepper == "ssprk(5,4)":
        # Evolve system by SSP-RK (5,4) method (4th-order); effective SSP coeff = 0.302
        # Computation of 1st register
        k1 = shockTube.domain + .39175222657189*dt*Lq0

        # Computation of 2nd register
        shockTube.eigmax = sys.float_info.epsilon
        flux1 = fn.evolveSpace(shockTube, k1)
        k2 = .444370493651235*shockTube.domain + .555629506348765*k1 + .368410593050371*dt*fn.getL(flux1, dx)

        # Computation of 3rd register
        shockTube.eigmax = sys.float_info.epsilon
        flux2 = fn.evolveSpace(shockTube, k2)
        k3 = .620101851488403*shockTube.domain + .379898148511597*k2 + .251891774271694*dt*fn.getL(flux2, dx)

        # Computation of 4th register
        shockTube.eigmax = sys.float_info.epsilon
        flux3 = fn.evolveSpace(shockTube, k3)
        k4 = .178079954393132*shockTube.domain + .821920045606868*k3 + .544974750228521*dt*fn.getL(flux3, dx)

        # Computation of the final update
        shockTube.eigmax = sys.float_info.epsilon
        flux4 = fn.evolveSpace(shockTube, k4)
        return .517231671970585*k2 + .096059710526147*k3 + .06369246866629*dt*fn.getL(flux3, dx) + .386708617503269*k4 + .226007483236906*dt*fn.getL(flux4, dx)
    
    elif stepper == "ssprk(5,3)":
        # Evolve system by SSP-RK (5,3) method (3rd-order); effective SSP coeff = 0.53
        # Computation of 1st register
        k1 = shockTube.domain + .3772689151171*dt*Lq0

        # Computation of 2nd register
        shockTube.eigmax = sys.float_info.epsilon
        flux1 = fn.evolveSpace(shockTube, k1)
        k2 = k1 + .3772689151171*dt*fn.getL(flux1, dx)

        # Computation of 3rd register
        shockTube.eigmax = sys.float_info.epsilon
        flux2 = fn.evolveSpace(shockTube, k2)
        k3 = .56656131914033*shockTube.domain + .43343868085967*k2 + .16352294089771*dt*fn.getL(flux2, dx)

        # Computation of 4th register
        shockTube.eigmax = sys.float_info.epsilon
        flux3 = fn.evolveSpace(shockTube, k3)
        k4 = .09299483444413*shockTube.domain + .0000209036962*k1 + .90698426185967*k3 + .00071997378654*dt*Lq0 + .34217696850008*dt*fn.getL(flux3, dx)

        # Computation of the final update
        shockTube.eigmax = sys.float_info.epsilon
        flux4 = fn.evolveSpace(shockTube, k4)
        return .0073613226092*shockTube.domain + .20127980325145*k1 + .00182955389682*k2 + .78952932024253*k4 + (dt * (.0027771981946*Lq0 + .00001567934613*fn.getL(flux1, dx) + .29786487010104*fn.getL(flux4, dx)))
    
    elif stepper == "ssprk(4,3)":
        # Evolve system by SSP-RK (4,3) method (3rd-order); effective SSP coeff = 0.5
        # Computation of 1st register
        k1 = shockTube.domain + .5*dt*Lq0

        # Computation of 2nd register
        shockTube.eigmax = sys.float_info.epsilon
        flux1 = fn.evolveSpace(shockTube, k1)
        k2 = k1 + .5*dt*fn.getL(flux1, dx)

        # Computation of 3rd register
        shockTube.eigmax = sys.float_info.epsilon
        flux2 = fn.evolveSpace(shockTube, k2)
        k3 = 1/6 * (4*shockTube.domain + 2*k2 + dt*fn.getL(flux2, dx))

        # Computation of the final update
        shockTube.eigmax = sys.float_info.epsilon
        flux3 = fn.evolveSpace(shockTube, k3)
        return k3 + .5*dt*fn.getL(flux3, dx)

    elif stepper == "ssprk(3,3)":
        # Evolve system by SSP-RK (3,3) method (3rd-order); effective SSP coeff = 0.333
        # Computation of 1st register
        k1 = shockTube.domain + dt*Lq0

        # Computation of 2nd register
        shockTube.eigmax = sys.float_info.epsilon
        flux1 = fn.evolveSpace(shockTube, k1)
        k2 = .25 * (3*shockTube.domain + k1 + dt*fn.getL(flux1, dx))

        # Computation of the final update
        shockTube.eigmax = sys.float_info.epsilon
        flux2 = fn.evolveSpace(shockTube, k2)
        return 1/3 * (shockTube.domain + 2*k2 + 2*dt*fn.getL(flux2, dx))

    elif stepper == "rk4":
        # Evolve the system by RK4 method (4th-order); effective SSP coeff = 0.25
        # Computation of 1st register
        k1 = shockTube.domain + .5*dt*Lq0

        # Computation of 2nd register
        shockTube.eigmax = sys.float_info.epsilon
        flux1 = fn.evolveSpace(shockTube, k1)
        k2 = shockTube.domain + .5*dt*fn.getL(flux1, dx)

        # Computation of 3rd register
        shockTube.eigmax = sys.float_info.epsilon
        flux2 = fn.evolveSpace(shockTube, k2)
        k3 = shockTube.domain + .5*dt*fn.getL(flux2, dx)

        # Computation of the final update
        shockTube.eigmax = sys.float_info.epsilon
        flux3 = fn.evolveSpace(shockTube, k3)
        return shockTube.domain + (dt * (Lq0 + 2*fn.getL(flux1, dx) + 2*fn.getL(flux2, dx) + fn.getL(flux3, dx)))/6

    else:
        # Evolve system by a full timestep (1st-order)
        return shockTube.domain + dt*Lq0