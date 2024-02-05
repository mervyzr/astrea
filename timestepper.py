import numpy as np

import functions as fn
import solvers as solv

##############################################################################

class TimeStepper:
    def __init__(self, domain, fluxes, dt, dx, boundary, gamma, solver):
        self.domain = domain
        self.fluxes = fluxes
        self.dt = dt
        self.dx = dx
        self.boundary = boundary
        self.gamma = gamma
        self.solver = solver
    
    # Evolve the system
    def evolveSystem(self, timestep):
        Lq = fn.getL(self.fluxes, self.dx)  # operator L as a function of the reconstruction values

        if timestep == "ssprk(3,3)":
            # Evolve system by SSP RK (3,3) method
            # Computation of 1st-term u_1 in Runge-Kutta (3,3)
            domain_u1 = self.domain + self.dt*Lq

            # Computation of 2nd-term u_2 in Runge-Kutta (3,3)
            hydroTube = solv.RiemannSolver(domain_u1, self.boundary, self.gamma)
            fluxes_u1 = hydroTube.calculateRiemannFlux(self.solver)
            Lq_u1 = fn.getL(fluxes_u1, self.dx)
            domain_u2 = .25 * (3*self.domain + domain_u1 + self.dt*Lq_u1)

            # Computation of the final update
            hydroTube = solv.RiemannSolver(domain_u2, self.boundary, self.gamma)
            fluxes_u2 = hydroTube.calculateRiemannFlux(self.solver)
            Lq_u2 = fn.getL(fluxes_u2, self.dx)
            return 1/3 * (self.domain + 2*domain_u2 + 2*self.dt*Lq_u2)

        
        elif timestep == "ssprk(5,4)":
            # Evolve system by SSP RK (5,4) method
            # Computation of 1st-term u_1 in Runge-Kutta (5,4)
            domain_u1 = self.domain + .39175222657189*self.dt*Lq

            # Computation of 2nd-term u_2 in Runge-Kutta (5,4)
            hydroTube = solv.RiemannSolver(domain_u1, self.boundary, self.gamma)
            fluxes_u1 = hydroTube.calculateRiemannFlux(self.solver)
            Lq_u1 = fn.getL(fluxes_u1, self.dx)
            domain_u2 = .444370493651235*self.domain + .555629506348765*domain_u1 + .368410593050371*self.dt*Lq_u1

            # Computation of 3rd-term u_3 in Runge-Kutta (5,4)
            hydroTube = solv.RiemannSolver(domain_u2, self.boundary, self.gamma)
            fluxes_u2 = hydroTube.calculateRiemannFlux(self.solver)
            Lq_u2 = fn.getL(fluxes_u2, self.dx)
            domain_u3 = .620101851488403*self.domain + .379898148511597*domain_u2 + .251891774271694*self.dt*Lq_u2

            # Computation of 4th-term u_4 in Runge-Kutta (5,4)
            hydroTube = solv.RiemannSolver(domain_u3, self.boundary, self.gamma)
            fluxes_u3 = hydroTube.calculateRiemannFlux(self.solver)
            Lq_u3 = fn.getL(fluxes_u3, self.dx)
            domain_u4 = .178079954393132*self.domain + .821920045606868*domain_u3 + .544974750228521*self.dt*Lq_u3

            # Computation of the final update
            hydroTube = solv.RiemannSolver(domain_u4, self.boundary, self.gamma)
            fluxes_u4 = hydroTube.calculateRiemannFlux(self.solver)
            Lq_u4 = fn.getL(fluxes_u4, self.dx)
            return .517231671970585*domain_u2 + .096059710526147*domain_u3 + .06369246866629*self.dt*Lq_u3 + .386708617503269*domain_u4 + .226007483236906*self.dt*Lq_u4


        elif timestep == "rk4":
            pass

        
        else:
            if timestep != "euler":
                print(f"Timestep procedure unknown; reverting to Forward Euler timestep\n")
            # Evolve system by a full timestep (1st-order)
            return (self.dt * Lq)