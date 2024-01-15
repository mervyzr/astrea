import numpy as np

import functions as fn
import flux_limiters as limiters

##############################################################################

class RiemannSolver:
    def __init__(self, domain, config, g):
        self.domain = domain
        self.config = config
        self.gamma = g
        self.eigmax = 0

    # Calculate Riemann flux (Lax-Friedrichs; similar to Roe)
    def calculateRiemannFlux(self, solver):
        # Impose boundary conditions
        if self.config == "sin":
            qLs, qRs = np.concatenate(([self.domain[-1]],self.domain)), np.concatenate((self.domain,[self.domain[0]]))  # Use periodic boundary for edge cells
        else:
            qLs, qRs = np.concatenate(([self.domain[0]],self.domain)), np.concatenate((self.domain,[self.domain[-1]]))  # Use outflow boundary for edge cells        

        # Select the reconstruction method
        if solver.lower() == "ppm" or solver.lower() == "parabolic":
            # Piecewise parabolic method solver (3rd-order stable for uneven grid; 4th-order stable for even grid)
            pass

        elif solver.lower() == "plm" or solver.lower() == "linear":
            # Piecewise linear method with minmod limiter (2nd-order stable)
            gradients = limiters.minmod(qLs, qRs)  # implement limiter here
            qLefts, qRights = np.copy(self.domain)-gradients, np.copy(self.domain)+gradients  # reconstruction step
            #avg_values = .5 * (qLefts+qRights)

            if self.config == "sin":
                # Use periodic boundary for edge cells
                leftInterfaces, rightInterfaces = np.concatenate((qLefts,[qLefts[0]])), np.concatenate(([qRights[-1]],qRights))
                #qLs, qRs = np.concatenate(([avg_values[-1]],avg_values)), np.concatenate((avg_values,[avg_values[0]]))
            else:
                # Use outflow boundary for edge cells
                leftInterfaces, rightInterfaces = np.concatenate((qLefts,[qRights[-1]])), np.concatenate(([qLefts[0]],qRights))
                #qLs, qRs = np.concatenate(([avg_values[0]],avg_values)), np.concatenate((avg_values,[avg_values[-1]]))

            wLs, wRs = fn.convertConservative(leftInterfaces, self.gamma), fn.convertConservative(rightInterfaces, self.gamma)
            #wLs, wRs = fn.convertConservative(qLs, self.gamma), fn.convertConservative(qRs, self.gamma)
            fLs, fRs = fn.makeFlux(wLs, self.gamma), fn.makeFlux(wRs, self.gamma)

            AL, AR = np.nan_to_num(fn.makeJacobian(wLs, self.gamma), copy=False), np.nan_to_num(fn.makeJacobian(wRs, self.gamma), copy=False)
            eigvalL, eigvalR = np.linalg.eigvals(AL), np.linalg.eigvals(AR)
            eigval = max(np.max(abs(eigvalL)), np.max(abs(eigvalR)))

            if eigval > self.eigmax:
                self.eigmax = eigval  # Compute the maximum wave speed; the maximum wave speed is the max eigenvalue between all cells i and i-1

            return .5 * ((fLs+fRs) - (eigval*(leftInterfaces-rightInterfaces)))

        else:
            # Piecewise constant method (1st-order stable)
            wLs, wRs = fn.convertConservative(qLs, self.gamma), fn.convertConservative(qRs, self.gamma)
            fLs, fRs = fn.makeFlux(wLs, self.gamma), fn.makeFlux(wRs, self.gamma)

            AL, AR = np.nan_to_num(fn.makeJacobian(wLs, self.gamma), copy=False), np.nan_to_num(fn.makeJacobian(wRs, self.gamma), copy=False)
            eigvalL, eigvalR = np.linalg.eigvals(AL), np.linalg.eigvals(AR)
            eigval = max(np.max(abs(eigvalL)), np.max(abs(eigvalR)))

            if eigval > self.eigmax:
                self.eigmax = eigval  # Compute the maximum wave speed; the maximum wave speed is the max eigenvalue between all cells i and i-1

            return .5 * ((fLs+fRs) - (eigval*(qRs-qLs)))