import numpy as np

import limiters
import functions as fn

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
        qLs, qRs = fn.makeBoundary(self.domain, self.config)

        # Select the reconstruction method
        # Piecewise parabolic method solver (3rd-order stable for uneven grid; 4th-order stable for even grid)
        if solver.lower() == "ppm" or solver.lower() == "parabolic":
            # Conversion of conservative variables to primitive variables
            wS = np.copy(fn.convertConservative(self.domain, self.gamma, self.config))

            # Reconstruction in primitive variables
            wLs, wRs = fn.makeBoundary(wS, self.config)
            if self.config == "sin":
                wL2s, wR2s = np.concatenate(([wLs[-2]],wLs))[:-1], np.concatenate((wRs,[wRs[1]]))[1:]  # Use periodic boundary for additional ghost box
            else:
                wL2s, wR2s = np.concatenate(([wLs[0]],wLs))[:-1], np.concatenate((wRs,[wRs[-1]]))[1:]  # Use outflow boundary for additional ghost box
            #wF = (7/12 * (wLs[:-1] + wS)) - (1/12 * (wRs[1:] + wL2s[:-1]))  # Compute face-averaged values (i-1/2)
            wF = (7/12 * (wS + wRs[1:])) - (1/12 * (wR2s[1:] + wLs[:-1]))  # Compute face-averaged values (i+1/2)
            
            # Apply limiters to avoid spurious oscillations at discontinuities
            """if (wF - wS)*(wRs[1:] - wF) < 0:
                wF_limit = limiters.faceValueLimit(wS, wF, wLs, wRs, wL2s, wR2s)
            else:
                wF_limit = 0

            wF_limit_L = makeBoundary(wF_limit, self.config)[:-1]
            if (wS - wF_limit_L)*(wF_limit - wS) < 0 or np.abs(wS - wF_limit_L) > 2*np.abs(wF_limit - wS) or np.abs(wF_limit - wS) > 2*np.abs(wS - wF_limit_L):
                pass
            else:
                pass
            """
            






            wLs, wRs = fn.makeBoundary(wF, self.config)
            fLs, fRs = fn.makeFlux(wLs, self.gamma), fn.makeFlux(wRs, self.gamma)

            AL, AR = np.nan_to_num(fn.makeJacobian(wLs, self.gamma), copy=False), np.nan_to_num(fn.makeJacobian(wRs, self.gamma), copy=False)
            eigvalL, eigvalR = np.linalg.eigvals(AL), np.linalg.eigvals(AR)
            eigval = max(np.max(abs(eigvalL)), np.max(abs(eigvalR)))

            if eigval > self.eigmax:
                self.eigmax = eigval  # Compute the maximum wave speed; the maximum wave speed is the max eigenvalue between all cells i and i-1

            return .5 * ((fLs+fRs) - (eigval*(qLs-qRs)))


            

        # Piecewise linear method with minmod limiter (2nd-order stable)
        elif solver.lower() == "plm" or solver.lower() == "linear":
            gradients = limiters.minmod(qLs, qRs)  # implement limiter here
            qLefts, qRights = np.copy(self.domain)-gradients, np.copy(self.domain)+gradients  # reconstruction step
            #avg_values = .5 * (qLefts+qRights)

            if self.config == "sin":
                # Use periodic boundary for ghost boxes
                leftInterfaces, rightInterfaces = np.concatenate((qLefts,[qLefts[0]])), np.concatenate(([qRights[-1]],qRights))
                #qLs, qRs = np.concatenate(([avg_values[-1]],avg_values)), np.concatenate((avg_values,[avg_values[0]]))
            else:
                # Use outflow boundary for ghost boxes
                leftInterfaces, rightInterfaces = np.concatenate((qLefts,[qRights[-1]])), np.concatenate(([qLefts[0]],qRights))
                #qLs, qRs = np.concatenate(([avg_values[0]],avg_values)), np.concatenate((avg_values,[avg_values[-1]]))

            wLs, wRs = fn.pointConvertConservative(leftInterfaces, self.gamma), fn.pointConvertConservative(rightInterfaces, self.gamma)
            #wLs, wRs = fn.pointConvertConservative(qLs, self.gamma), fn.pointConvertConservative(qRs, self.gamma)
            fLs, fRs = fn.makeFlux(wLs, self.gamma), fn.makeFlux(wRs, self.gamma)

            AL, AR = np.nan_to_num(fn.makeJacobian(wLs, self.gamma), copy=False), np.nan_to_num(fn.makeJacobian(wRs, self.gamma), copy=False)
            eigvalL, eigvalR = np.linalg.eigvals(AL), np.linalg.eigvals(AR)
            eigval = max(np.max(abs(eigvalL)), np.max(abs(eigvalR)))

            if eigval > self.eigmax:
                self.eigmax = eigval  # Compute the maximum wave speed; the maximum wave speed is the max eigenvalue between all cells i and i-1

            return .5 * ((fLs+fRs) - (eigval*(leftInterfaces-rightInterfaces)))

        # Piecewise constant method (1st-order stable)
        else:
            wLs, wRs = fn.pointConvertConservative(qLs, self.gamma), fn.pointConvertConservative(qRs, self.gamma)
            fLs, fRs = fn.makeFlux(wLs, self.gamma), fn.makeFlux(wRs, self.gamma)

            AL, AR = np.nan_to_num(fn.makeJacobian(wLs, self.gamma), copy=False), np.nan_to_num(fn.makeJacobian(wRs, self.gamma), copy=False)
            eigvalL, eigvalR = np.linalg.eigvals(AL), np.linalg.eigvals(AR)
            eigval = max(np.max(abs(eigvalL)), np.max(abs(eigvalR)))

            if eigval > self.eigmax:
                self.eigmax = eigval  # Compute the maximum wave speed; the maximum wave speed is the max eigenvalue between all cells i and i-1

            return .5 * ((fLs+fRs) - (eigval*(qRs-qLs)))