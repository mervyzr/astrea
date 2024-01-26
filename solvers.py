import numpy as np

import limiters
import functions as fn

##############################################################################

class RiemannSolver:
    def __init__(self, domain, boundary, g):
        self.domain = domain
        self.boundary = boundary
        self.gamma = g
        self.eigmax = 0

    # Calculate Riemann flux (Lax-Friedrichs; similar to Roe)
    def calculateRiemannFlux(self, solver):
        # Impose boundary conditions
        qLs, qRs = fn.makeBoundary(self.domain, self.boundary)

        # Piecewise parabolic method solver (3rd-order stable for uneven grid; 4th-order stable for even grid)
        if solver.lower() == "ppm" or solver.lower() == "parabolic":
            # Conversion of conservative variables to primitive variables
            wS = np.copy(fn.convertConservative(self.domain, self.gamma, self.boundary))

            # Reconstruction in primitive variables to 4th-order
            wLs, wRs = fn.makeBoundary(wS, self.boundary)
            if self.boundary == "periodic":
                wL2s, wR2s = np.concatenate(([wLs[-2]],wLs))[:-1], np.concatenate((wRs,[wRs[1]]))[1:]  # Periodic boundary for additional ghost box
            else:
                wL2s, wR2s = np.concatenate(([wLs[0]],wLs))[:-1], np.concatenate((wRs,[wRs[-1]]))[1:]  # Outflow boundary for additional ghost box
            #wF = (7/12 * (wLs[:-1] + wS)) - (1/12 * (wRs[1:] + wL2s[:-1]))  # Compute face-averaged values (i-1/2)
            wF = (7/12 * (wS + wRs[1:])) - (1/12 * (wR2s[1:] + wLs[:-1]))  # Compute face-averaged values (i+1/2)
            
            # Determine the limited face values
            wF_limit = limiters.limitFaceValues(wS, wF, wLs, wRs, wL2s, wR2s)

            # Determine the limited parabolic interpolants and complete the reconstruction
            wF_limit_L, wF_limit_R = fn.makeBoundary(wF_limit, self.boundary)
            if self.boundary == "periodic":
                wF_limit_L2, wF_limit_R2 = np.concatenate(([wF_limit_L[-2]],wF_limit_L))[:-1], np.concatenate((wF_limit_R,[wF_limit_R[1]]))[1:]
            else:
                wF_limit_L2, wF_limit_R2 = np.concatenate(([wF_limit_L[0]],wF_limit_L))[:-1], np.concatenate((wF_limit_R,[wF_limit_R[-1]]))[1:]
            wLefts, wRights = limiters.limitParabolicInterpolants(wS, wF, wLs, wRs, wL2s, wR2s, wF_limit, wF_limit_L, wF_limit_R, wF_limit_L2, wF_limit_R2)

            # Compute the 4th-order interface-averaged fluxes
            # Because the simulation is only 1D, the "normal" Laplacian (Taylor expansion) of the face-averaged states and fluxes are zero
            # Thus, the conversion between face-averaged and face-centred states and fluxes can be pointwise conversion

            # Solve the Riemann problem
            if self.boundary == "periodic":
                leftInterfaces, rightInterfaces = np.concatenate((wLefts,[wLefts[0]])), np.concatenate(([wRights[-1]],wRights))  # Periodic boundary for ghost boxes
            else:
                leftInterfaces, rightInterfaces = np.concatenate((wLefts,[wRights[-1]])), np.concatenate(([wLefts[0]],wRights))  # Outflow boundary for ghost boxes

            qLs, qRs = fn.pointConvertPrimitive(leftInterfaces, self.gamma), fn.pointConvertPrimitive(rightInterfaces, self.gamma)
            fLs, fRs = fn.makeFlux(leftInterfaces, self.gamma), fn.makeFlux(rightInterfaces, self.gamma)

            AL, AR = np.nan_to_num(fn.makeJacobian(leftInterfaces, self.gamma), copy=False), np.nan_to_num(fn.makeJacobian(rightInterfaces, self.gamma), copy=False)
            eigvalL, eigvalR = np.linalg.eigvals(AL), np.linalg.eigvals(AR)
            eigval = max(np.max(abs(eigvalL)), np.max(abs(eigvalR)))

            # In order to have a more stable simulation with the limited values, a constraint should be imposed: CFL <= 1.3925
            # But this constraint is for Runge-Kutta update methods; the LF Riemann solver might be stable enough for this
            if eigval > self.eigmax:
                self.eigmax = eigval  # Compute the maximum wave speed; the maximum wave speed is the max eigenvalue between all cells i and i-1

            return .5 * ((fLs+fRs) - (eigval*(qLs-qRs)))

        # Piecewise linear method with minmod limiter (2nd-order stable)
        elif solver.lower() == "plm" or solver.lower() == "linear":
            gradients = limiters.minmod(qLs, qRs)  # implement limiter here
            qLefts, qRights = np.copy(self.domain)-gradients, np.copy(self.domain)+gradients  # reconstruction step

            if self.boundary == "periodic":
                # Use periodic boundary for ghost boxes
                leftInterfaces, rightInterfaces = np.concatenate((qLefts,[qLefts[0]])), np.concatenate(([qRights[-1]],qRights))
            else:
                # Use outflow boundary for ghost boxes
                leftInterfaces, rightInterfaces = np.concatenate((qLefts,[qRights[-1]])), np.concatenate(([qLefts[0]],qRights))

            wLs, wRs = fn.pointConvertConservative(leftInterfaces, self.gamma), fn.pointConvertConservative(rightInterfaces, self.gamma)
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