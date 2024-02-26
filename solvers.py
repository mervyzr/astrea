import sys

import numpy as np

from functions import fv

##############################################################################

class RiemannSolver:
    def __init__(self, domain, solver, gamma, dx, boundary, limiter):
        self.domain = domain
        self.solver = solver
        self.gamma = gamma
        self.dx = dx
        self.boundary = boundary
        self.limiter = limiter
        self.eigmax = sys.float_info.epsilon


    # Interpolate the cell values
    def interpolate(self, tube):
        # Piecewise parabolic method solver (3rd-order stable for uneven grid; 4th-order stable for even grid)
        if self.solver in ["ppm", "parabolic", "p"]:
            # Conversion of conservative variables to primitive variables
            wS = np.copy(fv.convertConservative(tube, self.gamma, self.boundary))

            # Reconstruction in primitive variables to 4th-order
            wLs, wRs = fv.makeBoundary(wS, self.boundary)
            if self.boundary == "periodic":
                wL2s, wR2s = np.concatenate(([wLs[-2]],wLs))[:-1], np.concatenate((wRs,[wRs[1]]))[1:]  # Periodic boundary for additional ghost box
            else:
                wL2s, wR2s = np.concatenate(([wLs[0]],wLs))[:-1], np.concatenate((wRs,[wRs[-1]]))[1:]  # Outflow boundary for additional ghost box
            
            wFL = 7/12 * (wS+wLs[:-1]) - 1/12 * (wL2s[:-1]+wRs[1:])  # Compute face-averaged values (i-1/2)
            wFR = 7/12 * (wS+wRs[1:]) - 1/12 * (wLs[:-1]+wR2s[1:])  # Compute face-averaged values (i+1/2)
            return [wS, [wFL, wFR], [wLs, wRs], [wL2s, wR2s]]
        else:
            return fv.makeBoundary(tube, self.boundary)


    # Calculate Riemann flux (Lax-Friedrichs; similar to Roe)
    def calculateRiemannFlux(self, valueLefts, valueRights):
        # Impose boundary conditions for cell-interfaces
        if self.solver in ["ppm", "parabolic", "p", "plm", "linear", "l"]:
            if self.boundary == "periodic":
                leftValues, rightValues = np.concatenate((valueLefts,[valueLefts[0]])), np.concatenate(([valueRights[-1]],valueRights))  # Periodic boundary for ghost boxes
            else:
                leftValues, rightValues = np.concatenate((valueLefts,[valueLefts[-1]])), np.concatenate(([valueRights[0]],valueRights))  # Outflow boundary for ghost boxes
        else:
            leftValues, rightValues = valueLefts, valueRights

        # Solve the Riemann flux problem
        if self.solver in ["ppm", "parabolic", "p"]:
            # Ideally, the 4th-order interface-averaged fluxes should be computed for PPM
            # But because the simulation is only 1D, the "normal" Laplacian (Taylor expansion) of the face-averaged states and fluxes are zero
            # Thus, the conversion between face-averaged and face-centred states and fluxes can be pointwise conversion

            qLs, qRs = fv.pointConvertPrimitive(leftValues, self.gamma), fv.pointConvertPrimitive(rightValues, self.gamma)
            fLs, fRs = fv.makeFlux(leftValues, self.gamma), fv.makeFlux(rightValues, self.gamma)

            AL, AR = np.nan_to_num(fv.makeJacobian(leftValues, self.gamma), copy=False), np.nan_to_num(fv.makeJacobian(rightValues, self.gamma), copy=False)
        else:
            qLs, qRs = leftValues, rightValues
            wLs, wRs = fv.pointConvertConservative(leftValues, self.gamma), fv.pointConvertConservative(rightValues, self.gamma)
            fLs, fRs = fv.makeFlux(wLs, self.gamma), fv.makeFlux(wRs, self.gamma)

            AL, AR = np.nan_to_num(fv.makeJacobian(wLs, self.gamma), copy=False), np.nan_to_num(fv.makeJacobian(wRs, self.gamma), copy=False)

        eigvalL, eigvalR = np.linalg.eigvals(AL), np.linalg.eigvals(AR)
        eigval = max(np.max(abs(eigvalL)), np.max(abs(eigvalR)))
        if eigval > self.eigmax:
            self.eigmax = eigval  # Compute the maximum wave speed (max eigenvalue)
        # In order to have a more stable simulation with the limited values, a constraint should be imposed CFL <= 1.3925 for the PPM reconstruction
        # But this constraint is for Runge-Kutta update methods; the LF Riemann solver might be stable enough for this

        # Return the Riemann fluxes
        if self.solver in ["ppm", "parabolic", "p", "plm", "linear", "l"]:
            return .5 * ((fLs+fRs) - (eigval*(qLs-qRs)))
        else:
            return .5 * ((fLs+fRs) - (eigval*(qRs-qLs)))