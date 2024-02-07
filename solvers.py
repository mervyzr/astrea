import sys

import numpy as np

import functions as fn

##############################################################################

class RiemannSolver:
    def __init__(self, domain, solver, boundary, g):
        self.domain = domain
        self.solver = solver
        self.boundary = boundary
        self.gamma = g
        self.eigmax = sys.float_info.epsilon
        
        # Error condition
        if solver not in ["ppm", "parabolic", "p", "plm", "linear", "l", "pcm", "constant", "c"]:
            print(f"Solver unknown; reverting to piecewise constant reconstruction method\n")


    # Reconstruct the cell values
    def reconstruct(self):
        # Piecewise parabolic method solver (3rd-order stable for uneven grid; 4th-order stable for even grid)
        if self.solver in ["ppm", "parabolic", "p"]:
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
            return [wS, wF, wLs, wRs, wL2s, wR2s]
        else:
            return fn.makeBoundary(self.domain, self.boundary)


    # Calculate Riemann flux (Lax-Friedrichs; similar to Roe)
    def calculateRiemannFlux(self, valueLefts, valueRights):
        # Impose boundary conditions for cell-interfaces
        if self.solver in ["ppm", "parabolic", "p", "plm", "linear", "l"]:
            if self.boundary == "periodic":
                leftValues, rightValues = np.concatenate((valueLefts,[valueLefts[0]])), np.concatenate(([valueRights[-1]],valueRights))  # Periodic boundary for ghost boxes
            else:
                leftValues, rightValues = np.concatenate((valueLefts,[valueRights[-1]])), np.concatenate(([valueLefts[0]],valueRights))  # Outflow boundary for ghost boxes
        else:
            leftValues, rightValues = valueLefts, valueRights

        # Solve the Riemann flux problem
        if self.solver in ["ppm", "parabolic", "p"]:
            # Ideally, the 4th-order interface-averaged fluxes should be computed for PPM
            # But because the simulation is only 1D, the "normal" Laplacian (Taylor expansion) of the face-averaged states and fluxes are zero
            # Thus, the conversion between face-averaged and face-centred states and fluxes can be pointwise conversion

            qLs, qRs = fn.pointConvertPrimitive(leftValues, self.gamma), fn.pointConvertPrimitive(rightValues, self.gamma)
            fLs, fRs = fn.makeFlux(leftValues, self.gamma), fn.makeFlux(rightValues, self.gamma)

            AL, AR = np.nan_to_num(fn.makeJacobian(leftValues, self.gamma), copy=False), np.nan_to_num(fn.makeJacobian(rightValues, self.gamma), copy=False)
            eigvalL, eigvalR = np.linalg.eigvals(AL), np.linalg.eigvals(AR)
            eigval = max(np.max(abs(eigvalL)), np.max(abs(eigvalR)))

            # In order to have a more stable simulation with the limited values, a constraint should be imposed CFL <= 1.3925 for this PPM reconstruction
            # But this constraint is for Runge-Kutta update methods; the LF Riemann solver might be stable enough for this
            if eigval > self.eigmax:
                self.eigmax = eigval  # Compute the maximum wave speed; the maximum wave speed is the max eigenvalue between all cells i and i-1
        else:
            qLs, qRs = leftValues, rightValues
            wLs, wRs = fn.pointConvertConservative(leftValues, self.gamma), fn.pointConvertConservative(rightValues, self.gamma)
            fLs, fRs = fn.makeFlux(wLs, self.gamma), fn.makeFlux(wRs, self.gamma)

            AL, AR = np.nan_to_num(fn.makeJacobian(wLs, self.gamma), copy=False), np.nan_to_num(fn.makeJacobian(wRs, self.gamma), copy=False)
            eigvalL, eigvalR = np.linalg.eigvals(AL), np.linalg.eigvals(AR)
            eigval = max(np.max(abs(eigvalL)), np.max(abs(eigvalR)))

            if eigval > self.eigmax:
                self.eigmax = eigval  # Compute the maximum wave speed; the maximum wave speed is the max eigenvalue between all cells i and i-1
        
        # Return the Riemann fluxes
        if self.solver in ["ppm", "parabolic", "p", "plm", "linear", "l"]:
            return .5 * ((fLs+fRs) - (eigval*(qLs-qRs)))
        else:
            return .5 * ((fLs+fRs) - (eigval*(qRs-qLs)))