import sys

import numpy as np

from functions import fv

##############################################################################

# Extrapolate the cell-centre values to the face
def extrapolate(tube, gamma, solver, boundary):
    # Piecewise parabolic method solver (3rd-order stable for uneven grid; 4th-order stable for even grid)
    if solver in ["ppm", "parabolic", "p"]:
        # Conversion of conservative variables to primitive variables
        wS = fv.convertConservative(tube, gamma, boundary)

        w = fv.makeBoundary(wS, boundary)
        w2 = fv.makeBoundary(wS, boundary, 2)

        # Extrapolate in primitive variables to 4th-order face values
        wFL = 7/12 * (wS+w[:-2]) - 1/12 * (w2[:-4]+w[2:])  # face i-1/2
        wFR = 7/12 * (wS+w[2:]) - 1/12 * (w[:-2]+w2[4:])  # face i+1/2
        return [wS, [wFL, wFR], w, w2]
    else:
        return fv.makeBoundary(tube, boundary)


# Reconstruction of the monotonised functions
def reconstruct(tube):
    pass


# Calculate Riemann flux (Lax-Friedrichs; similar to Roe)
def calculateRiemannFlux(solution, gamma, solver, boundary):
    # Impose boundary conditions for cell-interfaces
    if solver in ["ppm", "parabolic", "p", "plm", "linear", "l"]:
        leftSolution, rightSolution = solution
        leftInterface, rightInterface = fv.makeBoundary(leftSolution, boundary)[1:], fv.makeBoundary(rightSolution, boundary)[:-1]
    else:
        leftInterface, rightInterface = solution[:-1], solution[1:]
    
    # Solve the Riemann flux problem
    if solver in ["ppm", "parabolic", "p"]:
        # Ideally, the 4th-order interface-averaged fluxes should be computed for PPM
        # But because the simulation is only 1D, the "normal" Laplacian (Taylor expansion) of the face-averaged states and fluxes are zero
        # Thus, the conversion between face-averaged and face-centred states and fluxes can be a point-wise conversion

        qLs, qRs = fv.pointConvertPrimitive(leftInterface, gamma), fv.pointConvertPrimitive(rightInterface, gamma)
        
        fLs, fRs = fv.makeFlux(leftInterface, gamma), fv.makeFlux(rightInterface, gamma)
        AL, AR = fv.makeJacobian(leftInterface, gamma), fv.makeJacobian(rightInterface, gamma)
    else:
        qLs, qRs = leftInterface, rightInterface
        wLs, wRs = fv.pointConvertConservative(leftInterface, gamma), fv.pointConvertConservative(rightInterface, gamma)

        fLs, fRs = fv.makeFlux(wLs, gamma), fv.makeFlux(wRs, gamma)
        AL, AR = fv.makeJacobian(wLs, gamma), fv.makeJacobian(wRs, gamma)

    eigvalL, eigvalR = np.linalg.eigvals(AL), np.linalg.eigvals(AR)
    eigmax = np.max([np.max(abs(eigvalL)), np.max(abs(eigvalR)), sys.float_info.epsilon])  # Compute the maximum wave speed (max eigenvalue)
    # In order to have a more stable simulation with the limited values, a constraint should be imposed CFL <= 1.3925 for the PPM reconstruction

    # Return the Riemann fluxes
    if solver in ["ppm", "parabolic", "p", "plm", "linear", "l"]:
        return .5 * ((fLs+fRs) - (eigmax*(qLs-qRs))), eigmax
    else:
        return .5 * ((fLs+fRs) - (eigmax*(qRs-qLs))), eigmax