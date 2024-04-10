import sys

import numpy as np

from functions import fv

##############################################################################

# Solve the Riemann (flux) problem (Local Lax-Friedrichs; approximate Roe solver)
def calculateRiemannFlux(solution, gamma, solver, boundary):
    # Create boundary for cell-interfaces
    if solver in ["ppm", "parabolic", "p", "plm", "linear", "l"]:
        leftSolution, rightSolution = solution
        leftInterface, rightInterface = fv.makeBoundary(leftSolution, boundary)[1:], fv.makeBoundary(rightSolution, boundary)[:-1]

    if solver in ["ppm", "parabolic", "p"]:
        # Ideally, the 4th-order averaged fluxes should be computed from the face-averaged variables
        # But because the simulation is only 1D, the "normal"-Laplacian (Taylor expansion) of the face-averaged states and fluxes are zero
        # Thus, the face-averaged and face-centred values are the same (<w>_i+1/2 = w_i+1/2)
        # Same for the averaged and centred fluxes (<F>_i+1/2 = F_i+1/2)

        qLs, qRs = fv.convertPrimitive(leftInterface, gamma, solver, boundary), fv.convertPrimitive(rightInterface, gamma, solver, boundary)

        fLs, fRs = fv.makeFlux(leftInterface, gamma), fv.makeFlux(rightInterface, gamma)
        AL, AR = fv.makeJacobian(leftInterface, gamma), fv.makeJacobian(rightInterface, gamma)
    elif solver in ["plm", "linear", "l"]:
        leftSolution, rightSolution = solution

        # Get the average of the solutions from the integral of the interpolated values



    else:
        wLs, wRs = solution[:-1], solution[1:]
        qLs, qRs = fv.convertPrimitive(wLs, gamma, solver, boundary), fv.convertPrimitive(wRs, gamma, solver, boundary)

        fLs, fRs = fv.makeFlux(wLs, gamma), fv.makeFlux(wRs, gamma)
        AL, AR = fv.makeJacobian(wLs, gamma), fv.makeJacobian(wRs, gamma)

    eigvalL, eigvalR = np.linalg.eigvals(AL), np.linalg.eigvals(AR)
    eigmax = np.max([np.max(abs(eigvalL)), np.max(abs(eigvalR)), sys.float_info.epsilon])  # Compute the maximum wave speed (max eigenvalue)

    # Return the Riemann fluxes
    if solver in ["ppm", "parabolic", "p", "plm", "linear", "l"]:
        return .5 * ((fLs+fRs) - (eigmax*(qLs-qRs))), eigmax
    else:
        return .5 * ((fLs+fRs) - (eigmax*(qRs-qLs))), eigmax