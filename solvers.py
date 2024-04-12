import sys

import numpy as np

from functions import fv

##############################################################################

# Solve the Riemann (flux) problem (Local Lax-Friedrichs; approximate Roe solver)
def calculateRiemannFlux(solutions, gamma, solver, boundary):
    if solver in ["ppm", "parabolic", "p", "plm", "linear", "l"]:
        leftSolution, rightSolution = solutions
        leftInterface, rightInterface = fv.makeBoundary(leftSolution, boundary)[1:], fv.makeBoundary(rightSolution, boundary)[:-1]

        # Get the average of the solutions by integrating the interpolated values
        if solver in ["ppm", "parabolic", "p"]:
            # [Colella & Woodward, 1984, eq. 1.4]
            avg_wS = (leftSolution + rightSolution)/2
        else:
            avg_wS = (leftSolution + rightSolution)/2
    else:
        avg_wS = solutions

    # Ideally, the 4th-order averaged fluxes should be computed from the face-averaged variables
    # But because the simulation is only 1D, the "normal"-Laplacian (Taylor expansion) of the face-averaged states and fluxes are zero
    # Thus, the face-averaged and face-centred values are the same (<w>_i+1/2 = w_i+1/2)
    # Same for the averaged and centred fluxes (<F>_i+1/2 = F_i+1/2)
    wS = fv.makeBoundary(avg_wS, boundary)
    fS = fv.makeFlux(wS, gamma)
    A = fv.makeJacobian(wS, gamma)

    if solver in ["ppm", "parabolic", "p", "plm", "linear", "l"]:
        # The conversion can be pointwise conversion for the face-averaged values
        qLs, qRs = fv.pointConvertPrimitive(leftInterface, gamma), fv.pointConvertPrimitive(rightInterface, gamma)
        qDiff = (qLs - qRs).T
    else:
        qLs, qRs = fv.pointConvertPrimitive(wS[:-1], gamma), fv.pointConvertPrimitive(wS[1:], gamma)
        qDiff = (qRs - qLs).T

    # Determine the eigenvalues for the computation of the flux and time stepping
    localEigvals = np.max(np.abs(np.linalg.eigvals(A)), axis=1)  # Local max eigenvalue for each cell
    eigvals = np.max([localEigvals[:-1], localEigvals[1:]], axis=0)  # Local max eigenvalue between consecutive pairs of cell
    eigmax = np.max([np.max(eigvals), sys.float_info.epsilon])  # Maximum wave speed (max eigenvalue) for system

    # Return the Riemann fluxes
    return .5 * ((fS[:-1]+fS[1:]) - ((eigvals * qDiff).T)), eigmax