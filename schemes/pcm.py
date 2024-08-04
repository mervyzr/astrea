import numpy as np

from functions import fv
from numerics import solvers

##############################################################################

# Piecewise constant reconstruction method (PCM)
def run(tube, simVariables):
    gamma, boundary, permutations = simVariables.gamma, simVariables.boundary, simVariables.permutations

    # Rotate grid and apply algorithm for each axis
    for axes in permutations:

        # Convert to primitive variables
        wS = fv.pointConvertConservative(tube.transpose(axes), gamma)
        qS = fv.makeBoundary(tube.transpose(axes), boundary)

        # Compute the fluxes and the Jacobian
        w = fv.makeBoundary(wS, boundary)
        f = fv.makeFluxTerm(w, gamma)
        A = fv.makeJacobian(w, gamma)
        characteristics = np.linalg.eigvals(A)

    return solvers.calculateRiemannFlux(simVariables, f=f, wS=wS, w=w, qS=qS, characteristics=characteristics)