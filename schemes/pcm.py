import numpy as np

from functions import fv, constructors
from numerics import solvers

##############################################################################
# Piecewise constant reconstruction method (PCM) [Godunov, 1959]
##############################################################################

def run(tube, simVariables):
    gamma, boundary, permutations = simVariables.gamma, simVariables.boundary, simVariables.permutations

    # Rotate grid and apply algorithm for each axis
    for axis, axes in enumerate(permutations):

        # Convert to primitive variables
        wS = fv.pointConvertConservative(tube.transpose(axes), gamma)
        qS = fv.addBoundary(tube.transpose(axes), boundary)

        # Compute the fluxes and the Jacobian
        w = fv.addBoundary(wS, boundary)
        f = constructors.makeFluxTerm(w, gamma)
        A = constructors.makeJacobian(w, gamma)
        characteristics = np.linalg.eigvals(A)

    return solvers.calculateRiemannFlux(simVariables, f=f, wS=wS, w=w, qS=qS, characteristics=characteristics)