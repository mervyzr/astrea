import numpy as np

from functions import fv
from numerics import solvers

##############################################################################

# Piecewise constant reconstruction method (PCM)
def run(tube, simVariables):
    gamma, boundary = simVariables.gamma, simVariables.boundary

    # Convert to primitive variables
    wS = fv.pointConvertConservative(tube, gamma)
    qS = fv.makeBoundary(tube, boundary)

    # Compute the fluxes and the Jacobian
    w = fv.makeBoundary(wS, boundary)
    f = fv.makeFluxTerm(w, gamma)
    A = fv.makeJacobian(w, gamma)
    characteristics = np.linalg.eigvals(A)

    return solvers.calculateRiemannFlux(simVariables, f=f, wS=wS, w=w, qS=qS, characteristics=characteristics)