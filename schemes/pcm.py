import numpy as np

from functions import fv
from numerics import solvers

##############################################################################

# Piecewise constant reconstruction method (PCM)
def run(tube, simVariables):
    gamma, precision, scheme, boundary = simVariables.gamma, simVariables.precision, simVariables.scheme, simVariables.boundary

    # Convert to primitive variables
    wS = fv.pointConvertConservative(tube, gamma)
    qS = fv.makeBoundary(tube, boundary)

    # Compute the fluxes and the Jacobian
    w = fv.makeBoundary(wS, boundary)
    f = fv.makeFlux(w, gamma)
    A = fv.makeJacobian(w, gamma)
    characteristics = np.linalg.eigvals(A)

    """# Entropy-stable flux component
    wS = fv.makeBoundary(avg_wS, boundary)
    wLs, wRs = fv.makeBoundary(leftSolution, boundary), fv.makeBoundary(rightSolution, boundary)
    fS = fv.makeFlux([wLs, wRs], gamma)

    A = fv.makeJacobian(wS, gamma)
    characteristics = np.linalg.eigvals(A)
    D = np.zeros((characteristics.shape[0], characteristics.shape[1], characteristics.shape[1]))
    _diag = np.arange(characteristics.shape[1])
    D[:, _diag, _diag] = characteristics

    sL, sR = fv.getEntropyVector(wLs, gamma), fv.getEntropyVector(wRs, gamma)
    dfS = .5 * np.einsum('ijk,ij->ik', A*(D*A.transpose([0,2,1])), sR-sL)
    fS -= dfS"""

    return solvers.calculateRiemannFlux(simVariables, f=f, wS=wS, w=w, qS=qS, characteristics=characteristics)