from collections import namedtuple

import numpy as np

from functions import fv
from numerics import solvers

##############################################################################

# Piecewise constant reconstruction method (PCM)
def run(tube, simVariables):
    gamma, precision, scheme, boundary = simVariables.gamma, simVariables.precision, simVariables.scheme, simVariables.boundary
    Data = namedtuple('Data', ['flux', 'eigmax'])

    # Convert to primitive variables
    wS = fv.pointConvertConservative(tube, gamma)

    # Compute state differences
    qS = fv.makeBoundary(tube, boundary)
    qDiff = (qS[1:]-qS[:-1]).T

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

    # Determine the eigenvalues for the computation of time stepping
    eigvals = np.max(np.abs(characteristics), axis=1)  # Local max eigenvalue for each cell (1- or 3-Riemann invariant; shock wave or rarefaction wave)
    maxEigvals = np.max([eigvals[:-1], eigvals[1:]], axis=0)  # Local max eigenvalue between consecutive pairs of cell

    eigmax = np.max([np.max(maxEigvals), np.finfo(precision).eps])  # Maximum wave speed (max eigenvalue) for time evolution

    if scheme in ["hllc", "c"]:
        data = Data(solvers.calculateHLLCFlux(w[:-1], w[1:], gamma, boundary), eigmax)
    elif scheme in ["os", "osher-solomon", "osher", "solomon"]:
        data = Data(solvers.calculateOSFlux([wS, wS], [qS, qS], gamma, boundary, simVariables.roots, simVariables.weights), eigmax)
    elif scheme in ["lw", "lax-wendroff", "wendroff"]:
        data = Data(solvers.calculateLaxWendroffFlux(f, qDiff, eigvals, characteristics), eigmax)
    else:
        data = Data(solvers.calculateLaxFriedrichFlux(f, qDiff, maxEigvals), eigmax)
    return data