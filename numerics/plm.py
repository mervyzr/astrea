from collections import namedtuple

import numpy as np

from functions import fv
from numerics import limiters, solvers

##############################################################################

# Piecewise linear reconstruction method (PLM)
# Current convention: |               w(i-1/2)                    w(i+1/2)              |
#                     | i-1          <-- | -->         i         <-- | -->          i+1 |
#                     |        w_R(i-1)  |   w_L(i)          w_R(i)  |  w_L(i+1)        |
def run(tube, simVariables):
    gamma, precision, scheme, boundary = simVariables.gamma, simVariables.precision, simVariables.scheme, simVariables.boundary
    Data = namedtuple('Data', ['flux', 'eigmax'])

    # Convert to primitive variables; able to use pointwise conversion as it is still 2nd-order
    wS = fv.pointConvertConservative(tube, gamma)

    # Pad array with boundary & apply (TVD) slope limiters
    w = fv.makeBoundary(wS, boundary)
    limitedValues = limiters.minmodLimiter(w)

    # Linear reconstruction [Derigs et al., 2017]
    gradients = .5 * limitedValues
    wL, wR = np.copy(wS-gradients), np.copy(wS+gradients)  # (eq. 4.13)

    # Pad the reconstructed interfaces
    wLs, wRs = fv.makeBoundary(wL, boundary)[1:], fv.makeBoundary(wR, boundary)[:-1]

    # Convert the primitive variables, and compute the state differences
    # The conversion can be pointwise conversion for face-average values as it is still 2nd-order
    qLs, qRs = fv.pointConvertPrimitive(wLs, gamma), fv.pointConvertPrimitive(wRs, gamma)
    qDiff = (qLs-qRs).T

    # Compute the fluxes and the Jacobian
    f = fv.makeFlux(w, gamma)
    A = fv.makeJacobian(w, gamma)
    characteristics = np.linalg.eigvals(A)

    # Determine the eigenvalues for the computation of time stepping
    eigvals = np.max(np.abs(characteristics), axis=1)  # Local max eigenvalue for each cell (1- or 3-Riemann invariant; shock wave or rarefaction wave)
    maxEigvals = np.max([eigvals[:-1], eigvals[1:]], axis=0)  # Local max eigenvalue between consecutive pairs of cell

    eigmax = np.max([np.max(maxEigvals), np.finfo(precision).eps])  # Maximum wave speed (max eigenvalue) for time evolution

    if scheme in ["lw", "lax-wendroff", "wendroff"]:
        data = Data(solvers.calculateLaxWendroffFlux(f, qDiff, eigvals, characteristics), eigmax)
    else:
        data = Data(solvers.calculateLaxFriedrichFlux(f, qDiff, maxEigvals), eigmax)
    return data