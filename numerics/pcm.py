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
    q = fv.makeBoundary(tube, boundary)
    qDiff = (q[1:]-q[:-1]).T

    # Compute the fluxes and the Jacobian
    w = fv.makeBoundary(wS, boundary)
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