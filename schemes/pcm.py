from collections import defaultdict

import numpy as np

from functions import fv, constructors
from numerics import solvers

##############################################################################
# Piecewise constant reconstruction method (PCM) [Godunov, 1959]
##############################################################################

def run(tube, simVariables):
    gamma, boundary, permutations = simVariables.gamma, simVariables.boundary, simVariables.permutations
    nested_dict = lambda: defaultdict(nested_dict)
    data = nested_dict()

    # Rotate grid and apply algorithm for each axis
    for axis, axes in enumerate(permutations):

        # Convert to primitive variables
        wS = fv.pointConvertConservative(tube.transpose(axes), gamma)
        q = fv.addBoundary(tube.transpose(axes), boundary)

        # Compute the fluxes and the Jacobian
        w = fv.addBoundary(wS, boundary)
        f = constructors.makeFluxTerm(w, gamma, axis)
        A = constructors.makeJacobian(w, gamma, axis)
        characteristics = np.linalg.eigvals(A)

        # Update dict
        data[axes]['wS'] = wS
        data[axes]['w'] = [w,]
        data[axes]['q'] = [q,]
        data[axes]['f'] = [f,]
        data[axes]['jacobian'] = A
        data[axes]['eigvals'] = characteristics

    return solvers.calculateRiemannFlux(simVariables, data, f=f, wS=wS, w=w, q=q, characteristics=characteristics)