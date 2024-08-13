from collections import defaultdict

import numpy as np

from functions import fv, constructors
from numerics import solvers

##############################################################################
# Piecewise constant reconstruction method (PCM) [Godunov, 1959]
##############################################################################

def run(tube, sim_variables):
    gamma, boundary, permutations = sim_variables.gamma, sim_variables.boundary, sim_variables.permutations
    nested_dict = lambda: defaultdict(nested_dict)
    data = nested_dict()

    # Rotate grid and apply algorithm for each axis
    for axis, axes in enumerate(permutations):

        # Convert to primitive variables
        wS = fv.point_convert_conservative(tube.transpose(axes), gamma)
        q = fv.add_boundary(tube.transpose(axes), boundary)

        # Compute the fluxes and the Jacobian
        w = fv.add_boundary(wS, boundary)
        f = constructors.make_flux_term(w, gamma, axis)
        A = constructors.make_Jacobian(w, gamma, axis)
        characteristics = np.linalg.eigvals(A)

        # Update dict
        data[axes]['cntr_primitive'] = wS
        data[axes]['face_primitive'] = w
        data[axes]['face_conserved'] = q
        data[axes]['fluxes'] = f
        data[axes]['jacobian'] = A

    return solvers.calculate_Riemann_flux(sim_variables, data, wS=wS, w=w, qS=q, f=f, characteristics=characteristics)