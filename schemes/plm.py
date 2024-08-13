from collections import defaultdict

import numpy as np

from functions import fv, constructors
from numerics import limiters, solvers

##############################################################################
# Piecewise linear reconstruction method (PLM) [van Leer, 1979]
##############################################################################

# Current convention: |               w(i-1/2)                    w(i+1/2)              |
#                     | i-1          <-- | -->         i         <-- | -->          i+1 |
#                     |        w_R(i-1)  |   w_L(i)          w_R(i)  |  w_L(i+1)        |
def run(tube, sim_variables):
    gamma, boundary, permutations = sim_variables.gamma, sim_variables.boundary, sim_variables.permutations
    nested_dict = lambda: defaultdict(nested_dict)
    data = nested_dict()

    # Rotate grid and apply algorithm for each axis
    for axis, axes in enumerate(permutations):

        # Convert to primitive variables; able to use pointwise conversion as it is still 2nd-order
        wS = fv.point_convert_conservative(tube.transpose(axes), gamma)

        # Pad array with boundary & apply (TVD) slope limiters
        w = fv.add_boundary(wS, boundary)
        limited_values = limiters.minmod_limiter(w)

        # Linear reconstruction [Derigs et al., 2017]
        gradients = .5 * limited_values
        wL, wR = np.copy(wS-gradients), np.copy(wS+gradients)  # (eq. 4.13)

        # Pad the reconstructed interfaces
        wLs, wRs = fv.add_boundary(wL, boundary)[1:], fv.add_boundary(wR, boundary)[:-1]

        # Convert the primitive variables
        # The conversion can be pointwise conversion for face-average values as it is still 2nd-order
        qLs, qRs = fv.point_convert_primitive(wLs, gamma), fv.point_convert_primitive(wRs, gamma)

        # Compute the fluxes and the Jacobian
        fLs, fRs = constructors.make_flux_term(wLs, gamma, axis), constructors.make_flux_term(wRs, gamma, axis)
        A = constructors.make_jacobian(w, gamma, axis)

        # Update dict
        data[axes]['cntr_primitive'] = wS
        data[axes]['face_primitive'] = [wLs,wRs]
        data[axes]['face_conserved'] = [qLs,qRs]
        data[axes]['fluxes'] = [fLs,fRs]
        data[axes]['jacobian'] = A
        characteristics = np.linalg.eigvals(A)

    return solvers.calculate_Riemann_flux(sim_variables, data, fLs=fLs, fRs=fRs, wLs=wLs, wRs=wRs, qLs=qLs, qRs=qRs, characteristics=characteristics)