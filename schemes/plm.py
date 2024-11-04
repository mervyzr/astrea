from collections import defaultdict

import numpy as np

from functions import fv, constructors
from numerics import limiters

##############################################################################
# Piecewise linear reconstruction method (PLM) [van Leer, 1979]
##############################################################################

def run(grid, sim_variables):
    gamma, boundary, permutations = sim_variables.gamma, sim_variables.boundary, sim_variables.permutations
    nested_dict = lambda: defaultdict(nested_dict)
    data = nested_dict()

    # Rotate grid and apply algorithm for each axis
    for axis, axes in enumerate(permutations):
        _grid = grid.transpose(axes)

        # Convert to primitive variables; able to use pointwise conversion as it is still 2nd-order
        wS = fv.point_convert_conservative(_grid, sim_variables)

        # Pad array with boundary & apply (TVD) slope limiters
        w = fv.add_boundary(wS, boundary)
        limited_values = limiters.minmod_limiter(w)

        """Linear reconstruction [Derigs et al., 2017]
        Current convention: |                        w(i-1/2)                    w(i+1/2)                       |
                            |-->         i-1         <--|-->          i          <--|-->         i+1         <--|
                            |   w_L(i-1)     w_R(i-1)   |   w_L(i)         w_R(i)   |   w_L(i+1)     w_R(i+1)   |
                    OR      |   w+(i-3/2)   w-(i-1/2)   |   w+(i-1/2)   w-(i+1/2)   |  w+(i+1/2)    w-(i+3/2)   |
        """
        gradients = .5 * limited_values
        wL, wR = np.copy(wS-gradients), np.copy(wS+gradients)  # (eq. 4.13)

        # Pad the reconstructed interfaces
        wLs, wRs = fv.add_boundary(wL, boundary)[1:], fv.add_boundary(wR, boundary)[:-1]

        # Convert the primitive variables
        # The conversion can be pointwise conversion for face-average values as it is still 2nd-order
        qLs, qRs = fv.convert_primitive(wLs, sim_variables, "face"), fv.convert_primitive(wRs, sim_variables, "face")

        # Compute the fluxes and the Jacobian
        fLs, fRs = constructors.make_flux_term(wLs, gamma, axis), constructors.make_flux_term(wRs, gamma, axis)
        A = constructors.make_Jacobian(w, gamma, axis)

        # Update dict
        data[axes]['wS'] = wS
        data[axes]['wLs'] = wLs
        data[axes]['wRs'] = wRs
        data[axes]['qLs'] = qLs
        data[axes]['qRs'] = qRs
        data[axes]['fLs'] = fLs
        data[axes]['fRs'] = fRs
        data[axes]['jacobian'] = A

    return data