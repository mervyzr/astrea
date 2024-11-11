from collections import defaultdict

import numpy as np

from functions import constructor, fv
from num_methods import limiters

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
        |                        w(i-1/2)                    w(i+1/2)                       |
        |<--         i-1         -->|<--          i          -->|<--         i+1         -->|
        |   w_L(i-1)     w_R(i-1)   |   w_L(i)         w_R(i)   |   w_L(i+1)     w_R(i+1)   |
        |   w+(i-3/2)   w-(i-1/2)   |   w+(i-1/2)   w-(i+1/2)   |  w+(i+1/2)    w-(i+3/2)   |
        """
        gradients = .5 * limited_values
        wL, wR = np.copy(wS-gradients), np.copy(wS+gradients)  # (eq. 4.13)

        # Re-align the interfaces so that cell wall is in between interfaces
        w_plus, w_minus = fv.add_boundary(wL, boundary)[1:], fv.add_boundary(wR, boundary)[:-1]

        # Get the average solution between the interfaces at the boundaries
        intf_avg = (.5 * (w_plus + w_minus))[1:]
        _intf_avg = fv.add_boundary(intf_avg, boundary)

        # Convert the primitive variables
        # The conversion can be pointwise conversion for face-average values as it is still 2nd-order
        q_plus, q_minus = fv.convert_primitive(w_plus, sim_variables, "face"), fv.convert_primitive(w_minus, sim_variables, "face")

        # Compute the fluxes and the Jacobian
        flux_plus, flux_minus = constructor.make_flux(w_plus, gamma, axis), constructor.make_flux(w_minus, gamma, axis)
        A = constructor.make_Jacobian(_intf_avg, gamma, axis)

        # Update dict
        data[axes]['wS'] = wS
        data[axes]['wFs'] = w_plus, w_minus
        data[axes]['qFs'] = q_plus, q_minus
        data[axes]['fluxFs'] = flux_plus, flux_minus
        data[axes]['Jacobian'] = A

    return data