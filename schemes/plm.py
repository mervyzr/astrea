from collections import defaultdict

import numpy as np

from functions import constructor, fv
from num_methods import limiters

##############################################################################
# Piecewise linear reconstruction method (PLM) [van Leer, 1979]
##############################################################################

def run(grids, sim_variables):
    gamma, boundary, axes = sim_variables.gamma, sim_variables.boundary, sim_variables.axes
    nested_dict = lambda: defaultdict(nested_dict)
    data = nested_dict()

    ax = 1

    # Convert to primitive variables; able to use pointwise conversion as it is still 2nd-order
    wS = fv.point_convert_conservative(grids, sim_variables)

    # Apply (TVD) slope limiters
    limited_values = limiters.gradient_limiters(wS, boundary, ax=ax, limiter="minmod")

    """Linear reconstruction [Derigs et al., 2017]
        Current convention: |                        w(i-1/2)                    w(i+1/2)                       |
                            |-->         i-1         <--|-->          i          <--|-->         i+1         <--|
                            |   w_L(i-1)     w_R(i-1)   |   w_L(i)         w_R(i)   |   w_L(i+1)     w_R(i+1)   |
                    OR      |   w+(i-3/2)   w-(i-1/2)   |   w+(i-1/2)   w-(i+1/2)   |  w+(i+1/2)    w-(i+3/2)   |
    """
    gradients = .5 * limited_values
    wL, wR = np.copy(wS-gradients), np.copy(wS+gradients)  # (eq. 4.13)

    # Re-align the reconstructed interfaces to correspond with the solvers
    w_minus, w_plus = wR, fv.add_boundary(wL, boundary, axis=ax).take(range(2,wL.shape[ax]+2), axis=ax)

    # Convert the primitive variables
    # The conversion can be pointwise conversion for face-average values as it is still 2nd-order
    q_minus, q_plus = fv.convert_primitive(w_minus, sim_variables, "face"), fv.convert_primitive(w_plus, sim_variables, "face")

    # Compute the fluxes and the Jacobian
    flux_minus, flux_plus = constructor.make_flux(w_minus, gamma, axes), constructor.make_flux(w_plus, gamma, axes)
    A = constructor.make_Jacobian(wS, gamma, axes)

    # Update dict
    data['wS'] = wS
    data['w_minus'] = w_minus
    data['w_plus'] = w_plus
    data['q_minus'] = q_minus
    data['q_plus'] = q_plus
    data['flux_minus'] = flux_minus
    data['flux_plus'] = flux_plus
    data['Jacobian'] = A

    return data