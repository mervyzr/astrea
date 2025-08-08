from collections import defaultdict

import numpy as np

from functions import constructor, fv
from num_methods import ct, limiters

##############################################################################
# Piecewise linear reconstruction method (PLM) [van Leer, 1979]
##############################################################################

def run(grid, sim_variables):
    boundary, axes, magnetic = sim_variables.boundary, sim_variables.axes, sim_variables.magnetic
    convert = sim_variables.convert
    Bx, By = sim_variables.Bx, sim_variables.By

    nested_dict = lambda: defaultdict(nested_dict)
    data = nested_dict()

    # Convert to primitive variables
    primitive = convert("conservative", grid, sim_variables, staggered=magnetic)

    for axis in axes:
        # Pad array with boundary & apply (TVD) slope limiters
        padded_primitive = fv.add_boundary(primitive, boundary, axis=axis)
        limited_values = limiters.minmod_limiter(padded_primitive, axis=axis)

        """Linear reconstruction [Derigs et al., 2017]
        |                        w(i-1/2)                    w(i+1/2)                       |
        |<--         i-1         -->|<--          i          -->|<--         i+1         -->|
        |   w_L(i-1)     w_R(i-1)   |   w_L(i)         w_R(i)   |   w_L(i+1)     w_R(i+1)   |
        |   w+(i-3/2)   w-(i-1/2)   |   w+(i-1/2)   w-(i+1/2)   |  w+(i+1/2)    w-(i+3/2)   |
        """
        gradients = .5 * limited_values
        wL, wR = primitive - gradients, primitive + gradients  # (eq. 4.13)

        # Magnetic component after computing to interface
        if magnetic:
            wR[...,(Bx,By)] = grid[...,(Bx,By)]
            data[axis]['ortho_interfaces'] = ct.reconstruct_transverse(wR, sim_variables, axis=axis)

        # Re-align the interfaces so that cell wall is in between interfaces
        prim_plus, prim_minus = fv.slice_(fv.add_boundary(wL, boundary, axis=axis), axis, start=1), fv.slice_(fv.add_boundary(wR, boundary, axis=axis), axis, end=-1)
        if magnetic:
            padded_grid = fv.add_boundary(grid, boundary, axis=axis)
            prim_plus[...,(Bx,By)] = prim_minus[...,(Bx,By)] = fv.slice_(padded_grid, axis, end=-1)[...,(Bx,By)]

        # Get the average solution between the interfaces at the boundaries
        intf_avg = fv.slice_(.5* (prim_plus + prim_minus), axis, start=1)
        padded_intf_avg = fv.add_boundary(intf_avg, boundary, axis=axis)

        # Convert the primitive interface variables
        cons_plus, cons_minus = fv.convert_interface("primitive", prim_plus, axis, sim_variables), fv.convert_interface("primitive", prim_minus, axis, sim_variables)

        # Compute the fluxes and the Jacobian
        flux_plus, flux_minus = constructor.make_flux(prim_plus, sim_variables, axis=axis), constructor.make_flux(prim_minus, sim_variables, axis=axis)
        jacobian = constructor.make_Jacobian(padded_intf_avg, sim_variables, axis=axis)

        # Update dict
        data[axis]['prim_interfaces'] = prim_plus, prim_minus
        data[axis]['cons_interfaces'] = cons_plus, cons_minus
        data[axis]['flux_interfaces'] = flux_plus, flux_minus
        data[axis]['characteristics'] = np.linalg.eigvals(jacobian)

    return data