from collections import defaultdict

import numpy as np

from functions import constructor, fv
from num_methods import ct

##############################################################################
# Piecewise constant reconstruction method (PCM) [Godunov, 1959]
##############################################################################

def run(grid, sim_variables):
    boundary, axes, magnetic = sim_variables.boundary, sim_variables.axes, sim_variables.magnetic
    convert_conservative = sim_variables.convert_conservative
    nested_dict = lambda: defaultdict(nested_dict)
    data = nested_dict()

    # Convert to primitive variables
    primitive = convert_conservative(grid, sim_variables, staggered=magnetic)

    for axis in axes:
        # Magnetic component after computing to interface (interface = centre for PCM)
        if magnetic:
            data[axis]['ortho_interfaces'] = ct.reconstruct_transverse(primitive, sim_variables, axis=axis)

        # Pad array with boundaries
        padded_conservative = fv.add_boundary(grid, boundary, axis=axis)
        padded_primitive = fv.add_boundary(primitive, boundary, axis=axis)

        # Compute the fluxes and the Jacobian
        fluxes = constructor.make_flux(padded_primitive, sim_variables, axis=axis)
        jacobian = constructor.make_Jacobian(padded_primitive, sim_variables, axis=axis)

        # Update data dictionary
        data[axis]['prim_interfaces'] = fv.slice_(padded_primitive, axis, start=1), fv.slice_(padded_primitive, axis, end=-1)
        data[axis]['cons_interfaces'] = fv.slice_(padded_conservative, axis, start=1), fv.slice_(padded_conservative, axis, end=-1)
        data[axis]['flux_interfaces'] = fv.slice_(fluxes, axis, start=1), fv.slice_(fluxes, axis, end=-1)
        data[axis]['characteristics'] = np.linalg.eigvals(jacobian)

    return data