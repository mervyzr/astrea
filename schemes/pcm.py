from collections import defaultdict

from functions import constructor, fv
from num_methods import mag_field

##############################################################################
# Piecewise constant reconstruction method (PCM) [Godunov, 1959]
##############################################################################

def run(grid, sim_variables):
    gamma, boundary, permutations, magnetic = sim_variables.gamma, sim_variables.boundary, sim_variables.permutations, sim_variables.magnetic
    convert_conservative = sim_variables.convert_conservative
    nested_dict = lambda: defaultdict(nested_dict)
    data = nested_dict()

    # Rotate grid and apply algorithm for each axis
    for axis, axes in permutations.items():
        staggered_grid = grid.transpose(axes)

        # Convert to primitive variables
        wS = convert_conservative(staggered_grid, sim_variables)
        q = fv.add_boundary(staggered_grid, boundary)

        if magnetic:
            data[axes]['wTs'] = mag_field.reconstruct_transverse(wS, sim_variables)

        # Compute the fluxes and the Jacobian
        w = fv.add_boundary(wS, boundary)
        f = constructor.make_flux(w, gamma, axis)
        A = constructor.make_Jacobian(w, gamma, axis)

        # Update dict
        data[axes]['wS'] = wS
        data[axes]['wFs'] = w[1:], w[:-1]
        data[axes]['qFs'] = q[1:], q[:-1]
        data[axes]['fluxFs'] = f[1:], f[:-1]
        data[axes]['Jacobian'] = A

    return data