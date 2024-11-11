from collections import defaultdict

from functions import constructor, fv

##############################################################################
# Piecewise constant reconstruction method (PCM) [Godunov, 1959]
##############################################################################

def run(grid, sim_variables):
    gamma, boundary, permutations = sim_variables.gamma, sim_variables.boundary, sim_variables.permutations
    nested_dict = lambda: defaultdict(nested_dict)
    data = nested_dict()

    # Rotate grid and apply algorithm for each axis
    for axis, axes in enumerate(permutations):
        _grid = grid.transpose(axes)

        # Convert to primitive variables
        wS = fv.point_convert_conservative(_grid, sim_variables)
        q = fv.add_boundary(_grid, boundary)

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