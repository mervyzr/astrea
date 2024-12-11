import numpy as np

from functions import fv

##############################################################################
# Fourth-order upwind constrained transport algorithm for MHD [Felker & Stone, 2018]
##############################################################################

# Compute the corner electric fields; gives 4-fold values for each corner
def corner_E(grid, sim_variables, method="ppm"):
    gamma, permutations = sim_variables.gamma, sim_variables.permutations

    for axis, axes in enumerate(permutations[1:] + permutations[:1]):
        _grid = grid.transpose(axes)
        pass


        if sim_variables.magnetic and sim_variables.dimension > 1:
            next_axes = permutations[(axis+1) % len(permutations)]
            wF_transverse = np.copy(wF.transpose(next_axes))
            wF_transverse_2 = fv.add_boundary(wF_transverse, boundary, 2)
            wF_transverse_1 = np.copy(wF_transverse_2[1:-1])

            wF_U = 7/12 * (wF_transverse + wF_transverse_1[2:]) - 1/12 * (wF_transverse_1[:-2] + wF_transverse_2[4:])
            pass