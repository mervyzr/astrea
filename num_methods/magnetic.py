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