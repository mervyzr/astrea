import numpy as np

from functions import fv
from num_methods import limiters

##############################################################################
# Fourth-order upwind constrained transport algorithm for MHD [Felker & Stone, 2018]
##############################################################################

# Compute the corner electric fields; gives 4-fold values for each corner
def corner_E(wFs, axes, sim_variables, method="ppm"):
    wF = np.copy(wFs.transpose(axes))
    wF2 = fv.add_boundary(wF, sim_variables.boundary, 2)
    wF1 = np.copy(wF2[1:-1])

    wF_D = 7/12 * (wF1[:-2] + wF) - 1/12 * (wF2[:-4] + wF1[2:])
    wF_U = 7/12 * (wF + wF1[2:]) - 1/12 * (wF1[:-2] + wF2[4:])

    limited_wFs_UD = limiters.interface_limiter(wF_D, wF2[:-4], wF1[:-2], wF, wF1[2:]), limiters.interface_limiter(wF_U, wF1[:-2], wF, wF1[2:], wF2[4:])

    kwargs = {"wF_pad2":fv.add_boundary(wF, sim_variables.boundary, 2) , "boundary":sim_variables.boundary}
    wD, wU = limiters.interpolant_limiter(wF, wF1, wF2, "mc", *limited_wFs_UD, **kwargs)