import numpy as np

from functions import grid as gutils
from functions import numeric
from numkit import c_transport as ct
from numkit import kernels, solvers

##############################################################################
# TENO reconstruction method [Fu et al., 2016; Fu, 2021]
##############################################################################

# Very sensitive to higher resolutions

# Linear weights by number of candidate stencils; note the arrangement vs. WENO: i, i+1, i-1, i+2, i-2, i+3, i-3, ...
LINEAR_WEIGHTS = {
    3: (1,),
    4: (3/6, 3/6),
    5: (6/10, 3/10, 1/10),
    6: (9/20, 6/20, 1/20, 4/20),
    7: (18/35, 9/35, 3/35, 4/35, 1/35),
    8: (30/70, 18/70, 4/70, 12/70, 1/70, 5/70),
}


def reconstruct(grid, sim_variables, axis, q=6, C_T=1e-7, adaptive=False):
    """TENO reconstruction from cell averages to face averages (both sides) [Fu et al., 2016; Fu, 2021]
    |                        w(i-1/2)                    w(i+1/2)                       |
    |<--         i-1         -->|<--          i          -->|<--         i+1         -->|
    |   w_L(i-1)     w_R(i-1)   |   w_L(i)         w_R(i)   |   w_L(i+1)     w_R(i+1)   |
    |   w+(i-3/2)   w-(i-1/2)   |   w+(i-1/2)   w-(i+1/2)   |   w+(i+1/2)   w-(i+3/2)   |
    """
    return kernels.teno_reconstruct(
        grid, axis, kernels.get_bci(sim_variables), LINEAR_WEIGHTS[5], q=q, C_T=C_T, adaptive=adaptive,
    )


def run(grid, sim_variables, axis):
    subgrid, multidimensional, magnetic, ds = sim_variables.subgrid, sim_variables.multidimensional, sim_variables.magnetic, sim_variables.ds[axis]

    # TENO reconstruction [Fu et al., 2016]
    if subgrid.endswith("a"):
        wL, wR = reconstruct(grid, sim_variables, axis, adaptive=True)
    else:
        wL, wR = reconstruct(grid, sim_variables, axis)

    # Re-align the interfaces so that cell wall is in between interfaces
    assign_interfaces = ct.assign_interfaces if magnetic else gutils.assign_interfaces
    prim_plus, prim_minus = assign_interfaces((wL, wR), grid, sim_variables, axis)

    # Compute the wavespeeds
    wavespeeds = numeric.compute_interface_wavespeed_bounds((prim_plus, prim_minus), sim_variables, axis)

    # Interface fluxes, at higher order from the face-centred states in multiple dimensions
    intf_fluxes = solvers.higher_order_flux(axis, sim_variables, wavespeeds, (prim_plus, prim_minus))

    # Compute flux difference for hydrodynamic components
    fluxes = np.diff(intf_fluxes, axis=axis)/ds

    if magnetic and multidimensional:
        return fluxes, wavespeeds, (gutils.slice_(prim_plus, axis, start=1), gutils.slice_(prim_minus, axis, end=-1))
    else:
        return fluxes, wavespeeds, None