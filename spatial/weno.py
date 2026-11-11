import numpy as np

from functions import grid as gutils
from functions import numeric
from numkit import c_transport as ct
from numkit import kernels, solvers

##############################################################################
# WENO reconstruction method [Jiang & Shu, 1996; Balsara & Shu, 2000; Shu, 2009; San & Kara, 2015]
##############################################################################

def reconstruct(grid, sim_variables, axis, order=5):
    """WENO reconstruction from cell averages to face averages (both sides) [Jiang & Shu, 1996; Balsara & Shu, 2000]
    |                        w(i-1/2)                    w(i+1/2)                       |
    |<--         i-1         -->|<--          i          -->|<--         i+1         -->|
    |   w_L(i-1)     w_R(i-1)   |   w_L(i)         w_R(i)   |   w_L(i+1)     w_R(i+1)   |
    |   w+(i-3/2)   w-(i-1/2)   |   w+(i-1/2)   w-(i+1/2)   |   w+(i+1/2)   w-(i+3/2)   |
    """
    if 0 < order <= 3:
        order = 3
    elif 5 < order <= 7:
        order = 7
    else:
        order = 5
    return kernels.weno_reconstruct(grid, axis, sim_variables.eps, kernels.get_bci(sim_variables), order=order)


def run(grid, sim_variables, axis):
    subgrid, multidimensional, magnetic, ds = sim_variables.subgrid, sim_variables.multidimensional, sim_variables.magnetic, sim_variables.ds[axis]

    # WENO reconstruction [Jiang & Shu, 1996]
    try:
        wL, wR = reconstruct(grid, sim_variables, axis, int(subgrid.replace('-','')[-1]))
    except ValueError:
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