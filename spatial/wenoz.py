import numpy as np

from functions import grid as gutils
from functions import numeric
from numkit import c_transport as ct
from numkit import kernels, solvers

##############################################################################
# WENO-Z reconstruction method [Borges et al., 2008]
##############################################################################

def reconstruct(grid, sim_variables, axis, power=1):
    """WENO-Z reconstruction from cell averages to face averages (both sides) [Borges et al., 2008]
    |                        w(i-1/2)                    w(i+1/2)                       |
    |<--         i-1         -->|<--          i          -->|<--         i+1         -->|
    |   w_L(i-1)     w_R(i-1)   |   w_L(i)         w_R(i)   |   w_L(i+1)     w_R(i+1)   |
    |   w+(i-3/2)   w-(i-1/2)   |   w+(i-1/2)   w-(i+1/2)   |   w+(i+1/2)   w-(i+3/2)   |
    """
    return kernels.weno_reconstruct(grid, axis, 1e-40, kernels.get_bci(sim_variables), wenoz=True, power=power)


def run(grid, sim_variables, axis):
    multidimensional, magnetic, ds = sim_variables.multidimensional, sim_variables.magnetic, sim_variables.ds[axis]

    # WENO-Z reconstruction [Borges et al., 2008]
    wL, wR = reconstruct(grid, sim_variables, axis)

    # Re-align the interfaces so that cell wall is in between interfaces
    assign_interfaces = ct.assign_interfaces if magnetic else gutils.assign_interfaces
    prim_plus, prim_minus = assign_interfaces((wL, wR), grid, sim_variables, axis)

    # Compute the wavespeeds
    wavespeeds = numeric.compute_interface_wavespeed_bounds((prim_plus, prim_minus), sim_variables, axis)

    # Calculate the interface fluxes; WENO-Z works in pointwise representation
    intf_fluxes = solvers.riemann_flux(axis, sim_variables, wavespeeds, (prim_plus, prim_minus))

    # Compute flux difference for hydrodynamic components
    fluxes = np.diff(intf_fluxes, axis=axis)/ds

    if magnetic and multidimensional:
        return fluxes, wavespeeds, (gutils.slice_(prim_plus, axis, start=1), gutils.slice_(prim_minus, axis, end=-1))
    else:
        return fluxes, wavespeeds, None