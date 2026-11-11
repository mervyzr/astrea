import numpy as np

from functions import grid as gutils
from functions import numeric
from numkit import solvers

##############################################################################
# Piecewise constant reconstruction method (PCM) [Godunov, 1959]
##############################################################################

def reconstruct(grid, sim_variables, axis):
    return grid, grid


def run(grid, sim_variables, axis):
    multidimensional, magnetic, ds = sim_variables.multidimensional, sim_variables.magnetic, sim_variables.ds[axis]

    padded_primitive = gutils.add_boundary(grid, sim_variables, axis=axis)
    prim_plus, prim_minus = gutils.slice_(padded_primitive, axis, start=1), gutils.slice_(padded_primitive, axis, end=-1)

    padded_conservative = gutils.variable_point_convert("primitive", padded_primitive, sim_variables)
    cons = gutils.slice_(padded_conservative, axis, start=1), gutils.slice_(padded_conservative, axis, end=-1)

    # Compute the wavespeeds
    wavespeeds = numeric.compute_cell_wavespeed_bounds(grid, sim_variables, axis)

    # Calculate the interface-averaged fluxes (pointwise & averaged values are the same for lower-order schemes)
    intf_fluxes = solvers.riemann_flux(axis, sim_variables, wavespeeds, (prim_plus, prim_minus), cons=cons)

    # Compute flux difference for hydrodynamic components
    fluxes = np.diff(intf_fluxes, axis=axis)/ds

    if magnetic and multidimensional:
        return fluxes, wavespeeds, (grid, grid)
    else:
        return fluxes, wavespeeds, None