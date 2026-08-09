import numpy as np

from functions import grid as gutils
from functions import numeric
from numkit import c_transport as ct
from numkit import limiters, solvers

##############################################################################
# Piecewise linear reconstruction method (PLM) [van Leer, 1979]
##############################################################################

def reconstruct(grid, sim_variables, axis):
    # Pad array with boundary
    padded_grid = gutils.add_boundary(grid, sim_variables, axis=axis)

    # Apply (TVD) slope limiters
    limited_values = limiters.minmod(padded_grid, axis=axis)
    gradients = .5 * limited_values

    """Reconstruct from cell averages to face averages (both sides)
    |                        w(i-1/2)                    w(i+1/2)                       |
    |<--         i-1         -->|<--          i          -->|<--         i+1         -->|
    |   w_L(i-1)     w_R(i-1)   |   w_L(i)         w_R(i)   |   w_L(i+1)     w_R(i+1)   |
    |   w+(i-3/2)   w-(i-1/2)   |   w+(i-1/2)   w-(i+1/2)   |  w+(i+1/2)    w-(i+3/2)   |
    """
    return grid - gradients, grid + gradients


def run(grid, sim_variables, axis):
    multidimensional, magnetic, ds = sim_variables.multidimensional, sim_variables.magnetic, sim_variables.ds[axis]

    Riemann_solver = solvers.get_Riemann_solver(sim_variables)

    # Linear reconstruction [Derigs et al., 2017]
    wL, wR = reconstruct(grid, sim_variables, axis)

    # Re-align the interfaces so that cell wall is in between interfaces
    assign_interfaces = ct.assign_interfaces if magnetic else gutils.assign_interfaces
    prim_plus, prim_minus = assign_interfaces((wL, wR), grid, sim_variables, axis)

    # Get the average solution between the interfaces at the boundaries
    intf_avg = .5 * (prim_plus + prim_minus)
    padded_intf_avg = gutils.add_boundary(intf_avg, sim_variables, axis=axis)

    # Convert the primitive variables at the interface
    cons_plus, cons_minus = gutils.variable_convert_intf("primitive", prim_plus, sim_variables, axis=axis), gutils.variable_convert_intf("primitive", prim_minus, sim_variables, axis=axis)

    # Compute the fluxes and the Jacobian
    flux_plus, flux_minus = numeric.compute_flux(prim_plus, sim_variables, axis=axis), numeric.compute_flux(prim_minus, sim_variables, axis=axis)
    jacobian = numeric.compute_jacobian(padded_intf_avg, sim_variables, axis=axis)

    # Resolve characteristics at interfaces
    try:
        characteristics = np.linalg.eigvals(jacobian)
    except np.linalg.LinAlgError:
        try:
            characteristics = numeric.compute_characteristics(padded_intf_avg, sim_variables, axis=axis)
        except np.linalg.LinAlgError:
            characteristics = np.full_like(padded_intf_avg, .01)

    # Calculate the interface-averaged fluxes (pointwise & averaged values are the same for lower-order schemes)
    intf_fluxes_avgd = intf_fluxes_cntrd = Riemann_solver(axis, sim_variables, **{
        'prim_interfaces': (prim_plus, prim_minus),
        'cons_interfaces': (cons_plus, cons_minus),
        'flux_interfaces': (flux_plus, flux_minus),
        'characteristics': characteristics,
        'jacobian': gutils.slice_(jacobian, axis, *[1,-1]),
    })

    # Compute flux difference for hydrodynamic components
    fluxes = np.diff(intf_fluxes_cntrd, axis=axis)/ds

    if magnetic and multidimensional:
        return fluxes, characteristics, (gutils.slice_(prim_plus, axis, start=1), gutils.slice_(prim_minus, axis, end=-1))
    else:
        return fluxes, characteristics, None