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

    Riemann_solver = solvers.get_Riemann_solver(sim_variables)
    needs_jacobian = solvers.needs_jacobian(sim_variables)

    # Pad array with boundaries
    padded_primitive_2 = gutils.add_boundary(grid, sim_variables, stencil=2, axis=axis)
    padded_primitive = gutils.slice_(padded_primitive_2, axis, *[1,-1])

    # Re-align the interfaces so that cell wall is in between interfaces
    prim_plus, prim_minus = gutils.slice_(padded_primitive, axis, start=1), gutils.slice_(padded_primitive, axis, end=-1)

    # Convert the primitive variables at the interface
    cons_plus, cons_minus = gutils.variable_point_convert("primitive", prim_plus, sim_variables), gutils.variable_point_convert("primitive", prim_minus, sim_variables)

    # Compute the fluxes and the Jacobian
    flux_plus, flux_minus = numeric.compute_flux(prim_plus, sim_variables, axis=axis), numeric.compute_flux(prim_minus, sim_variables, axis=axis)

    padded_intf = .5 * (gutils.slice_(padded_primitive_2, axis, end=-1) + gutils.slice_(padded_primitive_2, axis, start=1))

    # Resolve characteristics at interfaces from the analytic eigenvalues rather than an
    # np.linalg.eigvals over an (N,N,N,8,8) Jacobian; see spatial/cweno.py for the rationale
    characteristics = numeric.compute_characteristics(padded_intf, sim_variables, axis=axis)
    jacobian = numeric.compute_jacobian(padded_intf, sim_variables, axis=axis) if needs_jacobian else None

    # Calculate the interface-averaged fluxes (pointwise & averaged values are the same for lower-order schemes)
    intf_fluxes_avgd = intf_fluxes_cntrd = Riemann_solver(axis, sim_variables, **{
        'prim_interfaces': (prim_plus, prim_minus),
        'cons_interfaces': (cons_plus, cons_minus),
        'flux_interfaces': (flux_plus, flux_minus),
        'characteristics': characteristics,
        'jacobian': gutils.slice_(jacobian, axis, *[1,-1]) if needs_jacobian else None,
    })

    # Compute flux difference for hydrodynamic components
    fluxes = np.diff(intf_fluxes_cntrd, axis=axis)/ds

    if magnetic and multidimensional:
        return fluxes, characteristics, (grid, grid)
    else:
        return fluxes, characteristics, None