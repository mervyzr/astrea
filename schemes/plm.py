import numpy as np

from functions import constructor, fv
from num_methods import ct, limiters, solvers

##############################################################################
# Piecewise linear reconstruction method (PLM) [van Leer, 1979]
##############################################################################

def reconstruct(grid, sim_variables, axis):
    # Pad array with boundary
    padded_grid = fv.add_boundary(grid, sim_variables, axis=axis)

    # Apply (TVD) slope limiters
    limited_values = limiters.minmod_limiter(padded_grid, axis=axis)
    gradients = .5 * limited_values

    """Reconstruct from cell averages to face averages (both sides)
    |                        w(i-1/2)                    w(i+1/2)                       |
    |<--         i-1         -->|<--          i          -->|<--         i+1         -->|
    |   w_L(i-1)     w_R(i-1)   |   w_L(i)         w_R(i)   |   w_L(i+1)     w_R(i+1)   |
    |   w+(i-3/2)   w-(i-1/2)   |   w+(i-1/2)   w-(i+1/2)   |  w+(i+1/2)    w-(i+3/2)   |
    """
    return grid - gradients, grid + gradients


def run(grid, sim_variables, axis):
    convert, multidimensional, axes, magnetic, ds = sim_variables.convert, sim_variables.multidimensional, sim_variables.axes, sim_variables.magnetic, sim_variables.ds
    data = {}

    Riemann_solver = solvers.get_Riemann_solver(sim_variables)

    # Linear reconstruction [Derigs et al., 2017]
    wL, wR = reconstruct(grid, sim_variables, axis=axis)

    # Re-align the interfaces so that cell wall is in between interfaces
    prim_plus, prim_minus = fv.slice_(fv.add_boundary(wL, sim_variables, axis=axis), axis, start=1), fv.slice_(fv.add_boundary(wR, sim_variables, axis=axis), axis, end=-1)
    if magnetic:
        padded_grid = fv.add_boundary(grid, sim_variables, axis=axis)
        prim_plus[...,5+axes] = prim_minus[...,5+axes] = fv.slice_(padded_grid, axis, end=-1)[...,5+axes]
        data['interfaces'] = fv.slice_(prim_plus, axis, start=1), fv.slice_(prim_minus, axis, start=1)

    # Get the average solution between the interfaces at the boundaries
    intf_avg = .5 * (prim_plus + prim_minus)
    padded_intf_avg = fv.add_boundary(fv.slice_(intf_avg, axis, start=1), sim_variables, axis=axis)

    # Convert the primitive variables at the interface
    cons_plus, cons_minus = convert("primitive", prim_plus, sim_variables, axis=axis, pos='intf'), convert("primitive", prim_minus, sim_variables, axis=axis, pos='intf')

    # Compute the fluxes and the Jacobian
    flux_plus, flux_minus = constructor.make_flux(prim_plus, sim_variables, axis=axis), constructor.make_flux(prim_minus, sim_variables, axis=axis)
    jacobian = constructor.make_Jacobian(padded_intf_avg, sim_variables, axis=axis)

    # Resolve characteristics at interfaces
    try:
        characteristics = np.linalg.eigvals(jacobian)
    except np.linalg.LinAlgError:
        try:
            characteristics = constructor.make_characteristics(padded_intf_avg, sim_variables, axis=axis)
        except np.linalg.LinAlgError:
            characteristics = np.full_like(padded_intf_avg, .1)

    # Compute eigmax for time stepping limits
    data['eigmax'] = ds[axis]/fv.compute_eigmax(characteristics, axis=axis)

    # Compute alphas and save the reconstructed interfaces for CT computation
    if magnetic and multidimensional:
        data['alphas'] = ct.compute_alphas(characteristics, axis=axis)

    # Calculate the interface-averaged fluxes (pointwise & averaged values are the same for lower-order schemes)
    intf_fluxes_avgd = intf_fluxes_cntrd = Riemann_solver(axis, sim_variables, **{
        'prim_interfaces': [prim_plus, prim_minus],
        'cons_interfaces': [cons_plus, cons_minus],
        'flux_interfaces': [flux_plus, flux_minus],
        'characteristics': characteristics,
    })

    # Compute flux difference for hydrodynamic components
    data['fluxes'] = np.diff(intf_fluxes_cntrd, axis=axis)/ds[axis]

    return data