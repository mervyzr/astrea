import numpy as np

from functions import constructor, fv
from num_methods import ct, limiters, solvers

##############################################################################
# Piecewise linear reconstruction method (PLM) [van Leer, 1979]
##############################################################################

def run(grid, sim_variables, axis):
    boundary, multidimensional, axes, magnetic, ds = sim_variables.boundary, sim_variables.multidimensional, sim_variables.axes, sim_variables.magnetic, sim_variables.ds
    Bx, By = sim_variables.Bx, sim_variables.By
    data = {}

    Riemann_solver = solvers.get_Riemann_solver(sim_variables)
    ortho_axes = axes[axes != axis]

    # Pad array with boundary & apply (TVD) slope limiters
    padded_grid = fv.add_boundary(grid, boundary, axis=axis)
    limited_values = limiters.minmod_limiter(padded_grid, axis=axis)

    """Linear reconstruction [Derigs et al., 2017]
    |                        w(i-1/2)                    w(i+1/2)                       |
    |<--         i-1         -->|<--          i          -->|<--         i+1         -->|
    |   w_L(i-1)     w_R(i-1)   |   w_L(i)         w_R(i)   |   w_L(i+1)     w_R(i+1)   |
    |   w+(i-3/2)   w-(i-1/2)   |   w+(i-1/2)   w-(i+1/2)   |  w+(i+1/2)    w-(i+3/2)   |
    """
    gradients = .5 * limited_values
    wL, wR = grid - gradients, grid + gradients  # (eq. 4.13)
    if magnetic:
        wR[...,(Bx,By)] = grid[...,(Bx,By)]

        if multidimensional:
            ortho_axis = 1 - axis
            # Magnetic transverse interfaces reconstructed orthogonal to the axis
            ortho_plus, ortho_minus = ct.reconstruct_transverse(wR, sim_variables, axis=ortho_axis)
            data['ortho_interfaces'] = fv.slice_(ortho_plus, axis=ortho_axis, start=1), fv.slice_(ortho_minus, axis=ortho_axis, start=1)

    # Re-align the interfaces so that cell wall is in between interfaces
    prim_plus, prim_minus = fv.slice_(fv.add_boundary(wL, boundary, axis=axis), axis, start=1), fv.slice_(fv.add_boundary(wR, boundary, axis=axis), axis, end=-1)
    if magnetic:
        prim_plus[...,(Bx,By)] = prim_minus[...,(Bx,By)] = fv.slice_(padded_grid, axis, end=-1)[...,(Bx,By)]

    # Get the average solution between the interfaces at the boundaries
    intf_avg = fv.slice_(.5* (prim_plus + prim_minus), axis, start=1)
    padded_intf_avg = fv.add_boundary(intf_avg, boundary, axis=axis)

    # Convert the primitive interface variables
    cons_plus, cons_minus = fv.convert_interface("primitive", prim_plus, axis, sim_variables), fv.convert_interface("primitive", prim_minus, axis, sim_variables)

    # Compute the fluxes and the Jacobian
    flux_plus, flux_minus = constructor.make_flux(prim_plus, sim_variables, axis=axis), constructor.make_flux(prim_minus, sim_variables, axis=axis)
    jacobian = constructor.make_Jacobian(padded_intf_avg, sim_variables, axis=axis)

    # Compute eigmax for time stepping limits
    characteristics = np.linalg.eigvals(jacobian)
    data['eigmax'] = ds[axis]/fv.compute_eigmax(characteristics, axis=axis)

    # Magnetic alpha computation
    if magnetic:
        # alphas refers to the maximum(+)/minimum(-) eigenvalues respectively
        local_max, local_min = np.max(characteristics, axis=-1), np.min(characteristics, axis=-1)
        max_eigvals = np.maximum(fv.slice_(local_max, axis, end=-1), fv.slice_(local_max, axis, start=1))
        min_eigvals = np.minimum(fv.slice_(local_min, axis, end=-1), fv.slice_(local_min, axis, start=1))
        data['alphas'] = fv.slice_(np.maximum(0, max_eigvals), axis, start=1), fv.slice_(-np.minimum(0, min_eigvals), axis, start=1)

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