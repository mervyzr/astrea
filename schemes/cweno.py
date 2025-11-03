import concurrent.futures
from itertools import repeat

import numpy as np

from functions import constructor, fv
from num_methods import ct, solvers

##############################################################################
# CWENO reconstruction method [Levy et al., 1999, 2000]
##############################################################################

def reconstruct(grid, sim_variables, axis):
    # Define the frequently used terms
    padded_grid_2 = fv.add_boundary(grid, sim_variables, stencil=2, axis=axis)
    padded_grid = fv.slice_(padded_grid_2, axis, *[1,-1])

    zeroth = np.copy(grid)
    minus_one, minus_two = fv.slice_(padded_grid, axis, end=-2), fv.slice_(padded_grid_2, axis, end=-4)
    plus_one, plus_two = fv.slice_(padded_grid, axis, start=2), fv.slice_(padded_grid_2, axis, start=4)

    # Define the empirical parameters for Eq. 3.12
    eps, power = np.finfo(sim_variables.precision).eps, 2

    """CWENO reconstruction from cell averages to face averages (both sides) [Verma et al., 2018]
    |                        w(i-1/2)                    w(i+1/2)                       |
    |<--         i-1         -->|<--          i          -->|<--         i+1         -->|
    |   w_L(i-1)     w_R(i-1)   |   w_L(i)         w_R(i)   |   w_L(i+1)     w_R(i+1)   |
    |   w+(i-3/2)   w-(i-1/2)   |   w+(i-1/2)   w-(i+1/2)   |   w+(i+1/2)   w-(i+3/2)   |
    """
    # Define the linear weights C_k (5th-order) [Levy et al., 1999, tbl. 3.1]
    C_minus, C_zero, C_plus = 1/6, 2/3, 1/6

    # Determine the smoothness indicators (O(dx^4) at critical points but O(1) at discontinuities) [eq. 3.14]
    IS_minus = lambda stencils: 13/12 * (stencils[0] - 2*stencils[1] + stencils[2])**2 + 1/4 * (stencils[0] - 4*stencils[1] + 3*stencils[2])**2
    IS_zero = lambda stencils: 13/12 * (stencils[0] - 2*stencils[1] + stencils[2])**2 + 1/4 * (stencils[0] - stencils[2])**2
    IS_plus = lambda stencils: 13/12 * (stencils[0] - 2*stencils[1] + stencils[2])**2 + 1/4 * (3*stencils[0] - 4*stencils[1] + stencils[2])**2

    # Compute the alpha values [Levy et al., 1999, eq. 3.12]
    alpha = lambda C_k, IS_k: C_k/(eps + IS_k)**power

    # Compute the non-linear weights [Levy et al., 1999, eq. 3.11]
    denominator = (
        alpha(C_minus, IS_minus([minus_two, minus_one, zeroth]))
        + alpha(C_zero, IS_zero([minus_one, zeroth, plus_one]))
        + alpha(C_plus, IS_plus([zeroth, plus_one, plus_two]))
    )
    wj_minus = fv.divide(alpha(C_minus, IS_minus([minus_two, minus_one, zeroth])), denominator)
    wj_zero = fv.divide(alpha(C_zero, IS_zero([minus_one, zeroth, plus_one])), denominator)
    wj_plus = fv.divide(alpha(C_plus, IS_plus([zeroth, plus_one, plus_two])), denominator)

    wL = 1/6 * (
        wj_minus * (2*zeroth + 5*minus_one - minus_two)
        + wj_zero * (2*minus_one + 5*zeroth - plus_one)
        + wj_plus * (2*plus_two - 7*plus_one + 11*zeroth)
    )
    wR = 1/6 * (
        wj_minus * (11*zeroth - 7*minus_one + 2*minus_two)
        + wj_zero * (2*plus_one + 5*zeroth - minus_one)
        + wj_plus * (2*zeroth + 5*plus_one - plus_two)
    )

    return wL, wR


def run(grid, sim_variables, axis):
    convert, multidimensional, axes, magnetic, ds = sim_variables.convert, sim_variables.multidimensional, sim_variables.axes, sim_variables.magnetic, sim_variables.ds
    data = {}

    Riemann_solver = solvers.get_Riemann_solver(sim_variables)
    ortho_axes = axes[axes != axis] if (magnetic or multidimensional) else None

    # CWENO reconstruction [Levy et al., 1999; Verma et al., 2018]
    wL, wR = reconstruct(grid, sim_variables, axis=axis)

    # Re-align the interfaces so that cell wall is in between interfaces
    prim_plus, prim_minus = fv.slice_(fv.add_boundary(wL, sim_variables, axis=axis), axis, start=1), fv.slice_(fv.add_boundary(wR, sim_variables, axis=axis), axis, end=-1)
    if magnetic:
        padded_grid = fv.add_boundary(grid, sim_variables, axis=axis)
        prim_plus[...,5+axes] = prim_minus[...,5+axes] = fv.slice_(padded_grid, axis, end=-1)[...,5+axes]

    # Get the average solution between the interfaces at the boundaries
    intf_avg = fv.compute_Roe_average([prim_plus,prim_minus], sim_variables)
    padded_intf_avg = fv.add_boundary(fv.slice_(intf_avg, axis, start=1), sim_variables, axis=axis)

    # Convert the primitive variables at the interface
    cons_plus, cons_minus = convert("primitive", prim_plus, sim_variables, axis=axis, pos='intf'), convert("primitive", prim_minus, sim_variables, axis=axis, pos='intf')

    # Compute the fluxes and the Jacobian
    flux_plus, flux_minus = constructor.make_flux(prim_plus, sim_variables, axis=axis), constructor.make_flux(prim_minus, sim_variables, axis=axis)
    jacobian = constructor.make_Jacobian(padded_intf_avg, sim_variables, axis=axis)

    # Compute eigmax for time stepping limits
    characteristics = np.linalg.eigvals(jacobian)
    data['eigmax'] = ds[axis]/fv.compute_eigmax(characteristics, axis=axis)

    # Compute alphas and save the reconstructed interfaces for CT computation
    if magnetic and multidimensional:
        data['alphas'] = ct.compute_alphas(characteristics, axis=axis)
        data['interfaces'] = fv.slice_(prim_plus, axis, start=1), fv.slice_(prim_minus, axis, start=1)

    # Calculate the interface-averaged fluxes
    intf_fluxes_avgd = Riemann_solver(axis, sim_variables, **{
        'prim_interfaces': [prim_plus, prim_minus],
        'cons_interfaces': [cons_plus, cons_minus],
        'flux_interfaces': [flux_plus, flux_minus],
        'characteristics': characteristics,
    })

    # Compute the orthogonal L/R Riemann states and fluxes at higher-order accuracy
    if multidimensional:
        # Calculate the interface-centred fluxes
        intf_fluxes_cntrd = Riemann_solver(axis, sim_variables, **{
            'prim_interfaces': fv.approx_face_avg([prim_plus, prim_minus], sim_variables, axis),
            'cons_interfaces': fv.approx_face_avg([cons_plus, cons_minus], sim_variables, axis),
            'flux_interfaces': fv.approx_face_avg([flux_plus, flux_minus], sim_variables, axis),
            'characteristics': characteristics,
        })

        # Compute the 4th-order interface-centred fluxes from the interface-averaged fluxes via higher order approximation for each orthogonal axis
        with concurrent.futures.ThreadPoolExecutor() as inner_executor:
            jobs = inner_executor.map(fv.laplacian, repeat(intf_fluxes_avgd), repeat(sim_variables), ortho_axes)
            for idx, job in enumerate(jobs):
                intf_fluxes_cntrd -= (sim_variables.ds[ortho_axes[idx]]**2)/24 * job
    else:
        # Orthogonal Laplacian in 1d is zero
        intf_fluxes_cntrd = intf_fluxes_avgd

    # Compute flux difference for hydrodynamic components
    data['fluxes'] = np.diff(intf_fluxes_cntrd, axis=axis)/ds[axis]

    return data