import numpy as np

from functions import grid as gutils
from functions import math as mfuncs
from functions import numeric
from numkit import c_transport as ct
from numkit import solvers

##############################################################################
# WENO reconstruction method [Shu, 2009]
##############################################################################

def reconstruct(grid, sim_variables, axis, order=5):
    eps = sim_variables.eps

    # Define frequently used terms
    padded_grid = gutils.add_boundary(grid, sim_variables, axis=axis)

    zeroth = np.copy(grid)
    minus_one, plus_one = gutils.slice_(padded_grid, axis, end=-2), gutils.slice_(padded_grid, axis, start=2)

    """WENO reconstruction from cell averages to face averages (both sides) [Shu, 2009; San & Kara, 2015]
    |                        w(i-1/2)                    w(i+1/2)                       |
    |<--         i-1         -->|<--          i          -->|<--         i+1         -->|
    |   w_L(i-1)     w_R(i-1)   |   w_L(i)         w_R(i)   |   w_L(i+1)     w_R(i+1)   |
    |   w+(i-3/2)   w-(i-1/2)   |   w+(i-1/2)   w-(i+1/2)   |   w+(i+1/2)   w-(i+3/2)   |
    """
    if 0 < order <= 3:
        # Define the linear weights
        g_k = 1/3, 2/3
        inv_g_k = g_k[::-1]

        # Determine the smoothness indicators
        b_k = (zeroth - minus_one)**2, (plus_one - zeroth)**2

        # Compute the alpha
        alpha = lambda gk, k: gk[k]/(b_k[k] + eps)**2

        # Define the non-linear weights
        omega = lambda k: mfuncs.divide(alpha(g_k, k), alpha(g_k, 0)+alpha(g_k, 1))
        inv_omega = lambda k: mfuncs.divide(alpha(inv_g_k, k), alpha(inv_g_k, 0)+alpha(inv_g_k, 1))

        # Define the stencils
        wR = (
            omega(0) * (1.5*zeroth - .5*minus_one)
            + omega(1) * (.5*zeroth + .5*plus_one)
        )
        wL = (
            inv_omega(0) * (1.5*zeroth - .5*plus_one)
            + inv_omega(1) * (.5*zeroth + .5*minus_one)
        )

    else:
        padded_grid_2 = gutils.add_boundary(grid, sim_variables, stencil=2, axis=axis)
        minus_two, plus_two = gutils.slice_(padded_grid_2, axis, end=-4), gutils.slice_(padded_grid_2, axis, start=4)

        if 5 < order <= 7:
            padded_grid_3 = gutils.add_boundary(grid, sim_variables, stencil=3, axis=axis)
            minus_three, plus_three = gutils.slice_(padded_grid_3, axis, end=-6), gutils.slice_(padded_grid_3, axis, start=6)

            g_k = 1/35, 12/35, 18/35, 4/35
            inv_g_k = g_k[::-1]

            b0 = (
                minus_three * (547*minus_three - 3882*minus_two + 4642*minus_one - 1854*zeroth)
                + minus_two * (7043*minus_two - 17246*minus_one + 7042*zeroth)
                + minus_one * (11003*minus_one - 9402*zeroth)
                + zeroth * (2107*zeroth)
            )
            b1 = (
                minus_two * (267*minus_two - 1642*minus_one + 1602*zeroth - 494*plus_one)
                + minus_one * (2843*minus_one - 5966*zeroth + 1922*plus_one)
                + zeroth * (3443*zeroth - 2522*plus_one)
                + plus_one * (547*plus_one)
            )
            b2 = (
                minus_one * (547*minus_one - 2522*zeroth + 1922*plus_one - 494*plus_two)
                + zeroth * (3443*zeroth - 5966*plus_one + 1602*plus_two)
                + plus_one * (2843*plus_one - 1642*plus_two)
                + plus_two * (267*plus_two)
            )
            b3 = (
                zeroth * (2107*zeroth - 9402*plus_one + 7042*plus_two - 1854*plus_three)
                + plus_one * (11003*plus_one - 17246*plus_two + 4642*plus_three)
                + plus_two * (7043*plus_two - 3882*plus_three)
                + plus_three * (547*plus_three)
            )
            b_k = b0, b1, b2, b3

            alpha = lambda gk, k: gk[k]/(b_k[k] + eps)**2

            omega = lambda k: mfuncs.divide(alpha(g_k, k), alpha(g_k, 0)+alpha(g_k, 1)+alpha(g_k, 2)+alpha(g_k, 3))
            inv_omega = lambda k: mfuncs.divide(alpha(inv_g_k, k), alpha(inv_g_k, 0)+alpha(inv_g_k, 1)+alpha(inv_g_k, 2)+alpha(inv_g_k, 3))

            wR = (
                omega(0) * (-1/4*minus_three + 13/12*minus_two - 23/12*minus_one + 25/12*zeroth)
                + omega(1) * (1/12*minus_two - 5/12*minus_one + 13/12*zeroth + 1/4*plus_one)
                + omega(2) * (-1/12*minus_one + 7/12*zeroth + 7/12*plus_one - 1/12*plus_two)
                + omega(3) * (1/4*zeroth + 13/12*plus_one - 5/12*plus_two + 1/12*plus_three)
            )
            wL = (
                inv_omega(0) * (1/4*zeroth + 13/12*minus_one - 5/12*minus_two + 1/12*minus_three)
                + inv_omega(1) * (-1/12*plus_one + 7/12*zeroth + 7/12*minus_one - 1/12*minus_two)
                + inv_omega(2) * (1/12*plus_two - 5/12*plus_one + 13/12*zeroth + 1/4*minus_one)
                + inv_omega(3) * (-1/4*plus_three + 13/12*plus_two - 23/12*plus_one + 25/12*zeroth)
            )

        else:
            g_k = 1/10, 3/5, 3/10
            inv_g_k = g_k[::-1]

            b0 = (
                13/12 * (minus_two - 2*minus_one + zeroth)**2
                + 1/4 * (minus_two - 4*minus_one + 3*zeroth)**2
            )
            b1 = (
                13/12 * (minus_one - 2*zeroth + plus_one)**2
                + 1/4 * (minus_one - plus_one)**2
            )
            b2 = (
                13/12 * (zeroth - 2*plus_one + plus_two)**2
                + 1/4 * (3*zeroth - 4*plus_one + plus_two)**2
            )
            b_k = b0, b1, b2

            alpha = lambda gk, k: gk[k]/(b_k[k] + eps)**2

            omega = lambda k: mfuncs.divide(alpha(g_k, k), alpha(g_k, 0)+alpha(g_k, 1)+alpha(g_k, 2))
            inv_omega = lambda k: mfuncs.divide(alpha(inv_g_k, k), alpha(inv_g_k, 0)+alpha(inv_g_k, 1)+alpha(inv_g_k, 2))

            wR = (
                omega(0) * (1/3*minus_two - 7/6*minus_one + 11/6*zeroth)
                + omega(1) * (-1/6*minus_one + 5/6*zeroth + 1/3*plus_one)
                + omega(2) * (1/3*zeroth + 5/6*plus_one - 1/6*plus_two)
            )
            wL = (
                inv_omega(0) * (1/3*zeroth + 5/6*minus_one - 1/6*minus_two)
                + inv_omega(1) * (-1/6*plus_one + 5/6*zeroth + 1/3*minus_one)
                + inv_omega(2) * (1/3*plus_two - 7/6*plus_one + 11/6*zeroth)
            )

    return wL, wR


def run(grid, sim_variables, axis):
    subgrid, multidimensional, magnetic, ds = sim_variables.subgrid, sim_variables.multidimensional, sim_variables.magnetic, sim_variables.ds
    data = {}

    Riemann_solver = solvers.get_Riemann_solver(sim_variables)

    # WENO reconstruction [Shu, 2009; San & Kara, 2015]
    try:
        wL, wR = reconstruct(grid, sim_variables, axis, int(subgrid.replace('-','').split("weno")[-1]))
    except ValueError:
        wL, wR = reconstruct(grid, sim_variables, axis)

    # Re-align the interfaces so that cell wall is in between interfaces
    assign_interfaces = ct.assign_interfaces if magnetic else gutils.assign_interfaces
    prim_plus, prim_minus = assign_interfaces((wL, wR), grid, sim_variables, axis)

    # Get the average solution between the interfaces at the boundaries
    intf_avg = numeric.compute_Roe_average((prim_plus, prim_minus), sim_variables)
    padded_intf_avg = gutils.slice_(gutils.add_boundary(intf_avg, sim_variables, axis=axis), axis, end=-1)

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

    # Compute eigmax for time stepping limits
    data['eigmax'] = ds[axis]/numeric.compute_eigmax(characteristics, axis=axis)

    # Compute alphas and save the reconstructed interfaces for CT computation
    if magnetic and multidimensional:
        data['alphas'] = ct.compute_alphas(characteristics, axis=axis)
        data['interfaces'] = gutils.slice_(prim_plus, axis, start=1), gutils.slice_(prim_minus, axis, end=-1)

    # Calculate the interface-averaged fluxes
    intf_fluxes_avgd = Riemann_solver(axis, sim_variables, **{
        'prim_interfaces': (prim_plus, prim_minus),
        'cons_interfaces': (cons_plus, cons_minus),
        'flux_interfaces': (flux_plus, flux_minus),
        'characteristics': characteristics,
        'jacobian': gutils.slice_(jacobian, axis, start=1),
    })

    # Compute the orthogonal L/R Riemann states and fluxes at higher-order accuracy
    if multidimensional:
        # Calculate the interface-centred fluxes
        intf_fluxes_cntrd = Riemann_solver(axis, sim_variables, **{
            'prim_interfaces': gutils.approx_face_avg((prim_plus, prim_minus), sim_variables, axis),
            'cons_interfaces': gutils.approx_face_avg((cons_plus, cons_minus), sim_variables, axis),
            'flux_interfaces': gutils.approx_face_avg((flux_plus, flux_minus), sim_variables, axis),
            'characteristics': characteristics,
            'jacobian': gutils.slice_(jacobian, axis, start=1),
        })

        # Compute the higher-order fluxes
        intf_fluxes_cntrd = gutils.approx_flux_avg(intf_fluxes_cntrd, intf_fluxes_avgd, sim_variables, axis)
    else:
        # Orthogonal Laplacian in 1d is zero
        intf_fluxes_cntrd = intf_fluxes_avgd

    # Compute flux difference for hydrodynamic components
    data['fluxes'] = np.diff(intf_fluxes_cntrd, axis=axis)/ds[axis]

    return data