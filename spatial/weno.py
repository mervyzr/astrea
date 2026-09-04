import numpy as np

from functions import grid as gutils
from functions import math as mfuncs
from functions import numeric
from numkit import c_transport as ct
from numkit import solvers

##############################################################################
# WENO reconstruction method [Jiang & Shu, 1996; Balsara & Shu, 2000; Shu, 2009; San & Kara, 2015]
##############################################################################

def reconstruct(grid, sim_variables, axis, order=5):
    eps = sim_variables.eps

    # Define frequently used terms
    padded_grid = gutils.add_boundary(grid, sim_variables, axis=axis)

    # Read-only alias, not a copy. The caller's grid is shared by the concurrent axis
    # sweeps and this build runs without the GIL, so it must never be mutated here
    zeroth = grid
    minus_one, plus_one = gutils.slice_(padded_grid, axis, end=-2), gutils.slice_(padded_grid, axis, start=2)

    """WENO reconstruction from cell averages to face averages (both sides) [Jiang & Shu, 1996]
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
        wR = .5 * (
            omega(0) * (3*zeroth - minus_one)
            + omega(1) * (zeroth + plus_one)
        )
        wL = .5 * (
            inv_omega(0) * (3*zeroth - plus_one)
            + inv_omega(1) * (zeroth + minus_one)
        )

    else:
        padded_grid_2 = gutils.add_boundary(grid, sim_variables, stencil=2, axis=axis)
        minus_two, plus_two = gutils.slice_(padded_grid_2, axis, end=-4), gutils.slice_(padded_grid_2, axis, start=4)

        # [Balsara & Shu, 2000]
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

            wR = 1/12 * (
                omega(0) * (-3*minus_three + 13*minus_two - 23*minus_one + 25*zeroth)
                + omega(1) * (minus_two - 5*minus_one + 13*zeroth + 3*plus_one)
                + omega(2) * (-minus_one + 7*zeroth + 7*plus_one - plus_two)
                + omega(3) * (3*zeroth + 13*plus_one - 5*plus_two + plus_three)
            )
            wL = 1/12 * (
                inv_omega(0) * (3*zeroth + 13*minus_one - 5*minus_two + minus_three)
                + inv_omega(1) * (-plus_one + 7*zeroth + 7*minus_one - minus_two)
                + inv_omega(2) * (plus_two - 5*plus_one + 13*zeroth + 3*minus_one)
                + inv_omega(3) * (-3*plus_three + 13*plus_two - 23*plus_one + 25*zeroth)
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

            # The denominator depends only on the smoothness indicator, so it is shared by
            # the forward and reversed linear weights. Hoisting it takes the count of these
            # squarings and divisions from 24 to 3 and 6 per reconstruction
            denominator_k = tuple((b + eps)**2 for b in b_k)
            alpha_k = tuple(g_k[k]/denominator_k[k] for k in range(3))
            inv_alpha_k = tuple(inv_g_k[k]/denominator_k[k] for k in range(3))

            alpha_sum = alpha_k[0] + alpha_k[1] + alpha_k[2]
            inv_alpha_sum = inv_alpha_k[0] + inv_alpha_k[1] + inv_alpha_k[2]
            omega = tuple(mfuncs.divide(a, alpha_sum) for a in alpha_k)
            inv_omega = tuple(mfuncs.divide(a, inv_alpha_sum) for a in inv_alpha_k)

            wR = 1/6 * (
                omega[0] * (2*minus_two - 7*minus_one + 11*zeroth)
                + omega[1] * (-minus_one + 5*zeroth + 2*plus_one)
                + omega[2] * (2*zeroth + 5*plus_one - plus_two)
            )
            wL = 1/6 * (
                inv_omega[0] * (2*zeroth + 5*minus_one - minus_two)
                + inv_omega[1] * (-plus_one + 5*zeroth + 2*minus_one)
                + inv_omega[2] * (2*plus_two - 7*plus_one + 11*zeroth)
            )

    return wL, wR


def run(grid, sim_variables, axis):
    subgrid, multidimensional, magnetic, ds = sim_variables.subgrid, sim_variables.multidimensional, sim_variables.magnetic, sim_variables.ds[axis]

    Riemann_solver = solvers.get_Riemann_solver(sim_variables)
    needs_jacobian = solvers.needs_jacobian(sim_variables)

    # WENO reconstruction [Jiang & Shu, 1996]
    try:
        wL, wR = reconstruct(grid, sim_variables, axis, int(subgrid.replace('-','')[-1]))
    except ValueError:
        wL, wR = reconstruct(grid, sim_variables, axis)

    # Re-align the interfaces so that cell wall is in between interfaces
    assign_interfaces = ct.assign_interfaces if magnetic else gutils.assign_interfaces
    prim_plus, prim_minus = assign_interfaces((wL, wR), grid, sim_variables, axis)

    # Get the average solution between the interfaces at the boundaries
    intf_avg = numeric.compute_Roe_average((prim_plus, prim_minus), sim_variables)
    padded_intf_avg = gutils.add_boundary(intf_avg, sim_variables, axis=axis)

    # Convert the primitive variables at the interface
    cons_plus, cons_minus = gutils.variable_convert_intf("primitive", prim_plus, sim_variables, axis=axis), gutils.variable_convert_intf("primitive", prim_minus, sim_variables, axis=axis)

    # Compute the fluxes and the Jacobian
    flux_plus, flux_minus = numeric.compute_flux(prim_plus, sim_variables, axis=axis), numeric.compute_flux(prim_minus, sim_variables, axis=axis)

    # Resolve characteristics at interfaces from the analytic eigenvalues rather than an
    # np.linalg.eigvals over an (N,N,N,8,8) Jacobian; see spatial/cweno.py for the rationale
    characteristics = numeric.compute_characteristics(padded_intf_avg, sim_variables, axis=axis)
    jacobian = numeric.compute_jacobian(padded_intf_avg, sim_variables, axis=axis) if needs_jacobian else None

    # Calculate the interface-averaged fluxes
    intf_fluxes_avgd = Riemann_solver(axis, sim_variables, **{
        'prim_interfaces': (prim_plus, prim_minus),
        'cons_interfaces': (cons_plus, cons_minus),
        'flux_interfaces': (flux_plus, flux_minus),
        'characteristics': characteristics,
        'jacobian': gutils.slice_(jacobian, axis, *[1,-1]) if needs_jacobian else None,
    })

    # Compute the orthogonal L/R Riemann states and fluxes at higher-order accuracy
    if multidimensional:
        # Calculate the interface-centred fluxes
        intf_fluxes_cntrd = Riemann_solver(axis, sim_variables, **{
            'prim_interfaces': gutils.approx_face_avg((prim_plus, prim_minus), sim_variables, axis),
            'cons_interfaces': gutils.approx_face_avg((cons_plus, cons_minus), sim_variables, axis),
            'flux_interfaces': gutils.approx_face_avg((flux_plus, flux_minus), sim_variables, axis),
            'characteristics': characteristics,
            'jacobian': gutils.slice_(jacobian, axis, *[1,-1]) if needs_jacobian else None,
        })

        # Compute the higher-order fluxes
        intf_fluxes_cntrd = gutils.approx_flux_avg(intf_fluxes_cntrd, intf_fluxes_avgd, sim_variables, axis)
    else:
        # Orthogonal Laplacian in 1d is zero
        intf_fluxes_cntrd = intf_fluxes_avgd

    # Compute flux difference for hydrodynamic components
    fluxes = np.diff(intf_fluxes_cntrd, axis=axis)/ds

    if magnetic and multidimensional:
        return fluxes, characteristics, (gutils.slice_(prim_plus, axis, start=1), gutils.slice_(prim_minus, axis, end=-1))
    else:
        return fluxes, characteristics, None