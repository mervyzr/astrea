import numpy as np

from functions import grid as gutils
from functions import math as mfuncs
from functions import numeric
from numkit import c_transport as ct
from numkit import solvers

##############################################################################
# WENO-CU6 reconstruction method [Hu et al., 2010]
##############################################################################

def reconstruct(grid, sim_variables, axis, C=20, power=1):
    eps = 1e-40

    # Define frequently used terms
    padded_grid_3 = gutils.add_boundary(grid, sim_variables, stencil=3, axis=axis)
    padded_grid_2 = gutils.slice_(padded_grid_3, axis, *[1,-1])
    padded_grid = gutils.slice_(padded_grid_2, axis, *[1,-1])

    zeroth = np.copy(grid)
    minus_one, minus_two, minus_three = gutils.slice_(padded_grid, axis, end=-2), gutils.slice_(padded_grid_2, axis, end=-4), gutils.slice_(padded_grid_3, axis, end=-6)
    plus_one, plus_two, plus_three = gutils.slice_(padded_grid, axis, start=2), gutils.slice_(padded_grid_2, axis, start=4), gutils.slice_(padded_grid_3, axis, start=6)

    """WENO reconstruction from cell averages to face averages (both sides)
    |                        w(i-1/2)                    w(i+1/2)                       |
    |<--         i-1         -->|<--          i          -->|<--         i+1         -->|
    |   w_L(i-1)     w_R(i-1)   |   w_L(i)         w_R(i)   |   w_L(i+1)     w_R(i+1)   |
    |   w+(i-3/2)   w-(i-1/2)   |   w+(i-1/2)   w-(i+1/2)   |   w+(i+1/2)   w-(i+3/2)   |
    """
    # Determine the smoothness indicators
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
    b3 = 1/10080 * (
        271779 * minus_two**2
        + minus_two * (2380800*minus_one + 4086352*zeroth - 3462252*plus_one + 1458762*plus_two - 245620*plus_three)
        + minus_one * (5653317*minus_one - 20427884*zeroth + 17905032*plus_one - 7727988*plus_two + 1325006*plus_three)
        + zeroth * (19510972*zeroth - 35817664*plus_one + 15929912*plus_two - 2792660*plus_three)
        + plus_one * (17195652*plus_one - 15880404*plus_two + 2863984*plus_three)
        + plus_two * (3824847*plus_two - 1429976*plus_three)
        + 139633 * plus_three**2
    )
    b_k = b0, b1, b2, b3

    # Define the linear weights
    g_k = 1/20, 9/20, 9/20, 1/20
    inv_g_k = g_k[::-1]

    # Compute the tau value
    tau = b3 - 1/6 * (b0 + 4*b1 + b2)

    # Compute the alpha values
    alpha = lambda gk, k: gk[k] * (C + tau/(b_k[k] + eps))**power

    # Compute the non-linear weights
    omega = lambda k: mfuncs.divide(alpha(g_k, k), alpha(g_k, 0)+alpha(g_k, 1)+alpha(g_k, 2)+alpha(g_k, 3))
    inv_omega = lambda k: mfuncs.divide(alpha(inv_g_k, k), alpha(inv_g_k, 0)+alpha(inv_g_k, 1)+alpha(inv_g_k, 2)+alpha(inv_g_k, 3))

    # Define the stencils
    wR = 1/6 * (
        omega(0) * (2*minus_two - 7*minus_one + 11*zeroth)
        + omega(1) * (-minus_one + 5*zeroth + 2*plus_one)
        + omega(2) * (2*zeroth + 5*plus_one - plus_two)
        + omega(3) * (11*plus_one - 7*plus_two + 2*plus_three)
    )
    wL = 1/6 * (
        inv_omega(0) * (11*minus_one - 7*minus_two +2*minus_three)
        + inv_omega(1) * (2*zeroth + 5*minus_one - minus_two)
        + inv_omega(2) * (-plus_one + 5*zeroth + 2*minus_one)
        + inv_omega(3) * (2*zeroth + 5*plus_one - plus_two)
    )

    return wL, wR


def run(grid, sim_variables, axis):
    multidimensional, magnetic, ds = sim_variables.multidimensional, sim_variables.magnetic, sim_variables.ds
    data = {}

    Riemann_solver = solvers.get_Riemann_solver(sim_variables)

    # WENO-CU6 reconstruction [Hu et al., 2010]
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
        'jacobian': gutils.slice_(jacobian, axis, *[1,-1]),
    })

    # Compute the orthogonal L/R Riemann states and fluxes at higher-order accuracy
    if multidimensional:
        # Calculate the interface-centred fluxes
        intf_fluxes_cntrd = Riemann_solver(axis, sim_variables, **{
            'prim_interfaces': gutils.approx_face_avg((prim_plus, prim_minus), sim_variables, axis),
            'cons_interfaces': gutils.approx_face_avg((cons_plus, cons_minus), sim_variables, axis),
            'flux_interfaces': gutils.approx_face_avg((flux_plus, flux_minus), sim_variables, axis),
            'characteristics': characteristics,
            'jacobian': gutils.slice_(jacobian, axis, *[1,-1]),
        })

        # Compute the higher-order fluxes
        intf_fluxes_cntrd = gutils.approx_flux_avg(intf_fluxes_cntrd, intf_fluxes_avgd, sim_variables, axis)
    else:
        # Orthogonal Laplacian in 1d is zero
        intf_fluxes_cntrd = intf_fluxes_avgd

    # Compute flux difference for hydrodynamic components
    data['fluxes'] = np.diff(intf_fluxes_cntrd, axis=axis)/ds[axis]

    return data