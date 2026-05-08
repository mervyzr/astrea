import numpy as np

from functions import grid as gutils
from functions import math as mfuncs
from functions import numeric
from numkit import c_transport as ct
from numkit import solvers

##############################################################################
# TENO reconstruction method [Fu et al., 2016; Fu, 2021]
##############################################################################

# Very sensitive to higher resolutions
def reconstruct(grid, sim_variables, axis, q=6, C_T=1e-7, adaptive=False):
    eps = 1e-40

    # Note the arrangements vs. WENO: i, i+1, i-1, i+2, i-2, i+3, i-3, ...
    linear_weights = {
        3: (1,),
        4: (3/6, 3/6),
        5: (6/10, 3/10, 1/10),
        6: (9/20, 6/20, 1/20, 4/20),
        7: (18/35, 9/35, 3/35, 4/35, 1/35),
        8: (30/70, 18/70, 4/70, 12/70, 1/70, 5/70),
    }

    # Define frequently used terms
    padded_grid_2 = gutils.add_boundary(grid, sim_variables, stencil=2, axis=axis)
    padded_grid = gutils.slice_(padded_grid_2, axis, *[1,-1])

    zeroth = np.copy(grid)
    minus_one, minus_two = gutils.slice_(padded_grid, axis, end=-2), gutils.slice_(padded_grid_2, axis, end=-4)
    plus_one, plus_two = gutils.slice_(padded_grid, axis, start=2), gutils.slice_(padded_grid_2, axis, start=4)

    if adaptive:
        padded_grid_3 = gutils.add_boundary(grid, sim_variables, stencil=3, axis=axis)
        _, plus_three = gutils.slice_(padded_grid_3, axis, end=-6), gutils.slice_(padded_grid_3, axis, start=6)

    """TENO reconstruction from cell averages to face averages (both sides)
    |                        w(i-1/2)                    w(i+1/2)                       |
    |<--         i-1         -->|<--          i          -->|<--         i+1         -->|
    |   w_L(i-1)     w_R(i-1)   |   w_L(i)         w_R(i)   |   w_L(i+1)     w_R(i+1)   |
    |   w+(i-3/2)   w-(i-1/2)   |   w+(i-1/2)   w-(i+1/2)   |   w+(i+1/2)   w-(i+3/2)   |
    """
    # Determine the smoothness indicators: i, i+1, i-1
    b0 = (
        13/12 * (minus_one - 2*zeroth + plus_one)**2
        + 1/4 * (minus_one - plus_one)**2
    )
    b1 = (
        13/12 * (zeroth - 2*plus_one + plus_two)**2
        + 1/4 * (3*zeroth - 4*plus_one + plus_two)**2
    )
    b2 = (
        13/12 * (minus_two - 2*minus_one + zeroth)**2
        + 1/4 * (minus_two - 4*minus_one + 3*zeroth)**2
    )
    b_k = b0, b1, b2

    # Define the linear weights
    g_k = linear_weights[5]
    inv_g_k = g_k[::-1]

    # Compute the scale separation
    if adaptive:
        gamma_k = lambda k: (1/(b_k[k]+eps))**7
    else:
        gamma_k = lambda k: (1 + np.abs(b2 - b1)/(b_k[k]+eps))**q

    # Compute the adaptive cutoff threshold [Fu, 2021]
    if adaptive:
        C_r, alpha_1, alpha_2 = .265, 14, 6.4
        eta_j = lambda stencil: (
            (np.abs(
                2 * (stencil[2]-stencil[1]) * (stencil[1]-stencil[0])
                ) + eps
            )/(
                (stencil[2]-stencil[1])**2 + (stencil[1]-stencil[0])**2
                + eps
            )
        )
        eta = np.minimum(
            np.minimum(
                eta_j((minus_two, minus_two, zeroth)),
                eta_j((minus_one, zeroth, plus_one))
            ),
            np.minimum(
                eta_j((zeroth, plus_one, plus_two)),
                eta_j((plus_one, plus_two, plus_three))
            )
        )
        m = 1 - np.minimum(1, eta/C_r)
        g_m = (1 + 4*m) * (1 - m)**4
        beta = alpha_1 - alpha_2*(1-g_m)
        C_T = 10**-np.floor(beta)

    # Compute the smoothness measure with the sharp cutoff function
    delta_k = lambda gk, k: np.where(mfuncs.divide(gamma_k(k), gamma_k(0) + gamma_k(1) + gamma_k(2)) < C_T, 0., gk[k])

    # Compute the non-linear weights
    omega = lambda k: mfuncs.divide(delta_k(g_k, k), delta_k(g_k, 0) + delta_k(g_k, 1) + delta_k(g_k, 2))
    inv_omega = lambda k: mfuncs.divide(delta_k(inv_g_k, k), delta_k(inv_g_k, 0) + delta_k(inv_g_k, 1) + delta_k(inv_g_k, 2))

    # Define the stencils
    wR = 1/6 * (
        omega(0) * (-minus_one + 5*zeroth + 2*plus_one)
        + omega(1) * (2*zeroth + 5*plus_one - plus_two)
        + omega(2) * (2*minus_two - 7*minus_one + 11*zeroth)
    )
    wL = 1/6 * (
        inv_omega(0) * (-plus_one + 5*zeroth + 2*minus_one)
        + inv_omega(1) * (2*plus_two - 7*plus_one + 11*zeroth)
        + inv_omega(2) * (2*zeroth + 5*minus_one - minus_two)
    )

    return wL, wR


def run(grid, sim_variables, axis):
    subgrid, multidimensional, magnetic, ds = sim_variables.subgrid, sim_variables.multidimensional, sim_variables.magnetic, sim_variables.ds
    data = {}

    Riemann_solver = solvers.get_Riemann_solver(sim_variables)

    # TENO reconstruction [Fu et al., 2016]
    if subgrid.endswith("a"):
        wL, wR = reconstruct(grid, sim_variables, axis, adaptive=True)
    else:
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