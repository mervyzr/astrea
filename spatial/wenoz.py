import numpy as np

from functions import grid as gutils
from functions import math as mfuncs
from functions import numeric
from numkit import c_transport as ct
from numkit import solvers

##############################################################################
# WENO-Z reconstruction method [Borges et al., 2008]
##############################################################################

def reconstruct(grid, sim_variables, axis, power=1):
    eps = 1e-40

    # Define frequently used terms
    padded_grid_2 = gutils.add_boundary(grid, sim_variables, stencil=2, axis=axis)
    padded_grid = gutils.slice_(padded_grid_2, axis, *[1,-1])

    # Read-only alias, not a copy. The caller's grid is shared by the concurrent axis
    # sweeps and this build runs without the GIL, so it must never be mutated here
    zeroth = grid
    minus_one, minus_two = gutils.slice_(padded_grid, axis, end=-2), gutils.slice_(padded_grid_2, axis, end=-4)
    plus_one, plus_two = gutils.slice_(padded_grid, axis, start=2), gutils.slice_(padded_grid_2, axis, start=4)

    """WENO-Z reconstruction from cell averages to face averages (both sides) [Borges et al., 2008]
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
    b_k = b0, b1, b2

    # Define the linear weights
    g_k = 1/10, 3/5, 3/10
    inv_g_k = g_k[::-1]

    # Compute the alpha values
    # tau and the smoothness factor are independent of the linear weights, so they are shared
    # by the forward and reversed sets. Previously these were a lambda called eight times, each
    # call re-evaluating the whole alpha triple: 24 evaluations of np.abs(b0-b2) instead of 1
    tau = np.abs(b0-b2)
    factor_k = tuple(1 + (tau/(b + eps))**power for b in b_k)
    alpha_k = tuple(g_k[k] * factor_k[k] for k in range(3))
    inv_alpha_k = tuple(inv_g_k[k] * factor_k[k] for k in range(3))

    # Compute the non-linear weights
    alpha_sum = alpha_k[0] + alpha_k[1] + alpha_k[2]
    inv_alpha_sum = inv_alpha_k[0] + inv_alpha_k[1] + inv_alpha_k[2]
    omega = tuple(mfuncs.divide(a, alpha_sum) for a in alpha_k)
    inv_omega = tuple(mfuncs.divide(a, inv_alpha_sum) for a in inv_alpha_k)

    # Define the stencils
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
    multidimensional, magnetic, ds = sim_variables.multidimensional, sim_variables.magnetic, sim_variables.ds[axis]

    Riemann_solver = solvers.get_Riemann_solver(sim_variables)
    needs_jacobian = solvers.needs_jacobian(sim_variables)

    # WENO-Z reconstruction [Borges et al., 2008]
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
    wavespeeds = numeric.compute_wavespeed_bounds(padded_intf_avg, sim_variables, axis=axis)
    jacobian = numeric.compute_jacobian(padded_intf_avg, sim_variables, axis=axis) if needs_jacobian else None

    # Calculate the interface-averaged fluxes
    intf_fluxes_avgd = intf_fluxes_cntrd = Riemann_solver(axis, sim_variables, **{
        'prim_interfaces': (prim_plus, prim_minus),
        'cons_interfaces': (cons_plus, cons_minus),
        'flux_interfaces': (flux_plus, flux_minus),
        'wavespeeds': wavespeeds,
        'jacobian': gutils.slice_(jacobian, axis, *[1,-1]) if needs_jacobian else None,
    })

    # Compute flux difference for hydrodynamic components
    fluxes = np.diff(intf_fluxes_cntrd, axis=axis)/ds

    if magnetic and multidimensional:
        return fluxes, wavespeeds, (gutils.slice_(prim_plus, axis, start=1), gutils.slice_(prim_minus, axis, end=-1))
    else:
        return fluxes, wavespeeds, None