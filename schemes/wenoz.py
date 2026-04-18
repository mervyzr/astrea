import numpy as np

from functions import constructor, fv
from num_methods import ct, solvers

##############################################################################
# WENO-Z reconstruction method [Borges et al., 2008]
##############################################################################

def reconstruct(grid, sim_variables, axis):
    eps = 1e-40

    # Define frequently used terms
    padded_grid_2 = fv.add_boundary(grid, sim_variables, stencil=2, axis=axis)
    padded_grid = fv.slice_(padded_grid_2, axis, *[1,-1])

    zeroth = np.copy(grid)
    minus_one, minus_two = fv.slice_(padded_grid, axis, end=-2), fv.slice_(padded_grid_2, axis, end=-4)
    plus_one, plus_two = fv.slice_(padded_grid, axis, start=2), fv.slice_(padded_grid_2, axis, start=4)

    """WENO-Z reconstruction from cell averages to face averages (both sides) [Borges et al., 2008]
    |                        w(i-1/2)                    w(i+1/2)                       |
    |<--         i-1         -->|<--          i          -->|<--         i+1         -->|
    |   w_L(i-1)     w_R(i-1)   |   w_L(i)         w_R(i)   |   w_L(i+1)     w_R(i+1)   |
    |   w+(i-3/2)   w-(i-1/2)   |   w+(i-1/2)   w-(i+1/2)   |   w+(i+1/2)   w-(i+3/2)   |
    """
    # Define the linear weights
    g0, g1, g2 = 1/16, 5/8, 5/16

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

    # Define the non-linear weights
    a0 = lambda d0: d0 * (1 + np.abs(b0-b2)/(b0 + eps))
    a1 = lambda d1: d1 * (1 + np.abs(b0-b2)/(b1 + eps))
    a2 = lambda d2: d2 * (1 + np.abs(b0-b2)/(b2 + eps))

    # Define the stencils
    wR = .125 * (
        (a0(g0)/(a0(g0)+a1(g1)+a2(g2))) * (3*minus_two - 10*minus_one + 15*zeroth)
        + (a1(g1)/(a0(g0)+a1(g1)+a2(g2))) * (-minus_one + 6*zeroth + 3*plus_one)
        + (a2(g2)/(a0(g0)+a1(g1)+a2(g2))) * (3*zeroth + 6*plus_one - plus_two)
    )
    wL = .125 * (
        (a0(g2)/(a0(g2)+a1(g1)+a2(g0))) * (3*zeroth + 6*minus_one - minus_two)
        + (a1(g1)/(a0(g2)+a1(g1)+a2(g0))) * (-plus_one + 6*zeroth + 3*minus_one)
        + (a2(g0)/(a0(g2)+a1(g1)+a2(g0))) * (3*plus_two - 10*plus_one + 15*zeroth)
    )

    return wL, wR


def run(grid, sim_variables, axis):
    multidimensional, magnetic, ds = sim_variables.multidimensional, sim_variables.magnetic, sim_variables.ds
    data = {}

    Riemann_solver = solvers.get_Riemann_solver(sim_variables)

    # WENO-Z reconstruction [Borges et al., 2008]
    wL, wR = reconstruct(grid, sim_variables, axis)

    # Re-align the interfaces so that cell wall is in between interfaces
    assign_interfaces = ct.assign_interfaces if magnetic else fv.assign_interfaces
    prim_plus, prim_minus = assign_interfaces((wL, wR), grid, sim_variables, axis)

    # Get the average solution between the interfaces at the boundaries
    intf_avg = fv.compute_Roe_average((prim_plus, prim_minus), sim_variables)
    padded_intf_avg = fv.slice_(fv.add_boundary(intf_avg, sim_variables, axis=axis), axis, end=-1)

    # Convert the primitive variables at the interface
    cons_plus, cons_minus = fv.variable_convert_intf("primitive", prim_plus, sim_variables, axis=axis), fv.variable_convert_intf("primitive", prim_minus, sim_variables, axis=axis)

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
            characteristics = np.full_like(padded_intf_avg, .01)

    # Compute eigmax for time stepping limits
    data['eigmax'] = ds[axis]/fv.compute_eigmax(characteristics, axis=axis)

    # Compute alphas and save the reconstructed interfaces for CT computation
    if magnetic and multidimensional:
        data['alphas'] = ct.compute_alphas(characteristics, axis=axis)
        data['interfaces'] = fv.slice_(prim_plus, axis, start=1), fv.slice_(prim_minus, axis, end=-1)

    # Calculate the interface-averaged fluxes
    intf_fluxes_avgd = intf_fluxes_cntrd = Riemann_solver(axis, sim_variables, **{
        'prim_interfaces': (prim_plus, prim_minus),
        'cons_interfaces': (cons_plus, cons_minus),
        'flux_interfaces': (flux_plus, flux_minus),
        'characteristics': characteristics,
        'jacobian': fv.slice_(jacobian, axis, start=1),
    })

    # Compute flux difference for hydrodynamic components
    data['fluxes'] = np.diff(intf_fluxes_cntrd, axis=axis)/ds[axis]

    return data