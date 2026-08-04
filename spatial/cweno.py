import numpy as np

from functions import grid as gutils
from functions import math as mfuncs
from functions import numeric
from numkit import c_transport as ct
from numkit import limiters, solvers

##############################################################################
# CWENO reconstruction method [Levy et al., 1999, 2000; Verma et al., 2018]
##############################################################################

def reconstruct(grid, sim_variables, axis, power=2):
    # Define the frequently used terms
    padded_grid_2 = gutils.add_boundary(grid, sim_variables, stencil=2, axis=axis)
    padded_grid = gutils.slice_(padded_grid_2, axis, *[1,-1])

    zeroth = np.copy(grid)
    minus_one, minus_two = gutils.slice_(padded_grid, axis, end=-2), gutils.slice_(padded_grid_2, axis, end=-4)
    plus_one, plus_two = gutils.slice_(padded_grid, axis, start=2), gutils.slice_(padded_grid_2, axis, start=4)

    # Define the empirical parameters for Eq. 3.12
    eps = sim_variables.eps

    """CWENO reconstruction from cell averages to face averages (both sides) [Verma et al., 2018]
    |                        w(i-1/2)                    w(i+1/2)                       |
    |<--         i-1         -->|<--          i          -->|<--         i+1         -->|
    |   w_L(i-1)     w_R(i-1)   |   w_L(i)         w_R(i)   |   w_L(i+1)     w_R(i+1)   |
    |   w+(i-3/2)   w-(i-1/2)   |   w+(i-1/2)   w-(i+1/2)   |   w+(i+1/2)   w-(i+3/2)   |
    """
    # Define the linear weights C_k (5th-order & 4th-order accurate) [tbl. 3.1]
    C_k = 3/16, 5/8, 3/16
    dC_k = 1/6, 2/3, 1/6

    # Determine the smoothness indicators (O(dx^4) at critical points but O(1) at discontinuities) [eq. 3.14]
    SI_minus = (
        13/12 * (minus_two - 2*minus_one + zeroth)**2
        + 1/4 * (minus_two - 4*minus_one + 3*zeroth)**2
    )
    SI_zero = (
        13/12 * (minus_one - 2*zeroth + plus_one)**2
        + 1/4 * (minus_one - plus_one)**2
    )
    SI_plus = (
        13/12 * (zeroth - 2*plus_one + plus_two)**2
        + 1/4 * (3*zeroth - 4*plus_one + plus_two)**2
    )
    SI_k = SI_minus, SI_zero, SI_plus

    # Compute the alpha values [Levy et al., 1999, eq. 3.12]
    alpha = lambda k: dC_k[k]/(SI_k[k] + eps)**power

    # Compute the non-linear weights [Levy et al., 1999, eq. 3.11]
    omega = lambda k: mfuncs.divide(alpha(k), alpha(0)+alpha(1)+alpha(2))

    # Define the stencils (no need to flip linear weights in non-linear weights since C_k and dC_k are symmetrical)
    wR = 1/6 * (
        omega(0) * (2*minus_two - 7*minus_one + 11*zeroth)
        + omega(1) * (-minus_one + 5*zeroth + 2*plus_one)
        + omega(2) * (2*zeroth + 5*plus_one - plus_two)
    )
    wL = 1/6 * (
        omega(0) * (2*zeroth + 5*minus_one - minus_two)
        + omega(1) * (-plus_one + 5*zeroth + 2*minus_one)
        + omega(2) * (2*plus_two - 7*plus_one + 11*zeroth)
    )

    # Apply positivity limiter to densities
    wR[...,sim_variables.rho] = limiters.zs2010(grid, wR)[...,sim_variables.rho]
    wL[...,sim_variables.rho] = limiters.zs2010(grid, wL)[...,sim_variables.rho]

    return wL, wR


def run(grid, sim_variables, axis):
    multidimensional, magnetic, ds = sim_variables.multidimensional, sim_variables.magnetic, sim_variables.ds[axis]

    Riemann_solver = solvers.get_Riemann_solver(sim_variables)

    # CWENO reconstruction [Levy et al., 1999; Verma et al., 2018]
    wL, wR = reconstruct(grid, sim_variables, axis=axis)

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
    fluxes = np.diff(intf_fluxes_cntrd, axis=axis)/ds

    if magnetic and multidimensional:
        return fluxes, characteristics, (gutils.slice_(prim_plus, axis, start=1), gutils.slice_(prim_minus, axis, end=-1))
    else:
        return fluxes, characteristics, None


# [Levy et al., 1999]
def compute_cweno_interpolant(grid, sim_variables, axis, pos=.5):
    h = sim_variables.ds[axis]

    # Compute the reconstructed point-values with their derivatives (note that there are 9 equations) [eq. 3.8]
    def _reconstruct(order, stencil, cells):
        stencils = np.roll(cells, -stencil)[1:-1]
        if 'zeroth' in order or order in [0, '']:
            return stencils[1] - (stencils[0] - 2*stencils[1] + stencils[2])/24
        elif 'first' in order or order in [1, 'prime', 'p']:
            return (stencils[2] - stencils[0])/(2 * h)
        elif 'second' in order or order in [2, 'primeprime', 'pp']:
            return (stencils[2] - 2*stencils[1] + stencils[0])/h**2

    # Define the frequently used terms
    padded_grid_2 = gutils.add_boundary(grid, sim_variables, stencil=2, axis=axis)
    padded_grid = gutils.slice_(padded_grid_2, axis, *[1,-1])

    zeroth = np.copy(grid)
    minus_one, minus_two = gutils.slice_(padded_grid, axis, end=-2), gutils.slice_(padded_grid_2, axis, end=-4)
    plus_one, plus_two = gutils.slice_(padded_grid, axis, start=2), gutils.slice_(padded_grid_2, axis, start=4)

    # Define the empirical parameters for Eq. 3.12
    eps, power = np.finfo(float).eps, 2

    # Define the linear weights C_k (5th-order & 4th-order accurate) [tbl. 3.1]
    C_minus, C_zero, C_plus = 3/16, 5/8, 3/16
    dC_minus, dC_zero, dC_plus = 1/6, 2/3, 1/6

    # Determine the smoothness indicators (O(dx^4) at critical points but O(1) at discontinuities) [eq. 3.14]
    IS_minus = lambda stencils: 13/12 * (stencils[0] - 2*stencils[1] + stencils[2])**2 + 1/4 * (stencils[0] - 4*stencils[1] + 3*stencils[2])**2
    IS_zero = lambda stencils: 13/12 * (stencils[0] - 2*stencils[1] + stencils[2])**2 + 1/4 * (stencils[0] - stencils[2])**2
    IS_plus = lambda stencils: 13/12 * (stencils[0] - 2*stencils[1] + stencils[2])**2 + 1/4 * (3*stencils[0] - 4*stencils[1] + stencils[2])**2

    # Compute the alpha values [eq. 3.12]
    alpha = lambda C_k, IS_k: C_k/(eps + IS_k)**power

    # Compute the non-linear weights [eq. 3.11]
    denominator = (
        alpha(C_minus, IS_minus([minus_two, minus_one, zeroth]))
        + alpha(C_zero, IS_zero([minus_one, zeroth, plus_one]))
        + alpha(C_plus, IS_plus([zeroth, plus_one, plus_two]))
    )
    wj_minus = mfuncs.divide(alpha(C_minus, IS_minus([minus_two, minus_one, zeroth])), denominator)
    wj_zero = mfuncs.divide(alpha(C_zero, IS_zero([minus_one, zeroth, plus_one])), denominator)
    wj_plus = mfuncs.divide(alpha(C_plus, IS_plus([zeroth, plus_one, plus_two])), denominator)

    # Compute the coefficients in the parabolic interpolant R_j(x) [eq. 3.10]
    u_tilde = lambda _order, _stencil: _reconstruct(_order, _stencil, cells=[minus_two, minus_one, zeroth, plus_one, plus_two])
    uj_zeroth = (
        wj_minus * (u_tilde('', -1) + h*u_tilde('prime', -1) + .5*u_tilde('primeprime', -1)*h**2)
        + wj_zero * u_tilde('', 0)
        + wj_plus * (u_tilde('', +1) - h*u_tilde('prime', +1) + .5*u_tilde('primeprime', +1)*h**2)
    )
    uj_first = (
        wj_minus * (u_tilde('prime', -1) + h*u_tilde('primeprime', -1))
        + wj_zero * u_tilde('prime', 0)
        + wj_plus * (u_tilde('prime', +1) - h*u_tilde('primeprime', +1))
    )
    uj_second = (
        wj_minus * u_tilde('primeprime', -1)
        + wj_zero * u_tilde('primeprime', 0)
        + wj_plus * u_tilde('primeprime', +1)
    )

    # Compute the parabolic interpolant at the interfaces R_j(x+1/2) [eq. 3.9]
    Rj = uj_zeroth + uj_first*pos*h + .5*uj_second*(pos*h)**2

    """# Compute the fluxes (NO NEED FOR RIEMANN SOLVERS)
    flux = numeric.compute_flux(Rj, sim_variables, axis=axis)
    padded_flux_2 = fv.add_boundary(flux, sim_variables, stencil=2, axis=axis)
    padded_flux = fv.slice_(padded_flux_2, axis, *[1,-1])

    fz = np.copy(flux)
    fm1, fm2 = fv.slice_(padded_flux, axis, end=-2), fv.slice_(padded_flux_2, axis, end=-4)
    fp1, fp2 = fv.slice_(padded_flux, axis, start=2), fv.slice_(padded_flux_2, axis, start=4)

    # Compute the non-linear weights for fluxes
    denominator = (
        alpha(dC_minus, IS_minus([fm2, fm1, fz]))
        + alpha(dC_zero, IS_zero([fm1, fz, fp1]))
        + alpha(dC_plus, IS_plus([fz, fp1, fp2]))
    )
    wj_minus = fv.divide(alpha(dC_minus, IS_minus([fm2, fm1, fz])), denominator)
    wj_zero = fv.divide(alpha(dC_zero, IS_zero([fm1, fz, fp1])), denominator)
    wj_plus = fv.divide(alpha(dC_plus, IS_plus([fz, fp1, fp2])), denominator)

    # Compute the intermediate flux values [eq. 3.17]
    f_tilde = lambda _order, _stencil: reconstruct(_order, _stencil, cells=[fm2, fm1, fz, fp1, fp2])
    flux_first = (
        wj_minus * (f_tilde('prime', -1) + h*f_tilde('primeprime', -1))
        + wj_zero * f_tilde('prime', 0)
        + wj_plus * (f_tilde('prime', +1) - h*f_tilde('primeprime', +1))
    )"""

    return Rj