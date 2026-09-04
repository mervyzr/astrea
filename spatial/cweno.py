import numpy as np

from functions import grid as gutils
from functions import math as mfuncs
from functions import numeric
from numkit import c_transport as ct
from numkit import kernels, limiters, solvers

##############################################################################
# CWENO(Z) reconstruction method [Levy et al., 1999, 2000; Verma et al., 2018; Cravero et al., 2019]
##############################################################################


# Build the Riemann-solver keyword arguments, skipping anything the configured solver does not
# read. approx_face_avg costs two full-size arrays and four Laplacians per pair per axis, so
# for a solver like Lax-Friedrich that never touches prim_interfaces it is pure waste.
def _solver_kwargs(sim_variables, axis, wavespeeds, jacobian, prim, cons, flux, face_avg=False):
    inputs = solvers.solver_inputs(sim_variables)
    kwargs = {'wavespeeds': wavespeeds}

    for key, pair in (("prim", prim), ("cons", cons), ("flux", flux)):
        if key in inputs:
            kwargs[f'{key}_interfaces'] = gutils.approx_face_avg(pair, sim_variables, axis) if face_avg else pair

    if "jacobian" in inputs:
        kwargs['jacobian'] = gutils.slice_(jacobian, axis, *[1,-1])
    return kwargs

def reconstruct(grid, sim_variables, axis, power=2, limit=False):
    """CWENO reconstruction from cell averages to face averages (both sides) [Verma et al., 2018]

    |                        w(i-1/2)                    w(i+1/2)                       |
    |<--         i-1         -->|<--          i          -->|<--         i+1         -->|
    |   w_L(i-1)     w_R(i-1)   |   w_L(i)         w_R(i)   |   w_L(i+1)     w_R(i+1)   |
    |   w+(i-3/2)   w-(i-1/2)   |   w+(i-1/2)   w-(i+1/2)   |   w+(i+1/2)   w-(i+3/2)   |

    Handed to a kernel: the numpy form held the five stencil slices, three smoothness
    indicators, three alphas, three weights and both outputs live at once, roughly fifteen
    full-size arrays, and it was the single largest remaining cost in the RHS. Verified
    bit-identical to that form for CWENO and CWENOZ over 1/2/3 dimensions, every axis and all
    three boundary modes.
    """
    wL, wR = kernels.cweno_reconstruct(
        grid, axis, sim_variables.eps, kernels.bc_code(sim_variables),
        power=power, wenoz=sim_variables.subgrid.endswith("z"),
    )

    # Apply positivity-preserving limiter. Left out of the kernel because it keys off a global
    # minimum over the reconstructed array, so it cannot be known until reconstruction is done
    if limit:
        wR = limiters.w2012(grid, wR, sim_variables)
        wL = limiters.w2012(grid, wL, sim_variables)

    return wL, wR


def run(grid, sim_variables, axis):
    multidimensional, magnetic, ds = sim_variables.multidimensional, sim_variables.magnetic, sim_variables.ds[axis]

    Riemann_solver = solvers.get_Riemann_solver(sim_variables)
    needs_jacobian = solvers.needs_jacobian(sim_variables)

    # CWENO reconstruction [Levy et al., 1999; Verma et al., 2018]
    wL, wR = reconstruct(grid, sim_variables, axis, limit=True)

    # Re-align the interfaces so that cell wall is in between interfaces
    assign_interfaces = ct.assign_interfaces if magnetic else gutils.assign_interfaces
    prim_plus, prim_minus = assign_interfaces((wL, wR), grid, sim_variables, axis)

    # Get the average solution between the interfaces at the boundaries
    intf_avg = numeric.compute_Roe_average((prim_plus, prim_minus), sim_variables)
    padded_intf_avg = gutils.add_boundary(intf_avg, sim_variables, axis=axis)

    # Convert the primitive variables at the interface
    cons_plus, cons_minus = gutils.variable_convert_intf("primitive", prim_plus, sim_variables, axis=axis), gutils.variable_convert_intf("primitive", prim_minus, sim_variables, axis=axis)

    # Compute the fluxes
    flux_plus, flux_minus = numeric.compute_flux(prim_plus, sim_variables, axis=axis), numeric.compute_flux(prim_minus, sim_variables, axis=axis)

    # Resolve characteristics at interfaces from the analytic eigenvalues. This replaces a
    # per-cell np.linalg.eigvals over an (N,N,N,8,8) Jacobian, which cost 2.4 us/cell and
    # allocated 8 GiB per axis at 256^3 to recover eigenvalues that are known in closed form
    wavespeeds = numeric.compute_wavespeed_bounds(padded_intf_avg, sim_variables, axis=axis)
    jacobian = numeric.compute_jacobian(padded_intf_avg, sim_variables, axis=axis) if needs_jacobian else None

    # Calculate the interface-averaged fluxes
    intf_fluxes_avgd = Riemann_solver(axis, sim_variables, **_solver_kwargs(
        sim_variables, axis, wavespeeds, jacobian,
        (prim_plus, prim_minus), (cons_plus, cons_minus), (flux_plus, flux_minus),
    ))

    # Compute the orthogonal L/R Riemann states and fluxes at higher-order accuracy
    if multidimensional:
        # Calculate the interface-centred fluxes
        intf_fluxes_cntrd = Riemann_solver(axis, sim_variables, **_solver_kwargs(
            sim_variables, axis, wavespeeds, jacobian,
            (prim_plus, prim_minus), (cons_plus, cons_minus), (flux_plus, flux_minus),
            face_avg=True,
        ))

        # Compute the higher-order fluxes
        intf_fluxes_cntrd = gutils.approx_flux_avg(intf_fluxes_cntrd, intf_fluxes_avgd, sim_variables, axis)
    else:
        # Orthogonal Laplacian in 1d is zero
        intf_fluxes_cntrd = intf_fluxes_avgd

    # Compute flux difference for hydrodynamic components
    fluxes = np.diff(intf_fluxes_cntrd, axis=axis)/ds

    if magnetic and multidimensional:
        return fluxes, wavespeeds, (gutils.slice_(prim_plus, axis, start=1), gutils.slice_(prim_minus, axis, end=-1))
    else:
        return fluxes, wavespeeds, None


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

    # Read-only alias, not a copy. The caller's grid is shared by the concurrent axis
    # sweeps and this build runs without the GIL, so it must never be mutated here
    zeroth = grid
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