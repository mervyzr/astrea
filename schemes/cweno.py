import concurrent.futures
from itertools import repeat

import numpy as np

from functions import constructor, fv
from num_methods import ct, solvers

##############################################################################
# CWENO reconstruction method [Levy et al., 1999, 2000]
##############################################################################

def run(grid, sim_variables, axis):
    subgrid, multidimensional, axes, magnetic, ds = sim_variables.subgrid, sim_variables.multidimensional, sim_variables.axes, sim_variables.magnetic, sim_variables.ds
    convert, data = sim_variables.convert, {}

    Riemann_solver = solvers.get_Riemann_solver(sim_variables)
    ortho_axes = axes[axes != axis] if (magnetic or multidimensional) else 0

    # Compute the reconstructed point-values with their derivatives (note that there are 9 equations) [eq. 3.8]
    def reconstruct(order, stencil, cells):
        stencils = np.roll(cells, -stencil)[1:-1]
        if 'zeroth' in order or order in [0, '']:
            return stencils[1] - (stencils[0] - 2*stencils[1] + stencils[2])/24
        elif 'first' in order or order in [1, 'prime', 'p']:
            return (stencils[2] - stencils[0])/(2 * ds[axis])
        elif 'second' in order or order in [2, 'primeprime', 'pp']:
            return (stencils[2] - 2*stencils[1] + stencils[0])/ds[axis]**2

    # Define the frequently used terms
    padded_grid_2 = fv.add_boundary(grid, sim_variables, stencil=2, axis=axis)
    padded_grid = fv.slice_(padded_grid_2, axis, *[1,-1])

    zeroth = np.copy(grid)
    minus_one, minus_two = fv.slice_(padded_grid, axis, end=-2), fv.slice_(padded_grid_2, axis, end=-4)
    plus_one, plus_two = fv.slice_(padded_grid, axis, start=2), fv.slice_(padded_grid_2, axis, start=4)

    # Define the empirical parameters for Eq. 3.12
    eps, power = np.finfo(sim_variables.precision).eps, 2

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
    wj_minus = fv.divide(alpha(C_minus, IS_minus([minus_two, minus_one, zeroth])), denominator)
    wj_zero = fv.divide(alpha(C_zero, IS_zero([minus_one, zeroth, plus_one])), denominator)
    wj_plus = fv.divide(alpha(C_plus, IS_plus([zeroth, plus_one, plus_two])), denominator)


    ### EVERYTHING BELOW IS FOLLOWING THE PAPER BY VERMA 2018 ###
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
    if magnetic:
        wR[...,5+axes] = grid[...,5+axes]

        # Magnetic transverse interfaces reconstructed along orthogonal axis/axes (interface = centre for PCM)
        if multidimensional:
            data['ortho_interfaces'] = ct.reconstruct_transverse(grid, sim_variables, axis=axis)

    # Re-align the interfaces so that cell wall is in between interfaces
    prim_plus, prim_minus = fv.slice_(fv.add_boundary(wL, sim_variables, axis=axis), axis, start=1), fv.slice_(fv.add_boundary(wR, sim_variables, axis=axis), axis, end=-1)
    if magnetic:
        padded_grid = fv.add_boundary(grid, sim_variables, axis=axis)
        prim_plus[...,5+axes] = prim_minus[...,5+axes] = fv.slice_(padded_grid, axis, end=-1)[...,5+axes]

    # Get the average solution between the interfaces at the boundaries
    intf_avg = fv.compute_Roe_average([prim_plus,prim_minus], sim_variables)
    padded_intf_avg = fv.add_boundary(fv.slice_(intf_avg, axis, start=1), sim_variables, axis=axis)

    # Convert the primitive variables
    cons_plus, cons_minus = convert("primitive", prim_plus, sim_variables, axis=axis, pos='intf'), convert("primitive", prim_minus, sim_variables, axis=axis, pos='intf')

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



    ### EVERYTHING BELOW IS FOLLOWING THE PAPER BY LEVY 1999 ###

    """# Compute the coefficients in the parabolic interpolant R_j(x) [eq. 3.10]
    u_tilde = lambda _order, _stencil: reconstruct(_order, _stencil, cells=[minus_two, minus_one, zeroth, plus_one, plus_two])
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

    # CWENO reconstruction [Levy et al., 1999, 2000]
    # |               w(i-1/2)            w(i+1/2)                |
    # |  i-1           -->|   i            -->|  i+1           -->|
    # |        w_R(i-1)   |          w_R(i)   |        w_R(i+1)   |

    # Compute the parabolic interpolant at the interfaces R_j(x+1/2) [eq. 3.9]
    Rj = uj_zeroth + uj_first*h/2 + .5*uj_second*(h/2)**2
    if magnetic:
        Rj[...,5+axes] = grid[...,5+axes]

        # Magnetic transverse interfaces reconstructed longitudinal to the axis (returns [ prim_plus, prim_minus ]); will be used for orthogonal axes later
        if multidimensional:
            data['ortho_interfaces'] = ct.reconstruct_transverse(grid, sim_variables, axis=axis)


    # Compute the fluxes (NO NEED FOR RIEMANN SOLVERS)
    flux = constructor.make_flux(Rj, sim_variables, axis=axis)
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

















    """WENO reconstruction [Shu, 2009; San & Kara, 2015]
    |                        w(i-1/2)                    w(i+1/2)                       |
    |<--         i-1         -->|<--          i          -->|<--         i+1         -->|
    |   w_L(i-1)     w_R(i-1)   |   w_L(i)         w_R(i)   |   w_L(i+1)     w_R(i+1)   |
    |   w+(i-3/2)   w-(i-1/2)   |   w+(i-1/2)   w-(i+1/2)   |   w+(i+1/2)   w-(i+3/2)   |
    """
    def _reconstruct(_grid, _sim_variables, _axis, _order=5):
        eps = 1e-6

        # Define frequently used terms    
        padded_grid_3 = fv.add_boundary(_grid, _sim_variables, stencil=3, axis=_axis)
        padded_grid_2 = fv.slice_(padded_grid_3, _axis, *[1,-1])
        padded_grid = fv.slice_(padded_grid_2, _axis, *[1,-1])

        zeroth = np.copy(_grid)
        minus_one, minus_two, minus_three = fv.slice_(padded_grid, _axis, end=-2), fv.slice_(padded_grid_2, _axis, end=-4), fv.slice_(padded_grid_3, _axis, end=-6)
        plus_one, plus_two, plus_three = fv.slice_(padded_grid, _axis, start=2), fv.slice_(padded_grid_2, _axis, start=4), fv.slice_(padded_grid_3, _axis, start=6)

        if _order == 3:
            # Define the linear weights
            g0, g1 = 1/3, 2/3

            # Determine the smoothness indicators
            b0 = (zeroth - minus_one)**2
            b1 = (plus_one - zeroth)**2

            # Define the non-linear weights
            a0 = lambda d0: d0/(b0 + eps)**2
            a1 = lambda d1: d1/(b1 + eps)**2

            # Define the stencils
            wR = (a0(g0)/(a0(g0) + a1(g1)))*(1.5*zeroth - .5*minus_one) + (a1(g1)/(a0(g0) + a1(g1)))*(.5*zeroth + .5*plus_one)
            wL = (a1(g0)/(a0(g1) + a1(g0)))*(1.5*zeroth - .5*plus_one) + (a0(g1)/(a0(g1) + a1(g0)))*(.5*zeroth + .5*minus_one)

        elif _order == 7:
            # Define the linear weights
            g0, g1, g2, g3 = 1/35, 12/35, 18/35, 4/35

            # Determine the smoothness indicators
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
                + plus_two * (267* plus_two)
            )
            b3 = (
                zeroth * (2107*zeroth - 9402*plus_one + 7042*plus_two - 1854*plus_three)
                + plus_one * (11003*plus_one - 17246*plus_two + 4642*plus_three)
                + plus_two * (7043*plus_two - 3882*plus_three)
                + plus_three * (547*plus_three)
            )

            # Define the non-linear weights
            a0 = lambda d0: d0/(b0 + eps)**2
            a1 = lambda d1: d1/(b1 + eps)**2
            a2 = lambda d2: d2/(b2 + eps)**2
            a3 = lambda d3: d3/(b3 + eps)**2

            # Define the stencils
            wR = (
                (a0(g0)/(a0(g0)+a1(g1)+a2(g2)+a3(g3))) * (-1/4*minus_three + 13/12*minus_two - 23/12*minus_one + 25/12*zeroth)
                + (a1(g1)/(a0(g0)+a1(g1)+a2(g2)+a3(g3))) * (1/12*minus_two - 5/12*minus_one + 13/12*zeroth + 1/4*plus_one)
                + (a2(g2)/(a0(g0)+a1(g1)+a2(g2)+a3(g3))) * (-1/12*minus_one + 7/12*zeroth + 7/12*plus_one - 1/12*plus_two)
                + (a3(g3)/(a0(g0)+a1(g1)+a2(g2)+a3(g3))) * (1/4*zeroth + 13/12*plus_one - 5/12*plus_two + 1/12*plus_three)
            )
            wL = (
                (a0(g3)/(a0(g3)+a1(g2)+a2(g1)+a3(g0))) * (1/4*zeroth + 13/12*minus_one - 5/12*minus_two + 1/12*minus_three)
                + (a1(g2)/(a0(g3)+a1(g2)+a2(g1)+a3(g0))) * (-1/12*plus_one + 7/12*zeroth + 7/12*minus_one - 1/12*minus_two)
                + (a2(g1)/(a0(g3)+a1(g2)+a2(g1)+a3(g0))) * (1/12*plus_two - 5/12*plus_one + 13/12*zeroth + 1/4*minus_one)
                + (a3(g0)/(a0(g3)+a1(g2)+a2(g1)+a3(g0))) * (-1/4*plus_three + 13/12*plus_two - 23/12*plus_one + 25/12*zeroth)
            )

        else:
            # Define the linear weights
            g0, g1, g2 = 1/10, 3/5, 3/10

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
            a0 = lambda d0: d0/(b0 + eps)**2
            a1 = lambda d1: d1/(b1 + eps)**2
            a2 = lambda d2: d2/(b2 + eps)**2

            # Define the stencils
            wR = (
                (a0(g0)/(a0(g0)+a1(g1)+a2(g2))) * (1/3*minus_two - 7/6*minus_one + 11/6*zeroth)
                + (a1(g1)/(a0(g0)+a1(g1)+a2(g2))) * (-1/6*minus_one + 5/6*zeroth + 1/3*plus_one)
                + (a2(g2)/(a0(g0)+a1(g1)+a2(g2))) * (1/3*zeroth + 5/6*plus_one - 1/6*plus_two)
            )
            wL = (
                (a0(g2)/(a0(g2)+a1(g1)+a2(g0))) * (1/3*zeroth + 5/6*minus_one - 1/6*minus_two)
                + (a1(g1)/(a0(g2)+a1(g1)+a2(g0))) * (-1/6*plus_one + 5/6*zeroth + 1/3*minus_one)
                + (a2(g0)/(a0(g2)+a1(g1)+a2(g0))) * (1/3*plus_two - 7/6*plus_one + 11/6*zeroth)
            )

        return wL, wR


    # Reconstruct the interface states
    if len(subgrid.split("weno")) == 2:
        try:
            wL, wR = reconstruct(grid, sim_variables, axis, int(subgrid.replace('-','').split("weno")[-1]))
        except Exception as e:
            wL, wR = reconstruct(grid, sim_variables, axis)
    else:
        wL, wR = reconstruct(grid, sim_variables, axis)
    if magnetic:
        wR[...,5+axes] = grid[...,5+axes]

        # Magnetic transverse interfaces reconstructed longitudinal to the axis (returns [ prim_plus, prim_minus ]); will be used for orthogonal axes later
        if multidimensional:
            data['ortho_interfaces'] = ct.reconstruct_transverse(grid, sim_variables, axis=axis)

    # Re-align the interfaces so that cell wall is in between interfaces
    prim_plus, prim_minus = fv.slice_(fv.add_boundary(wL, sim_variables, axis=axis), axis, start=1), fv.slice_(fv.add_boundary(wR, sim_variables, axis=axis), axis, end=-1)
    if magnetic:
        padded_grid = fv.add_boundary(grid, sim_variables, axis=axis)
        prim_plus[...,5+axes] = prim_minus[...,5+axes] = fv.slice_(padded_grid, axis, end=-1)[...,5+axes]

    # Get the average solution between the interfaces at the boundaries
    intf_avg = fv.compute_Roe_average([prim_plus,prim_minus], sim_variables)
    padded_intf_avg = fv.add_boundary(fv.slice_(intf_avg, axis, start=1), sim_variables, axis=axis)

    # Convert the primitive variables
    cons_plus, cons_minus = sim_variables.convert("primitive", prim_plus, sim_variables, axis=axis, pos='intf'), sim_variables.convert("primitive", prim_minus, sim_variables, axis=axis, pos='intf')

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
            'prim_interfaces': fv.approx_face_avg(ortho_axes, sim_variables, *[prim_plus, prim_minus]),
            'cons_interfaces': fv.approx_face_avg(ortho_axes, sim_variables, *[cons_plus, cons_minus]),
            'flux_interfaces': fv.approx_face_avg(ortho_axes, sim_variables, *[flux_plus, flux_minus]),
            'characteristics': characteristics,
        })

        # Compute the 4th-order interface-centred fluxes from the interface-averaged fluxes via higher order approximation for each orthogonal axis
        with concurrent.futures.ThreadPoolExecutor() as inner_executor:
            jobs = inner_executor.map(fv.laplacian, repeat(intf_fluxes_avgd), repeat(sim_variables), ortho_axes)
            for job_idx, ortho_axis in enumerate(ortho_axes):
                intf_fluxes_cntrd -= (sim_variables.ds[ortho_axis]**2)/24 * jobs[job_idx]
    else:
        # Orthogonal Laplacian in 1d is zero
        intf_fluxes_cntrd = intf_fluxes_avgd

    # Compute flux difference for hydrodynamic components
    data['fluxes'] = np.diff(intf_fluxes_cntrd, axis=axis)/ds[axis]

    return data