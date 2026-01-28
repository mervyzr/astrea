import concurrent.futures
from itertools import repeat

import numpy as np

from functions import constructor, fv
from num_methods import ct, solvers

##############################################################################
# WENO reconstruction method [Shu, 2009]
##############################################################################

def reconstruct(grid, sim_variables, axis, order=5):
    eps = sim_variables.eps

    # Define frequently used terms
    padded_grid = fv.add_boundary(grid, sim_variables, axis=axis)

    zeroth = np.copy(grid)
    minus_one, plus_one = fv.slice_(padded_grid, axis, end=-2), fv.slice_(padded_grid, axis, start=2)

    """WENO reconstruction from cell averages to face averages (both sides) [Shu, 2009; San & Kara, 2015]
    |                        w(i-1/2)                    w(i+1/2)                       |
    |<--         i-1         -->|<--          i          -->|<--         i+1         -->|
    |   w_L(i-1)     w_R(i-1)   |   w_L(i)         w_R(i)   |   w_L(i+1)     w_R(i+1)   |
    |   w+(i-3/2)   w-(i-1/2)   |   w+(i-1/2)   w-(i+1/2)   |   w+(i+1/2)   w-(i+3/2)   |
    """
    wL, wR = None, None
    if 0 < order <= 3:
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

    else:
        padded_grid_2 = fv.add_boundary(grid, sim_variables, stencil=2, axis=axis)
        minus_two, plus_two = fv.slice_(padded_grid_2, axis, end=-4), fv.slice_(padded_grid_2, axis, start=4)

        if 3 < order <= 5:
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

        elif 5 < order <= 7:
            padded_grid_3 = fv.add_boundary(grid, sim_variables, stencil=3, axis=axis)
            minus_three, plus_three = fv.slice_(padded_grid_3, axis, end=-6), fv.slice_(padded_grid_3, axis, start=6)

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

    return wL, wR


def run(grid, sim_variables, axis):
    convert, subgrid, multidimensional, axes, magnetic, ds = sim_variables.convert, sim_variables.subgrid, sim_variables.multidimensional, sim_variables.axes, sim_variables.magnetic, sim_variables.ds
    data = {}

    Riemann_solver = solvers.get_Riemann_solver(sim_variables)
    ortho_axes = axes[axes != axis] if (magnetic or multidimensional) else None

    # WENO reconstruction [Shu, 2009; San & Kara, 2015]
    if len(subgrid.split("weno")) == 2:
        try:
            wL, wR = reconstruct(grid, sim_variables, axis, int(subgrid.replace('-','').split("weno")[-1]))
        except Exception as e:
            wL, wR = reconstruct(grid, sim_variables, axis)
    else:
        wL, wR = reconstruct(grid, sim_variables, axis)

    # Re-align the interfaces so that cell wall is in between interfaces
    prim_plus, prim_minus = fv.slice_(fv.add_boundary(wL, sim_variables, axis=axis), axis, start=1), fv.slice_(fv.add_boundary(wR, sim_variables, axis=axis), axis, end=-1)
    if magnetic:
        padded_grid = fv.add_boundary(grid, sim_variables, axis=axis)
        prim_plus[...,5+axes] = prim_minus[...,5+axes] = fv.slice_(padded_grid, axis, end=-1)[...,5+axes]
        data['interfaces'] = fv.slice_(prim_plus, axis, start=1), fv.slice_(prim_minus, axis, start=1)

    # Get the average solution between the interfaces at the boundaries
    intf_avg = fv.compute_Roe_average([prim_plus,prim_minus], sim_variables)
    padded_intf_avg = fv.add_boundary(fv.slice_(intf_avg, axis, start=1), sim_variables, axis=axis)

    # Convert the primitive variables at the interface
    cons_plus, cons_minus = convert("primitive", prim_plus, sim_variables, axis=axis, pos='intf'), convert("primitive", prim_minus, sim_variables, axis=axis, pos='intf')

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
            characteristics = np.full_like(padded_intf_avg, .1)

    # Compute eigmax for time stepping limits
    data['eigmax'] = ds[axis]/fv.compute_eigmax(characteristics, axis=axis)

    # Compute alphas and save the reconstructed interfaces for CT computation
    if magnetic and multidimensional:
        data['alphas'] = ct.compute_alphas(characteristics, axis=axis)

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