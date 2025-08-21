import concurrent.futures as cfutures
from itertools import repeat

import numpy as np

from functions import fv
from num_methods import limiters

##############################################################################
# Fourth-order upwind constrained transport algorithm for MHD [Felker & Stone, 2018]
##############################################################################

# Reconstruct the transverse values for each face average (computation done entirely for orthogonal axis)
# Returns array aligned with input axis, e.g., ax=1 means returned array is aligned with y-axis-transposed grid
def reconstruct_transverse(interface, sim_variables, axis, method=None, extras=None):
    if not method:
        method = sim_variables.subgrid
    boundary = sim_variables.boundary

    padded_grid_2 = fv.add_boundary(interface, boundary, stencil=2, axis=axis)
    padded_grid = fv.slice_(padded_grid_2, axis, *[1,-1])

    zeroth = np.copy(interface)
    minus_one, minus_two = fv.slice_(padded_grid, axis, end=-2), fv.slice_(padded_grid_2, axis, end=-4)
    plus_one, plus_two = fv.slice_(padded_grid, axis, start=2), fv.slice_(padded_grid_2, axis, start=4)

    # 5th-order WENO reconstruction
    if "weno" in method:
        eps = 1e-6

        """Interpolate the face averages to both corners (upwards & downwards)
        |                w(i-1/2)            w(i+1/2)               |
        |-------------------|-------------------|-------------------|
        |           w_U(i-1/2,j+1/2)    w_U(i+1/2,j+1/2)            |
        |                  ^|                  ^|                  ^|
        |                  ||                  ||                  ||
        |                  ||                  ||                  ||
        |  o (i-1,j)     -->|  o (i,j)       -->|  o (i+1,j)     -->|
        |                  ||                  ||                  ||
        |                  ||                  ||                  ||
        |                  v|                  v|                  v|
        |           w_D(i-1/2,j-1/2)    w_D(i+1/2,j-1/2)            |
        |-------------------|-------------------|-------------------|
        """
        g0, g1, g2 = 1/10, 3/5, 3/10

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

        a0 = lambda d0: d0/(b0 + eps)**2
        a1 = lambda d1: d1/(b1 + eps)**2
        a2 = lambda d2: d2/(b2 + eps)**2

        wU = (
            (a0(g0)/(a0(g0)+a1(g1)+a2(g2))) * (1/3*minus_two - 7/6*minus_one + 11/6*zeroth)
            + (a1(g1)/(a0(g0)+a1(g1)+a2(g2))) * (-1/6*minus_one + 5/6*zeroth + 1/3*plus_one)
            + (a2(g2)/(a0(g0)+a1(g1)+a2(g2))) * (1/3*zeroth + 5/6*plus_one - 1/6*plus_two)
        )
        wD = (
            (a0(g2)/(a0(g2)+a1(g1)+a2(g0))) * (1/3*zeroth + 5/6*minus_one - 1/6*minus_two)
            + (a1(g1)/(a0(g2)+a1(g1)+a2(g0))) * (-1/6*plus_one + 5/6*zeroth + 1/3*minus_one)
            + (a2(g0)/(a0(g2)+a1(g1)+a2(g0))) * (1/3*plus_two - 7/6*plus_one + 11/6*zeroth)
        )

    elif method == "ppm":

        grid_slices = [minus_one, zeroth, plus_one, plus_two]

        """Interpolate the face averages to the top corners (upwards) [McCorquodale & Colella, 2011, eq. 17; Colella et al., 2011, eq. 67]
        |                w(i-1/2)            w(i+1/2)               |
        |-------------------|-------------------|-------------------|
        |           w_U(i-1/2,j+1/2)    w_U(i+1/2,j+1/2)            |
        |                  ^|                  ^|                  ^|
        |                  ||                  ||                  ||
        |                  ||                  ||                  ||
        |  o (i-1,j)     -->|  o (i,j)       -->|  o (i+1,j)     -->|
        """
        wU = 7/12 * (zeroth + plus_one) - 1/12 * (minus_one + plus_two)

        if sim_variables.ppm_author.startswith(("peterson", "p", "x")):
            """Interpolate the face averages to both corners (upwards & downwards)
            |                w(i-1/2)            w(i+1/2)               |
            |-------------------|-------------------|-------------------|
            |           w_U(i-1/2,j+1/2)    w_U(i+1/2,j+1/2)            |
            |                  ^|                  ^|                  ^|
            |                  ||                  ||                  ||
            |                  ||                  ||                  ||
            |  o (i-1,j)     -->|  o (i,j)       -->|  o (i+1,j)     -->|
            |                  ||                  ||                  ||
            |                  ||                  ||                  ||
            |                  v|                  v|                  v|
            |           w_D(i-1/2,j-1/2)    w_D(i+1/2,j-1/2)            |
            |-------------------|-------------------|-------------------|
            """
            wD = 7/12 * (minus_one + zeroth) - 1/12 * (minus_two + plus_one)

            # Limit interface values [Peterson & Hammett, 2008, eq. 3.33-3.34]
            limited_wUs = limiters.interface_limiter(wD, *[minus_two, minus_one, zeroth, plus_one]), limiters.interface_limiter(wU, *grid_slices)
            padded_wU_2 = np.zeros_like(fv.add_boundary(wU, boundary, stencil=2, axis=axis))
        else:
            if sim_variables.ppm_author.startswith(("colella", "c")):
                # Limit interface values [Colella et al., 2011, p. 25-26]
                wU = limiters.interface_limiter(wU, *grid_slices)

            # Define the top and bottom parabolic extrapolants
            padded_wU_2 = fv.add_boundary(wU, boundary, stencil=2, axis=axis)
            limited_wUs = fv.slice_(padded_wU_2, axis, *[1,-3]), fv.slice_(padded_wU_2, axis, *[2,-2])

        """Reconstruct the limited extrapolants from the interface values. Returns the face averages in the form of w+(y) & w-(y) when considering x-axis, and w+(x) & w-(x) when considering y-axis
        |                w(i-1/2)            w(i+1/2)               |
        |  o (i-1,j+1)      |  o (i,j+1)        |  o (i+1,j+1)      |
        |                   |                   |                   |
        |                   |                   |                   |
        |           w_D(i-1/2,j+1/2)    w_D(i+1/2,j+1/2)            |
        |                 w+(y)               w+(y)               w+(y)
        |                   ^                   ^                   ^
        |-------------------|-------------------|-------------------|
        |                   v                   v                   v
        |                 w-(y)               w-(y)               w-(y)
        |           w_U(i-1/2,j+1/2)    w_U(i+1/2,j+1/2)            |
        |                   |                   |                   |
        |                   |                   |                   |
        |  o (i-1,j)     -->|  o (i,j)       -->|  o (i+1,j)     -->|
        """
        wD, wU = limiters.extrapolant_limiter(zeroth, sim_variables, axis, *limited_wUs, **{
            'padded_grid':padded_grid, 'padded_grid_2':padded_grid_2, 'padded_interface_2':padded_wU_2
            })

        if sim_variables.ppm_dissipate:
            grid, eta = extras
            wD = wD * eta[...,None] + grid * (1-eta)[...,None]
            wU = wU * eta[...,None] + grid * (1-eta)[...,None]

    elif method == "plm":
        limited_values = limiters.minmod_limiter(padded_grid, axis)
        gradients = .5 * limited_values
        wD, wU = zeroth - gradients, zeroth + gradients

    else:
        wD, wU = zeroth, zeroth

    # Re-align the interfaces so that cell wall is in between interfaces
    prim_plus, prim_minus = fv.slice_(fv.add_boundary(wD, boundary, axis=axis), axis, start=1), fv.slice_(fv.add_boundary(wU, boundary, axis=axis), axis, end=-1)

    return prim_plus, prim_minus


# Compute the corner electric fields wrt to corner; gives 4-fold values for each corner [Mignone & del Zanna, 2021]
def compute_corner(data, sim_variables):
    vx, vy, Bx, By = sim_variables.vx, sim_variables.vy, sim_variables.Bx, sim_variables.By

    # [ (m1, m2), (m1, m2) ] refers to the [ x(w+, w-), y(w+, w-) ] axis, which corresponds to [ x(N,S), y(E,W) ]; take note of the order in the axes
    [north, south], [east, west] = data[0]['ortho_interfaces'], data[1]['ortho_interfaces']
    [ap_x, am_x], [ap_y, am_y] = data[0]['alphas'], data[1]['alphas']

    # Compute the corner B-fields wrt to corner
    SW = .5*(west[...,vy]+south[...,vy])*south[...,Bx] - .5*(west[...,vx]+south[...,vx])*west[...,By]
    SE = .5*(east[...,vy]+south[...,vy])*south[...,Bx] - .5*(east[...,vx]+south[...,vx])*east[...,By]
    NW = .5*(west[...,vy]+north[...,vy])*north[...,Bx] - .5*(west[...,vx]+north[...,vx])*west[...,By]
    NE = .5*(east[...,vy]+north[...,vy])*north[...,Bx] - .5*(east[...,vx]+north[...,vx])*east[...,By]

    return fv.divide(ap_x*ap_y*SW + am_x*ap_y*SE + ap_x*am_y*NW + am_x*am_y*NE, (ap_x+am_x)*(ap_y+am_y)) - fv.divide(ap_y*am_y, ap_y+am_y)*(north[...,Bx]-south[...,Bx]) + fv.divide(ap_x*am_x, ap_x+am_x)*(east[...,By]-west[...,By])


# Compute constrained transport flux using corners; the hydro fluxes are unaltered, and the CT fluxes are automatically allocated to their respective axes
def compute_ct_flux(corners, flux, sim_variables, axis):
    ortho_axis = 1 - axis
    padded_e3U = fv.add_boundary(corners, sim_variables.boundary, axis=ortho_axis)

    flux[...,5+axis] = (-1)**axis * np.diff(fv.slice_(padded_e3U, axis=ortho_axis, end=-1), axis=ortho_axis)/sim_variables.ds[ortho_axis]
    flux[...,5+ortho_axis] = 0

    return flux