import concurrent.futures
from itertools import repeat

import numpy as np

from functions import fv
from num_methods import limiters

##############################################################################
# Fourth-order upwind constrained transport algorithm for MHD [Felker & Stone, 2018]
##############################################################################

# Reconstruct the transverse values for each face average (computation done entirely for orthogonal axis)
def reconstruct_transverse(interface, sim_variables, axis, method=None, extras=None):
    if not method:
        method = sim_variables.subgrid

    padded_grid_2 = fv.add_boundary(interface, sim_variables, stencil=2, axis=axis)
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
            padded_wU_2 = np.zeros_like(fv.add_boundary(wU, sim_variables, stencil=2, axis=axis))
        else:
            if sim_variables.ppm_author.startswith(("colella", "c")):
                # Limit interface values [Colella et al., 2011, p. 25-26]
                wU = limiters.interface_limiter(wU, *grid_slices)

            # Define the top and bottom parabolic extrapolants
            padded_wU_2 = fv.add_boundary(wU, sim_variables, stencil=2, axis=axis)
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
    prim_plus, prim_minus = fv.slice_(fv.add_boundary(wD, sim_variables, axis=axis), axis, start=1), fv.slice_(fv.add_boundary(wU, sim_variables, axis=axis), axis, end=-1)

    # Remove the 'leftmost' interface since only the upwind corners/lines are needed
    prim_plus, prim_minus = fv.slice_(prim_plus, axis=axis, start=1), fv.slice_(prim_minus, axis=axis, start=1)

    return prim_plus, prim_minus


# Compute the corner/line electric fields wrt to corner/line; gives 4-fold values for each corner/line in each axis [Mignone & Del Zanna, 2020]
# data = {axis: results_dict for axis, results in enumerate(thread_jobs)}
def compute_emf(data, axis):
    abscissa, ordinate, applicate = (axis + np.array(range(3)))%3
    vx, vy = 1+ordinate, 1+applicate
    Bx, By = 5+ordinate, 5+applicate

    # For -v x B in x-axis (axis=0, thumb), y (axis=1, middle finger) becomes (N,S) & z (axis=2, index finger) becomes (E,W)
    # For -v x B in y-axis (axis=1, thumb), z (axis=2, middle finger) becomes (N,S) & x (axis=0, index finger) becomes (E,W)
    # For -v x B in z-axis (axis=2, thumb), x (axis=0, middle finger) becomes (N,S) & y (axis=1, index finger) becomes (E,W)
    [north, south], [east, west] = data[applicate]['ortho_interfaces'], data[ordinate]['ortho_interfaces']
    [ap_x, am_x], [ap_y, am_y] = data[ordinate]['alphas'], data[applicate]['alphas']

    # Compute the corner mag. fields wrt to corner/line
    SW = .5*(west[...,vy]+south[...,vy])*south[...,Bx] - .5*(west[...,vx]+south[...,vx])*west[...,By]
    SE = .5*(east[...,vy]+south[...,vy])*south[...,Bx] - .5*(east[...,vx]+south[...,vx])*east[...,By]
    NW = .5*(west[...,vy]+north[...,vy])*north[...,Bx] - .5*(west[...,vx]+north[...,vx])*west[...,By]
    NE = .5*(east[...,vy]+north[...,vy])*north[...,Bx] - .5*(east[...,vx]+north[...,vx])*east[...,By]

    return fv.divide(ap_x*ap_y*SW + am_x*ap_y*SE + ap_x*am_y*NW + am_x*am_y*NE, (ap_x+am_x)*(ap_y+am_y)) - fv.divide(ap_y*am_y, ap_y+am_y)*(north[...,Bx]-south[...,Bx]) + fv.divide(ap_x*am_x, ap_x+am_x)*(east[...,By]-west[...,By])


# Compute constrained transport flux using corner/line emfs [Mignone & Del Zanna, 2020];
# the hydro fluxes are unaltered, and the CT fluxes are automatically allocated to their respective axes
def compute_ct_flux(emfs, flux, sim_variables, axis):
    # Get the orthogonal axes & emfs from the axis being computed
    ortho_axes = sim_variables.axes[sim_variables.axes != axis]
    ortho_emfs = emfs[sim_variables.axes != axis]

    # Use normal axis to orthogonal axes for derivatives
    normal_axes = np.roll(ortho_axes, shift=-1)

    def per_normal_axis(emf, _sim_variables, normal_axis):
        padded_emf = fv.add_boundary(emf, _sim_variables, axis=normal_axis)
        return np.diff(fv.slice_(padded_emf, axis=normal_axis, end=-1), axis=normal_axis)/_sim_variables.ds[normal_axis]

    # Update CT flux in axis; set the other axes fluxes to zero (for summation later)
    with concurrent.futures.ThreadPoolExecutor() as inner_executor:
        jobs = inner_executor.map(per_normal_axis, ortho_emfs, repeat(sim_variables), normal_axes)
        emf_fluxes = [emf_flux for emf_flux in jobs]
        if sim_variables.dimension > 2:
            flux[...,5+axis] = -np.diff(emf_fluxes, axis=0)
        else:
            flux[...,5+axis] = (-1)**axis * emf_fluxes[0]
        flux[...,5+ortho_axes] = 0

    return flux