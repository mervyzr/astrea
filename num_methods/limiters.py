import numpy as np

from functions import fv

##############################################################################
# Limiter functions for the interface and cell values
##############################################################################

# Calculate minmod (slope) limiter [Derigs et al., 2017]. Returns an array of gradients for each parameter in each cell
def minmod_limiter(padded_grid, axis):
    a, b = np.diff(fv.slice_(padded_grid, axis, end=-1), axis=axis), np.diff(fv.slice_(padded_grid, axis, start=1), axis=axis)
    arr = np.zeros_like(b)

    # (eq. 4.17)
    mask = np.where((np.abs(a) < np.abs(b)) & (a*b > 0))
    arr[mask] = a[mask]
    mask = np.where((np.abs(a) >= np.abs(b)) & (a*b > 0))
    arr[mask] = b[mask]
    return arr


# Calculate the van Leer/harmonic parameter [van Leer, 1974]
def vanLeer_limiter(padded_grid, axis):
    r = fv.divide(np.diff(fv.slice_(padded_grid, axis, end=-1), axis=axis), np.diff(fv.slice_(padded_grid, axis, start=1), axis=axis))
    return (r + np.abs(r))/(1 + np.abs(r)) * np.diff(fv.slice_(padded_grid, axis, start=1), axis=axis)


# Calculate the Ospre parameter [Waterson & Deconinck, 1995]
def ospre_limiter(padded_grid, axis):
    r = fv.divide(np.diff(fv.slice_(padded_grid, axis, end=-1), axis=axis), np.diff(fv.slice_(padded_grid, axis, start=1), axis=axis))
    return 1.5 * ((r**2 + r)/(r**2 + r + 1)) * np.diff(fv.slice_(padded_grid, axis, start=1), axis=axis)


# Calculate the van Albada "1" parameter [van Albada, 1982]
def vanAlbada_one_limiter(padded_grid, axis):
    r = fv.divide(np.diff(fv.slice_(padded_grid, axis, end=-1), axis=axis), np.diff(fv.slice_(padded_grid, axis, start=1), axis=axis))
    return (r**2 + r)/(r**2 + 1) * np.diff(fv.slice_(padded_grid, axis, start=1), axis=axis)


# Calculate the Koren parameter [Vreugdenhil & Koren, 1993]
def koren_limiter(padded_grid, axis):
    r = fv.divide(np.diff(fv.slice_(padded_grid, axis, end=-1), axis=axis), np.diff(fv.slice_(padded_grid, axis, start=1), axis=axis))
    return np.maximum(np.zeros_like(r), np.minimum(np.minimum(2*r, (2+r)/3), np.full_like(r,2))) * np.diff(fv.slice_(padded_grid, axis, start=1), axis=axis)


# Calculate the superbee parameter [Roe, 1986]
def superbee_limiter(padded_grid, axis):
    r = fv.divide(np.diff(fv.slice_(padded_grid, axis, end=-1), axis=axis), np.diff(fv.slice_(padded_grid, axis, start=1), axis=axis))
    return np.maximum(np.zeros_like(r), np.maximum(np.minimum(2*r, np.ones_like(r)), np.minimum(r, np.full_like(r,2)))) * np.diff(fv.slice_(padded_grid, axis, start=1), axis=axis)


# Function for limiting the interface values interpolated from cell centre for PPM [Colella et al., 2011, p. 26; Peterson & Hammett, 2008, eq. 3.33-3.34]
def interface_limiter(interface, *grid_slices):
    minus_one, zeroth, plus_one, plus_two = grid_slices
    C = 5/4

    # Initial check for local extrema (eq. 84)
    local_extrema = (interface - zeroth)*(plus_one - interface) < 0

    if local_extrema.any():
        D2w = np.zeros_like(interface)

        # Approximation to the second derivatives (eq. 85)
        D2w_L = minus_one - 2*zeroth + plus_one
        D2w_C = 3 * (zeroth - 2*interface + plus_one)
        D2w_R = zeroth - 2*plus_one + plus_two

        # Get the curvatures that have the same signs
        non_monotonic = (np.sign(D2w_L) == np.sign(D2w_R)) & (np.sign(D2w_C) == np.sign(D2w_R)) & (np.sign(D2w_C) == np.sign(D2w_L))
        #advanced_non_monotonic = ((D2w_R - D2w_C)*(D2w_C - D2w_L) < 0) & (np.sign(D2w_L) == np.sign(D2w_R)) & (np.sign(D2w_C) == np.sign(D2w_R))

        # Determine the limited curvature with the sign of each element in the 'centre' array (eq. 87)
        limited_curvature = np.sign(D2w_C) * np.minimum(np.abs(D2w_C), np.minimum(np.abs(C*D2w_L), np.abs(C*D2w_R)))

        # Update the limited local curvature estimates based on the conditions
        D2w[non_monotonic] = limited_curvature[non_monotonic]

        return .5*(zeroth+plus_one) - D2w/6
    else:
        return interface


# Parabolic extrapolant limiter for PPM [McCorquodale & Colella, 2011; Colella et al., 2011; Peterson & Hammett, 2008]
def extrapolant_limiter(grid, sim_variables, axis, *args, **kwargs):
    left_of_centre, right_of_centre = args
    padded_grid, padded_grid_2, padded_interface_2 = kwargs['padded_grid'], kwargs['padded_grid_2'], kwargs['padded_interface_2']
    C = 5/4

    # Set differences
    dw_minus, dw_plus = grid - left_of_centre, right_of_centre - grid

    if sim_variables.ppm_author.casefold().startswith(("mccorquodale", "m", "mc")):
        # Define functions
        wL, wR = np.copy(left_of_centre), np.copy(right_of_centre)
        d2w = 6 * (left_of_centre - 2*grid + right_of_centre)
        d2w_C = fv.slice_(padded_grid, axis, end=-2) - 2*grid + fv.slice_(padded_grid, axis, start=2)

        # Approximation to the third derivative [McCorquodale & Colella, 2011, eq. 23]
        d3w = fv.slice_(np.diff(fv.add_boundary(d2w_C, sim_variables, axis=axis), axis=axis), axis, start=1)

        # Check for cell extrema in cells [McCorquodale & Colella, 2011, eq. 24-25]
        cell_extrema = (dw_minus*dw_plus <= 0) | ((grid-fv.slice_(padded_grid_2, axis, end=-4)) * (fv.slice_(padded_grid_2, axis, start=4)-grid) <= 0)

        # If there are extrema in the cells
        if cell_extrema.any():
            d2w_Cw = fv.add_boundary(d2w_C, sim_variables, axis=axis)
            d2w_lim = np.zeros_like(grid)

            # Get the curvatures that have the same signs
            non_monotonic = (np.sign(fv.slice_(d2w_Cw, axis, end=-2)) == np.sign(fv.slice_(d2w_Cw, axis, start=2))) \
                & (np.sign(d2w) == np.sign(d2w_C)) \
                & (np.sign(d2w_C) == np.sign(fv.slice_(d2w_Cw, axis, end=-2))) \
                & (np.sign(d2w_C) == np.sign(fv.slice_(d2w_Cw, axis, start=2))) \
                & (np.sign(d2w) == np.sign(fv.slice_(d2w_Cw, axis, end=-2))) \
                & (np.sign(d2w) == np.sign(fv.slice_(d2w_Cw, axis, start=2)))

            # Determine the limited curvature with the sign of each element in the 'main' array [McCorquodale & Colella, 2011, eq. 26]
            limited_curvature = np.sign(d2w_C) * np.minimum(
                np.minimum(np.abs(d2w), C*np.abs(d2w_C)), \
                np.minimum(C*np.abs(fv.slice_(d2w_Cw, axis, start=2)), C*np.abs(fv.slice_(d2w_Cw, axis, end=-2))) \
                )

            # Update the limited local curvature estimates based on the conditions
            d2w_lim[cell_extrema] = limited_curvature[cell_extrema]

            # Determine the limited values that are sensitive to roundoff errors
            rho_limiter = np.zeros_like(grid)

            # Get the cells where the limited values fulfil the condition
            rho_sensitive = np.abs(d2w) > 1e-12 * np.maximum(
                np.abs(grid), 
                np.maximum(
                    np.maximum(np.abs(fv.slice_(padded_grid, axis, end=-2)), np.abs(fv.slice_(padded_grid, axis, start=2))), 
                    np.maximum(np.abs(fv.slice_(padded_grid_2, axis, end=-4)), np.abs(fv.slice_(padded_grid_2, axis, start=4)))
                    )
                )

            # Update the limited estimates based on the condition [McCorquodale & Colella, 2011, eq. 27]
            phi = fv.divide(d2w_lim, d2w)
            rho_limiter[rho_sensitive] = phi[rho_sensitive]

            # Apply additional limiters
            d3w_w2 = fv.add_boundary(d3w, sim_variables, stencil=2, axis=axis)
            d3w_w = fv.slice_(d3w_w2, axis, *[1,-1])
            d3w_min = np.minimum(
                np.minimum(fv.slice_(d3w_w, axis, end=-2), d3w), \
                np.minimum(fv.slice_(d3w_w2, axis, end=-4), fv.slice_(d3w_w2, axis, start=4)) \
                )
            d3w_max = np.maximum(
                np.maximum(fv.slice_(d3w_w, axis, end=-2), d3w), \
                np.maximum(fv.slice_(d3w_w2, axis, end=-4), fv.slice_(d3w_w2, axis, start=4)) \
                )

            # [McCorquodale & Colella, 2011, eq. 28]
            roundoff_limiters = (rho_limiter < (1-1e-12)) | (.1*np.maximum(np.abs(d3w_max), np.abs(d3w_min)) <= (d3w_max-d3w_min))

            # [McCorquodale & Colella, 2011, eq. 29-30]
            wL[(dw_minus*dw_plus < 0) & (roundoff_limiters)] = (grid - rho_limiter*dw_minus)[(dw_minus*dw_plus < 0) & (roundoff_limiters)]
            wR[(dw_minus*dw_plus < 0) & (roundoff_limiters)] = (grid + rho_limiter*dw_plus)[(dw_minus*dw_plus < 0) & (roundoff_limiters)]

            # [McCorquodale & Colella, 2011, eq. 31-32]
            wL[(roundoff_limiters) & (np.abs(dw_minus) >= 2*np.abs(dw_plus))] = (grid - 2*(1-rho_limiter)*dw_plus - rho_limiter*dw_minus)[(roundoff_limiters) & (np.abs(dw_minus) >= 2*np.abs(dw_plus))]
            wR[(roundoff_limiters) & (np.abs(dw_plus) >= 2*np.abs(dw_minus))] = (grid + 2*(1-rho_limiter)*dw_minus + rho_limiter*dw_plus)[(roundoff_limiters) & (np.abs(dw_plus) >= 2*np.abs(dw_minus))]
        else:
            wL[np.abs(dw_minus) >= 2*np.abs(dw_plus)] = (grid - 2*dw_plus)[np.abs(dw_minus) >= 2*np.abs(dw_plus)]
            wR[np.abs(dw_plus) >= 2*np.abs(dw_minus)] = (grid + 2*dw_minus)[np.abs(dw_plus) >= 2*np.abs(dw_minus)]
    else:
        # Check for cell extrema in cells [Colella et al., 2011, eq. 89; Peterson & Hammett, 2008, eq. 3.31]
        cell_extrema = dw_minus*dw_plus <= 0

        if sim_variables.ppm_author.casefold().startswith(("peterson", "p", "ph", "x")):
            extrapolant_extrema = (fv.slice_(padded_grid, axis, end=-2)-grid)*(grid-fv.slice_(padded_grid, axis, start=2)) <= 0
        else:
            # Check for overshoot in cells [Colella et al., 2011, eq. 90]
            overshoot = (np.abs(dw_minus) > 2*np.abs(dw_plus)) | (np.abs(dw_plus) > 2*np.abs(dw_minus))

            # Check for extrema in extrapolants [Colella et al., 2011, eq. 91-94]
            d_wF_minmod_L, d_wF_minmod_R = left_of_centre - fv.slice_(padded_interface_2, axis, end=-4), fv.slice_(padded_interface_2, axis, start=4) - right_of_centre
            d_wS_minmod_L, d_wS_minmod_R = grid - fv.slice_(padded_grid, axis, end=-2), fv.slice_(padded_grid, axis, start=2) - grid

            d_wF_minmod = np.minimum(np.abs(d_wF_minmod_L), np.abs(d_wF_minmod_R))
            d_wS_minmod = np.minimum(np.abs(d_wS_minmod_L), np.abs(d_wS_minmod_R))

            extrapolant_extrema = ((d_wF_minmod >= d_wS_minmod) & (d_wF_minmod_L*d_wF_minmod_R < 0)) | ((d_wS_minmod >= d_wF_minmod) & (d_wS_minmod_L*d_wS_minmod_R < 0))

        # If there are extrema in the cells or extrapolants
        if cell_extrema.any() or extrapolant_extrema.any():
            D2w_lim = np.zeros_like(grid)

            # Approximation to the second derivative [Colella et al., 2011, eq. 95; Peterson & Hammett, 2008, eq. 3.37]
            D2w = 6 * (left_of_centre - 2*grid + right_of_centre)
            D2w_L = fv.slice_(padded_grid_2, axis, end=-4) - 2*fv.slice_(padded_grid, axis, end=-2) + grid
            D2w_C = fv.slice_(padded_grid, axis, end=-2) - 2*grid + fv.slice_(padded_grid, axis, start=2)
            D2w_R = grid - 2*fv.slice_(padded_grid, axis, start=2) + fv.slice_(padded_grid_2, axis, start=4)

            # Get the curvatures that have the same signs
            non_monotonic = (np.sign(D2w) == np.sign(D2w_C)) \
                & (np.sign(D2w) == np.sign(D2w_L)) \
                & (np.sign(D2w) == np.sign(D2w_R)) \
                & (np.sign(D2w_C) == np.sign(D2w_L)) \
                & (np.sign(D2w_C) == np.sign(D2w_R)) \
                & (np.sign(D2w_L) == np.sign(D2w_R))

            # Determine the limited curvature with the sign of each element in the 'main' array [Colella et al., 2011, eq. 96]
            limited_curvature = np.sign(D2w) * np.minimum(np.minimum(np.abs(D2w), np.abs(C*D2w_C)), np.minimum(np.abs(C*D2w_L), np.abs(C*D2w_R)))

            # Update the limited local curvature estimates based on the conditions [Peterson & Hammett, 2008, eq. 3.38]
            D2w_lim[cell_extrema & non_monotonic] = limited_curvature[cell_extrema & non_monotonic]

            if sim_variables.ppm_author.casefold().startswith(("peterson", "p", "ph", "x")):
                # Get the final limited values [Peterson & Hammett, 2008, eq. 3.39]
                phi = fv.divide(D2w_lim, D2w)

                wL, wR = grid + phi*(left_of_centre-grid), grid + phi*(right_of_centre-grid)
            else:
                D2w_lim[extrapolant_extrema & non_monotonic] = limited_curvature[extrapolant_extrema & non_monotonic]

                phi = fv.divide(D2w_lim, D2w)

                # Further update if there are local extrema [Colella et al., 2011, eq. 97-98]
                d_uL_bar, d_uR_bar = np.copy(dw_minus), np.copy(dw_plus)
                if overshoot.any():
                    d_uL_bar[np.abs(dw_minus) > 2*np.abs(dw_plus)] = 2*dw_plus[np.abs(dw_minus) > 2*np.abs(dw_plus)]
                    d_uR_bar[np.abs(dw_plus) > 2*np.abs(dw_minus)] = 2*dw_minus[np.abs(dw_plus) > 2*np.abs(dw_minus)]

                # [Colella et al., 2011, eq. 98]
                wL, wR = grid - phi*d_uL_bar, grid + phi*d_uR_bar
        else:
            wL, wR = left_of_centre, right_of_centre
    return wL, wR