import numpy as np

from functions import fv

##############################################################################
# Limiter functions for the interface and cell values
##############################################################################

# Gradient-based limiters. Returns an array of gradients for each parameter in each cell
def gradient_limiters(wS, boundary, ax, limiter="minmod"):
    w = fv.add_boundary(wS, boundary, axis=ax)
    width = w.shape[ax]
    a, b = np.diff(w.take(range(0,width-1), axis=ax), axis=ax), np.diff(w.take(range(1,width), axis=ax), axis=ax)

    # minmod (slope) limiter [Derigs et al., 2017]
    if limiter.lower() == "minmod":
        arr = np.zeros_like(wS)

        mask = np.where((np.abs(a) < np.abs(b)) & (a*b > 0))
        arr[mask] = a[mask]
        mask = np.where((np.abs(a) >= np.abs(b)) & (a*b > 0))
        arr[mask] = b[mask]
    else:
        r = fv.divide(a, b)

        # van Leer/harmonic parameter [van Leer, 1974]
        if "leer" in limiter.lower():
            arr = (r + np.abs(r))/(1 + np.abs(r)) * b
        # Ospre parameter [Waterson & Deconinck, 1995]
        elif "ospre" in limiter.lower():
            arr = 1.5 * ((r**2 + r)/(r**2 + r + 1)) * b
        # van Albada "1" parameter [van Albada, 1982]
        elif "albada" in limiter.lower():
            arr = (r**2 + r)/(r**2 + 1) * b
        # Koren parameter [Vreugdenhil & Koren, 1993]
        elif "koren" in limiter.lower():
            arr = np.maximum(np.zeros_like(r), np.minimum(np.minimum(2*r, (2+r)/3), np.full_like(r,2))) * b
        # Superbee parameter [Roe, 1986]
        elif "superbee" in limiter.lower():
            arr = np.maximum(np.zeros_like(r), np.maximum(np.minimum(2*r, np.ones_like(r)), np.minimum(r, np.full_like(r,2)))) * b
    return arr


# Function for limiting the interface values extrapolated from cell centre for PPM [Colella et al., 2011, p. 26; Peterson & Hammett, 2008, eq. 3.33-3.34]
def interface_limiter(w_face, w_minus_one, w_cell, w_plus_one, w_plus_two, C=5/4):
    # Initial check for local extrema (eq. 84)
    local_extrema = (w_face - w_cell)*(w_plus_one - w_face) < 0

    if local_extrema.any():
        D2w = np.zeros_like(w_face)

        # Approximation to the second derivatives (eq. 85)
        D2w_L = w_minus_one - 2*w_cell + w_plus_one
        D2w_C = 3 * (w_cell - 2*w_face + w_plus_one)
        D2w_R = w_cell - 2*w_plus_one + w_plus_two

        # Get the curvatures that have the same signs
        non_monotonic = (np.sign(D2w_L) == np.sign(D2w_R)) & (np.sign(D2w_C) == np.sign(D2w_R)) & (np.sign(D2w_C) == np.sign(D2w_L))
        #advanced_non_monotonic = ((D2w_R - D2w_C)*(D2w_C - D2w_L) < 0) & (np.sign(D2w_L) == np.sign(D2w_R)) & (np.sign(D2w_C) == np.sign(D2w_R))

        # Determine the limited curvature with the sign of each element in the 'centre' array (eq. 87)
        limited_curvature = np.sign(D2w_C) * np.minimum(np.abs(D2w_C), np.minimum(np.abs(C*D2w_L), np.abs(C*D2w_R)))

        # Update the limited local curvature estimates based on the conditions
        D2w[non_monotonic] = limited_curvature[non_monotonic]

        return .5*(w_cell+w_plus_one) - D2w/6
    else:
        return w_face


# Parabolic interpolant limiter for PPM [McCorquodale & Colella, 2011; Colella et al., 2011; Peterson & Hammett, 2008]
def interpolant_limiter(wS, w, w2, author, ax, *args, **kwargs):
    C = 5/4
    wF_L, wF_R = args

    # Set differences
    dw_minus, dw_plus = wS - wF_L, wF_R - wS
    w_minusOne, w_plusOne = w.take(range(0,w.shape[ax]-2), axis=ax), w.take(range(2,w.shape[ax]), axis=ax)
    w_minusTwo, w_plusTwo = w2.take(range(0,w2.shape[ax]-4), axis=ax), w2.take(range(4,w2.shape[ax]), axis=ax)

    if author == "mc" or "mccorquodale" in author:
        # Define functions
        boundary = kwargs['boundary']
        wL, wR = np.copy(wF_L), np.copy(wF_R)
        d2w = 6 * (wF_L - 2*wS + wF_R)
        d2w_C = w_minusOne - 2*wS + w_plusOne

        # Approximation to the third derivative [McCorquodale & Colella, 2011, eq. 23]
        _d3w = np.diff(fv.add_boundary(d2w_C, boundary, axis=ax), axis=ax)
        d3w = _d3w.take(range(1,_d3w.shape[ax]), axis=ax)

        # Check for cell extreme in cells [McCorquodale & Colella, 2011, eq. 24-25]
        cell_extrema = (dw_minus*dw_plus <= 0) | ((wS-w_minusTwo)*(w_plusTwo-wS) <= 0)

        # If there are extrema in the cells
        if cell_extrema.any():
            d2w_Cw = fv.add_boundary(d2w_C, boundary, axis=ax)
            d2w_Cw_minusOne, d2w_Cw_plusOne = d2w_Cw.take(range(0,d2w_Cw.shape[ax]-2), axis=ax), d2w_Cw.take(range(2,d2w_Cw.shape[ax]), axis=ax)
            d2w_lim = np.zeros_like(wS)

            # Get the curvatures that have the same signs
            non_monotonic = (np.sign(d2w_Cw_minusOne) == np.sign(d2w_Cw_plusOne)) & (np.sign(d2w) == np.sign(d2w_C)) & (np.sign(d2w_C) == np.sign(d2w_Cw_minusOne)) & (np.sign(d2w_C) == np.sign(d2w_Cw_plusOne)) & (np.sign(d2w) == np.sign(d2w_Cw_minusOne)) & (np.sign(d2w) == np.sign(d2w_Cw_plusOne))

            # Determine the limited curvature with the sign of each element in the 'main' array [McCorquodale & Colella, 2011, eq. 26]
            limited_curvature = np.sign(d2w_C) * np.minimum(np.minimum(np.abs(d2w), C*np.abs(d2w_C)), np.minimum(C*np.abs(d2w_Cw_plusOne), C*np.abs(d2w_Cw_minusOne)))

            # Update the limited local curvature estimates based on the conditions
            d2w_lim[cell_extrema] = limited_curvature[cell_extrema]

            # Determine the limited values that are sensitive to roundoff errors
            rho_limiter = np.zeros_like(wS)

            # Get the cells where the limited values fulfil the condition
            rho_sensitive = np.abs(d2w) > 1e-12 * np.maximum(np.abs(wS), np.maximum(np.maximum(np.abs(w_minusOne), np.abs(w_plusOne)), np.maximum(np.abs(w_minusTwo), np.abs(w_plusTwo))))

            # Update the limited estimates based on the condition [McCorquodale & Colella, 2011, eq. 27]
            phi = fv.divide(d2w_lim, d2w)
            rho_limiter[rho_sensitive] = phi[rho_sensitive]

            # Apply additional limiters
            d3w_w = fv.add_boundary(d3w, boundary, axis=ax)
            d3w_w2 = fv.add_boundary(d3w, boundary, stencil=2, axis=ax)
            d3w_minusOne = d3w_w.take(range(0,d3w_w.shape[ax]-2), axis=ax)
            d3w_minusTwo, d3w_plusTwo = d3w_w2.take(range(0,d3w_w2.shape[ax]-4), axis=ax), d3w_w2.take(range(4,d3w_w2.shape[ax]), axis=ax)
            d3w_min = np.minimum(np.minimum(d3w_minusOne, d3w), np.minimum(d3w_minusTwo, d3w_plusTwo))
            d3w_max = np.maximum(np.maximum(d3w_minusOne, d3w), np.maximum(d3w_minusTwo, d3w_plusTwo))

            # [McCorquodale & Colella, 2011, eq. 28]
            roundoff_limiters = (rho_limiter < (1-1e-12)) | (.1*np.maximum(np.abs(d3w_max), np.abs(d3w_min)) <= (d3w_max-d3w_min))

            # [McCorquodale & Colella, 2011, eq. 29-30]
            wL[(dw_minus*dw_plus < 0) & (roundoff_limiters)] = (wS - rho_limiter*dw_minus)[(dw_minus*dw_plus < 0) & (roundoff_limiters)]
            wR[(dw_minus*dw_plus < 0) & (roundoff_limiters)] = (wS + rho_limiter*dw_plus)[(dw_minus*dw_plus < 0) & (roundoff_limiters)]

            # [McCorquodale & Colella, 2011, eq. 31-32]
            wL[(roundoff_limiters) & (np.abs(dw_minus) >= 2*np.abs(dw_plus))] = (wS - 2*(1-rho_limiter)*dw_plus - rho_limiter*dw_minus)[(roundoff_limiters) & (np.abs(dw_minus) >= 2*np.abs(dw_plus))]
            wR[(roundoff_limiters) & (np.abs(dw_plus) >= 2*np.abs(dw_minus))] = (wS + 2*(1-rho_limiter)*dw_minus + rho_limiter*dw_plus)[(roundoff_limiters) & (np.abs(dw_plus) >= 2*np.abs(dw_minus))]
        else:
            wL[np.abs(dw_minus) >= 2*np.abs(dw_plus)] = (wS - 2*dw_plus)[np.abs(dw_minus) >= 2*np.abs(dw_plus)]
            wR[np.abs(dw_plus) >= 2*np.abs(dw_minus)] = (wS + 2*dw_minus)[np.abs(dw_plus) >= 2*np.abs(dw_minus)]
    else:
        # Check for cell extrema in cells [Colella et al., 2011, eq. 89; Peterson & Hammett, 2008, eq. 3.31]
        cell_extrema = dw_minus*dw_plus <= 0

        if "x" in author or "ph" in author or author in ["peterson", "hammett"]:
            interpolant_extrema = (w_minusOne-wS)*(wS-w_plusOne) <= 0
        else:
            wF_minusTwo, wF_plusTwo = kwargs['wF_pad2'].take(range(0,kwargs['wF_pad2'].shape[ax]-4), axis=ax), kwargs['wF_pad2'].take(range(4,kwargs['wF_pad2'].shape[ax]), axis=ax)

            # Check for overshoot in cells [Colella et al., 2011, eq. 90]
            overshoot = (np.abs(dw_minus) > 2*np.abs(dw_plus)) | (np.abs(dw_plus) > 2*np.abs(dw_minus))

            # Check for extrema in interpolants [Colella et al., 2011, eq. 91-94]
            d_wF_minmod_L, d_wF_minmod_R = wF_L - np.copy(wF_minusTwo), np.copy(wF_plusTwo) - wF_R
            d_wS_minmod_L, d_wS_minmod_R = wS - w_minusOne, w_plusOne - wS

            d_wF_minmod = np.minimum(np.abs(d_wF_minmod_L), np.abs(d_wF_minmod_R))
            d_wS_minmod = np.minimum(np.abs(d_wS_minmod_L), np.abs(d_wS_minmod_R))

            interpolant_extrema = ((d_wF_minmod >= d_wS_minmod) & (d_wF_minmod_L*d_wF_minmod_R < 0)) | ((d_wS_minmod >= d_wF_minmod) & (d_wS_minmod_L*d_wS_minmod_R < 0))

        # If there are extrema in the cells or interpolants
        if cell_extrema.any() or interpolant_extrema.any():
            D2w_lim = np.zeros_like(wS)

            # Approximation to the second derivative [Colella et al., 2011, eq. 95; Peterson & Hammett, 2008, eq. 3.37]
            D2w = 6 * (wF_L - 2*wS + wF_R)
            D2w_L = w_minusTwo - 2*w_minusOne + wS
            D2w_C = w_minusOne - 2*wS + w_plusOne
            D2w_R = wS - 2*w_plusOne + w_plusTwo

            # Get the curvatures that have the same signs
            non_monotonic = (np.sign(D2w) == np.sign(D2w_C)) & (np.sign(D2w) == np.sign(D2w_L)) & (np.sign(D2w) == np.sign(D2w_R)) & (np.sign(D2w_C) == np.sign(D2w_L)) & (np.sign(D2w_C) == np.sign(D2w_R)) & (np.sign(D2w_L) == np.sign(D2w_R))

            # Determine the limited curvature with the sign of each element in the 'main' array [Colella et al., 2011, eq. 96]
            limited_curvature = np.sign(D2w) * np.minimum(np.minimum(np.abs(D2w), np.abs(C*D2w_C)), np.minimum(np.abs(C*D2w_L), np.abs(C*D2w_R)))

            # Update the limited local curvature estimates based on the conditions [Peterson & Hammett, 2008, eq. 3.38]
            D2w_lim[cell_extrema & non_monotonic] = limited_curvature[cell_extrema & non_monotonic]

            if "x" in author or "ph" in author or author in ["peterson", "hammett"]:
                # Get the final limited values [Peterson & Hammett, 2008, eq. 3.39]
                phi = fv.divide(D2w_lim, D2w)

                wL, wR = wS + phi*(wF_L-wS), wS + phi*(wF_R-wS)
            else:
                D2w_lim[interpolant_extrema & non_monotonic] = limited_curvature[interpolant_extrema & non_monotonic]

                phi = fv.divide(D2w_lim, D2w)

                # Further update if there are local extrema [Colella et al., 2011, eq. 97-98]
                d_uL_bar, d_uR_bar = np.copy(dw_minus), np.copy(dw_plus)
                if overshoot.any():
                    d_uL_bar[np.abs(dw_minus) > 2*np.abs(dw_plus)] = 2*dw_plus[np.abs(dw_minus) > 2*np.abs(dw_plus)]
                    d_uR_bar[np.abs(dw_plus) > 2*np.abs(dw_minus)] = 2*dw_minus[np.abs(dw_plus) > 2*np.abs(dw_minus)]

                # [Colella et al., 2011, eq. 98]
                wL, wR = wS - phi*d_uL_bar, wS + phi*d_uR_bar
        else:
            wL, wR = wF_L, wF_R
    return wL, wR