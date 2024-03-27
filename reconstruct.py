import numpy as np

from functions import fv

##############################################################################

# Extrapolate the cell-centre values to the cell-faces
def extrapolate(tube, gamma, solver, boundary):
    # Piecewise parabolic method solver (3rd-order stable for uneven grid; 4th-order stable for even grid)
    if solver in ["ppm", "parabolic", "p"]:
        # Conversion of conservative variables to primitive variables
        wS = fv.convertConservative(tube, gamma, boundary)

        w = fv.makeBoundary(wS, boundary)
        w2 = fv.makeBoundary(wS, boundary, 2)

        # Extrapolate in primitive variables to 4th-order face values
        wFL = 7/12 * (wS+w[:-2]) - 1/12 * (w2[:-4]+w[2:])  # face i-1/2
        wFR = 7/12 * (wS+w[2:]) - 1/12 * (w[:-2]+w2[4:])  # face i+1/2
        return [wS, [wFL, wFR], w, w2]
    else:
        return fv.makeBoundary(tube, boundary)


# Reconstruct the interpolants using the limited values
def interpolate(extrapolatedValues, limitedValues, solver):
    # Reconstruction of parabolic interpolant
    if solver in ["ppm", "parabolic", "p"]:
        C = 5/4
        wS, wF, w, w2 = extrapolatedValues
        wFL, wFR = wF
        wF_limit_L, wF_limit_R = limitedValues

        # XPPM parabolic interpolant [Peterson & Hammett, 2013, p. B586]; preserves behaviour at smooth-extrema
        if True:
            # Check for local extrema away from smooth extrema (eq. 3.31)
            local_extrema = ((wFR - wS)*(wS - wFL) <= 0) | ((w[:-2] - wS)*(wS - w[2:]) <= 0)
            wF_limit_L[local_extrema] = wS[local_extrema]
            wF_limit_R[local_extrema] = wS[local_extrema]

            # Calculate the limited smooth extrema
            D2w_lim = np.zeros(wS.shape)

            # Approximation to the second derivatives (eq. 3.37)
            D2w = 6 * (wFL - 2*wS + wFR)
            D2w_L = w2[:-4] - 2*w[:-2] + wS
            D2w_C = w[:-2] - 2*wS + w[2:]
            D2w_R = wS - 2*w[2:] + w2[4:]

            # Get the curvatures that have the same signs
            non_monotonic = (np.sign(D2w_L) == np.sign(D2w_R)) & (np.sign(D2w_C) == np.sign(D2w)) & (np.sign(D2w_C) == np.sign(D2w_R))

            # Determine the limited curvature with the sign of each element in the 'main' array (eq. 3.38)
            limited_curvature = np.sign(D2w) * np.minimum(np.minimum(np.abs(D2w), C*np.abs(D2w_C)), C * np.minimum(np.abs(D2w_L), np.abs(D2w_R)))

            # Update the limited local curvature estimates based on the conditions
            D2w_lim[non_monotonic] = limited_curvature[non_monotonic]

            D2w[D2w == 0] = np.inf  # removes divide-by-zero issue; causes wF -> wS (i.e. piecewise constant) when D2w -> 0

            return [wS + ((D2w_lim/D2w) * (wF_limit_L - wS)), wS + ((D2w_lim/D2w) * (wF_limit_R - wS))]  # (eq. 3.39)
        
        # Limited parabolic interpolant [Colella et al., 2011, p. 26]
        else:
            # Calculate the limited parabolic interpolant values (eq. 89)
            d_uL, d_uR = wS - wF_limit_L, wF_limit_R - wS
            
            # Check for local and cell extrema in cells (eq. 90)
            local_extrema = (np.abs(d_uL) > 2*np.abs(d_uR)) | (np.abs(d_uR) > 2*np.abs(d_uL))
            cell_extrema = d_uL*d_uR < 0

            wF_limit_L2 = fv.makeBoundary(wF_limit_L, boundary)
            wF_limit_R2 = fv.makeBoundary(wF_limit_R, boundary)

            # Check for local extrema in interpolants (eq. 91-94)
            d_wF_minmod = np.minimum(np.abs(wF_limit_L - wF_limit_L2[:-2]), np.abs(wF_limit_R2[2:] - wF_limit_R))
            d_wS_minmod = np.minimum(np.abs(wS - w[:-2]), np.abs(w[2:] - wS))
            interpolant_extrema = ((d_wF_minmod >= d_wS_minmod) & ((wF_limit_L - wF_limit_L2[:-2])*(wF_limit_R2[2:] - wF_limit_R) < 0)) | ((d_wS_minmod >= d_wF_minmod) & ((wS - w[:-2])*(w[2:] - wS) < 0))

            # If there are extrema in either the cells or interpolants
            if local_extrema.any() or cell_extrema.any() or interpolant_extrema.any():
                D2w_lim = np.zeros(wS.shape)
                
                # Approximation to the second derivative (eq. 95)
                D2w = 6 * (wF_limit_L - 2*wS + wF_limit_R)
                D2w_L = w2[:-4] - 2*w[:-2] + wS
                D2w_C = w[:-2] - 2*wS + w[2:]
                D2w_R = wS - 2*w[2:] + w2[4:]

                # Get the curvatures that have the same signs
                non_monotonic = (np.sign(D2w_L) == np.sign(D2w_R)) & (np.sign(D2w_C) == np.sign(D2w)) & (np.sign(D2w_C) == np.sign(D2w_R))

                # Determine the limited curvature with the sign of each element in the 'main' array (eq. 96)
                limited_curvature = np.sign(D2w) * np.minimum(np.minimum(np.abs(D2w), C*np.abs(D2w_C)), C * np.minimum(np.abs(D2w_L), np.abs(D2w_R)))

                # Update the limited local curvature estimates based on the conditions
                D2w_lim[cell_extrema & non_monotonic] = limited_curvature[cell_extrema & non_monotonic]
                D2w_lim[interpolant_extrema & non_monotonic] = limited_curvature[interpolant_extrema & non_monotonic]

                D2w[D2w == 0] = np.inf  # removes divide by zero issue; causes wF -> wS (i.e. piecewise constant) when D2w -> 0

                # Further update if there is local extrema (eq. 97-98)
                if local_extrema.any():
                    d_uL_bar, d_uR_bar = np.copy(d_uL), np.copy(d_uR)
                    d_uL_bar[np.abs(d_uL) > 2*np.abs(d_uR)] = 2*d_uR[np.abs(d_uL) > 2*np.abs(d_uR)]
                    d_uR_bar[np.abs(d_uR) > 2*np.abs(d_uL)] = 2*d_uL[np.abs(d_uR) > 2*np.abs(d_uL)]
                    return [wS - d_uL_bar*(D2w_lim/D2w), wS + d_uR_bar*(D2w_lim/D2w)]
                else:
                    return [wS - d_uL*(D2w_lim/D2w), wS + d_uR*(D2w_lim/D2w)]
            else:
                return [wF_limit_L, wF_limit_R]

    # Linear reconstruction [Derigs et al., 2017]
    elif solver in ["plm", "linear", "l"]:
        tube = extrapolatedValues[1:-1]
        gradients = .5 * limitedValues
        return [tube - gradients, tube + gradients]  # (eq. 4.13)

    # No reconstruction, i.e. piecewise constant
    else:
        return extrapolatedValues