import numpy as np

from functions import fv

##############################################################################

# Calculate parabolic-interpolant and face-value limiters
def xppmLimiter(reconstructedValues, boundary, C=5/4):
    wS, wF, w1, w2 = reconstructedValues
    wFL, wFR = wF
    wLs, wRs = w1
    wL2s, wR2s = w2

    # Function for calculating the limited face-values
    def limitFaceValues(w_face, w_minusOne, w_cell, w_plusOne, w_plusTwo):
        # Initial check for local extrema
        local_extrema = (w_face - w_cell)*(w_plusOne - w_face) < 0

        if local_extrema.any():
            D2w = np.zeros(w_face.shape)

            # Approximation to the second derivatives
            D2w_L = w_minusOne - 2*w_cell + w_plusOne
            D2w_C = 3 * (w_cell - 2*w_face + w_plusOne)
            D2w_R = w_cell - 2*w_plusOne + w_plusTwo

            # Get the curvatures that have the same signs
            monotonic = (np.sign(D2w_L) == np.sign(D2w_R)) & (np.sign(D2w_C) == np.sign(D2w_R))
            
            # Determine the limited curvature with the sign of each element in the 'centre' array
            limited_curvature = np.sign(D2w_C) * np.minimum(np.abs(D2w_C), C * np.minimum(np.abs(D2w_L), np.abs(D2w_R)))

            # Update the limited local curvature estimates based on the conditions
            D2w[monotonic] = limited_curvature[monotonic]

            return (.5 * (w_cell+w_plusOne)) - (D2w/6)
        else:
            return w_face

    # Limited face-values
    wF_limit_L = limitFaceValues(wFL, wL2s[:-1], wLs[:-1], wS, wRs[1:])
    wF_limit_R = limitFaceValues(wFR, wLs[:-1], wS, wRs[1:], wR2s[1:])

    # Check for local extrema away from smooth extrema
    local_extrema = ((wFR - wS)*(wS - wFL) <= 0) | ((wLs[:-1] - wS)*(wS - wRs[1:]) <= 0)
    wF_limit_L[local_extrema] = wS[local_extrema]
    wF_limit_R[local_extrema] = wS[local_extrema]

    # Calculate the limited smooth extrema
    D2w_lim = np.zeros(wS.shape)

    # Approximation to the second derivatives
    D2w = 6 * (wFL - 2*wS + wFR)
    D2w_L = wL2s[:-1] - 2*wLs[:-1] + wS
    D2w_C = wLs[:-1] - 2*wS + wRs[1:]
    D2w_R = wS - 2*wRs[1:] + wR2s[1:]

    # Get the curvatures that have the same signs
    monotonic = (np.sign(D2w_L) == np.sign(D2w_R)) & (np.sign(D2w_C) == np.sign(D2w)) & (np.sign(D2w_C) == np.sign(D2w_R))

    # Determine the limited curvature with the sign of each element in the 'main' array
    limited_curvature = np.sign(D2w) * np.minimum(np.minimum(np.abs(D2w), C*np.abs(D2w_C)), C * np.minimum(np.abs(D2w_L), np.abs(D2w_R)))

    # Update the limited local curvature estimates based on the conditions
    D2w_lim[monotonic] = limited_curvature[monotonic]

    D2w[D2w == 0] = np.inf  # removes divide-by-zero issue; causes wF -> wS (i.e. piecewise constant) when D2w -> 0

    return wS + ((D2w_lim/D2w) * (wF_limit_L - wS)), wS + ((D2w_lim/D2w) * (wF_limit_R - wS))


# Calculate parabolic-interpolant and face-value limiters
def modppmLimiter(reconstructedValues, boundary, C=5/4):
    wS, wF, w1, w2 = reconstructedValues
    wFL, wFR = wF
    wLs, wRs = w1
    wL2s, wR2s = w2

    # Function for calculating the limited face-values
    def limitFaceValues(w_face, w_minusOne, w_cell, w_plusOne, w_plusTwo):
        # Initial check for local extrema
        local_extrema = (w_face - w_cell)*(w_plusOne - w_face) < 0

        if local_extrema.any():
            D2w = np.zeros(w_face.shape)

            # Approximation to the second derivatives
            D2w_L = w_minusOne - 2*w_cell + w_plusOne
            D2w_C = 3 * (w_cell - 2*w_face + w_plusOne)
            D2w_R = w_cell - 2*w_plusOne + w_plusTwo

            # Check for non-monotonic curvatures, provided they have the same signs
            non_monotonic = ((D2w_R - D2w_C)*(D2w_C - D2w_L) < 0) & (np.sign(D2w_L) == np.sign(D2w_R)) & (np.sign(D2w_C) == np.sign(D2w_R))

            # Determine the limited curvature with the sign of each element in the 'centre' array
            limited_curvature = np.sign(D2w_C) * np.minimum(np.abs(D2w_C), C * np.minimum(np.abs(D2w_L), np.abs(D2w_R)))

            # Update the limited local curvature estimates based on the conditions
            D2w[non_monotonic] = limited_curvature[non_monotonic]

            return (.5 * (w_cell+w_plusOne)) - (D2w/6)
        else:
            return w_face

    wF_limit_L = limitFaceValues(wFL, wL2s[:-1], wLs[:-1], wS, wRs[1:])
    wF_limit_R = limitFaceValues(wFR, wLs[:-1], wS, wRs[1:], wR2s[1:])

    # Calculate the limited parabolic interpolant values
    d_uL, d_uR = wS - wF_limit_L, wF_limit_R - wS
    
    # Check for local and cell extrema in cells
    local_extrema = (np.abs(d_uL) > 2*np.abs(d_uR)) | (np.abs(d_uR) > 2*np.abs(d_uL))
    cell_extrema = d_uL*d_uR < 0

    wL_limit_L2, wL_limit_R2 = fv.makeBoundary(wF_limit_L, boundary)
    wR_limit_L2, wR_limit_R2 = fv.makeBoundary(wF_limit_R, boundary)

    # Check for local extrema in interpolants
    d_wF_minmod = np.minimum(np.abs(wF_limit_L - wL_limit_L2[:-1]), np.abs(wR_limit_R2[1:] - wF_limit_R))
    d_wS_minmod = np.minimum(np.abs(wS - wLs[:-1]), np.abs(wRs[1:] - wS))
    interpolant_extrema = ((d_wF_minmod >= d_wS_minmod) & ((wF_limit_L - wL_limit_L2[:-1])*(wR_limit_R2[1:] - wF_limit_R) < 0))\
                        | ((d_wS_minmod >= d_wF_minmod) & ((wS - wLs[:-1])*(wRs[1:] - wS) < 0))
    
    # If there are extrema in either the cells or interpolants
    if local_extrema.any() or cell_extrema.any() or interpolant_extrema.any():
        D2w_lim = np.zeros(wS.shape)
        
        # Approximation to the second derivative
        D2w = 6 * (wF_limit_L - 2*wS + wF_limit_R)
        D2w_L = wL2s[:-1] - 2*wLs[:-1] + wS
        D2w_C = wLs[:-1] - 2*wS + wRs[1:]
        D2w_R = wS - 2*wRs[1:] + wR2s[1:]

        # Get the curvatures that have the same signs
        monotonic = (np.sign(D2w_L) == np.sign(D2w_R)) & (np.sign(D2w_C) == np.sign(D2w)) & (np.sign(D2w_C) == np.sign(D2w_R))

        # Determine the limited curvature with the sign of each element in the 'main' array
        limited_curvature = np.sign(D2w) * np.minimum(np.minimum(np.abs(D2w), C*np.abs(D2w_C)), C * np.minimum(np.abs(D2w_L), np.abs(D2w_R)))

        # Update the limited local curvature estimates based on the conditions
        D2w_lim[cell_extrema & monotonic] = limited_curvature[cell_extrema & monotonic]
        D2w_lim[interpolant_extrema & monotonic] = limited_curvature[interpolant_extrema & monotonic]

        D2w[D2w == 0] = np.inf  # removes divide by zero issue; causes wF -> wS (i.e. piecewise constant) when D2w -> 0

        # Further update if there is local extrema
        if local_extrema.any():
            d_uL_bar, d_uR_bar = np.copy(d_uL), np.copy(d_uR)
            d_uL_bar[np.abs(d_uL) > 2*np.abs(d_uR)] = 2*d_uR[np.abs(d_uL) > 2*np.abs(d_uR)]
            d_uR_bar[np.abs(d_uR) > 2*np.abs(d_uL)] = 2*d_uL[np.abs(d_uR) > 2*np.abs(d_uL)]
            return wS - d_uL_bar*(D2w_lim/D2w), wS + d_uR_bar*(D2w_lim/D2w)
        else:
            return wS - d_uL*(D2w_lim/D2w), wS + d_uR*(D2w_lim/D2w)
    else:
        return wF_limit_L, wF_limit_R
    

# Calculate minmod (slope) limiter. Returns an array of gradients for each parameter in each cell
def minmodLimiter(reconstructedValues, tube, C=.5):
    qLs, qRs = reconstructedValues
    a, b = np.diff(qLs, axis=0), np.diff(qRs, axis=0)
    arr = np.zeros(b.shape)

    mask = np.where((np.abs(a) < np.abs(b)) & (a*b > 0))
    arr[mask] = a[mask]

    mask = np.where((np.abs(a) >= np.abs(b)) & (a*b > 0))
    arr[mask] = b[mask]

    gradients = C * arr
    return tube - gradients, tube + gradients


# Calculate the van Leer/harmonic parameter. Returns an array of gradients for each parameter in each cell
def harmonicLimiter(reconstructedValues):
    qLs, qRs = reconstructedValues
    r = np.nan_to_num((qLs[1:] - qLs[:-1])/(qRs[1:] - qRs[:-1]))
    return (r + np.abs(r))/(1 + np.abs(r))


# Calculate the ospre parameter. Returns an array of gradients for each parameter in each cell
def ospreLimiter(reconstructedValues):
    qLs, qRs = reconstructedValues
    r = np.nan_to_num((qLs[1:] - qLs[:-1])/(qRs[1:] - qRs[:-1]))
    return 1.5 * ((r**2 + r)/(r**2 + r + 1))


# Calculate the van Albada parameter. Returns an array of gradients for each parameter in each cell
def vanAlbadaLimiter(reconstructedValues):
    qLs, qRs = reconstructedValues
    r = np.nan_to_num((qLs[1:] - qLs[:-1])/(qRs[1:] - qRs[:-1]))
    return (r**2 + r)/(r**2 + 1)