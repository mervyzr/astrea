import sys

import numpy as np

import functions as fn

##############################################################################

# Apply limiters based on the reconstruction method
def applyLimiter(shockTube, reconstructedValues, tube):
    if shockTube.solver in ["ppm", "parabolic", "p"]:
        # Apply the face-values and parabolic-interpolant limiter
        return parabolicLimiter(reconstructedValues, shockTube.boundary)
    elif shockTube.solver in ["plm", "linear", "l"]:
        # Apply the minmod limiter
        return minmodLimiter(reconstructedValues, tube)
    else:
        # Do not apply any limiters
        return reconstructedValues


# Calculate parabolic-interpolant and face-value limters
def parabolicLimiter(reconstructedValues, boundary, C=5/4):
    wS, wF, wLs, wRs, wL2s, wR2s = reconstructedValues
    
    # Calculate the limited face-values
    local_extrema = (wF - wS)*(wRs[1:] - wF) < 0  # Initial check for local extrema

    if local_extrema.any():
        d2_wF = np.zeros((len(wF), len(wF[0])))

        d2_wF_L = wLs[:-1] - 2*wS + wRs[1:]
        d2_wF_C = 3 * (wS - 2*wF + wRs[1:])
        d2_wF_R = wS - 2*wRs[1:] + wR2s[1:]

        # Get the curvatures that are not monotonic
        non_monotonic = (d2_wF_R - d2_wF_C)*(d2_wF_C - d2_wF_L) < 0

        # Get sign of each element in the 'centre' array, provided the signs of all arrays are the same
        signage = np.ones((len(d2_wF_C), len(d2_wF_C[0])))
        if ((d2_wF_L < 0) == (d2_wF_R < 0)).all() and ((d2_wF_C < 0) == (d2_wF_R < 0)).all():
            signage[d2_wF_C < 0] = -1

        # Determine the limited curvature
        limited_curvature = signage * C * np.minimum(d2_wF_C, np.minimum(d2_wF_L, d2_wF_R))

        # Update the limited local curvature estimates based on the conditions
        d2_wF[non_monotonic] = limited_curvature[non_monotonic]

        wF_limit = (.5 * (wS + wRs[1:])) - (1/6 * d2_wF)
    else:
        wF_limit = wF
    
    # Calculate the limited parabolic interpolant values
    wF_limit_L, wF_limit_R = fn.makeBoundary(wF_limit, boundary)
    if boundary == "periodic":
        wF_limit_L2, wF_limit_R2 = np.concatenate(([wF_limit_L[-2]],wF_limit_L))[:-1], np.concatenate((wF_limit_R,[wF_limit_R[1]]))[1:]
    else:
        wF_limit_L2, wF_limit_R2 = np.concatenate(([wF_limit_L[0]],wF_limit_L))[:-1], np.concatenate((wF_limit_R,[wF_limit_R[-1]]))[1:]

    # First determine if there is a local extremum in cells
    d_uL, d_uR = wS - wF_limit_L[:-1], wF_limit - wS
    cell_extrema = (d_uL*d_uR < 0) | (np.abs(d_uL) > 2*np.abs(d_uR)) | (np.abs(d_uR) > 2*np.abs(d_uL))

    # Next determine if there is a local extremum in the parabolic interpolants
    d_wF_minmod = np.minimum(np.abs(wF_limit_L[:-1] - wF_limit_L2[:-1]), np.abs(wF_limit_R[1:] - wF_limit))
    d_wS_minmod = np.minimum(np.abs(wS - wLs[:-1]), np.abs(wRs[1:] - wS))
    interpolant_extrema = ((d_wF_minmod >= d_wS_minmod) & (d_wF_minmod*d_wS_minmod < 0)) | ((d_wS_minmod >= d_wF_minmod) & (d_wF_minmod*d_wS_minmod < 0))

    # If there are local extrema in either the cells or interpolants
    if cell_extrema.any() or interpolant_extrema.any():
        d2_wS_bar = np.zeros((len(wS), len(wS[0])))

        d2_wS = -2 * (6*wS - 3*(wF_limit + wF_limit_L[:-1]))
        d2_wS_L = wL2s[:-1] - 2*wLs[:-1] + wS
        d2_wS_C = wLs[:-1] - 2*wS + wRs[1:]
        d2_wS_R = wS - 2*wRs[1:] + wR2s[1:]

        # Get sign of each element in the cells array, provided the signs of all arrays are the same
        signage = np.ones((len(d2_wS), len(d2_wS[0])))
        if ((d2_wS < 0) == (d2_wS_L < 0)).all() and ((d2_wS < 0) == (d2_wS_C < 0)).all() and ((d2_wS < 0) == (d2_wS_R < 0)).all():
            signage[d2_wS < 0] = -1
        
        # Determine the limited curvature
        limited_curvature = signage * C * np.minimum(np.minimum(d2_wS, d2_wS_C), np.minimum(d2_wS_L, d2_wS_R))

        # Update the limited local curvature estimates based on the conditions
        d2_wS_bar[interpolant_extrema] = limited_curvature[interpolant_extrema]  # local extrema in parabolic interpolants
        d2_wS_bar[cell_extrema] = limited_curvature[cell_extrema]  # local extrema in cells

        d_uL_bar, d_uR_bar = np.copy(d_uL), np.copy(d_uR)
        d_uL_bar[np.abs(d_uL) > 2*np.abs(d_uR)] = 2*d_uR[np.abs(d_uL) > 2*np.abs(d_uR)]
        d_uR_bar[np.abs(d_uR) > 2*np.abs(d_uL)] = 2*d_uL[np.abs(d_uR) > 2*np.abs(d_uL)]

        d2_wS[d2_wS == 0] = sys.float_info.epsilon

        return wS - d_uL_bar*(d2_wS_bar/d2_wS), wS + d_uR_bar*(d2_wS_bar/d2_wS)
    else:
        return wF_limit_L[:-1], wF_limit_R[1:]
    

# Calculate minmod limiter. Returns an array of gradients for each parameter in each cell
def minmodLimiter(reconstructedValues, tube, C=.5):
    qLs, qRs = reconstructedValues
    a, b = np.diff(qLs, axis=0), np.diff(qRs, axis=0)
    arr = np.zeros(b.shape)

    mask = np.where((np.abs(a) < np.abs(b)) & (a*b > 0))
    arr[mask] = a[mask]

    mask = np.where((np.abs(a) >= np.abs(b)) & (a*b > 0))
    arr[mask] = b[mask]

    gradients = C * arr
    return np.copy(tube) - gradients, np.copy(tube) + gradients


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