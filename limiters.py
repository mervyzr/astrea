import numpy as np

from functions import fv

##############################################################################

# Apply limiters based on the reconstruction method
def applyLimiter(reconstructedValues, solver):
    if solver in ["ppm", "parabolic", "p"]:
        # Apply the limiter for parabolic or XPPM
        return xppmLimiter(reconstructedValues)
    elif solver in ["plm", "linear", "l"]:
        # Apply the minmod limiter
        return minmodLimiter(reconstructedValues)
    else:
        # Do not apply any limiters
        return reconstructedValues


# Calculate parabolic-interpolant and face-value limiters; preserves behaviour at smooth-extrema
def xppmLimiter(reconstructedValues, C=5/4):
    wS, wF, w, w2 = reconstructedValues
    wFL, wFR = wF

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
            non_monotonic = (np.sign(D2w_L) == np.sign(D2w_R)) & (np.sign(D2w_C) == np.sign(D2w_R))
            
            # Determine the limited curvature with the sign of each element in the 'centre' array
            limited_curvature = np.sign(D2w_C) * np.minimum(np.abs(D2w_C), C * np.minimum(np.abs(D2w_L), np.abs(D2w_R)))

            # Update the limited local curvature estimates based on the conditions
            D2w[non_monotonic] = limited_curvature[non_monotonic]

            return (.5 * (w_cell+w_plusOne)) - (D2w/6)
        else:
            return w_face

    # Limited face-values
    wF_limit_L = limitFaceValues(wFL, w2[:-4], w[:-2], wS, w[2:])
    wF_limit_R = limitFaceValues(wFR, w[:-2], wS, w[2:], w2[4:])

    # Check for local extrema away from smooth extrema
    local_extrema = ((wFR - wS)*(wS - wFL) <= 0) | ((w[:-2] - wS)*(wS - w[2:]) <= 0)
    wF_limit_L[local_extrema] = wS[local_extrema]
    wF_limit_R[local_extrema] = wS[local_extrema]

    # Calculate the limited smooth extrema
    D2w_lim = np.zeros(wS.shape)

    # Approximation to the second derivatives
    D2w = 6 * (wFL - 2*wS + wFR)
    D2w_L = w2[:-4] - 2*w[:-2] + wS
    D2w_C = w[:-2] - 2*wS + w[2:]
    D2w_R = wS - 2*w[2:] + w2[4:]

    # Get the curvatures that have the same signs
    non_monotonic = (np.sign(D2w_L) == np.sign(D2w_R)) & (np.sign(D2w_C) == np.sign(D2w)) & (np.sign(D2w_C) == np.sign(D2w_R))

    # Determine the limited curvature with the sign of each element in the 'main' array
    limited_curvature = np.sign(D2w) * np.minimum(np.minimum(np.abs(D2w), C*np.abs(D2w_C)), C * np.minimum(np.abs(D2w_L), np.abs(D2w_R)))

    # Update the limited local curvature estimates based on the conditions
    D2w_lim[non_monotonic] = limited_curvature[non_monotonic]

    D2w[D2w == 0] = np.inf  # removes divide-by-zero issue; causes wF -> wS (i.e. piecewise constant) when D2w -> 0

    return [wS + ((D2w_lim/D2w) * (wF_limit_L - wS)), wS + ((D2w_lim/D2w) * (wF_limit_R - wS))]


# Calculate parabolic-interpolant and face-value limiters
def modppmLimiter(reconstructedValues, C=5/4):
    wS, wF, w, w2 = reconstructedValues
    wFL, wFR = wF

    if (w[0] != wS[0]) and (w[-1] != wS[-1]):
        boundary = "wrap"
    else:
        boundary = "edge"

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

    # Limited face-values
    wF_limit_L = limitFaceValues(wFL, w2[:-4], w[:-2], wS, w[2:])
    wF_limit_R = limitFaceValues(wFR, w[:-2], wS, w[2:], w2[4:])

    # Calculate the limited parabolic interpolant values
    d_uL, d_uR = wS - wF_limit_L, wF_limit_R - wS
    
    # Check for local and cell extrema in cells
    local_extrema = (np.abs(d_uL) > 2*np.abs(d_uR)) | (np.abs(d_uR) > 2*np.abs(d_uL))
    cell_extrema = d_uL*d_uR < 0

    wF_limit_L2 = fv.makeBoundary(wF_limit_L, boundary)
    wF_limit_R2 = fv.makeBoundary(wF_limit_R, boundary)

    # Check for local extrema in interpolants
    d_wF_minmod = np.minimum(np.abs(wF_limit_L - wF_limit_L2[:-2]), np.abs(wF_limit_R2[2:] - wF_limit_R))
    d_wS_minmod = np.minimum(np.abs(wS - w[:-2]), np.abs(w[2:] - wS))
    interpolant_extrema = ((d_wF_minmod >= d_wS_minmod) & ((wF_limit_L - wF_limit_L2[:-2])*(wF_limit_R2[2:] - wF_limit_R) < 0)) | ((d_wS_minmod >= d_wF_minmod) & ((wS - w[:-2])*(w[2:] - wS) < 0))

    # If there are extrema in either the cells or interpolants
    if local_extrema.any() or cell_extrema.any() or interpolant_extrema.any():
        D2w_lim = np.zeros(wS.shape)
        
        # Approximation to the second derivative
        D2w = 6 * (wF_limit_L - 2*wS + wF_limit_R)
        D2w_L = w2[:-4] - 2*w[:-2] + wS
        D2w_C = w[:-2] - 2*wS + w[2:]
        D2w_R = wS - 2*w[2:] + w2[4:]

        # Get the curvatures that have the same signs
        non_monotonic = (np.sign(D2w_L) == np.sign(D2w_R)) & (np.sign(D2w_C) == np.sign(D2w)) & (np.sign(D2w_C) == np.sign(D2w_R))

        # Determine the limited curvature with the sign of each element in the 'main' array
        limited_curvature = np.sign(D2w) * np.minimum(np.minimum(np.abs(D2w), C*np.abs(D2w_C)), C * np.minimum(np.abs(D2w_L), np.abs(D2w_R)))

        # Update the limited local curvature estimates based on the conditions
        D2w_lim[cell_extrema & non_monotonic] = limited_curvature[cell_extrema & non_monotonic]
        D2w_lim[interpolant_extrema & non_monotonic] = limited_curvature[interpolant_extrema & non_monotonic]

        D2w[D2w == 0] = np.inf  # removes divide by zero issue; causes wF -> wS (i.e. piecewise constant) when D2w -> 0

        # Further update if there is local extrema
        if local_extrema.any():
            d_uL_bar, d_uR_bar = np.copy(d_uL), np.copy(d_uR)
            d_uL_bar[np.abs(d_uL) > 2*np.abs(d_uR)] = 2*d_uR[np.abs(d_uL) > 2*np.abs(d_uR)]
            d_uR_bar[np.abs(d_uR) > 2*np.abs(d_uL)] = 2*d_uL[np.abs(d_uR) > 2*np.abs(d_uL)]
            return [wS - d_uL_bar*(D2w_lim/D2w), wS + d_uR_bar*(D2w_lim/D2w)]
        else:
            return [wS - d_uL*(D2w_lim/D2w), wS + d_uR*(D2w_lim/D2w)]
    else:
        return [wF_limit_L, wF_limit_R]
    

# Calculate minmod (slope) limiter. Returns an array of gradients for each parameter in each cell
def minmodLimiter(reconstructedValues, C=.5):
    tube = reconstructedValues[1:-1]

    a, b = np.diff(reconstructedValues[:-1], axis=0), np.diff(reconstructedValues[1:], axis=0)
    arr = np.zeros(b.shape)

    mask = np.where((np.abs(a) < np.abs(b)) & (a*b > 0))
    arr[mask] = a[mask]

    mask = np.where((np.abs(a) >= np.abs(b)) & (a*b > 0))
    arr[mask] = b[mask]

    gradients = C * arr
    return [tube - gradients, tube + gradients]


# Calculate the van Leer/harmonic parameter. Returns an array of gradients for each parameter in each cell
def harmonicLimiter(reconstructedValues):
    r = np.diff(reconstructedValues[:-1])/np.diff(reconstructedValues[1:])
    return (r + np.abs(r))/(1 + np.abs(r))


# Calculate the ospre parameter. Returns an array of gradients for each parameter in each cell
def ospreLimiter(reconstructedValues):
    r = np.diff(reconstructedValues[:-1])/np.diff(reconstructedValues[1:])
    return 1.5 * ((r**2 + r)/(r**2 + r + 1))


# Calculate the van Albada parameter. Returns an array of gradients for each parameter in each cell
def vanAlbadaLimiter(reconstructedValues):
    r = np.diff(reconstructedValues[:-1])/np.diff(reconstructedValues[1:])
    return (r**2 + r)/(r**2 + 1)


# Function that returns the coefficient of the slope flattener
def getSlopeCoeff(tube, boundary, g, slope_determinants=[.75, .85, .33]):
    z0, z1, delta = slope_determinants
    domain = fv.pointConvertConservative(tube, g)
    arr, chi = np.ones(len(domain)), np.ones(len(domain))

    w = fv.makeBoundary(domain, boundary)
    w2 = fv.makeBoundary(domain, boundary, 2)

    z = np.abs((w[2:][:,4] - w[:-2][:,4]) / (w2[4:][:,4] - w2[:-4][:,4]))  # define the linear function
    eta = np.minimum(np.ones(len(z)), np.maximum(np.zeros(len(z)), 1 - ((z-z0)/(z1-z0))))  # limit the range between 0 and 1
    criteria = (w[:-2][:,1] - w[2:][:,1] > 0) & (np.abs(w[2:][:,4] - w[:-2][:,4])/np.minimum(w[2:][:,4], w[:-2][:,4]) > delta)

    chi[criteria] = eta[criteria]
    chiB = fv.makeBoundary(chi, boundary)

    signage = np.sign(w[2:][:,4] - w[:-2][:,4])
    arr[signage < 0] = np.minimum(chi, chiB[2:])[signage < 0]
    arr[signage > 0] = np.minimum(chi, chiB[:-2])[signage > 0]

    return np.tile(np.reshape(arr, (len(arr),1)), (1,5))