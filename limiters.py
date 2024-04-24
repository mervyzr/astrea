import numpy as np

from functions import fv

##############################################################################

from reconstruct import modified

# Apply limiters based on the reconstruction method
def applyLimiter(extrapolatedValues, solver):
    # Apply the limiter for parabolic or XPPM
    if solver in ["ppm", "parabolic", "p"]:
        wS, wF, w, w2 = extrapolatedValues
        if modified:
            return wF
        else:
            return faceValueLimiter(wF, w[:-2], wS, w[2:], w2[4:])

    # Apply the minmod limiter
    elif solver in ["plm", "linear", "l"]:
        return minmodLimiter(extrapolatedValues)

    # Do not apply any limiters
    else:
        return extrapolatedValues


# Function for limiting the face-values for PPM [Colella et al., 2011, p. 26; Peterson & Hammett, 2013, p. B585]
def faceValueLimiter(w_face, w_minusOne, w_cell, w_plusOne, w_plusTwo, C=5/4):
    # Initial check for local extrema (eq. 84)
    local_extrema = (w_face - w_cell)*(w_plusOne - w_face) < 0

    if local_extrema.any():
        D2w = np.zeros(w_face.shape)

        # Approximation to the second derivatives (eq. 85)
        D2w_L = w_minusOne - 2*w_cell + w_plusOne
        D2w_C = 3 * (w_cell - 2*w_face + w_plusOne)
        D2w_R = w_cell - 2*w_plusOne + w_plusTwo

        # Get the curvatures that have the same signs
        non_monotonic = (np.sign(D2w_L) == np.sign(D2w_R)) & (np.sign(D2w_C) == np.sign(D2w_R))
        #advanced_non_monotonic = ((D2w_R - D2w_C)*(D2w_C - D2w_L) < 0) & (np.sign(D2w_L) == np.sign(D2w_R)) & (np.sign(D2w_C) == np.sign(D2w_R))

        # Determine the limited curvature with the sign of each element in the 'centre' array (eq. 87)
        limited_curvature = np.sign(D2w_C) * np.minimum(np.abs(D2w_C), np.minimum(np.abs(C*D2w_L), np.abs(C*D2w_R)))

        # Update the limited local curvature estimates based on the conditions
        D2w[non_monotonic] = limited_curvature[non_monotonic]

        return .5*(w_cell+w_plusOne) - D2w/6
    else:
        return w_face


# Calculate minmod (slope) limiter [Derigs et al., 2017]. Returns an array of gradients for each parameter in each cell
def minmodLimiter(extrapolatedValues):
    a, b = np.diff(extrapolatedValues[:-1], axis=0), np.diff(extrapolatedValues[1:], axis=0)
    arr = np.zeros(b.shape)

    # (eq. 4.17)
    mask = np.where((np.abs(a) < np.abs(b)) & (a*b > 0))
    arr[mask] = a[mask]
    mask = np.where((np.abs(a) >= np.abs(b)) & (a*b > 0))
    arr[mask] = b[mask]
    return arr


# Calculate the van Leer/harmonic parameter. Returns an array of gradients for each parameter in each cell
def harmonicLimiter(extrapolatedValues):
    r = np.diff(extrapolatedValues[:-1])/np.diff(extrapolatedValues[1:])
    return (r + np.abs(r))/(1 + np.abs(r))


# Calculate the ospre parameter. Returns an array of gradients for each parameter in each cell
def ospreLimiter(extrapolatedValues):
    r = np.diff(extrapolatedValues[:-1])/np.diff(extrapolatedValues[1:])
    return 1.5 * ((r**2 + r)/(r**2 + r + 1))


# Calculate the van Albada parameter. Returns an array of gradients for each parameter in each cell
def vanAlbadaLimiter(extrapolatedValues):
    r = np.diff(extrapolatedValues[:-1])/np.diff(extrapolatedValues[1:])
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