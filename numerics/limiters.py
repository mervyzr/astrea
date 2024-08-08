import numpy as np

from functions import fv

##############################################################################
# Limiter functions for the interface and cell values
##############################################################################

# Function for limiting the face-values for PPM [Colella et al., 2011, p. 26]
def interfaceLimiter(w_face, w_minusOne, w_cell, w_plusOne, w_plusTwo, C):
    # Initial check for local extrema (eq. 84)
    local_extrema = (w_face - w_cell)*(w_plusOne - w_face) < 0

    if local_extrema.any():
        D2w = np.zeros_like(w_face)

        # Approximation to the second derivatives (eq. 85)
        D2w_L = w_minusOne - 2*w_cell + w_plusOne
        D2w_C = 3 * (w_cell - 2*w_face + w_plusOne)
        D2w_R = w_cell - 2*w_plusOne + w_plusTwo

        # Get the curvatures that have the same signs
        non_monotonic = (np.sign(D2w_L) == np.sign(D2w_R)) & (np.sign(D2w_C) == np.sign(D2w_R)) & (np.sign(D2w_C) == np.sign(D2w_L))
        #advanced_non_monotonic = ((D2w_R - D2w_C)*(D2w_C - D2w_L) < 0) & (np.sign(D2w_L) == np.sign(D2w_R)) & (np.sign(D2w_C) == np.sign(D2w_R))

        # Determine the limited curvature with the sign of each element in the 'centre' array (eq. 87)
        limited_curvature = np.sign(D2w_C) * np.minimum(np.abs(D2w_C), np.minimum(np.abs(C*D2w_L), np.abs(C*D2w_R)))

        # Update the limited local curvature estimates based on the conditions
        D2w[non_monotonic] = limited_curvature[non_monotonic]

        return .5*(w_cell+w_plusOne) - D2w/6
    else:
        return w_face


# Calculate minmod (slope) limiter [Derigs et al., 2017]. Returns an array of gradients for each parameter in each cell
def minmodLimiter(w):
    a, b = np.diff(w[:-1], axis=0), np.diff(w[1:], axis=0)
    arr = np.zeros_like(b)

    # (eq. 4.17)
    mask = np.where((np.abs(a) < np.abs(b)) & (a*b > 0))
    arr[mask] = a[mask]
    mask = np.where((np.abs(a) >= np.abs(b)) & (a*b > 0))
    arr[mask] = b[mask]
    return arr


# Calculate the van Leer/harmonic parameter [van Leer, 1974]
def vanLeerLimiter(w):
    r = fv.divide(np.diff(w[:-1], axis=0), np.diff(w[1:], axis=0))
    return (r + np.abs(r))/(1 + np.abs(r)) * np.diff(w[1:], axis=0)


# Calculate the Ospre parameter [Waterson & Deconinck, 1995]
def ospreLimiter(w):
    r = fv.divide(np.diff(w[:-1], axis=0), np.diff(w[1:], axis=0))
    return 1.5 * ((r**2 + r)/(r**2 + r + 1)) * np.diff(w[1:], axis=0)


# Calculate the van Albada "1" parameter [van Albada, 1982]
def vanAlbadaOneLimiter(w):
    r = fv.divide(np.diff(w[:-1], axis=0), np.diff(w[1:], axis=0))
    return (r**2 + r)/(r**2 + 1) * np.diff(w[1:], axis=0)


# Calculate the Koren parameter [Vreugdenhil & Koren, 1993]
def korenLimiter(w):
    r = fv.divide(np.diff(w[:-1], axis=0), np.diff(w[1:], axis=0))
    return np.maximum(np.zeros_like(r), np.minimum(np.minimum(2*r, (2+r)/3), np.full_like(r,2))) * np.diff(w[1:], axis=0)


# Calculate the superbee parameter [Roe, 1986]
def superbeeLimiter(w):
    r = fv.divide(np.diff(w[:-1], axis=0), np.diff(w[1:], axis=0))
    return np.maximum(np.zeros_like(r), np.maximum(np.minimum(2*r, np.ones_like(r)), np.minimum(r, np.full_like(r,2)))) * np.diff(w[1:], axis=0)