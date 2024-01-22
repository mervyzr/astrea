import numpy as np

##############################################################################


# Calculate minmod parameter. Returns an array of gradients for each parameter in each cell
def minmod(qLs, qRs):
    a, b = qLs[1:] - qLs[:-1], qRs[1:] - qRs[:-1]
    arr = np.zeros(b.shape)

    mask = np.where((np.abs(a) < np.abs(b)) & (a*b > 0))
    arr[mask] = a[mask]

    mask = np.where((np.abs(a) >= np.abs(b)) & (a*b > 0))
    arr[mask] = b[mask]

    return .5*arr


# Calculate the van Leer/harmonic parameter. Returns an array of gradients for each parameter in each cell
def harmonic(qLs, qRs):
    r = np.nan_to_num((qLs[1:] - qLs[:-1])/(qRs[1:] - qRs[:-1]))
    return (r + np.abs(r))/(1 + np.abs(r))


# Calculate the ospre parameter. Returns an array of gradients for each parameter in each cell
def ospre(qLs, qRs):
    r = np.nan_to_num((qLs[1:] - qLs[:-1])/(qRs[1:] - qRs[:-1]))
    return 1.5 * ((r**2 + r)/(r**2 + r + 1))


# Calculate the van Albada parameter. Returns an array of gradients for each parameter in each cell
def vanAlbada(qLs, qRs):
    r = np.nan_to_num((qLs[1:] - qLs[:-1])/(qRs[1:] - qRs[:-1]))
    return (r**2 + r)/(r**2 + 1)


# Calculate the limited face-values
def faceValueLimit(wS, wF, wLs, wRs, wL2s, wR2s):
    C = 5/4

    d2_wF_L = wLs[:-1] - 2*wS + wRs[1:]
    d2_wF_C = 3 * (wS - 2*wF + wRs[1:])
    d2_wF_R = wS - 2*wRs[1:] + wR2s[1:]

    if (d2_wF_R - d2_wF_C) * (d2_wF_C - d2_wF_L) < 0:
        signage = np.ones((len(d2_wF_C), len(d2_wF_C[0])))
        signage[np.where(d2_wF_C < 0)] = -1
        d2_wF = signage * np.minimum(C*d2_wF_C, np.minimum(C*d2_wF_L, C*d2_wF_R))
    else:
        d2_wF = 0
    return (.5 * (wS + wRs[1:])) - (1/6 * d2_wF)


# Calculate the limited parabolic interpolant values
def parabolicInterpolantLimit(wS, wF, wLs, wRs, wL2s, wR2s):
    C = 5/4

    d2_wF_L = wLs[:-1] - 2*wS + wRs[1:]
    d2_wF_C = 3 * (wS - 2*wF + wRs[1:])
    d2_wF_R = wS - 2*wRs[1:] + wR2s[1:]

    if (d2_wF_R - d2_wF_C) * (d2_wF_C - d2_wF_L) < 0:
        signage = np.ones((len(d2_wF_C), len(d2_wF_C[0])))
        signage[np.where(d2_wF_C < 0)] = -1
        d2_wF = signage * np.minimum(C*d2_wF_C, np.minimum(C*d2_wF_L, C*d2_wF_R))
    else:
        d2_wF = 0
    return (.5 * (wS + wRs[1:])) - (1/6 * d2_wF)