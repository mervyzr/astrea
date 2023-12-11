import numpy as np


# Calculate minmod parameter. Returns an array of gradients for each parameter in each cell
def minmod(qLs, qRs):
    a, b = qLs[1:] - qLs[:-1], qRs[1:] - qRs[:-1]
    arr = np.zeros(b.shape)

    mask = np.where((np.abs(a) < np.abs(b)) & (a*b > 0))
    arr[mask] = a[mask]

    mask = np.where((np.abs(a) >= np.abs(b)) & (a*b > 0))
    arr[mask] = b[mask]

    return arr