import numpy as np

from functions import fv
from num_methods import limiters

##############################################################################
# Fourth-order upwind constrained transport algorithm for MHD [Felker & Stone, 2018]
##############################################################################

# Compute the corner electric fields wrt to cell centre; gives 4-fold values for each corner
def reconstruct_corner(_wF, next_ax, boundary, method="ppm", **kwargs):
    wF = np.copy(_wF.transpose(next_ax))

    wF_pad2 = fv.add_boundary(wF, boundary, 2)
    wF_pad1 = np.copy(wF_pad2[1:-1])

    if method == "ppm":
        try:
            author = kwargs['author']
        except KeyError:
            author = "mc"

        """Extrapolate the face averages to the top corners (upwards) [McCorquodale & Colella, 2011, eq. 17; Colella et al., 2011, eq. 67]
        |                w(i-1/2)            w(i+1/2)               |
        |-------------------|-------------------|-------------------|
        |                  ^|                  ^|                  ^|
        |                  ||                  ||                  ||
        |                  ||                  ||                  ||
        |  o (i-1,j)     -->|  o (i,j)       -->|  o (i+1,j)     -->|
        """
        wU = 7/12 * (wF + wF_pad1[2:]) - 1/12 * (wF_pad1[:-2] + wF_pad2[4:])

        if "x" in author or "ph" in author or author in ["peterson", "hammett"]:
            """Extrapolate the face averages to both corners (upwards & downwards)
            |                w(i-1/2)            w(i+1/2)               |
            |-------------------|-------------------|-------------------|
            |                  ^|                  ^|                  ^|
            |                  ||                  ||                  ||
            |                  ||                  ||                  ||
            |  o (i-1,j)     -->|  o (i,j)       -->|  o (i+1,j)     -->|
            |                  ||                  ||                  ||
            |                  ||                  ||                  ||
            |                  v|                  v|                  v|
            |-------------------|-------------------|-------------------|
            """
            wD = 7/12 * (wF_pad1[:-2] + wF) - 1/12 * (wF_pad2[:-4] + wF_pad1[2:])

            # Limit interface values [Peterson & Hammett, 2008, eq. 3.33-3.34]
            limited_wUs = limiters.interface_limiter(wD, wF_pad2[:-4], wF_pad1[:-2], wF, wF_pad1[2:]), limiters.interface_limiter(wU, wF_pad1[:-2], wF, wF_pad1[2:], wF_pad2[4:])
            wU_pad2 = np.zeros_like(fv.add_boundary(wU, boundary, 2))
        else:
            if author == "c" or author == "collela":
                # Limit interface values [Colella et al., 2011, p. 25-26]
                wU = limiters.interface_limiter(wU, wF_pad1[:-2], wF, wF_pad1[2:], wF_pad2[4:])

            # Define the top and bottom parabolic interpolants
            wU_pad2 = fv.add_boundary(wU, boundary, 2)
            limited_wUs = np.copy(wU_pad2[1:-3]), np.copy(wU_pad2[2:-2])

        """Reconstruct the limited interpolants from the interface values. Returns the face averages in the form of w+(y) & w-(y) when considering x-axis, and w+(x) & w-(x) when considering y-axis
        |                w(i-1/2)            w(i+1/2)               |
        |  o (i-1,j+1)      |  o (i,j+1)        |  o (i+1,j+1)      |
        |                   |                   |                   |
        |                   |                   |                   |
        |                 w+(y)               w+(y)               w+(y)
        |                   ^                   ^                   ^
        |-------------------|-------------------|-------------------|
        |                   v                   v                   v
        |                 w-(y)               w-(y)               w-(y)
        |                   |                   |                   |
        |                   |                   |                   |
        |  o (i-1,j)     -->|  o (i,j)       -->|  o (i+1,j)     -->|
        """
        return limiters.interpolant_limiter(wF, wF_pad1, wF_pad2, wU_pad2, author, boundary, *limited_wUs)











    """wL, wR = np.copy(LR[0].transpose(next_ax)), np.copy(LR[1].transpose(next_ax))

    wL2, wR2 = fv.add_boundary(wL, boundary, 2), fv.add_boundary(wR, boundary, 2)
    wL1, wR1 = np.copy(wL2[1:-1]), np.copy(wR2[1:-1])

    # Extrapolate the face-averages to the top (up) and bottom (down) corners
    if method == "ppm":
        author = "mc"

        wL_U, wR_U = 7/12 * (wL + wL1[2:]) - 1/12 * (wL1[:-2] + wL2[4:]), 7/12 * (wR + wR1[2:]) - 1/12 * (wR1[:-2] + wR2[4:])

        if "c" in author:
            # Collela method
            if author == "c" or author == "collela":
                wL_U, wR_U = limiters.interface_limiter(wL_U, wL1[:-2], wL, wL1[2:], wL2[4:]), limiters.interface_limiter(wR_U, wR1[:-2], wR, wR1[2:], wR2[4:])

            wL_U_pad2, wR_U_pad2 = fv.add_boundary(wL_U, boundary, 2), fv.add_boundary(wR_U, boundary, 2)

            limited_wLs = np.copy(wL_U_pad2[1:-3]), np.copy(wL_U_pad2[2:-2])
            limited_wRs = np.copy(wR_U_pad2[1:-3]), np.copy(wR_U_pad2[2:-2])

        # Peterson & Hammett method
        else:
            wL_D, wR_D = 7/12 * (wL1[:-2] + wL) - 1/12 * (wL2[:-4] + wL1[2:]), 7/12 * (wR1[:-2] + wR) - 1/12 * (wR2[:-4] + wR1[2:])

            wL_U_pad2, wR_U_pad2 = np.zeros_like(fv.add_boundary(wL_U, boundary, 2)), np.zeros_like(fv.add_boundary(wR_U, boundary, 2))

            limited_wLs = limiters.interface_limiter(wL_D, wL2[:-4], wL1[:-2], wL, wL1[2:]), limiters.interface_limiter(wL_U, wL1[:-2], wL, wL1[2:], wL2[4:])
            limited_wRs = limiters.interface_limiter(wR_D, wR2[:-4], wR1[:-2], wR, wR1[2:]), limiters.interface_limiter(wR_U, wR1[:-2], wR, wR1[2:], wR2[4:])

        SW, NW = limiters.interpolant_limiter(wL, wL1, wL2, wL_U_pad2, author, boundary, *limited_wLs)
        SE, NE = limiters.interpolant_limiter(wR, wR1, wR2, wR_U_pad2, author, boundary, *limited_wRs)



    return [wL_D, wL_U], [wR_D, wR_U]"""