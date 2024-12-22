import numpy as np

from functions import fv, constructor
from num_methods import limiters

##############################################################################
# Fourth-order upwind constrained transport algorithm for MHD [Felker & Stone, 2018]
##############################################################################

# Reconstruct the transverse values for each face average
def reconstruct_transverse(_wF, sim_variables, method="ppm", author="mc"):
    alt_axes, boundary = sim_variables.permutations[-1], sim_variables.boundary

    # Compute with orthogonal axes
    wF = np.copy(_wF.transpose(alt_axes))

    wF_pad2 = fv.add_boundary(wF, boundary, 2)
    wF_pad1 = np.copy(wF_pad2[1:-1])

    if method == "ppm":
        """Extrapolate the face averages to the top corners (upwards) [McCorquodale & Colella, 2011, eq. 17; Colella et al., 2011, eq. 67]
        |                w(i-1/2)            w(i+1/2)               |
        |-------------------|-------------------|-------------------|
        |           w_U(i-1/2,j+1/2)    w_U(i+1/2,j+1/2)            |
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
            |           w_U(i-1/2,j+1/2)    w_U(i+1/2,j+1/2)            |
            |                  ^|                  ^|                  ^|
            |                  ||                  ||                  ||
            |                  ||                  ||                  ||
            |  o (i-1,j)     -->|  o (i,j)       -->|  o (i+1,j)     -->|
            |                  ||                  ||                  ||
            |                  ||                  ||                  ||
            |                  v|                  v|                  v|
            |           w_D(i-1/2,j-1/2)    w_D(i+1/2,j-1/2)            |
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
        |           w_D(i-1/2,j+1/2)    w_D(i+1/2,j+1/2)            |
        |                 w+(y)               w+(y)               w+(y)
        |                   ^                   ^                   ^
        |-------------------|-------------------|-------------------|
        |                   v                   v                   v
        |                 w-(y)               w-(y)               w-(y)
        |           w_U(i-1/2,j+1/2)    w_U(i+1/2,j+1/2)            |
        |                   |                   |                   |
        |                   |                   |                   |
        |  o (i-1,j)     -->|  o (i,j)       -->|  o (i+1,j)     -->|
        """
        wD, wU = limiters.interpolant_limiter(wF, wF_pad1, wF_pad2, wU_pad2, author, boundary, *limited_wUs)
    
    # 5th-order
    elif method == "weno":
        """Extrapolate the face averages to both corners (upwards & downwards)
        |                w(i-1/2)            w(i+1/2)               |
        |-------------------|-------------------|-------------------|
        |           w_U(i-1/2,j+1/2)    w_U(i+1/2,j+1/2)            |
        |                  ^|                  ^|                  ^|
        |                  ||                  ||                  ||
        |                  ||                  ||                  ||
        |  o (i-1,j)     -->|  o (i,j)       -->|  o (i+1,j)     -->|
        |                  ||                  ||                  ||
        |                  ||                  ||                  ||
        |                  v|                  v|                  v|
        |           w_D(i-1/2,j-1/2)    w_D(i+1/2,j-1/2)            |
        |-------------------|-------------------|-------------------|
        """
        eps = 1e-6

        zeroth = np.copy(wF_pad1[1:-1])
        minus_one, minus_two = wF_pad1[:-2], wF_pad2[:-4]
        plus_one, plus_two = wF_pad1[2:], wF_pad2[4:]

        g0, g1, g2 = 1/10, 3/5, 3/10

        b0 = (13/12 * (minus_two - 2*minus_one + zeroth)**2 + 1/4 * (minus_two - 4*minus_one + 3*zeroth)**2)
        b1 = (13/12 * (minus_one - 2*zeroth + plus_one)**2 + 1/4 * (minus_one - plus_one)**2)
        b2 = (13/12 * (zeroth - 2*plus_one + plus_two)**2 + 1/4 * (3*zeroth - 4*plus_one + plus_two)**2)

        a0 = lambda d0: d0/(b0 + eps)**2
        a1 = lambda d1: d1/(b1 + eps)**2
        a2 = lambda d2: d2/(b2 + eps)**2

        wD = (
            (a0(g2)/(a0(g2)+a1(g1)+a2(g0))) * (1/3*zeroth + 5/6*minus_one - 1/6*minus_two)
            + (a1(g1)/(a0(g2)+a1(g1)+a2(g0))) * (-1/6*plus_one + 5/6*zeroth + 1/3*minus_one)
            + (a2(g0)/(a0(g2)+a1(g1)+a2(g0))) * (1/3*plus_two - 7/6*plus_one + 11/6*zeroth)
        )
        wU = (
            (a0(g0)/(a0(g0)+a1(g1)+a2(g2))) * (1/3*minus_two - 7/6*minus_one + 11/6*zeroth)
            + (a1(g1)/(a0(g0)+a1(g1)+a2(g2))) * (-1/6*minus_one + 5/6*zeroth + 1/3*plus_one)
            + (a2(g2)/(a0(g0)+a1(g1)+a2(g2))) * (1/3*zeroth + 5/6*plus_one - 1/6*plus_two)
        )

    return np.copy(_wF), wD, wU


##### !!!!!! Currently the axes are not aligned. The axes need to be re-transposed before the corner calculations can begin !!!!!!! #####
# Compute the corner electric fields wrt to corner; gives 4-fold values for each corner for now
def compute_corner(magnetic_components: list, sim_variables):
    gamma, boundary, reversed_axes = sim_variables.gamma, sim_variables.boundary, sim_variables.permutations[::-1]
    magnetic_components = np.asarray(magnetic_components)

    def compute_corners(_data):
        _NE = np.average(_data[:,0,...,2], axis=0)*_data[0,0,...,5] - np.average(_data[:,0,...,1], axis=0)*_data[1,0,...,6]
        _NW = np.average(_data[[0,1],[0,1],...,2], axis=0)*_data[0,0,...,5] - np.average(_data[[0,1],[0,1],...,1], axis=0)*_data[1,1,...,6]
        _SW = np.average(_data[:,1,...,2], axis=0)*_data[0,1,...,5] - np.average(_data[:,1,...,1], axis=0)*_data[1,1,...,6]
        _SE = np.average(_data[[0,1],[1,0],...,2], axis=0)*_data[0,1,...,5] - np.average(_data[[0,1],[1,0],...,1], axis=0)*_data[1,0,...,6]
        return _NE, _NW, _SW, _SE

    NE, NW, SW, SE = compute_corners(magnetic_components)

    alphas = []
    for ax, wTs in enumerate(magnetic_components):
        # Re-align the interfaces and calculate Roe average between the interfaces
        w_plus, w_minus = fv.add_boundary(wTs[0], boundary)[1:], fv.add_boundary(wTs[1], boundary)[:-1]
        grid_intf = constructor.make_Roe_average(w_plus, w_minus)[1:]

        # Define the variables
        rhos, vels, pressures, B_fields = grid_intf[...,0], grid_intf[...,1:4], grid_intf[...,4], grid_intf[...,5:8]/np.sqrt(4*np.pi)
        vx, Bx = vels[...,ax%3], B_fields[...,ax%3]

        # Define speeds
        sound_speed = np.sqrt(gamma * fv.divide(pressures, rhos))
        alfven_speed = fv.divide(fv.norm(B_fields), np.sqrt(rhos))
        alfven_speed_x = fv.divide(Bx, np.sqrt(rhos))
        fast_magnetosonic_wave = np.sqrt(.5 * (sound_speed**2 + alfven_speed**2 + np.sqrt(((sound_speed**2 + alfven_speed**2)**2) - (4*(sound_speed**2)*(alfven_speed_x**2)))))

        """Determine the alphas. The convention here uses L & R states, i.e. L state = w-, R state = w+
            |                        w(i-1/2)                    w(i+1/2)                       |
            |-->         i-1         <--|-->          i          <--|-->         i+1         <--|
            |   w_R1(i-1)   w_L1(i-1)   |   w_R1(i)       w_L1(i)   |  w_R1(i+1)    w_L1(i+1)   |
        --> |   w+(i-3/2)   w-(i-1/2)   |   w+(i-1/2)   w-(i+1/2)   |  w+(i+1/2)    w-(i+3/2)   |
        """
        alpha_minus = -np.minimum(np.zeros_like(vx), vx-fast_magnetosonic_wave)
        alpha_plus = np.maximum(np.zeros_like(vx), vx+fast_magnetosonic_wave)
        alphas.append([alpha_plus, alpha_minus])

    [ap2,am2], [ap1,am1] = alphas

    return fv.divide(ap1*ap2*SW + am1*ap2*SE + ap1*am2*NW + am1*am2*NE, (ap1+am1)*(ap2+am2)) - fv.divide(ap2*am2*np.squeeze(np.diff(magnetic_components[0,...,5], axis=0)), ap2+am2) + fv.divide(ap1*am1*np.squeeze(np.diff(magnetic_components[1,...,6], axis=0)), ap1+am1)