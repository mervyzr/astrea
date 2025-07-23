import numpy as np

from functions import fv, constructor
from num_methods import limiters

##############################################################################
# Fourth-order upwind constrained transport algorithm for MHD [Felker & Stone, 2018]
##############################################################################

# Reconstruct the transverse values for each face average (computation done entirely for orthogonal axis)
# Returns array aligned with input axis, e.g., ax=1 means returned array is aligned with y-axis-transposed grid
def reconstruct_transverse(interface, sim_variables, axis, method=None):
    if not method:
        method = sim_variables.subgrid
    boundary = sim_variables.boundary

    ortho_axis = 1 - axis

    padded_grid = fv.add_boundary(interface, boundary, axis=ortho_axis)
    padded_grid_2 = fv.add_boundary(interface, boundary, stencil=2, axis=ortho_axis)

    zeroth = np.copy(interface)
    minus_one, minus_two = fv.slice_along_axis(padded_grid, ortho_axis, end=-2), fv.slice_along_axis(padded_grid_2, ortho_axis, end=-4)
    plus_one, plus_two = fv.slice_along_axis(padded_grid, ortho_axis, start=2), fv.slice_along_axis(padded_grid_2, ortho_axis, start=4)

    # 5th-order WENO reconstruction
    if "weno" in method:
        eps = 1e-6

        """Interpolate the face averages to both corners (upwards & downwards)
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
        g0, g1, g2 = 1/10, 3/5, 3/10

        b0 = (
            13/12 * (minus_two - 2*minus_one + zeroth)**2
            + 1/4 * (minus_two - 4*minus_one + 3*zeroth)**2
        )
        b1 = (
            13/12 * (minus_one - 2*zeroth + plus_one)**2
            + 1/4 * (minus_one - plus_one)**2
        )
        b2 = (
            13/12 * (zeroth - 2*plus_one + plus_two)**2
            + 1/4 * (3*zeroth - 4*plus_one + plus_two)**2
        )

        a0 = lambda d0: d0/(b0 + eps)**2
        a1 = lambda d1: d1/(b1 + eps)**2
        a2 = lambda d2: d2/(b2 + eps)**2

        wU = (
            (a0(g0)/(a0(g0)+a1(g1)+a2(g2))) * (1/3*minus_two - 7/6*minus_one + 11/6*zeroth)
            + (a1(g1)/(a0(g0)+a1(g1)+a2(g2))) * (-1/6*minus_one + 5/6*zeroth + 1/3*plus_one)
            + (a2(g2)/(a0(g0)+a1(g1)+a2(g2))) * (1/3*zeroth + 5/6*plus_one - 1/6*plus_two)
        )
        wD = (
            (a0(g2)/(a0(g2)+a1(g1)+a2(g0))) * (1/3*zeroth + 5/6*minus_one - 1/6*minus_two)
            + (a1(g1)/(a0(g2)+a1(g1)+a2(g0))) * (-1/6*plus_one + 5/6*zeroth + 1/3*minus_one)
            + (a2(g0)/(a0(g2)+a1(g1)+a2(g0))) * (1/3*plus_two - 7/6*plus_one + 11/6*zeroth)
        )

    elif method == "ppm":

        author = "mc"

        """Interpolate the face averages to the top corners (upwards) [McCorquodale & Colella, 2011, eq. 17; Colella et al., 2011, eq. 67]
        |                w(i-1/2)            w(i+1/2)               |
        |-------------------|-------------------|-------------------|
        |           w_U(i-1/2,j+1/2)    w_U(i+1/2,j+1/2)            |
        |                  ^|                  ^|                  ^|
        |                  ||                  ||                  ||
        |                  ||                  ||                  ||
        |  o (i-1,j)     -->|  o (i,j)       -->|  o (i+1,j)     -->|
        """
        wU = 7/12 * (zeroth + plus_one) - 1/12 * (minus_one + plus_two)

        if "x" in author or "ph" in author or author in ["peterson", "hammett"]:
            """Interpolate the face averages to both corners (upwards & downwards)
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
            wD = 7/12 * (minus_one + zeroth) - 1/12 * (minus_two + plus_one)

            # Limit interface values [Peterson & Hammett, 2008, eq. 3.33-3.34]
            limited_wUs = limiters.interface_limiter(wD, minus_two, minus_one, zeroth, plus_one), limiters.interface_limiter(wU, minus_one, zeroth, plus_one, plus_two)
            padded_wU_2 = np.zeros_like(fv.add_boundary(wU, boundary, stencil=2, axis=ortho_axis))
        else:
            if author == "c" or author == "collela":
                # Limit interface values [Colella et al., 2011, p. 25-26]
                wU = limiters.interface_limiter(wU, minus_one, zeroth, plus_one, plus_two)

            # Define the top and bottom parabolic extrapolants
            padded_wU_2 = fv.add_boundary(wU, boundary, stencil=2, axis=ortho_axis)
            limited_wUs = fv.slice_along_axis(padded_wU_2, ortho_axis, *[1,-3]), fv.slice_along_axis(padded_wU_2, ortho_axis, *[2,-2])

        """Reconstruct the limited extrapolants from the interface values. Returns the face averages in the form of w+(y) & w-(y) when considering x-axis, and w+(x) & w-(x) when considering y-axis
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
        wD, wU = limiters.extrapolant_limiter(zeroth, padded_grid, padded_grid_2, padded_wU_2, author, boundary, ortho_axis, *limited_wUs)

    elif method == "plm":
        limited_values = limiters.minmod_limiter(padded_grid, ortho_axis)
        gradients = .5 * limited_values
        wD, wU = zeroth - gradients, zeroth + gradients

    else:
        wD, wU = zeroth, zeroth

    return wD, wU



# Compute the corner electric fields wrt to corner; gives 4-fold values for each corner for now [Mignone & del Zanna, 2021]
def compute_corner(data, sim_variables):

    # Calculate the eigenvalues for the Riemann problem at the corner; crucial for selecting the corner
    def get_wavespeeds(_wD, _wU, _sim_variables, _axis):
        # Re-align the interfaces so that cell wall is in between interfaces
        plus, minus = fv.add_boundary(_wD, _sim_variables.boundary)[1:], fv.add_boundary(_wU, _sim_variables.boundary)[:-1]

        # Get the average solution between the interfaces at the boundaries
        intf_avg = constructor.make_Roe_average(plus, minus)[1:]

        # HLL-family solver
        if _sim_variables.solver_category == "hll":
            # Define the variables
            rhos, vels, pressures, B_fields = intf_avg[...,0], intf_avg[...,1:4], intf_avg[...,4], intf_avg[...,5:8]
            vx, Bx = vels[...,_axis%3], B_fields[...,_axis%3]

            # Define speeds
            sound_speed = np.sqrt(_sim_variables.gamma * fv.divide(pressures, rhos))
            alfven_speed = fv.divide(fv.norm(B_fields), np.sqrt(rhos))
            alfven_speed_x = fv.divide(Bx, np.sqrt(rhos))
            fast_magnetosonic_wave = np.sqrt(.5 * (sound_speed**2 + alfven_speed**2 + np.sqrt(((sound_speed**2 + alfven_speed**2)**2) - (4*(sound_speed**2)*(alfven_speed_x**2)))))

            # Local min/max characteristic waves for each cell
            local_max_eigvals = np.maximum(np.zeros_like(vx), vx+fast_magnetosonic_wave)
            local_min_eigvals = -np.minimum(np.zeros_like(vx), vx-fast_magnetosonic_wave)

        # Default to Local Lax-Friedrich solver
        else:
            # Compute the eigenvalues for the Riemann fan at the corner; crucial in selecting the corner
            A = constructor.make_Jacobian(intf_avg, _sim_variables.gamma, _axis%3)
            characteristics = np.linalg.eigvals(A)

            # Local min/max eigenvalues for each cell
            local_max_eigvals = np.max(characteristics, axis=-1)
            local_min_eigvals = -np.min(characteristics, axis=-1)

        return local_max_eigvals, local_min_eigvals

    alphas, magnetic_components = [], []
    for axis, axes in sim_variables.swapped_permutations.items():
        wD, wU = data[axes]['wTs']

        # Solve Riemann problem for corners; compute with transverse axes, but the 'axis' must match the 'axes'
        a_plus, a_minus = get_wavespeeds(wD, wU, sim_variables, axis)

        # Collate and align the magnetic components and the alphas (rotate array to make x-axis as 'reference axis')
        alignment_axes = sim_variables.permutations[axis]
        alphas.append([a_plus.transpose(alignment_axes[:-1]), a_minus.transpose(alignment_axes[:-1])])
        magnetic_components.append([wD.transpose(alignment_axes), wU.transpose(alignment_axes)])

    # Compute the corner B-fields wrt to corner
    [north, south], [east, west] = magnetic_components
    SW = .5*(west[...,2]+south[...,2])*south[...,5] - .5*(west[...,1]+south[...,1])*west[...,6]
    SE = .5*(east[...,2]+south[...,2])*south[...,5] - .5*(east[...,1]+south[...,1])*east[...,6]
    NW = .5*(west[...,2]+north[...,2])*north[...,5] - .5*(west[...,1]+north[...,1])*west[...,6]
    NE = .5*(east[...,2]+north[...,2])*north[...,5] - .5*(east[...,1]+north[...,1])*east[...,6]

    # Determine the alphas
    [ap_y, am_y], [ap_x, am_x] = alphas

    return fv.divide(ap_x*ap_y*SW + am_x*ap_y*SE + ap_x*am_y*NW + am_x*am_y*NE, (ap_x+am_x)*(ap_y+am_y)) - fv.divide(ap_y*am_y, ap_y+am_y)*(north[...,5]-south[...,5]) + fv.divide(ap_x*am_x, ap_x+am_x)*(east[...,6]-west[...,6])
    #return -(west[...,1]*west[...,6] + east[...,1]*east[...,6]) + (north[...,2]*north[...,5] + south[...,2]*south[...,5])