import numpy as np

from functions import fv, constructor
from num_methods import limiters

##############################################################################
# Fourth-order upwind constrained transport algorithm for MHD [Felker & Stone, 2018]
##############################################################################

# Reconstruct the transverse values for each face average (computation done entirely for orthogonal axis)
def reconstruct_transverse(wF, sim_variables, **kwargs):
    if kwargs:
        method = kwargs.get("method").lower()
    else:
        method = sim_variables.subgrid

    ortho_axis, boundary = sim_variables.ortho_axis, sim_variables.boundary

    ortho_wF = np.copy(wF.transpose(ortho_axis))
    wF_pad2 = fv.add_boundary(ortho_wF, boundary, 2)
    wF_pad1 = np.copy(wF_pad2[1:-1])

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
        zeroth = ortho_wF
        minus_one, minus_two = wF_pad1[:-2], wF_pad2[:-4]
        plus_one, plus_two = wF_pad1[2:], wF_pad2[4:]

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
        wU = 7/12 * (ortho_wF + wF_pad1[2:]) - 1/12 * (wF_pad1[:-2] + wF_pad2[4:])

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
            wD = 7/12 * (wF_pad1[:-2] + ortho_wF) - 1/12 * (wF_pad2[:-4] + wF_pad1[2:])

            # Limit interface values [Peterson & Hammett, 2008, eq. 3.33-3.34]
            limited_wUs = limiters.interface_limiter(wD, wF_pad2[:-4], wF_pad1[:-2], ortho_wF, wF_pad1[2:]), limiters.interface_limiter(wU, wF_pad1[:-2], ortho_wF, wF_pad1[2:], wF_pad2[4:])
            wU_pad2 = np.zeros_like(fv.add_boundary(wU, boundary, 2))
        else:
            if author == "c" or author == "collela":
                # Limit interface values [Colella et al., 2011, p. 25-26]
                wU = limiters.interface_limiter(wU, wF_pad1[:-2], ortho_wF, wF_pad1[2:], wF_pad2[4:])

            # Define the top and bottom parabolic extrapolants
            wU_pad2 = fv.add_boundary(wU, boundary, 2)
            limited_wUs = np.copy(wU_pad2[1:-3]), np.copy(wU_pad2[2:-2])

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
        wD, wU = limiters.extrapolant_limiter(ortho_wF, wF_pad1, wF_pad2, wU_pad2, author, boundary, *limited_wUs)

    elif method == "plm":
        limited_values = limiters.minmod_limiter(wF_pad1)
        gradients = .5 * limited_values
        wD, wU = np.copy(ortho_wF-gradients), np.copy(ortho_wF+gradients)

    else:
        wD, wU = np.copy(ortho_wF), np.copy(ortho_wF)

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


# 'Inverse reconstruct' the cell-averages from the face-averages after the induction difference [Felker & Stone, 2018]
def inverse_reconstruct(grid, sim_variables):
    new_grid = np.copy(grid)

    for axis, axes in sim_variables.permutations.items():
        axis %= 3
        reversed_axes = np.argsort(axes)

        # Approximate the face-averaged values to face-centred values (eq. 38)
        face_cntrd = fv.high_order_convert('face', 'avg', grid.transpose(axes), sim_variables)

        # Interpolate the face-centred values to cell-centred values (eq. 39)
        face_cntrd_pad2 = fv.add_boundary(face_cntrd, sim_variables.boundary, 2)
        face_cntrd_pad1 = np.copy(face_cntrd_pad2[1:-1])
        cell_cntrd = -1/16*(face_cntrd_pad1[:-2] + face_cntrd_pad2[4:]) + 9/16*(face_cntrd + face_cntrd_pad1[2:])

        # Apply Laplacian operator to convert cell-centred values to cell-averaged values (eq. 40)
        cell_avgd = fv.high_order_convert('cell', 'cntr', cell_cntrd, sim_variables)

        # Update the grid values with the updated B-field values
        new_grid[...,5+axis] = cell_avgd.transpose(reversed_axes)[...,5+axis]

    return new_grid