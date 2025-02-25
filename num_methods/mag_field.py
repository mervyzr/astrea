import numpy as np

from functions import fv, constructor
from num_methods import limiters

##############################################################################
# Fourth-order upwind constrained transport algorithm for MHD [Felker & Stone, 2018]
##############################################################################

# Reconstruct the transverse values for each face average
def reconstruct_transverse(wF, sim_variables, method="ppm", author="mc"):
    ortho_axis, boundary = sim_variables.ortho_axis, sim_variables.boundary

    # Compute in orthogonal axis
    ortho_wF = np.copy(wF.transpose(ortho_axis))
    wF_pad2 = fv.add_boundary(ortho_wF, boundary, 2)
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
        wU = 7/12 * (ortho_wF + wF_pad1[2:]) - 1/12 * (wF_pad1[:-2] + wF_pad2[4:])

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
            wD = 7/12 * (wF_pad1[:-2] + ortho_wF) - 1/12 * (wF_pad2[:-4] + wF_pad1[2:])

            # Limit interface values [Peterson & Hammett, 2008, eq. 3.33-3.34]
            limited_wUs = limiters.interface_limiter(wD, wF_pad2[:-4], wF_pad1[:-2], ortho_wF, wF_pad1[2:]), limiters.interface_limiter(wU, wF_pad1[:-2], ortho_wF, wF_pad1[2:], wF_pad2[4:])
            wU_pad2 = np.zeros_like(fv.add_boundary(wU, boundary, 2))
        else:
            if author == "c" or author == "collela":
                # Limit interface values [Colella et al., 2011, p. 25-26]
                wU = limiters.interface_limiter(wU, wF_pad1[:-2], ortho_wF, wF_pad1[2:], wF_pad2[4:])

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
        wD, wU = limiters.interpolant_limiter(ortho_wF, wF_pad1, wF_pad2, wU_pad2, author, boundary, *limited_wUs)
    
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

    # Return result back in 'original' axis
    return wD.transpose(ortho_axis), wU.transpose(ortho_axis)


# Compute the corner electric fields wrt to corner; gives 4-fold values for each corner for now [Mignone & del Zanna, 2021]
def compute_corner(data, sim_variables):
    gamma, boundary, permutations = sim_variables.gamma, sim_variables.boundary, sim_variables.permutations
    reversed_axes = dict(reversed(list(sim_variables.permutations.items())))

    # Collate and align the magnetic components (use the x-axis as 'reference axis')
    magnetic_components = []
    for axes, arrays in data.items():
        wD, wU = arrays['wTs']
        magnetic_components.append([wD.transpose(axes), wU.transpose(axes)])
    [R2, L2], [R1, L1] = magnetic_components
    [north, south], [east, west] = magnetic_components

    # Compute the corner B-fields wrt to corner
    NE = .5*(west[...,2]+south[...,2])*south[...,5] - .5*(west[...,1]+south[...,1])*west[...,6]
    NW = .5*(east[...,2]+south[...,2])*south[...,5] - .5*(east[...,1]+south[...,1])*east[...,6]
    SE = .5*(west[...,2]+north[...,2])*north[...,5] - .5*(west[...,1]+north[...,1])*west[...,6]
    SW = .5*(east[...,2]+north[...,2])*north[...,5] - .5*(east[...,1]+north[...,1])*east[...,6]


    # NEED TO CHECK IF THE ALIGNMENT IS RIGHT; WHEN COMPUTING THE JACOBIAN AND EIGENVALUES, IS THE ALIGNMENT CORRECT FOR THE CORRECT ARRAY?
    def get_wavespeeds(_R, _L, _sim_variables, _axis):
        # Re-align the interfaces so that cell wall is in between interfaces
        plus, minus = fv.add_boundary(_R, _sim_variables.boundary)[1:], fv.add_boundary(_L, _sim_variables.boundary)[:-1]

        # Get the average solution between the interfaces at the boundaries
        intf_avg = constructor.make_Roe_average(plus, minus)[1:]

        # Compute the eigenvalues for the Riemann fan at the corner; crucial in selecting the corner
        A = constructor.make_Jacobian(intf_avg, _sim_variables.gamma, _axis)
        characteristics = np.linalg.eigvals(A)

        # Local min/max eigenvalues for each cell
        local_max_eigvals = np.max(characteristics, axis=-1)
        local_min_eigvals = -np.min(characteristics, axis=-1)

        return local_max_eigvals, local_min_eigvals

    # Determine the alphas
    alphas = []
    for axis, axes in permutations.item():
        alphas.append(list(get_wavespeeds(north, south, sim_variables, axis)))
    #[ap2,am2], [ap1,am1] = alphas




    # Re-align the interfaces so that cell wall is in between interfaces
    #NS_plus, NS_minus = fv.add_boundary(north, boundary)[1:], fv.add_boundary(south, boundary)[:-1]
    #EW_plus, EW_minus = fv.add_boundary(east, boundary)[1:], fv.add_boundary(west, boundary)[:-1]



    # {0: (0,1,2) , 1: (1,0,2)}

    # {(0,1,2): [ R2(up), L2(down) ] , (1,0,2): [ R1(right), L1(left) ]}

    # ortho axis,  normal axis



    # {1: (1,0,2) , 0: (0,1,2)}

    # {(1,0,2): [ R1(right), L1(left) ] , (0,1,2): [ R2(up), L2(down) ]}

    # normal axis, ortho axis


def compute_corner(magnetic_components, sim_variables):
    gamma, boundary = sim_variables.gamma, sim_variables.boundary
    magnetic_components = np.asarray(magnetic_components)
    reversed_axes = dict(reversed(list(sim_variables.permutations.items())))

    # Transpose the magnetic components back into the original arrangement (use the x-axis as 'reference axis')
    data = []
    for axis, axes in reversed_axes.items():
        wD, wU = magnetic_components[axis]
        data.append([wD.transpose(axes), wU.transpose(axes)])
    data = np.asarray(data)

    # Compute the corner B-fields wrt to corner
    NE = np.average(data[:,0,...,2], axis=0)*data[0,0,...,5] - np.average(data[:,0,...,1], axis=0)*data[1,0,...,6]
    NW = np.average(data[[0,1],[0,1],...,2], axis=0)*data[0,0,...,5] - np.average(data[[0,1],[0,1],...,1], axis=0)*data[1,1,...,6]
    SW = np.average(data[:,1,...,2], axis=0)*data[0,1,...,5] - np.average(data[:,1,...,1], axis=0)*data[1,1,...,6]
    SE = np.average(data[[0,1],[1,0],...,2], axis=0)*data[0,1,...,5] - np.average(data[[0,1],[1,0],...,1], axis=0)*data[1,0,...,6]

    # Calculate the eigenvalues for the Riemann problem at the corner; crucial for selecting the corner
    alphas = []
    for axis, axes in reversed_axes.items():
        _w_plus, _w_minus = magnetic_components[axis][0], magnetic_components[axis][1]
        _axes = axes[:-1]

        # Re-align the interfaces and calculate Roe average between the interfaces
        w_plus, w_minus = fv.add_boundary(_w_plus, boundary)[1:], fv.add_boundary(_w_minus, boundary)[:-1]
        grid_intf = constructor.make_Roe_average(w_plus, w_minus)[1:]

        # Define the variables
        rhos, vels, pressures, B_fields = grid_intf[...,0], grid_intf[...,1:4], grid_intf[...,4], grid_intf[...,5:8]/np.sqrt(4*np.pi)
        vx, Bx = vels[...,axis%3], B_fields[...,axis%3]

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
        alphas.append([alpha_plus.transpose(_axes), alpha_minus.transpose(_axes)])

    [ap2,am2], [ap1,am1] = alphas

    return fv.divide(ap1*ap2*SW + am1*ap2*SE + ap1*am2*NW + am1*am2*NE, (ap1+am1)*(ap2+am2)) - fv.divide(ap2*am2*np.squeeze(np.diff(data[0,...,5], axis=0)), ap2+am2) + fv.divide(ap1*am1*np.squeeze(np.diff(data[1,...,6], axis=0)), ap1+am1)


# 'Inverse reconstruct' the cell-average values from the face-average values with the induction difference [Felker & Stone, 2018]
def inverse_reconstruct(grid, sim_variables):
    _grid = np.copy(grid)

    for axis, axes in sim_variables.permutations.items():
        reversed_axes = np.argsort(axes)

        # Approximate the face-averaged values to face-centred values (eq. 38)
        face_cntrd = fv.high_order_convert('avg', grid.transpose(axes), sim_variables, 'face')

        # Interpolate the face-centred values to cell-centred values (eq. 39)
        face_cntrd_pad2 = fv.add_boundary(face_cntrd, sim_variables.boundary, 2)
        face_cntrd_pad1 = np.copy(face_cntrd_pad2[1:-1])
        cell_cntrd = -1/16*(face_cntrd_pad1[:-2] + face_cntrd_pad2[4:]) + 9/16*(face_cntrd + face_cntrd_pad1[2:])

        # Apply Laplacian operator to convert cell-centred values to cell-averaged values (eq. 40)
        cell_avgd = fv.high_order_convert('cntr', cell_cntrd, sim_variables, 'cell')

        # Update the grid values with the updated B-field values
        _grid[...,5+axis] = cell_avgd.transpose(reversed_axes)[...,5+axis]
    return _grid