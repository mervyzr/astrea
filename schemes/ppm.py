from collections import defaultdict

import numpy as np

from functions import constructor, fv
from num_methods import limiters, mag_field

##############################################################################
# Piecewise parabolic reconstruction method (PPM) [Colella & Woodward, 1984]
##############################################################################

# [McCorquodale & Colella, 2011; Colella et al., 2011; Peterson & Hammett, 2008]
def run(grid, sim_variables, author="mc", dissipate=False):
    gamma, boundary, permutations = sim_variables.gamma, sim_variables.boundary, sim_variables.permutations
    magnetic, dimension = sim_variables.magnetic, sim_variables.dimension
    convert_primitive, convert_conservative = sim_variables.convert_primitive, sim_variables.convert_conservative
    nested_dict = lambda: defaultdict(nested_dict)
    data = nested_dict()

    author = author.lower()

    # Rotate grid and apply algorithm for each axis
    for axis, axes in enumerate(permutations):
        _grid = grid.transpose(axes)

        # Convert to primitive variables
        wS = convert_conservative(_grid, sim_variables)

        # Pad array with boundary; PPM requires additional ghost cells
        w2 = fv.add_boundary(wS, boundary, 2)
        w = np.copy(w2[1:-1])

        """Extrapolate the cell averages to face averages (forward/upwind)
        |               w(i-1/2)            w(i+1/2)                |
        |  i-1           -->|   i            -->|  i+1           -->|
        |        w_R(i-1)   |          w_R(i)   |        w_R(i+1)   |
        """
        # Face i+1/2 (4th-order) [McCorquodale & Colella, 2011, eq. 17; Colella et al., 2011, eq. 67]
        wF = 7/12 * (wS + w[2:]) - 1/12 * (w[:-2] + w2[4:])

        if magnetic and dimension == 2:
            next_axes = permutations[(axis+1) % len(permutations)]
            data[axes]['wTs'] = mag_field.reconstruct_transverse(wF, next_axes, boundary)

        if "x" in author or "ph" in author or author in ["peterson", "hammett"]:
            """Extrapolate the cell averages to face averages (both sides)
            |                        w(i-1/2)                    w(i+1/2)                       |
            |<--         i-1         -->|<--          i          -->|<--         i+1         -->|
            |   w_L(i-1)     w_R(i-1)   |   w_L(i)         w_R(i)   |   w_L(i+1)     w_R(i+1)   |
            """
            # Face i+1/2 (4th-order) (eq. 3.26-3.27)
            wF_L = 7/12 * (w[:-2] + wS) - 1/12 * (w2[:-4] + w[2:])
            wF_R = wF

            # Face i+1/2 (5th-order) [Peterson & Hammett, 2008, eq. 3.40]
            #wF_R = 1/60 * (2*w2[:-4] - 13*w[:-2] + 47*wS + 27*w[2:] - 2*w2[4:])

            # Limit interface values [Peterson & Hammett, 2008, eq. 3.33-3.34]
            limited_wFs = limiters.interface_limiter(wF_L, w2[:-4], w[:-2], wS, w[2:]), limiters.interface_limiter(wF_R, w[:-2], wS, w[2:], w2[4:])
            wF_pad2 = np.zeros_like(fv.add_boundary(wF_R, boundary, 2))

        else:
            if author == "c" or author == "colella":
                # Limit interface values [Colella et al., 2011, p. 25-26]
                wF = limiters.interface_limiter(wF, w[:-2], wS, w[2:], w2[4:])

            if (author == "mc" or "mccorquodale" in author) and dissipate:
                eta = apply_flattener(wS, axis, boundary)
                wF = wF * eta[...,None] + wS * (1-eta)[...,None]

            # Define the left and right parabolic interpolants
            wF_pad2 = fv.add_boundary(wF, boundary, 2)
            limited_wFs = np.copy(wF_pad2[1:-3]), np.copy(wF_pad2[2:-2])

        """Reconstruct the limited parabolic interpolants from the interface values [McCorquodale & Colella, 2011; Colella et al., 2011; Peterson & Hammett, 2008]
        |                        w(i-1/2)                    w(i+1/2)                       |
        |-->         i-1         <--|-->          i          <--|-->         i+1         <--|
        |   w_L(i-1)     w_R(i-1)   |   w_L(i)         w_R(i)   |   w_L(i+1)     w_R(i+1)   |
        |   w+(i-3/2)   w-(i-1/2)   |   w+(i-1/2)   w-(i+1/2)   |  w+(i+1/2)    w-(i+3/2)   |
        """
        wL, wR = limiters.interpolant_limiter(wS, w, w2, wF_pad2, author, boundary, *limited_wFs)

        # Re-align the interfaces so that cell wall is in between interfaces
        w_plus, w_minus = fv.add_boundary(wL, boundary)[1:], fv.add_boundary(wR, boundary)[:-1]

        # Get the average solution between the interfaces at the boundaries
        intf_avg = constructor.make_Roe_average(w_plus, w_minus)[1:]
        _intf_avg = fv.add_boundary(intf_avg, boundary)

        # Convert the primitive variables
        q_plus, q_minus = convert_primitive(w_plus, sim_variables, "face"), convert_primitive(w_minus, sim_variables, "face")

        # Compute the fluxes and the Jacobian
        flux_plus, flux_minus = constructor.make_flux(w_plus, gamma, axis), constructor.make_flux(w_minus, gamma, axis)

        if (author == "mc" or "mccorquodale" in author) and dissipate:
            data[axes]['mu'] = apply_artificial_viscosity(wS, axis, sim_variables)

        A = constructor.make_Jacobian(_intf_avg, gamma, axis)

        # Update dict
        data[axes]['wS'] = wS
        data[axes]['wFs'] = w_plus, w_minus
        data[axes]['qFs'] = q_plus, q_minus
        data[axes]['fluxFs'] = flux_plus, flux_minus
        data[axes]['Jacobian'] = A

    return data


# Calculate the coefficient of the slope flattener for the parabolic extrapolants [Colella, 1990]
def apply_flattener(wS, axis, boundary, slope_determinants=[.33, .75, .85]):
    delta, z0, z1 = slope_determinants

    w2 = fv.add_boundary(wS, boundary, 2)
    w = np.copy(w2[1:-1])

    def zeta_func(_z, _z0, _z1):
        _arr = np.copy(1 - fv.divide(_z-_z0, _z1-_z0))
        _arr[_z > _z1] = 0
        _arr[_z < _z0] = 1
        return _arr

    chi_bar = zeta_func(fv.divide(np.abs(w[...,4][2:]-w[...,4][:-2]), np.abs(w2[...,4][4:]-w2[...,4][:-4])), z0, z1)
    chi_bar[((w[...,axis+1][:-2]-w[...,axis+1][2:]) <= 0) & (fv.divide(np.abs(w[...,4][2:]-w[...,4][:-2]), np.minimum(w[...,4][2:], w[...,4][:-2])) <= delta)] = 0
    chi_bar_padded = fv.add_boundary(chi_bar, boundary)

    signage = np.sign(w[...,4][2:]-w[...,4][:-2])

    chi = np.copy(chi_bar)
    chi[signage < 0] = np.minimum(chi_bar, chi_bar_padded[2:])[signage < 0]
    chi[signage > 0] = np.minimum(chi_bar, chi_bar_padded[:-2])[signage > 0]

    arr_expander = np.ones_like(wS)
    return arr_expander * chi[...,None]


# Implement artificial viscosity [McCorquodale & Colella, 2011]
def apply_artificial_viscosity(wS, axis, sim_variables, viscosity_determinants=[.3, .3]):
    alpha, beta = viscosity_determinants
    dimension, gamma, boundary, dx = sim_variables.dimension, sim_variables.gamma, sim_variables.boundary, sim_variables.dx

    w = fv.add_boundary(wS, boundary)

    velocity = wS[...,axis+1]
    velocity_w = w[...,axis+1]

    # Calculate face-centred divergence of velocity [eq. 35]
    lambda_R = velocity_w[2:] - velocity_w[1:-1]
    if velocity.ndim != 1:
        for ax in range(1, dimension):
            padded_velocity = fv.add_boundary(velocity, boundary, axis=ax)
            padded_w = fv.add_boundary(velocity_w, boundary, axis=ax)

            lambda_R += .25 * (np.diff(padded_w.take(range(1,padded_w.shape[ax]), axis=ax), axis=ax) + np.diff(padded_velocity.take(range(1,padded_velocity.shape[ax]), axis=ax), axis=ax))

    # Calculate sound speed
    cs = np.sqrt(fv.divide(gamma*w[...,4], w[...,0]))
    c_min = np.minimum(cs[1:-1], cs[2:])

    # Calculate artificial viscosity coefficient [eq. 36]
    reference = np.copy(lambda_R)
    nu = np.minimum(1, fv.divide((dx * lambda_R)**2, beta * c_min**2)) * lambda_R[...,None]
    nu[reference >= 0] = 0

    # Calculate the coefficient [eq. 38]
    arr_expander = np.ones_like(wS)
    coeff = nu * arr_expander
    mu = alpha * (coeff * np.diff(w[1:], axis=0))

    return mu