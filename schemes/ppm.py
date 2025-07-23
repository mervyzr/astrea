from collections import defaultdict

import numpy as np

from functions import constructor, fv
from num_methods import limiters, mag_field

##############################################################################
# Piecewise parabolic reconstruction method (PPM) [Colella & Woodward, 1984]
##############################################################################

# [McCorquodale & Colella, 2011; Colella et al., 2011; Peterson & Hammett, 2008]
def run(grid, sim_variables, author="mc", dissipate=False):
    gamma, boundary, axes, magnetic = sim_variables.gamma, sim_variables.boundary, sim_variables.axes, sim_variables.magnetic
    convert_primitive, convert_conservative = sim_variables.convert_primitive, sim_variables.convert_conservative
    nested_dict = lambda: defaultdict(nested_dict)
    data = nested_dict()

    author = author.lower()
    Bx, By, Bz = range(5,8)

    for axis in axes:
        # Convert to primitive variables
        primitive = convert_conservative(grid, sim_variables, staggered=magnetic)

        # Pad array with boundary; PPM requires additional ghost cells
        padded_primitive_2 = fv.add_boundary(primitive, boundary, stencil=2, axis=axis)
        padded_primitive = fv.slice_along_axis(padded_primitive_2, axis, *[1,-1])

        minus_one, minus_two = fv.slice_along_axis(padded_primitive, axis, end=-2), fv.slice_along_axis(padded_primitive_2, axis, end=-4)
        plus_one, plus_two = fv.slice_along_axis(padded_primitive, axis, start=2), fv.slice_along_axis(padded_primitive_2, axis, start=4)

        """Interpolate the cell averages to face averages (forward/upwind)
        |               w(i-1/2)            w(i+1/2)                |
        |  i-1           -->|   i            -->|  i+1           -->|
        |        w_R(i-1)   |          w_R(i)   |        w_R(i+1)   |
        """
        # Face i+1/2 (4th-order) [McCorquodale & Colella, 2011, eq. 17; Colella et al., 2011, eq. 67]
        interface = 7/12 * (primitive + plus_one) - 1/12 * (minus_one + plus_two)

        # Face i+1/2 (5th-order, less robust) [Suresh & Huynh, 1997, eq. 2.1]
        #interface = 1/60 * (2*minus_two - 13*minus_one + 47*primitive + 27*plus_one - 3*plus_two)

        # Magnetic component after computing to interface
        if magnetic:
            padded_grid = fv.add_boundary(grid, boundary, axis=axis)
            interface[...,(Bx,By)] = grid[...,(Bx,By)]
            data[axis]['ortho_interfaces'] = mag_field.reconstruct_transverse(interface, sim_variables, axis=axis)

        if "x" in author or "ph" in author or author in ["peterson", "hammett"]:
            """Interpolate the cell averages to face averages (both sides)
            |                        w(i-1/2)                    w(i+1/2)                       |
            |<--         i-1         -->|<--          i          -->|<--         i+1         -->|
            |   w_L(i-1)     w_R(i-1)   |   w_L(i)         w_R(i)   |   w_L(i+1)     w_R(i+1)   |
            """
            # Face i+1/2 (4th-order) (eq. 3.26-3.27)
            left_of_centre = 7/12 * (minus_one + primitive) - 1/12 * (minus_two + plus_one)
            right_of_centre = interface

            # Limit interface values [Peterson & Hammett, 2008, eq. 3.33-3.34]
            limited_wFs = limiters.interface_limiter(left_of_centre, minus_two, minus_one, primitive, plus_one), limiters.interface_limiter(right_of_centre, minus_one, primitive, plus_one, plus_two)
            padded_interface_2 = np.zeros_like(fv.add_boundary(right_of_centre, boundary, stencil=2, axis=axis))

        else:
            if author == "c" or author == "colella":
                # Limit interface values [Colella et al., 2011, p. 25-26]
                interface = limiters.interface_limiter(interface, minus_one, primitive, plus_one, plus_two)

            if (author == "mc" or "mccorquodale" in author) and dissipate:
                eta = apply_flattener(primitive, axis, boundary)
                interface = interface * eta[...,None] + primitive * (1-eta)[...,None]

            # Define the left and right parabolic extrapolants
            padded_interface_2 = fv.add_boundary(interface, boundary, stencil=2, axis=axis)
            limited_wFs = fv.slice_along_axis(padded_interface_2, axis, *[1,-3]), fv.slice_along_axis(padded_interface_2, axis, *[2,-2])

        """Reconstruct the limited parabolic extrapolants from the interface values [McCorquodale & Colella, 2011; Colella et al., 2011; Peterson & Hammett, 2008]
        |                        w(i-1/2)                    w(i+1/2)                       |
        |-->         i-1         <--|-->          i          <--|-->         i+1         <--|
        |   w_L(i-1)     w_R(i-1)   |   w_L(i)         w_R(i)   |   w_L(i+1)     w_R(i+1)   |
        |   w+(i-3/2)   w-(i-1/2)   |   w+(i-1/2)   w-(i+1/2)   |  w+(i+1/2)    w-(i+3/2)   |
        """
        wL, wR = limiters.extrapolant_limiter(primitive, padded_primitive, padded_primitive_2, padded_interface_2, author, boundary, axis, *limited_wFs)

        # Re-align the interfaces so that cell wall is in between interfaces
        prim_plus, prim_minus = fv.slice_along_axis(fv.add_boundary(wL, boundary, axis=axis), axis, start=1), fv.slice_along_axis(fv.add_boundary(wR, boundary, axis=axis), axis, end=-1)
        if magnetic:
            prim_plus[...,(Bx,By)] = prim_minus[...,(Bx,By)] = fv.slice_along_axis(padded_grid, axis, end=-1)[...,(Bx,By)]

        # Get the average solution between the interfaces at the boundaries
        intf_avg = fv.slice_along_axis(constructor.make_Roe_average(prim_plus, prim_minus), axis, start=1)
        padded_intf_avg = fv.add_boundary(intf_avg, boundary, axis=axis)

        # Convert the primitive variables
        cons_plus, cons_minus = convert_primitive(prim_plus, sim_variables, compute_face=True), convert_primitive(prim_minus, sim_variables, compute_face=True)

        # Compute the fluxes and the Jacobian
        flux_plus, flux_minus = constructor.make_flux(prim_plus, gamma, axis=axis), constructor.make_flux(prim_minus, gamma, axis=axis)

        if (author == "mc" or "mccorquodale" in author) and dissipate:
            data[axis]['mu'] = apply_artificial_viscosity(primitive, axis, sim_variables)

        jacobian = constructor.make_Jacobian(padded_intf_avg, gamma, axis=axis)

        # Update dict
        data[axis]['primitive'] = primitive
        data[axis]['prim_interfaces'] = prim_plus, prim_minus
        data[axis]['cons_interfaces'] = cons_plus, cons_minus
        data[axis]['flux_interfaces'] = flux_plus, flux_minus
        data[axis]['characteristics'] = np.linalg.eigvals(jacobian)

    return data


# Calculate the coefficient of the slope flattener for the parabolic interpolants/extrapolants [Colella, 1990]
def apply_flattener(grid, axis, boundary, slope_determinants=[.33, .75, .85]):
    delta, z0, z1 = slope_determinants

    padded_grid_2 = fv.add_boundary(grid, boundary, stencil=2, axis=axis)
    padded_grid = fv.add_boundary(grid, boundary, axis=axis)

    minus_one, minus_two = fv.slice_along_axis(padded_grid, axis, end=-2), fv.slice_along_axis(padded_grid_2, axis, end=-4)
    plus_one, plus_two = fv.slice_along_axis(padded_grid, axis, start=2), fv.slice_along_axis(padded_grid_2, axis, start=4)

    def zeta_func(_z, _z0, _z1):
        _arr = np.copy(1 - fv.divide(_z-_z0, _z1-_z0))
        _arr[_z > _z1] = 0
        _arr[_z < _z0] = 1
        return _arr

    chi_bar = zeta_func(fv.divide(np.abs(plus_one[...,4]-minus_one[...,4]), np.abs(plus_two[...,4]-minus_two[...,4])), z0, z1)
    chi_bar[((minus_one[...,1+axis]-plus_one[...,1+axis]) <= 0) & (fv.divide(np.abs(plus_one[...,4]-minus_one[...,4]), np.minimum(plus_one[...,4], minus_one[...,4])) <= delta)] = 0
    chi_bar_padded = fv.add_boundary(chi_bar, boundary, axis=axis)

    signage = np.sign(plus_one[...,4]-minus_one[...,4])

    chi = np.copy(chi_bar)
    chi[signage < 0] = np.minimum(chi_bar, fv.slice_along_axis(chi_bar_padded, axis, start=2))[signage < 0]
    chi[signage > 0] = np.minimum(chi_bar, fv.slice_along_axis(chi_bar_padded, axis, end=-2))[signage > 0]

    arr_expander = np.ones_like(grid)
    return arr_expander * chi[...,None]


# Implement artificial viscosity [McCorquodale & Colella, 2011]
def apply_artificial_viscosity(grid, axis, sim_variables, viscosity_determinants=[.3, .3]):
    alpha, beta = viscosity_determinants
    dimension, gamma, boundary, dx = sim_variables.dimension, sim_variables.gamma, sim_variables.boundary, sim_variables.dx

    padded_grid = fv.add_boundary(grid, boundary, axis=axis)

    velocity = grid[...,1+axis]
    velocity_w = padded_grid[...,1+axis]

    # Calculate face-centred divergence of velocity [eq. 35]
    lambda_R = fv.slice_along_axis(velocity_w, axis, start=2) - fv.slice_along_axis(velocity_w, axis, [1,-1])

    for ax in range(1, dimension):
        padded_velocity = fv.add_boundary(velocity, boundary, axis=ax)
        padded_w = fv.add_boundary(velocity_w, boundary, axis=ax)

        lambda_R += .25 * (np.diff(fv.slice_along_axis(padded_w, ax, start=1), axis=ax) + np.diff(fv.slice_along_axis(padded_velocity, ax, start=1), axis=ax))

    # Calculate sound speed
    cs = np.sqrt(fv.divide(gamma*padded_grid[...,4], padded_grid[...,0]))
    c_min = np.minimum(fv.slice_along_axis(cs, axis, [1,-1]) , fv.slice_along_axis(cs, axis, start=2))

    # Calculate artificial viscosity coefficient [eq. 36]
    reference = np.copy(lambda_R)
    nu = np.minimum(1, fv.divide((dx * lambda_R)**2, beta * c_min**2)) * lambda_R[...,None]
    nu[reference >= 0] = 0

    # Calculate the coefficient [eq. 38]
    arr_expander = np.ones_like(grid)
    coeff = nu * arr_expander
    mu = alpha * (coeff * np.diff(fv.slice_along_axis(padded_grid, axis, start=1), axis=axis))

    return mu