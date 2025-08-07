from collections import defaultdict

import numpy as np

from functions import constructor, fv
from num_methods import ct, limiters

##############################################################################
# Piecewise parabolic reconstruction method (PPM) [Colella & Woodward, 1984]
##############################################################################

# [McCorquodale & Colella, 2011 (McCorquodale&Colella2011); Colella et al., 2011 (Colella+2011); Peterson & Hammett, 2008 (Peterson&Hammett2008)]
def run(grid, sim_variables, author="McCorquodale&Colella2011", dissipate=False):
    boundary, axes, magnetic = sim_variables.boundary, sim_variables.axes, sim_variables.magnetic
    convert_primitive, convert_conservative = sim_variables.convert_primitive, sim_variables.convert_conservative
    Bx, By, Bz = range(5,8)

    nested_dict = lambda: defaultdict(nested_dict)
    data = nested_dict()

    # Convert to primitive variables
    primitive = convert_conservative(grid, sim_variables, staggered=magnetic)

    if dissipate:
        eta = get_flattening_coeff(primitive, sim_variables)

    for axis in axes:
        # Pad array with boundary; PPM requires additional ghost cells
        padded_primitive_2 = fv.add_boundary(primitive, boundary, stencil=2, axis=axis)
        padded_primitive = fv.slice_(padded_primitive_2, axis, *[1,-1])

        minus_one, minus_two = fv.slice_(padded_primitive, axis, end=-2), fv.slice_(padded_primitive_2, axis, end=-4)
        plus_one, plus_two = fv.slice_(padded_primitive, axis, start=2), fv.slice_(padded_primitive_2, axis, start=4)

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
            data[axis]['ortho_interfaces'] = ct.reconstruct_transverse(interface, sim_variables, axis=axis)

        if author.lower().startswith(("peterson", "p", "x")):
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
            if author.lower().startswith(("colella", "c")):
                # Limit interface values [Colella et al., 2011, p. 25-26]
                interface = limiters.interface_limiter(interface, minus_one, primitive, plus_one, plus_two)

            # Define the left and right parabolic extrapolants
            padded_interface_2 = fv.add_boundary(interface, boundary, stencil=2, axis=axis)
            limited_wFs = fv.slice_(padded_interface_2, axis, *[1,-3]), fv.slice_(padded_interface_2, axis, *[2,-2])

        """Reconstruct the limited parabolic extrapolants from the interface values [McCorquodale & Colella, 2011; Colella et al., 2011; Peterson & Hammett, 2008]
        |                        w(i-1/2)                    w(i+1/2)                       |
        |-->         i-1         <--|-->          i          <--|-->         i+1         <--|
        |   w_L(i-1)     w_R(i-1)   |   w_L(i)         w_R(i)   |   w_L(i+1)     w_R(i+1)   |
        |   w+(i-3/2)   w-(i-1/2)   |   w+(i-1/2)   w-(i+1/2)   |  w+(i+1/2)    w-(i+3/2)   |
        """
        wL, wR = limiters.extrapolant_limiter(primitive, padded_primitive, padded_primitive_2, padded_interface_2, author, boundary, axis, *limited_wFs)

        if dissipate:
            wL = wL * eta[...,None] + primitive * (1-eta)[...,None]
            wR = wR * eta[...,None] + primitive * (1-eta)[...,None]

        # Re-align the interfaces so that cell wall is in between interfaces
        prim_plus, prim_minus = fv.slice_(fv.add_boundary(wL, boundary, axis=axis), axis, start=1), fv.slice_(fv.add_boundary(wR, boundary, axis=axis), axis, end=-1)
        if magnetic:
            prim_plus[...,(Bx,By)] = prim_minus[...,(Bx,By)] = fv.slice_(padded_grid, axis, end=-1)[...,(Bx,By)]

        # Get the average solution between the interfaces at the boundaries
        intf_avg = fv.slice_(fv.compute_Roe_average(prim_plus, prim_minus), axis, start=1)
        padded_intf_avg = fv.add_boundary(intf_avg, boundary, axis=axis)

        # Convert the primitive variables
        cons_plus, cons_minus = convert_primitive(prim_plus, sim_variables, compute_face=True), convert_primitive(prim_minus, sim_variables, compute_face=True)

        # Compute the fluxes and the Jacobian
        flux_plus, flux_minus = constructor.make_flux(prim_plus, sim_variables, axis=axis), constructor.make_flux(prim_minus, sim_variables, axis=axis)

        if author.lower().startswith(("mccorquodale", "m")) and dissipate:
            data[axis]['artf_visc'] = get_artificial_viscosity([primitive, plus_one], axis, sim_variables)

        jacobian = constructor.make_Jacobian(padded_intf_avg, sim_variables, axis=axis)

        # Update dict
        data[axis]['prim_interfaces'] = prim_plus, prim_minus
        data[axis]['cons_interfaces'] = cons_plus, cons_minus
        data[axis]['flux_interfaces'] = flux_plus, flux_minus
        data[axis]['characteristics'] = np.linalg.eigvals(jacobian)

    return data


# Calculate the coefficient of the slope flattener for the parabolic interpolants/extrapolants [Colella, 1990]
def get_flattening_coeff(grid, sim_variables, slope_determinants=[.33, .75, .85]):
    delta, z0, z1 = slope_determinants
    pressure = 4
    chi_min = np.ones_like(grid[...,pressure])

    for axis in sim_variables.axes:
        padded_primitive_2 = fv.add_boundary(grid, sim_variables.boundary, stencil=2, axis=axis)
        padded_primitive = fv.slice_(padded_primitive_2, axis, *[1,-1])

        minus_one, minus_two = fv.slice_(padded_primitive, axis, end=-2), fv.slice_(padded_primitive_2, axis, end=-4)
        plus_one, plus_two = fv.slice_(padded_primitive, axis, start=2), fv.slice_(padded_primitive_2, axis, start=4)

        # zeta function
        z = fv.divide(np.abs(plus_one[...,pressure]-minus_one[...,pressure]), np.abs(plus_two[...,pressure]-minus_two[...,pressure]))
        chi_bar = 1 - fv.divide(z-z0, z1-z0)
        chi_bar[z > z1] = 0
        chi_bar[z < z0] = 1

        # Update chi_bar based on condition [eq. 4.9]
        otherwise_condition = np.where(
            ((minus_one[...,1+axis]-plus_one[...,1+axis]) <= 0)
            & (fv.divide(np.abs(plus_one[...,pressure]-minus_one[...,pressure]), np.minimum(plus_one[...,pressure], minus_one[...,pressure])) <= delta))
        chi_bar[otherwise_condition] = 0

        # Create flattening coefficient for axis
        chi = np.copy(chi_bar)
        chi_bar_padded = fv.add_boundary(chi_bar, sim_variables.boundary, axis=axis)
        signage = np.sign(plus_one[...,pressure]-minus_one[...,pressure])
        chi[signage < 0] = np.minimum(chi_bar, fv.slice_(chi_bar_padded, axis, start=2))[signage < 0]
        chi[signage > 0] = np.minimum(chi_bar, fv.slice_(chi_bar_padded, axis, end=-2))[signage > 0]

        chi_min = np.minimum(chi_min, chi)

    return chi_min


# Implement artificial viscosity [McCorquodale & Colella, 2011]
def get_artificial_viscosity(grid_slices, axis, sim_variables, viscosity_determinants=[.3, .3]):
    alpha, beta = viscosity_determinants
    rho, pressure, Bfields = 0, 4, slice(5,8)

    grid, plus_one = grid_slices
    ortho_axis = 1 - axis

    # Calculate face-centred divergence of velocity [eq. 35]
    lambda_d = plus_one - grid
    if sim_variables.dimension == 2:
        ortho_padded_grid = fv.add_boundary(grid, sim_variables.boundary, axis=ortho_axis)
        ortho_padded_plus_one = fv.add_boundary(plus_one, sim_variables.boundary, axis=ortho_axis)

        lambda_d += .25 * (
            fv.slice_(ortho_padded_plus_one, ortho_axis, start=2) - fv.slice_(ortho_padded_plus_one, ortho_axis, end=-2) 
            + fv.slice_(ortho_padded_grid, ortho_axis, start=2) - fv.slice_(ortho_padded_grid, ortho_axis, end=-2)
            )

    # Calculate minimum sound speed
    cs_grid = fv.divide(sim_variables.gamma * grid[...,pressure], grid[...,rho])
    cs_plus_one = fv.divide(sim_variables.gamma * plus_one[...,pressure], plus_one[...,rho])
    c_min = np.minimum(cs_grid, cs_plus_one)

    # Calculate artificial viscosity coefficient [eq. 36]
    nu = np.minimum(1, fv.divide(lambda_d[...,1+axis]**2, beta*c_min)) * lambda_d[...,1+axis]
    nu[lambda_d[...,1+axis] >= 0] = 0

    # Calculate the artificial viscosity [eq. 38]
    mu = alpha * (plus_one - grid) * nu[...,None]
    if sim_variables.magnetic:
        mu[...,Bfields] = 0
    mu = fv.add_boundary(mu, sim_variables.boundary, axis=axis)

    return fv.slice_(mu, axis=axis, start=1)