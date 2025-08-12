import numpy as np

from functions import constructor, fv
from num_methods import ct, limiters, solvers

##############################################################################
# Piecewise parabolic reconstruction method (PPM) [Colella & Woodward, 1984]
# [McCorquodale & Colella, 2011 (MC:2011); Colella et al., 2011 (C+:2011); Peterson & Hammett, 2008 (PH:2008)]
##############################################################################

def run(grid, sim_variables, axis, author="MC:2011", **kwargs):
    boundary, dimension, magnetic, ds, dissipate = sim_variables.boundary, sim_variables.dimension, sim_variables.magnetic, sim_variables.ds, sim_variables.ppm_dissipate
    Bx, By = sim_variables.Bx, sim_variables.By
    data = {}

    Riemann_solver = solvers.get_Riemann_solver(sim_variables)
    ortho_axis = 1 - axis if (magnetic or dimension == 2) else 0

    # Approximate the face-averaged values to face-centred values (for higher-order flux calculations)
    def approx_face_avg(_axis, _boundary, *_interfaces):
        plus_intf, minus_intf = _interfaces
        padded_plus_intf, padded_minus_intf = fv.add_boundary(plus_intf, _boundary, axis=_axis), fv.add_boundary(minus_intf, _boundary, axis=_axis)
        return np.copy(plus_intf) - 1/24 * fv.derivative(padded_plus_intf, axis=_axis), np.copy(minus_intf) - 1/24 * fv.derivative(padded_minus_intf, axis=_axis)


    # Pad array with boundary; PPM requires additional ghost cells
    padded_grid_2 = fv.add_boundary(grid, boundary, stencil=2, axis=axis)
    padded_grid = fv.slice_(padded_grid_2, axis, *[1,-1])

    minus_one, minus_two = fv.slice_(padded_grid, axis, end=-2), fv.slice_(padded_grid_2, axis, end=-4)
    plus_one, plus_two = fv.slice_(padded_grid, axis, start=2), fv.slice_(padded_grid_2, axis, start=4)
    grid_slices = [minus_one, grid, plus_one, plus_two]

    """Interpolate the cell averages to face averages (forward/upwind)
    |               w(i-1/2)            w(i+1/2)                |
    |  i-1           -->|   i            -->|  i+1           -->|
    |        w_R(i-1)   |          w_R(i)   |        w_R(i+1)   |
    """
    interface = 7/12 * (grid + plus_one) - 1/12 * (minus_one + plus_two)  # Face i+1/2 (4th-order) [McCorquodale & Colella, 2011, eq. 17; Colella et al., 2011, eq. 67]
    #interface = 1/60 * (2*minus_two - 13*minus_one + 47*grid + 27*plus_one - 3*plus_two)  # Face i+1/2 (5th-order, less robust) [Suresh & Huynh, 1997, eq. 2.1]
    if magnetic:
        interface[...,(Bx,By)] = grid[...,(Bx,By)]

        # Magnetic transverse interfaces reconstructed orthogonal to the axis
        ortho_plus, ortho_minus = ct.reconstruct_transverse(interface, sim_variables, axis=ortho_axis)
        data['ortho_interfaces'] = fv.slice_(ortho_plus, axis=ortho_axis, start=1), fv.slice_(ortho_minus, axis=ortho_axis, start=1)

    if author.lower().startswith(("peterson", "p", "ph", "x")):
        """Interpolate the cell averages to face averages (both sides)
        |                        w(i-1/2)                    w(i+1/2)                       |
        |<--         i-1         -->|<--          i          -->|<--         i+1         -->|
        |   w_L(i-1)     w_R(i-1)   |   w_L(i)         w_R(i)   |   w_L(i+1)     w_R(i+1)   |
        """
        # Face i+1/2 (4th-order) (eq. 3.26-3.27)
        left_of_centre = 7/12 * (minus_one + grid) - 1/12 * (minus_two + plus_one)
        right_of_centre = interface

        # Limit interface values [Peterson & Hammett, 2008, eq. 3.33-3.34]
        limited_wFs = limiters.interface_limiter(left_of_centre, *[minus_two, minus_one, grid, plus_one]), limiters.interface_limiter(right_of_centre, *grid_slices)
        padded_interface_2 = np.zeros_like(fv.add_boundary(right_of_centre, boundary, stencil=2, axis=axis))

    else:
        # Limit interface values [Colella et al., 2011, p. 25-26]
        if author.lower().startswith(("colella", "c", "c+")):
            interface = limiters.interface_limiter(interface, *grid_slices)

        # Define the left and right parabolic extrapolants
        padded_interface_2 = fv.add_boundary(interface, boundary, stencil=2, axis=axis)
        limited_wFs = fv.slice_(padded_interface_2, axis, *[1,-3]), fv.slice_(padded_interface_2, axis, *[2,-2])

    """Reconstruct the limited parabolic extrapolants from the interface values [McCorquodale & Colella, 2011; Colella et al., 2011; Peterson & Hammett, 2008]
    |                        w(i-1/2)                    w(i+1/2)                       |
    |-->         i-1         <--|-->          i          <--|-->         i+1         <--|
    |   w_L(i-1)     w_R(i-1)   |   w_L(i)         w_R(i)   |   w_L(i+1)     w_R(i+1)   |
    |   w+(i-3/2)   w-(i-1/2)   |   w+(i-1/2)   w-(i+1/2)   |  w+(i+1/2)    w-(i+3/2)   |
    """
    wL, wR = limiters.extrapolant_limiter(grid, boundary, axis, author, *limited_wFs, **{
        'padded_grid':padded_grid, 'padded_grid_2':padded_grid_2, 'padded_interface_2':padded_interface_2
        })

    if dissipate:
        wL = wL * kwargs['eta'][...,None] + grid * (1-kwargs['eta'])[...,None]
        wR = wR * kwargs['eta'][...,None] + grid * (1-kwargs['eta'])[...,None]

    # Re-align the interfaces so that cell wall is in between interfaces
    prim_plus, prim_minus = fv.slice_(fv.add_boundary(wL, boundary, axis=axis), axis, start=1), fv.slice_(fv.add_boundary(wR, boundary, axis=axis), axis, end=-1)
    if magnetic:
        prim_plus[...,(Bx,By)] = prim_minus[...,(Bx,By)] = fv.slice_(padded_grid, axis, end=-1)[...,(Bx,By)]

    # Get the average solution between the interfaces at the boundaries
    intf_avg = fv.slice_(fv.compute_Roe_average(prim_plus, prim_minus), axis, start=1)
    padded_intf_avg = fv.add_boundary(intf_avg, boundary, axis=axis)

    # Convert the primitive variables
    cons_plus, cons_minus = fv.convert_interface("primitive", prim_plus, axis, sim_variables), fv.convert_interface("primitive", prim_minus, axis, sim_variables)

    # Compute the fluxes and the Jacobian
    flux_plus, flux_minus = constructor.make_flux(prim_plus, sim_variables, axis=axis), constructor.make_flux(prim_minus, sim_variables, axis=axis)
    jacobian = constructor.make_Jacobian(padded_intf_avg, sim_variables, axis=axis)

    # Compute eigmax for time stepping limits
    characteristics = np.linalg.eigvals(jacobian)
    data['eigmax'] = ds[axis]/fv.compute_eigmax(characteristics, axis=axis)

    # Magnetic alpha computation
    if magnetic:
        # alphas refers to the maximum(+)/minimum(-) eigenvalues respectively
        local_max, local_min = np.max(characteristics, axis=-1), np.min(characteristics, axis=-1)
        max_eigvals = np.maximum(fv.slice_(local_max, axis, end=-1), fv.slice_(local_max, axis, start=1))
        min_eigvals = np.minimum(fv.slice_(local_min, axis, end=-1), fv.slice_(local_min, axis, start=1))
        data['alphas'] = fv.slice_(np.maximum(0, max_eigvals), axis, start=1), fv.slice_(-np.minimum(0, min_eigvals), axis, start=1)

    # Calculate the interface-averaged fluxes
    intf_fluxes_avgd = Riemann_solver(axis, sim_variables, **{
        'prim_interfaces': [prim_plus, prim_minus],
        'cons_interfaces': [cons_plus, cons_minus],
        'flux_interfaces': [flux_plus, flux_minus],
        'characteristics': characteristics,
    })

    # Compute the orthogonal L/R Riemann states and fluxes at higher-order accuracy
    if dimension == 2:
        # Calculate the interface-centred fluxes
        intf_fluxes_cntrd = Riemann_solver(axis, sim_variables, **{
            'prim_interfaces': approx_face_avg(ortho_axis, sim_variables.boundary, *[prim_plus, prim_minus]),
            'cons_interfaces': approx_face_avg(ortho_axis, sim_variables.boundary, *[cons_plus, cons_minus]),
            'flux_interfaces': approx_face_avg(ortho_axis, sim_variables.boundary, *[flux_plus, flux_minus]),
            'characteristics': characteristics,
        })

        # Compute the 4th-order interface-centred fluxes from the interface-averaged fluxes via higher order approximation
        padded_avg_flux = fv.add_boundary(intf_fluxes_avgd, sim_variables.boundary, axis=ortho_axis)
        intf_fluxes_cntrd -= 1/24 * fv.derivative(padded_avg_flux, axis=ortho_axis)
    else:
        # Orthogonal Laplacian in 1D is zero
        intf_fluxes_cntrd = intf_fluxes_avgd

    # Add additional dissipation for strong shocks, if switched on (should not apply for mag. fields) [McCorquodale & Colella, 2011]
    if dissipate:
        intf_fluxes_cntrd += get_artificial_viscosity([grid, plus_one], axis, sim_variables)

    # Compute flux difference for hydrodynamic components
    data['fluxes'] = np.diff(intf_fluxes_cntrd, axis=axis)/ds[axis]

    return data



# Calculate the coefficient of the slope flattener for the parabolic interpolants/extrapolants [Colella, 1990]
def get_flattening_coeff(grid, sim_variables, axis, slope_determinants=[.33, .75, .85]):
    delta, z0, z1 = slope_determinants
    pressure = sim_variables.pressure

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

    return chi



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