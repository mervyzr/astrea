import concurrent.futures
from itertools import repeat

import numpy as np

from functions import constructor, fv
from num_methods import ct, limiters, solvers

##############################################################################
# Piecewise parabolic reconstruction method (PPM) [Colella & Woodward, 1984]
# [McCorquodale & Colella, 2011 (MC:2011); Colella et al., 2011 (C+:2011); Peterson & Hammett, 2008 (PH:2008)]
##############################################################################

def run(grid, sim_variables, axis, eta=None, author="MC:2011"):
    multidimensional, axes, magnetic, ds, dissipate = sim_variables.multidimensional, sim_variables.axes, sim_variables.magnetic, sim_variables.ds, sim_variables.ppm_dissipate
    convert, data = sim_variables.convert, {}

    author = author.lower()
    sim_variables.ppm_author = author

    Riemann_solver = solvers.get_Riemann_solver(sim_variables)
    ortho_axes = axes[axes != axis] if (magnetic or multidimensional) else 0

    # Pad array with boundary; PPM requires additional ghost cells
    padded_grid_2 = fv.add_boundary(grid, sim_variables, stencil=2, axis=axis)
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
        interface[...,5+axes] = grid[...,5+axes]

        # Magnetic transverse interfaces reconstructed along orthogonal axis/axes (interface = centre for PCM)
        if multidimensional:
            if dissipate:
                data['ortho_interfaces'] = ct.reconstruct_transverse(grid, sim_variables, axis=axis, extras=[grid, eta])
            else:
                data['ortho_interfaces'] = ct.reconstruct_transverse(grid, sim_variables, axis=axis)

    if author.startswith(("peterson", "p", "ph", "x")):
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
        padded_interface_2 = np.zeros_like(fv.add_boundary(right_of_centre, sim_variables, stencil=2, axis=axis))

    else:
        # Limit interface values [Colella et al., 2011, p. 25-26]
        if author.startswith(("colella", "c", "c+")):
            interface = limiters.interface_limiter(interface, *grid_slices)

        # Define the left and right parabolic extrapolants
        padded_interface_2 = fv.add_boundary(interface, sim_variables, stencil=2, axis=axis)
        limited_wFs = fv.slice_(padded_interface_2, axis, *[1,-3]), fv.slice_(padded_interface_2, axis, *[2,-2])

    """Reconstruct the limited parabolic extrapolants from the interface values [McCorquodale & Colella, 2011; Colella et al., 2011; Peterson & Hammett, 2008]
    |                        w(i-1/2)                    w(i+1/2)                       |
    |-->         i-1         <--|-->          i          <--|-->         i+1         <--|
    |   w_L(i-1)     w_R(i-1)   |   w_L(i)         w_R(i)   |   w_L(i+1)     w_R(i+1)   |
    |   w+(i-3/2)   w-(i-1/2)   |   w+(i-1/2)   w-(i+1/2)   |  w+(i+1/2)    w-(i+3/2)   |
    """
    wL, wR = limiters.extrapolant_limiter(grid, sim_variables, axis, *limited_wFs, **{
        'padded_grid':padded_grid, 'padded_grid_2':padded_grid_2, 'padded_interface_2':padded_interface_2
        })

    if dissipate:
        wL = wL * eta[...,None] + grid * (1-eta)[...,None]
        wR = wR * eta[...,None] + grid * (1-eta)[...,None]

    # Re-align the interfaces so that cell wall is in between interfaces
    prim_plus, prim_minus = fv.slice_(fv.add_boundary(wL, sim_variables, axis=axis), axis, start=1), fv.slice_(fv.add_boundary(wR, sim_variables, axis=axis), axis, end=-1)
    if magnetic:
        prim_plus[...,5+axes] = prim_minus[...,5+axes] = fv.slice_(padded_grid, axis, end=-1)[...,5+axes]

    # Get the average solution between the interfaces at the boundaries
    intf_avg = fv.compute_Roe_average([prim_plus,prim_minus], sim_variables)
    padded_intf_avg = fv.add_boundary(fv.slice_(intf_avg, axis, start=1), sim_variables, axis=axis)

    # Convert the primitive variables
    cons_plus, cons_minus = convert("primitive", prim_plus, sim_variables, axis=axis, pos='intf'), convert("primitive", prim_minus, sim_variables, axis=axis, pos='intf')

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
    if multidimensional:
        # Calculate the interface-centred fluxes
        intf_fluxes_cntrd = Riemann_solver(axis, sim_variables, **{
            'prim_interfaces': fv.approx_face_avg([prim_plus, prim_minus], sim_variables, axis),
            'cons_interfaces': fv.approx_face_avg([cons_plus, cons_minus], sim_variables, axis),
            'flux_interfaces': fv.approx_face_avg([flux_plus, flux_minus], sim_variables, axis),
            'characteristics': characteristics,
        })

        # Compute the 4th-order interface-centred fluxes from the interface-averaged fluxes via higher order approximation for each orthogonal axis
        with concurrent.futures.ThreadPoolExecutor() as inner_executor:
            jobs = inner_executor.map(fv.laplacian, repeat(intf_fluxes_avgd), repeat(sim_variables), ortho_axes)
            for idx, job in enumerate(jobs):
                intf_fluxes_cntrd -= (sim_variables.ds[ortho_axes[idx]]**2)/24 * job
    else:
        # Orthogonal Laplacian in 1d is zero
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

    padded_primitive_2 = fv.add_boundary(grid, sim_variables, stencil=2, axis=axis)
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

    # Create flattening coefficient
    chi = np.copy(chi_bar)
    chi_bar_padded = fv.add_boundary(chi_bar, sim_variables, axis=axis)
    signage = np.sign(plus_one[...,pressure]-minus_one[...,pressure])
    chi[signage < 0] = np.minimum(chi_bar, fv.slice_(chi_bar_padded, axis, start=2))[signage < 0]
    chi[signage > 0] = np.minimum(chi_bar, fv.slice_(chi_bar_padded, axis, end=-2))[signage > 0]

    return chi



# Implement artificial viscosity [McCorquodale & Colella, 2011]
def get_artificial_viscosity(grid_slices, axis, sim_variables, viscosity_determinants=[.3, .3]):
    alpha, beta = viscosity_determinants
    rho, pressure, Bfields, axes = sim_variables.rho, sim_variables.pressure, sim_variables.Bfields, sim_variables.axes

    zeroth, plus_one = grid_slices
    ortho_axes = axes[axes != axis]

    def per_ortho_axis(_grid_slices, _sim_variables, _axis):
        _zeroth, _plus_one = _grid_slices
        padded_zeroth = fv.add_boundary(_zeroth, _sim_variables, axis=_axis)
        padded_plus_one = fv.add_boundary(_plus_one, _sim_variables, axis=_axis)
        return .25 * (
            fv.slice_(padded_plus_one, _axis, start=2) - fv.slice_(padded_plus_one, _axis, end=-2) 
            + fv.slice_(padded_zeroth, _axis, start=2) - fv.slice_(padded_zeroth, _axis, end=-2)
        )

    # Calculate face-centred divergence of velocity [eq. 35]
    lambda_d = plus_one - zeroth
    if sim_variables.multidimensional:
        with concurrent.futures.ThreadPoolExecutor() as inner_executor:
            jobs = inner_executor.map(per_ortho_axis, repeat(grid_slices), repeat(sim_variables), ortho_axes)
            lambda_d += np.sum([job for job in jobs], axis=0)

    # Calculate minimum sound speed
    cs_grid = fv.divide(sim_variables.gamma * zeroth[...,pressure], zeroth[...,rho])
    cs_plus_one = fv.divide(sim_variables.gamma * plus_one[...,pressure], plus_one[...,rho])
    c_min = np.minimum(cs_grid, cs_plus_one)

    # Calculate artificial viscosity coefficient [eq. 36]
    nu = np.minimum(1, fv.divide(lambda_d[...,1+axis]**2, beta*c_min)) * lambda_d[...,1+axis]
    nu[lambda_d[...,1+axis] >= 0] = 0

    # Calculate the artificial viscosity [eq. 38]
    mu = alpha * (plus_one - zeroth) * nu[...,None]
    if sim_variables.magnetic:
        mu[...,Bfields] = 0
    mu = fv.add_boundary(mu, sim_variables, axis=axis)

    return fv.slice_(mu, axis=axis, start=1)