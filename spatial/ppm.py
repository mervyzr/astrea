import numpy as np

from functions import grid as gutils
from functions import math as mfuncs
from functions import numeric
from numkit import c_transport as ct
from numkit import limiters, solvers

##############################################################################
# Piecewise parabolic reconstruction method (PPM) [Colella & Woodward, 1984; McCorquodale & Colella, 2011; Felker & Stone, 2018]
##############################################################################

def reconstruct(grid, sim_variables, axis):
    # Pad array with boundary; PPM requires additional ghost cells
    padded_grid_2 = gutils.add_boundary(grid, sim_variables, stencil=2, axis=axis)
    padded_grid = gutils.slice_(padded_grid_2, axis, *[1,-1])

    minus_one, minus_two = gutils.slice_(padded_grid, axis, end=-2), gutils.slice_(padded_grid_2, axis, end=-4)
    plus_one, plus_two = gutils.slice_(padded_grid, axis, start=2), gutils.slice_(padded_grid_2, axis, start=4)

    """Interpolate (forward/upwind) the cell averages to face averages <w>_{i+1/2,j} [McCorquodale & Colella, 2011, eq. 17; Colella et al., 2011, eq. 67]
    |               w(i-1/2)            w(i+1/2)                |
    |  i-1           -->|   i            -->|  i+1           -->|
    |        w_R(i-1)   |          w_R(i)   |        w_R(i+1)   |
    """
    interface = 7/12 * (grid + plus_one) - 1/12 * (minus_one + plus_two)  # 4th-order [Felker & Stone, 2018, eq. 10]
    #interface = 1/60 * (2*minus_two - 13*minus_one + 47*grid + 27*plus_one - 3*plus_two)  # 5th-order [Suresh & Huynh, 1997, eq. 2.1]

    if sim_variables.magnetic:
        interface[...,5+sim_variables.axes] = grid[...,5+sim_variables.axes]

    if sim_variables.ppm_author.casefold().startswith(("peterson", "p", "ph", "x")):
        """Reconstruction from cell averages to face averages (both sides)
        |                        w(i-1/2)                    w(i+1/2)                       |
        |<--         i-1         -->|<--          i          -->|<--         i+1         -->|
        |   w_L(i-1)     w_R(i-1)   |   w_L(i)         w_R(i)   |   w_L(i+1)     w_R(i+1)   |
        """
        left_of_centre = 7/12 * (minus_one + grid) - 1/12 * (minus_two + plus_one)
        right_of_centre = interface

        # Limit interface values [Peterson & Hammett, 2008, eq. 3.33-3.34]
        padded_interface_2 = np.zeros_like(gutils.add_boundary(right_of_centre, sim_variables, stencil=2, axis=axis))
        limited_wFs = (
            limiters.interface_limit(left_of_centre, *(minus_two, minus_one, grid, plus_one)), 
            limiters.interface_limit(right_of_centre, *(minus_one, grid, plus_one, plus_two))
        )

    else:
        # Limit interface values [Colella et al., 2011, p. 25-26]
        if sim_variables.ppm_author.casefold().startswith(("colella", "c", "c+")):
            interface = limiters.interface_limit(interface, *(minus_one, grid, plus_one, plus_two))

        # Define the left and right parabolic extrapolants
        padded_interface_2 = gutils.add_boundary(interface, sim_variables, stencil=2, axis=axis)
        limited_wFs = (
            gutils.slice_(padded_interface_2, axis, *[1,-3]), 
            gutils.slice_(padded_interface_2, axis, *[2,-2])
        )

    """Reconstruct the limited parabolic extrapolants from the interface values [McCorquodale & Colella, 2011; Colella et al., 2011; Peterson & Hammett, 2008]
    |                        w(i-1/2)                    w(i+1/2)                       |
    |-->         i-1         <--|-->          i          <--|-->         i+1         <--|
    |   w_L(i-1)     w_R(i-1)   |   w_L(i)         w_R(i)   |   w_L(i+1)     w_R(i+1)   |
    |   w+(i-3/2)   w-(i-1/2)   |   w+(i-1/2)   w-(i+1/2)   |  w+(i+1/2)    w-(i+3/2)   |
    """
    wL, wR = limiters.extrapolant_limit(grid, sim_variables, axis, *limited_wFs, **{
        'padded_grid':padded_grid, 'padded_grid_2':padded_grid_2, 'padded_interface_2':padded_interface_2
        })

    return wL, wR


def run(grid, sim_variables, axis, eta=None):
    multidimensional, magnetic, ds = sim_variables.multidimensional, sim_variables.magnetic, sim_variables.ds[axis]

    Riemann_solver = solvers.get_Riemann_solver(sim_variables)

    # Parabolic piecewise reconstruction [McCorquodale & Colella, 2011; Colella et al., 2011; Peterson & Hammett, 2008]
    wL, wR = reconstruct(grid, sim_variables, axis)
    if sim_variables.ppm_dissipate:
        wL = wL * eta[...,None] + grid * (1-eta)[...,None]
        wR = wR * eta[...,None] + grid * (1-eta)[...,None]

    # Re-align the interfaces so that cell wall is in between interfaces
    assign_interfaces = ct.assign_interfaces if magnetic else gutils.assign_interfaces
    prim_plus, prim_minus = assign_interfaces((wL, wR), grid, sim_variables, axis)

    # Get the average solution between the interfaces at the boundaries
    intf_avg = numeric.compute_Roe_average((prim_plus, prim_minus), sim_variables)
    padded_intf_avg = gutils.add_boundary(intf_avg, sim_variables, axis=axis)

    # Convert the primitive variables at the interface
    cons_plus, cons_minus = gutils.variable_convert_intf("primitive", prim_plus, sim_variables, axis=axis), gutils.variable_convert_intf("primitive", prim_minus, sim_variables, axis=axis)

    # Compute the fluxes and the Jacobian
    flux_plus, flux_minus = numeric.compute_flux(prim_plus, sim_variables, axis=axis), numeric.compute_flux(prim_minus, sim_variables, axis=axis)
    jacobian = numeric.compute_jacobian(padded_intf_avg, sim_variables, axis=axis)

    # Resolve characteristics at interfaces
    try:
        characteristics = np.linalg.eigvals(jacobian)
    except np.linalg.LinAlgError:
        try:
            characteristics = numeric.compute_characteristics(padded_intf_avg, sim_variables, axis=axis)
        except np.linalg.LinAlgError:
            characteristics = np.full_like(padded_intf_avg, .01)

    # Calculate the interface-averaged fluxes
    intf_fluxes_avgd = Riemann_solver(axis, sim_variables, **{
        'prim_interfaces': (prim_plus, prim_minus),
        'cons_interfaces': (cons_plus, cons_minus),
        'flux_interfaces': (flux_plus, flux_minus),
        'characteristics': characteristics,
        'jacobian': gutils.slice_(jacobian, axis, *[1,-1]),
    })

    # Compute the orthogonal L/R Riemann states and fluxes at higher-order accuracy
    if multidimensional:
        # Calculate the interface-centred fluxes
        intf_fluxes_cntrd = Riemann_solver(axis, sim_variables, **{
            'prim_interfaces': gutils.approx_face_avg((prim_plus, prim_minus), sim_variables, axis),
            'cons_interfaces': gutils.approx_face_avg((cons_plus, cons_minus), sim_variables, axis),
            'flux_interfaces': gutils.approx_face_avg((flux_plus, flux_minus), sim_variables, axis),
            'characteristics': characteristics,
            'jacobian': gutils.slice_(jacobian, axis, *[1,-1]),
        })

        # Compute the higher-order fluxes
        intf_fluxes_cntrd = gutils.approx_flux_avg(intf_fluxes_cntrd, intf_fluxes_avgd, sim_variables, axis)
    else:
        # Orthogonal Laplacian in 1d is zero
        intf_fluxes_cntrd = intf_fluxes_avgd

    # Add additional dissipation for strong shocks, if switched on (should not apply for mag. fields) [McCorquodale & Colella, 2011]
    if sim_variables.ppm_dissipate:
        plus_one = gutils.slice_(gutils.add_boundary(grid, sim_variables, axis=axis), axis, start=2)
        intf_fluxes_cntrd += get_artificial_viscosity((grid, plus_one), axis, sim_variables)

    # Compute flux difference for hydrodynamic components
    fluxes = np.diff(intf_fluxes_cntrd, axis=axis)/ds

    if magnetic and multidimensional:
        return fluxes, characteristics, (gutils.slice_(prim_plus, axis, start=1), gutils.slice_(prim_minus, axis, end=-1))
    else:
        return fluxes, characteristics, None


# Calculate the coefficient of the slope flattener for the parabolic interpolants/extrapolants [Colella, 1990]
def get_flattening_coeff(grid, sim_variables, slope_determinants=[.33, .75, .85]):
    delta, z0, z1 = slope_determinants
    axes, pressure = sim_variables.axes, sim_variables.pressure

    def coefficient_per_axis(_grid, _sim_variables, axis):
        padded_primitive_2 = gutils.add_boundary(_grid, _sim_variables, stencil=2, axis=axis)
        padded_primitive = gutils.slice_(padded_primitive_2, axis, *[1,-1])

        minus_one, minus_two = gutils.slice_(padded_primitive, axis, end=-2), gutils.slice_(padded_primitive_2, axis, end=-4)
        plus_one, plus_two = gutils.slice_(padded_primitive, axis, start=2), gutils.slice_(padded_primitive_2, axis, start=4)

        # zeta function
        z = mfuncs.divide(np.abs(plus_one[...,pressure]-minus_one[...,pressure]), np.abs(plus_two[...,pressure]-minus_two[...,pressure]))
        chi_bar = 1 - mfuncs.divide(z-z0, z1-z0)
        chi_bar[z > z1] = 0
        chi_bar[z < z0] = 1

        # Update chi_bar based on condition [eq. 4.9]
        otherwise_condition = np.where(
            ((minus_one[...,1+axis]-plus_one[...,1+axis]) <= 0)
            & (mfuncs.divide(np.abs(plus_one[...,pressure]-minus_one[...,pressure]), np.minimum(plus_one[...,pressure], minus_one[...,pressure])) <= delta))
        chi_bar[otherwise_condition] = 0

        # Create flattening coefficient
        chi = np.copy(chi_bar)
        chi_bar_padded = gutils.add_boundary(chi_bar, _sim_variables, axis=axis)
        signage = np.sign(plus_one[...,pressure]-minus_one[...,pressure])
        chi[signage < 0] = np.minimum(chi_bar, gutils.slice_(chi_bar_padded, axis, start=2))[signage < 0]
        chi[signage > 0] = np.minimum(chi_bar, gutils.slice_(chi_bar_padded, axis, end=-2))[signage > 0]

        return chi
    
    return np.minimum(1, np.min([coefficient_per_axis(grid, sim_variables, axis) for axis in axes], axis=0))


# Implement artificial viscosity [McCorquodale & Colella, 2011]
def get_artificial_viscosity(grid_slices, axis, sim_variables, viscosity_determinants=[.3, .3]):
    alpha, beta = viscosity_determinants
    rho, pressure, Bfields, axes = sim_variables.rho, sim_variables.pressure, sim_variables.Bfields, sim_variables.axes

    zeroth, plus_one = grid_slices
    ortho_axes = axes[axes != axis]

    def per_ortho_axis(_grid_slices, _sim_variables, _axis):
        _zeroth, _plus_one = _grid_slices
        padded_zeroth = gutils.add_boundary(_zeroth, _sim_variables, axis=_axis)
        padded_plus_one = gutils.add_boundary(_plus_one, _sim_variables, axis=_axis)
        return .25 * (
            gutils.slice_(padded_plus_one, _axis, start=2) - gutils.slice_(padded_plus_one, _axis, end=-2) 
            + gutils.slice_(padded_zeroth, _axis, start=2) - gutils.slice_(padded_zeroth, _axis, end=-2)
        )

    # Calculate face-centred divergence of velocity [eq. 35]
    lambda_d = plus_one - zeroth
    if sim_variables.multidimensional:
        lambda_d += np.sum([per_ortho_axis(grid_slices, sim_variables, ortho_axis) for ortho_axis in ortho_axes], axis=0)

    # Calculate minimum sound speed
    cs_grid = mfuncs.divide(sim_variables.gamma * zeroth[...,pressure], zeroth[...,rho])
    cs_plus_one = mfuncs.divide(sim_variables.gamma * plus_one[...,pressure], plus_one[...,rho])
    c_min = np.minimum(cs_grid, cs_plus_one)

    # Calculate artificial viscosity coefficient [eq. 36]
    nu = np.minimum(1, mfuncs.divide(lambda_d[...,1+axis]**2, beta*c_min)) * lambda_d[...,1+axis]
    nu[lambda_d[...,1+axis] >= 0] = 0

    # Calculate the artificial viscosity [eq. 38]
    mu = alpha * (plus_one - zeroth) * nu[...,None]
    if sim_variables.magnetic:
        mu[...,Bfields] = 0
    mu = gutils.add_boundary(mu, sim_variables, axis=axis)

    return gutils.slice_(mu, axis=axis, start=1)