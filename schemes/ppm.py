from collections import defaultdict

import numpy as np

from functions import fv, constructors
from numerics import limiters

##############################################################################
# Piecewise parabolic reconstruction method (PPM) [Colella & Woodward, 1984]
##############################################################################


# [Colella et al., 2011]
def run_C(grid, sim_variables, C=5/4):
    gamma, boundary, permutations = sim_variables.gamma, sim_variables.boundary, sim_variables.permutations
    nested_dict = lambda: defaultdict(nested_dict)
    data = nested_dict()

    # Rotate grid and apply algorithm for each axis
    for axis, axes in enumerate(permutations):
        _grid = grid.transpose(axes)

        # Convert to primitive variables
        wS = fv.convert_conservative(_grid, sim_variables)

        # Pad array with boundary; PPM requires additional ghost cells
        w2 = fv.add_boundary(wS, boundary, 2)
        w = np.copy(w2[1:-1])

        """Extrapolate the cell averages to face averages
        Current convention: |  i-1     ---> |  i       ---> |  i+1     ---> |
                            |       w(i-1/2)|       w(i+1/2)|       w(i+3/2)|
        """
        # Face i+1/2 (4th-order) [McCorquodale & Colella, 2011, eq. 17; Colella et al., 2011, eq. 67]
        wF = 7/12 * (wS + w[2:]) - 1/12 * (w[:-2] + w2[4:])

        # Limit interface values [Colella et al., 2011, p. 25-26]
        wF_limit = limiters.interface_limiter(wF, w[:-2], wS, w[2:], w2[4:], C)

        """Reconstruct the interpolants using the limited values
        Current convention: |               w(i-1/2)                    w(i+1/2)              |
                            | i-1          <-- | -->         i         <-- | -->          i+1 |
                            |        w_R(i-1)  |   w_L(i)          w_R(i)  |  w_L(i+1)        |
                    OR      |       w-(i-1/2)  |   w+(i-1/2)    w-(i+1/2)  |  w+(i+1/2)       |
        """
        # Limited parabolic interpolant [Colella et al., 2011, p. 26]
        wF_pad2 = fv.add_boundary(wF_limit, boundary, 2)
        wF_limit_L, wF_limit_R = np.copy(wF_pad2[1:-3]), np.copy(wF_pad2[2:-2])

        # Check for cell extrema in cells (eq. 89)
        d_uL, d_uR = wS - wF_limit_L, wF_limit_R - wS
        cell_extrema = d_uL*d_uR < 0

        # Check for overshoot in cells (eq. 90)
        overshoot = (np.abs(d_uL) > 2*np.abs(d_uR)) | (np.abs(d_uR) > 2*np.abs(d_uL))

        # Check for extrema in interpolants (eq. 91-94)
        d_wF_minmod_L, d_wF_minmod_R = wF_limit_L - np.copy(wF_pad2[:-4]), np.copy(wF_pad2[4:]) - wF_limit_R
        d_wS_minmod_L, d_wS_minmod_R = wS - w[:-2], w[2:] - wS

        d_wF_minmod = np.minimum(np.abs(d_wF_minmod_L), np.abs(d_wF_minmod_R))
        d_wS_minmod = np.minimum(np.abs(d_wS_minmod_L), np.abs(d_wS_minmod_R))

        interpolant_extrema = ((d_wF_minmod >= d_wS_minmod) & (d_wF_minmod_L*d_wF_minmod_R < 0)) | ((d_wS_minmod >= d_wF_minmod) & (d_wS_minmod_L*d_wS_minmod_R < 0))

        # If there are extrema in either the cells or interpolants
        if cell_extrema.any() or interpolant_extrema.any():
            D2w_lim = np.zeros_like(wS)

            # Approximation to the second derivative (eq. 95)
            D2w = 6 * (wF_limit_L - 2*wS + wF_limit_R)
            D2w_L = w2[:-4] - 2*w[:-2] + wS
            D2w_C = w[:-2] - 2*wS + w[2:]
            D2w_R = wS - 2*w[2:] + w2[4:]

            # Get the curvatures that have the same signs
            non_monotonic = (np.sign(D2w) == np.sign(D2w_C)) & (np.sign(D2w) == np.sign(D2w_L)) & (np.sign(D2w) == np.sign(D2w_R)) & (np.sign(D2w_C) == np.sign(D2w_L)) & (np.sign(D2w_C) == np.sign(D2w_R)) & (np.sign(D2w_L) == np.sign(D2w_R))

            # Determine the limited curvature with the sign of each element in the 'main' array (eq. 96)
            limited_curvature = np.sign(D2w) * np.minimum(np.minimum(np.abs(D2w), np.abs(C*D2w_C)), np.minimum(np.abs(C*D2w_L), np.abs(C*D2w_R)))

            # Update the limited local curvature estimates based on the conditions
            D2w_lim[cell_extrema & non_monotonic] = limited_curvature[cell_extrema & non_monotonic]
            D2w_lim[interpolant_extrema & non_monotonic] = limited_curvature[interpolant_extrema & non_monotonic]

            phi = fv.divide(D2w_lim, D2w)

            # Further update if there is local extrema (eq. 97-98)
            d_uL_bar, d_uR_bar = np.copy(d_uL), np.copy(d_uR)
            if overshoot.any():
                d_uL_bar[np.abs(d_uL) > 2*np.abs(d_uR)] = 2*d_uR[np.abs(d_uL) > 2*np.abs(d_uR)]
                d_uR_bar[np.abs(d_uR) > 2*np.abs(d_uL)] = 2*d_uL[np.abs(d_uR) > 2*np.abs(d_uL)]
            wL, wR = wS - phi*d_uL_bar, wS + phi*d_uR_bar  # (eq. 98)
        else:
            wL, wR = wF_limit_L, wF_limit_R

        # Get the average solution
        avg_wS = constructors.make_Roe_average(wL, wR)

        # Pad the reconstructed interfaces
        wLs, wRs = fv.add_boundary(wL, boundary)[1:], fv.add_boundary(wR, boundary)[:-1]

        # Convert the primitive variables
        qLs, qRs = fv.convert_primitive(wLs, sim_variables, "face"), fv.convert_primitive(wRs, sim_variables, "face")

        # Compute the fluxes and the Jacobian
        _w = fv.add_boundary(avg_wS, boundary)
        fLs, fRs = constructors.make_flux_term(wLs, gamma, axis), constructors.make_flux_term(wRs, gamma, axis)
        A = constructors.make_Jacobian(_w, gamma, axis)

        # Update dict
        data[axes]['wS'] = wS
        data[axes]['wLs'] = wLs
        data[axes]['wRs'] = wRs
        data[axes]['qLs'] = qLs
        data[axes]['qRs'] = qRs
        data[axes]['fLs'] = fLs
        data[axes]['fRs'] = fRs
        data[axes]['jacobian'] = A

    return data


# [Peterson & Hammett, 2008]
def run_XPPM(grid, sim_variables, C=5/4):
    gamma, boundary, permutations = sim_variables.gamma, sim_variables.boundary, sim_variables.permutations
    nested_dict = lambda: defaultdict(nested_dict)
    data = nested_dict()

    # Rotate grid and apply algorithm for each axis
    for axis, axes in enumerate(permutations):
        _grid = grid.transpose(axes)

        # Convert to primitive variables
        wS = fv.convert_conservative(_grid, sim_variables)

        # Pad array with boundary; PPM requires additional ghost cells
        w2 = fv.add_boundary(wS, boundary, 2)
        w = np.copy(w2[1:-1])

        """Extrapolate the cell averages to face averages
        Current convention: | <---     i-1     ---> | <---      i      ---> | <---     i+1     ---> |
                            |w_L(i-1)       w_R(i-1)|w_L(i)           w_R(i)|w_L(i+1)       w_R(i+1)|
        """
        # Face i+1/2 (4th-order) (eq. 3.26-3.27)
        wF_L = 7/12 * (w[:-2] + wS) - 1/12 * (w2[:-4] + w[2:])
        wF_R = 7/12 * (wS + w[2:]) - 1/12 * (w[:-2] + w2[4:])

        # Face i+1/2 (5th-order, eq. 3.40)
        #wF_R = 1/60 * (2*w2[:-4] - 13*w[:-2] + 47*wS + 27*w[2:] - 2*w2[4:])

        # Limit the interface values (eq. 3.33-3.34)
        wF_limit_L = limiters.interface_limiter(wF_L, w2[:-4], w[:-2], wS, w[2:], C)
        wF_limit_R = limiters.interface_limiter(wF_R, w[:-2], wS, w[2:], w2[4:], C)

        # Check for cell extrema in cells (eq. 3.31)
        d_uL, d_uR = wS - wF_limit_L, wF_limit_R - wS
        cell_extrema = d_uL*d_uR <= 0
        overshoot = (w[:-2]-wS)*(wS-w[2:]) <= 0

        # If there are extrema in the cells or overshoots
        if cell_extrema.any() or overshoot.any():
            D2w_lim = np.zeros_like(wS)

            # Approximation to the second derivative (eq. 3.37)
            D2w = 6 * (wF_limit_L - 2*wS + wF_limit_R)
            D2w_L = w2[:-4] - 2*w[:-2] + wS
            D2w_C = w[:-2] - 2*wS + w[2:]
            D2w_R = wS - 2*w[2:] + w2[4:]

            # Get the curvatures that have the same signs
            non_monotonic = (np.sign(D2w) == np.sign(D2w_C)) & (np.sign(D2w) == np.sign(D2w_L)) & (np.sign(D2w) == np.sign(D2w_R)) & (np.sign(D2w_C) == np.sign(D2w_L)) & (np.sign(D2w_C) == np.sign(D2w_R)) & (np.sign(D2w_L) == np.sign(D2w_R))

            # Determine the limited curvature with the sign of each element in the 'main' array
            limited_curvature = np.sign(D2w) * np.minimum(np.minimum(np.abs(D2w), np.abs(C*D2w_C)), np.minimum(np.abs(C*D2w_L), np.abs(C*D2w_R)))

            # Update the limited local curvature estimates based on the conditions (eq. 3.38)
            D2w_lim[cell_extrema & non_monotonic] = limited_curvature[cell_extrema & non_monotonic]

            # Get the final limited values (eq. 3.39)
            phi = fv.divide(D2w_lim, D2w)
            wL, wR = wS + phi*(wF_limit_L-wS), wS + phi*(wF_limit_R-wS)
        else:
            wL, wR = wF_limit_L, wF_limit_R

        # Get the average solution
        avg_wS = constructors.make_Roe_average(wL, wR)

        # Pad the reconstructed interfaces
        wLs, wRs = fv.add_boundary(wL, boundary)[1:], fv.add_boundary(wR, boundary)[:-1]

        # Convert the primitive variables
        qLs, qRs = fv.convert_primitive(wLs, sim_variables, "face"), fv.convert_primitive(wRs, sim_variables, "face")

        # Compute the fluxes and the Jacobian
        _w = fv.add_boundary(avg_wS, boundary)
        fLs, fRs = constructors.make_flux_term(wLs, gamma, axis), constructors.make_flux_term(wRs, gamma, axis)
        A = constructors.make_Jacobian(_w, gamma, axis)

        # Update dict
        data[axes]['wS'] = wS
        data[axes]['wLs'] = wLs
        data[axes]['wRs'] = wRs
        data[axes]['qLs'] = qLs
        data[axes]['qRs'] = qRs
        data[axes]['fLs'] = fLs
        data[axes]['fRs'] = fRs
        data[axes]['jacobian'] = A

    return data


# [McCorquodale & Colella, 2011]
def run_MC(grid, sim_variables, dissipate=False, C=5/4):
    
    def modify_stencil(_wF, _wS):
        _wF[0] = 1/12 * (25*_wS[1] - 23*_wS[2] + 13*_wS[3] - 3*_wS[4])
        _wF[-1] = 1/12 * (25*_wS[-1] - 23*_wS[-2] + 13*_wS[-3] - 3*_wS[-4])

        _wF[1] = 1/12 * (3*_wS[1] + 13*_wS[2] - 5*_wS[3] + _wS[4])
        _wF[-2] = 1/12 * (3*_wS[-1] + 13*_wS[-2] - 5*_wS[-3] + _wS[-4])
        return _wF

    gamma, boundary, permutations = sim_variables.gamma, sim_variables.boundary, sim_variables.permutations
    nested_dict = lambda: defaultdict(nested_dict)
    data = nested_dict()

    # Rotate grid and apply algorithm for each axis
    for axis, axes in enumerate(permutations):
        _grid = grid.transpose(axes)

        # Convert to primitive variables
        wS = fv.convert_conservative(_grid, sim_variables)

        # Pad array with boundary; PPM requires additional ghost cells
        w2 = fv.add_boundary(wS, boundary, 2)
        w = np.copy(w2[1:-1])

        """Extrapolate the cell averages to face averages
        Current convention: |  i-1     ---> |  i       ---> |  i+1     ---> |
                            |       w(i-1/2)|       w(i+1/2)|       w(i+3/2)|
        """
        # Face i+1/2 (4th-order) [McCorquodale & Colella, 2011, eq. 17; Colella et al., 2011, eq. 67]
        wF = 7/12 * (wS + w[2:]) - 1/12 * (w[:-2] + w2[4:])

        # Modified stencil [McCorquodale & Colella, 2013, eq. 21-22]
        #wF = modify_stencil(wF, wS)

        """Reconstruct the interpolants using the limited values
        Current convention: |               w(i-1/2)                    w(i+1/2)              |
                            | i-1          <-- | -->         i         <-- | -->          i+1 |
                            |        w_R(i-1)  |   w_L(i)          w_R(i)  |  w_L(i+1)        |
                    OR      |       w-(i-1/2)  |   w+(i-1/2)    w-(i+1/2)  |  w+(i+1/2)       |
        """
        # Limited modified parabolic interpolant [McCorquodale & Colella, 2011]
        # Define the left and right parabolic interpolants
        wF_pad = fv.add_boundary(wF, boundary)
        wF_limit_L, wF_limit_R = np.copy(wF_pad[:-2]), np.copy(wF_pad[1:-1])

        # Set differences
        dw_minus, dw_plus = wS - wF_limit_L, wF_limit_R - wS
        d2w = 6 * (wF_limit_L - 2*wS + wF_limit_R)
        d2w_C = w[:-2] - 2*wS + w[2:]

        # Approximation to the third derivative (eq. 23)
        d3w = np.diff(fv.add_boundary(d2w_C, boundary), axis=0)[1:]

        # Check for cell extreme in cells (eq. 24-25)
        cell_extrema = (dw_minus*dw_plus <= 0) | ((wS-w2[:-4])*(w2[4:]-wS) <= 0)

        # If there are extrema in the cells
        if cell_extrema.any():
            d2w_Cw = fv.add_boundary(d2w_C, boundary)
            d2w_lim = np.zeros_like(wS)

            # Get the curvatures that have the same signs
            non_monotonic = (np.sign(d2w_Cw[:-2]) == np.sign(d2w_Cw[2:])) & (np.sign(d2w) == np.sign(d2w_C)) & (np.sign(d2w_C) == np.sign(d2w_Cw[:-2])) & (np.sign(d2w_C) == np.sign(d2w_Cw[2:])) & (np.sign(d2w) == np.sign(d2w_Cw[:-2])) & (np.sign(d2w) == np.sign(d2w_Cw[2:]))
            # Determine the limited curvature with the sign of each element in the 'main' array (eq. 26)
            limited_curvature = np.sign(d2w_C) * np.minimum(np.minimum(np.abs(d2w), C*np.abs(d2w_C)), np.minimum(C*np.abs(d2w_Cw[2:]), C*np.abs(d2w_Cw[:-2])))
            # Update the limited local curvature estimates based on the conditions
            d2w_lim[cell_extrema] = limited_curvature[cell_extrema]

            # Determine the limited values that are sensitive to roundoff errors
            rho_limiter = np.zeros_like(wS)
            # Get the cells where the limited values fulfil the condition
            rho_sensitive = np.abs(d2w) > 1e-12 * np.maximum(np.abs(wS), np.maximum(np.maximum(np.abs(w[:-2]), np.abs(w[2:])), np.maximum(np.abs(w2[:-4]), np.abs(w2[4:]))))
            # Update the limited estimates based on the condition (eq. 27)
            phi = fv.divide(d2w_lim, d2w)
            rho_limiter[rho_sensitive] = phi[rho_sensitive]

            # Apply additional limiters
            d3w_w2 = fv.add_boundary(d3w, boundary, 2)
            d3w_w = d3w_w2[1:-1]
            d3w_min = np.minimum(np.minimum(d3w_w[:-2], d3w), np.minimum(d3w_w2[:-4], d3w_w2[4:]))
            d3w_max = np.maximum(np.maximum(d3w_w[:-2], d3w), np.maximum(d3w_w2[:-4], d3w_w2[4:]))

            # (eq. 28)
            roundoff_limiters = (rho_limiter < (1-1e-12)) | (.1*np.maximum(np.abs(d3w_max), np.abs(d3w_min)) <= (d3w_max-d3w_min))

            # (eq. 29-30)
            wF_limit_L[(dw_minus*dw_plus < 0) & (roundoff_limiters)] = (wS - rho_limiter*dw_minus)[(dw_minus*dw_plus < 0) & (roundoff_limiters)]
            wF_limit_R[(dw_minus*dw_plus < 0) & (roundoff_limiters)] = (wS + rho_limiter*dw_plus)[(dw_minus*dw_plus < 0) & (roundoff_limiters)]

            # (eq. 31-32)
            wF_limit_L[(roundoff_limiters) & (np.abs(dw_minus) >= 2*np.abs(dw_plus))] = (wS - 2*(1-rho_limiter)*dw_plus - rho_limiter*dw_minus)[(roundoff_limiters) & (np.abs(dw_minus) >= 2*np.abs(dw_plus))]
            wF_limit_R[(roundoff_limiters) & (np.abs(dw_plus) >= 2*np.abs(dw_minus))] = (wS + 2*(1-rho_limiter)*dw_minus + rho_limiter*dw_plus)[(roundoff_limiters) & (np.abs(dw_plus) >= 2*np.abs(dw_minus))]
        else:
            wF_limit_L[np.abs(dw_minus) >= 2*np.abs(dw_plus)] = (wS - 2*dw_plus)[np.abs(dw_minus) >= 2*np.abs(dw_plus)]
            wF_limit_R[np.abs(dw_plus) >= 2*np.abs(dw_minus)] = (wS + 2*dw_minus)[np.abs(dw_plus) >= 2*np.abs(dw_minus)]
        if dissipate:
            eta = calculate_flatten_coeff(wS, boundary)
            wL, wR = (eta*wF_limit_L) + wS*(1-eta), (eta*wF_limit_R) + wS*(1-eta)
        else:
            wL, wR = wF_limit_L, wF_limit_R

        # Get the average solution
        avg_wS = constructors.make_Roe_average(wL, wR)

        # Pad the reconstructed interfaces
        wLs, wRs = fv.add_boundary(wL, boundary)[1:], fv.add_boundary(wR, boundary)[:-1]

        # Convert the primitive variables
        qLs, qRs = fv.convert_primitive(wLs, sim_variables, "face"), fv.convert_primitive(wRs, sim_variables, "face")

        # Compute the fluxes and the Jacobian
        _w = fv.add_boundary(avg_wS, boundary)
        fLs, fRs = constructors.make_flux_term(wLs, gamma, axis), constructors.make_flux_term(wRs, gamma, axis)

        if dissipate:
            qS = fv.add_boundary(_grid, boundary)
            mu = apply_artificial_viscosity(wS, gamma, boundary) * np.diff(qS, axis=0)[1:]
            _mu = fv.add_boundary(mu, boundary)
            f += _mu

        A = constructors.make_Jacobian(_w, gamma, axis)

        # Update dict
        data[axes]['wS'] = wS
        data[axes]['wLs'] = wLs
        data[axes]['wRs'] = wRs
        data[axes]['qLs'] = qLs
        data[axes]['qRs'] = qRs
        data[axes]['fLs'] = fLs
        data[axes]['fRs'] = fRs
        data[axes]['jacobian'] = A

    return data


# Calculate the coefficients of the slope flattener for the parabolic extrapolants using pressure and v_x [Colella, 1990]
def calculate_flatten_coeff(wS, boundary, slope_determinants=[.33, .75, .85]):
    delta, z0, z1 = slope_determinants

    chi_bar = np.zeros_like(wS[:,4])

    vxs = np.pad(wS[:,1], 1, mode=boundary)
    Ps = np.pad(wS[:,4], 2, mode=boundary)

    z = fv.divide(np.abs(Ps[3:-1]-Ps[1:-3]), np.abs(Ps[4:]-Ps[:-4]))

    eta = np.minimum(np.ones_like(z), np.maximum(np.zeros_like(z), 1-((z-z0)/(z1-z0))))
    criteria = ((vxs[:-2]-vxs[2:]) > 0) & (np.abs(Ps[3:-1]-Ps[1:-3])/np.minimum(Ps[3:-1],Ps[1:-3]) > delta)
    chi_bar[criteria] = eta[criteria]
    chi_plus_one = np.pad(chi_bar, 1, mode=boundary)

    chi = np.copy(chi_bar)
    signage = np.sign(Ps[3:-1]-Ps[1:-3])
    chi[signage < 0] = np.minimum(chi_plus_one[2:], chi_bar)[signage < 0]
    chi[signage > 0] = np.minimum(chi_plus_one[:-2], chi_bar)[signage > 0]

    arr = np.ones_like(wS)
    return (chi*arr.T).T


# Implement artificial viscosity
def apply_artificial_viscosity(wS, gamma, boundary, viscosity_determinants=[.3, .3]):
    alpha, beta = viscosity_determinants

    vxs = np.pad(wS[:,1], 1, mode=boundary)
    cs = np.sqrt((gamma*wS[:,4])/wS[:,0])
    Gamma = np.diff(vxs)[1:]

    nu = np.zeros_like(Gamma)
    c_min = np.minimum(cs, np.pad(cs, 1, mode=boundary)[2:])
    nu[Gamma < 0] = (Gamma * np.minimum(np.ones_like(Gamma), (Gamma**2)/(beta*c_min**2)))[Gamma < 0]

    arr = np.ones_like(wS)
    return (alpha*nu*arr.T).T