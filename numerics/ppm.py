from collections import namedtuple

import numpy as np

from functions import fv
from numerics import limiters, solvers

##############################################################################

modified = 1
dissipate = 0

# Piecewise parabolic reconstruction method (PPM)
def run(tube, simVariables):
    gamma, precision, scheme, boundary = simVariables.gamma, simVariables.precision, simVariables.scheme, simVariables.boundary
    Data = namedtuple('Data', ['flux', 'eigmax'])

    # Extrapolate the cell averages to face averages
    # Current convention: |  i-1     ---> |  i       ---> |  i+1     ---> |
    #                     |       w(i-1/2)|       w(i+1/2)|       w(i+3/2)|

    # Convert to primitive variables
    wS = fv.convertConservative(tube, gamma, boundary)

    # Pad array with boundary & apply (TVD) slope limiters
    w = fv.makeBoundary(wS, boundary)
    limitedValues = limiters.minmodLimiter(w)




    pass
# Extrapolate the cell averages to face averages
# Current convention: |  i-1     ---> |  i       ---> |  i+1     ---> |
#                     |       w(i-1/2)|       w(i+1/2)|       w(i+3/2)|
def extrapolate(tube, simVariables):
    gamma, subgrid, boundary = simVariables.gamma, simVariables.subgrid, simVariables.boundary

    # Conversion of conservative variables to primitive variables
    if subgrid in ["ppm", "parabolic", "p", "weno", "w"]:
        wS = fv.convertConservative(tube, gamma, boundary)
    else:
        wS = fv.pointConvertConservative(tube, gamma)

    if subgrid in ["ppm", "parabolic", "p", "plm", "linear", "l"]:
        # Pad array with boundary
        w = fv.makeBoundary(wS, boundary)

        if subgrid in ["ppm", "parabolic", "p"]:
            # PPM requires additional ghost cells
            w2 = fv.makeBoundary(wS, boundary, 2)
            w3 = fv.makeBoundary(wS, boundary, 3)

            # Face i+1/2 (4th-order) [McCorquodale & Colella, 2011, eq. 17; Colella et al., 2011, eq. 67]
            wF = 7/12 * (wS+w[2:]) - 1/12 * (w[:-2]+w2[4:])
            # Face i+1/2 (6th-order) [Colella & Sekora, 2008, eq. 17]
            #wF = 1/60 * (37*(wS+w[2:]) - 8*(w[:-2]+w2[4:]) + (w2[:-4]+w3[6:]))

            # Modified stencil [McCorquodale & Colella, 2011, eq. 21-22]
            if modified:
                wF[0] = 1/12 * (25*wS[1] - 23*wS[2] + 13*wS[3] - 3*wS[4])
                wF[-1] = 1/12 * (25*wS[-1] - 23*wS[-2] + 13*wS[-3] - 3*wS[-4])

                wF[1] = 1/12 * (3*wS[1] + 13*wS[2] - 5*wS[3] + wS[4])
                wF[-2] = 1/12 * (3*wS[-1] + 13*wS[-2] - 5*wS[-3] + wS[-4])

                if dissipate:
                    wS_point = fv.pointConvertConservative(tube, gamma)
                    eta = calculateFlattenCoeff(wS_point, boundary)

                    q = fv.makeBoundary(tube, boundary)
                    mu = applyArtificialViscosity(wS_point, gamma, boundary) * np.diff(q, axis=0)[1:]
                    return [wS, wF, w, w2, eta, mu]
                else:
                    return [wS, wF, w, w2]
            else:
                return [wS, wF, w, w2]
        else:
            return w
    elif subgrid in ["weno", "w"]:
        # Pad array with boundary
        w = fv.makeBoundary(wS, simVariables.boundary)
        w2 = fv.makeBoundary(wS, simVariables.boundary, 2)

        # Define the stencils
        u1 = (w2[:-4]/3) - (w[:-2]*7/6) + (wS*11/6)
        u2 = -(w[:-2]/6) + (wS*5/6) + (w[2:]/3)
        u3 = (wS/3) + (w[2:]*5/6) - (w2[4:]/6)

        # Define the linear weights
        x1, x2, x3 = 1/10, 3/5, 3/10

        # Determine the smoothness indicators
        b1 = (13/12 * (w2[:-4] - 2*w[:-2] + wS)**2) + (.25 * (w2[:-4] - 4*w[:-2] + 3*wS)**2)
        b2 = (13/12 * (w[:-2] - 2*wS + w[2:])**2) + (.25 * (w[:-2] - w[2:])**2)
        b3 = (13/12 * (wS - 2*w[2:] + w2[4:])**2) + (.25 * (3*wS - 4*w[2:] + w2[4:])**2)

        # Determine the non-linear weights
        alpha1 = x1/((1e-6 + b1)**2)
        alpha2 = x2/((1e-6 + b2)**2)
        alpha3 = x3/((1e-6 + b3)**2)

        weight1 = fv.divide(alpha1, alpha1+alpha2+alpha3)
        weight2 = fv.divide(alpha2, alpha1+alpha2+alpha3)
        weight3 = fv.divide(alpha3, alpha1+alpha2+alpha3)

        wF = weight1*u1 + weight2*u2 + weight3*u3
        return [wS, wF, w, w2]
    else:
        return wS


# Reconstruct the interpolants using the limited values
# Current convention: |               w(i-1/2)                    w(i+1/2)              |
#                     | i-1          <-- | -->         i         <-- | -->          i+1 |
#                     |        w_R(i-1)  |   w_L(i)          w_R(i)  |  w_L(i+1)        |
def interpolate(extrapolatedValues, limitedValues, simVariables):
    subgrid, boundary = simVariables.subgrid, simVariables.boundary

    # Reconstruction of parabolic interpolant
    if subgrid in ["ppm", "parabolic", "p"]:
        C = 5/4

        # Limited modified parabolic interpolant [McCorquodale & Colella, 2011]
        if modified:
            if dissipate:
                wS, wF, w, w2, eta, mu = extrapolatedValues
            else:
                wS, wF, w, w2 = extrapolatedValues
            # Define the left and right parabolic interpolants
            wF_limit = fv.makeBoundary(limitedValues, boundary)
            wF_limit_L, wF_limit_R = wF_limit[:-2], limitedValues

            # Set differences
            dw_minus, dw_plus = wS - wF_limit_L, wF_limit_R - wS
            d2w = 6 * (wF_limit_L - 2*wS + wF_limit_R)
            d2w_C = w[:-2] - 2*wS + w[2:]

            # Approximation to the third derivative (eq. 23)
            d3w = np.diff(fv.makeBoundary(d2w_C, boundary), axis=0)[1:]

            # Check for cell extreme in cells (eq. 24-25)
            cell_extrema = (dw_minus*dw_plus <= 0) | ((wS-w2[:-4])*(w2[4:]-wS) <= 0)

            # If there are extrema in the cells
            if cell_extrema.any():
                d2w_Cw = fv.makeBoundary(d2w_C, boundary)
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
                sensitive = np.abs(d2w) > 1e-12 * np.maximum(np.abs(wS), np.maximum(np.maximum(np.abs(w[:-2]), np.abs(w[2:])), np.maximum(np.abs(w2[:-4]), np.abs(w2[4:]))))
                # Update the limited estimates based on the condition (eq. 27)
                phi = fv.divide(d2w_lim, d2w)
                rho_limiter[sensitive] = phi[sensitive]

                # Apply additional limiters
                d3w_w = fv.makeBoundary(d3w, boundary)
                d3w_w2 = fv.makeBoundary(d3w, boundary, 2)
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
                return [(eta*wF_limit_L) + wS*(1-eta), (eta*wF_limit_R) + wS*(1-eta)], mu
            else:
                return [wF_limit_L, wF_limit_R]

        # Limited parabolic interpolant [Colella et al., 2011, p. 26]
        else:
            wS, wF, w, w2 = extrapolatedValues

            wF_limit = fv.makeBoundary(limitedValues, boundary)
            wF_limit_2 = fv.makeBoundary(limitedValues, boundary, 2)

            wF_limit_L, wF_limit_R = wF_limit[:-2], limitedValues

            # Check for cell extrema in cells (eq. 89)
            d_uL, d_uR = wS - wF_limit_L, wF_limit_R - wS
            cell_extrema = d_uL*d_uR < 0

            # Check for overshoot in cells (eq. 90)
            overshoot = (np.abs(d_uL) > 2*np.abs(d_uR)) | (np.abs(d_uR) > 2*np.abs(d_uL))

            # Check for extrema in interpolants (eq. 91-94)
            d_wF_minmod = np.minimum(np.abs(wF_limit_L - wF_limit_2[:-4]), np.abs(wF_limit_2[4:] - wF_limit_R))
            d_wS_minmod = np.minimum(np.abs(wS - w[:-2]), np.abs(w[2:] - wS))
            interpolant_extrema = ((d_wF_minmod >= d_wS_minmod) & ((wF_limit_L - wF_limit_2[:-4])*(wF_limit_2[4:] - wF_limit_R) < 0)) | ((d_wS_minmod >= d_wF_minmod) & ((wS - w[:-2])*(w[2:] - wS) < 0))

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
                return [wS - phi*d_uL_bar, wS + phi*d_uR_bar]  # (eq. 98)
            else:
                return [wF_limit_L, wF_limit_R]

    # Linear reconstruction [Derigs et al., 2017]
    elif subgrid in ["plm", "linear", "l"]:
        tube = np.copy(extrapolatedValues[1:-1])
        gradients = .5 * limitedValues
        return [tube - gradients, tube + gradients]  # (eq. 4.13)

    # No reconstruction, i.e. piecewise constant
    else:
        return extrapolatedValues


# Calculate the coefficients of the slope flattener for the parabolic extrapolants using pressure and v_x [Colella, 1990]
def calculateFlattenCoeff(wS, boundary, slope_determinants=[.33, .75, .85]):
    delta, z0, z1 = slope_determinants

    chiBar = np.zeros_like(wS[:,4])

    vxs = np.pad(wS[:,1], 1, mode=boundary)
    Ps = np.pad(wS[:,4], 2, mode=boundary)

    z = fv.divide(np.abs(Ps[3:-1]-Ps[1:-3]), np.abs(Ps[4:]-Ps[:-4]))

    eta = np.minimum(np.ones_like(z), np.maximum(np.zeros_like(z), 1-((z-z0)/(z1-z0))))
    criteria = ((vxs[:-2]-vxs[2:]) > 0) & (np.abs(Ps[3:-1]-Ps[1:-3])/np.minimum(Ps[3:-1],Ps[1:-3]) > delta)
    chiBar[criteria] = eta[criteria]
    chiPlusOne = np.pad(chiBar, 1, mode=boundary)

    chi = np.copy(chiBar)
    signage = np.sign(Ps[3:-1]-Ps[1:-3])
    chi[signage < 0] = np.minimum(chiPlusOne[2:], chiBar)[signage < 0]
    chi[signage > 0] = np.minimum(chiPlusOne[:-2], chiBar)[signage > 0]

    arr = np.ones_like(wS)
    return (chi*arr.T).T


# Implement artificial viscosity
def applyArtificialViscosity(wS, gamma, boundary, viscosity_determinants=[.3, .3]):

    def calculateSoundSpeed(pressures, densities, g):
        return np.sqrt((g*pressures)/densities)

    alpha, beta = viscosity_determinants

    vxs = np.pad(wS[:,1], 1, mode=boundary)
    cs = np.sqrt((gamma*wS[:,4])/wS[:,0])
    Gamma = np.diff(vxs)[1:]

    nu = np.zeros_like(Gamma)
    c_min = np.minimum(cs, np.pad(cs, 1, mode=boundary)[2:])
    nu[Gamma < 0] = (Gamma * np.minimum(np.ones_like(Gamma), (Gamma**2)/(beta*c_min**2)))[Gamma < 0]

    arr = np.ones_like(wS)
    return (alpha*nu*arr.T).T