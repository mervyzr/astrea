from collections import defaultdict

import numpy as np

from functions import fv, constructors
from numerics import limiters

##############################################################################
# Piecewise parabolic reconstruction method (PPM) [Colella & Woodward, 1984]
##############################################################################

# [McCorquodale & Colella, 2011; Colella et al., 2011; Peterson & Hammett, 2008]
def run(grid, sim_variables, paper="mc", dissipate=False):
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

        if "x" in paper.lower() or "ph" in paper.lower() or paper.lower() in ["peterson", "hammett"]:
            """Extrapolate the cell averages to face averages (both sides) 
            Current convention: | <---     i-1     ---> | <---      i      ---> | <---     i+1     ---> |
                                |w_L(i-1)       w_R(i-1)|w_L(i)           w_R(i)|w_L(i+1)       w_R(i+1)|
            """
            # Face i+1/2 (4th-order) (eq. 3.26-3.27)
            wF_L = 7/12 * (w[:-2] + wS) - 1/12 * (w2[:-4] + w[2:])
            wF_R = 7/12 * (wS + w[2:]) - 1/12 * (w[:-2] + w2[4:])

            # Face i+1/2 (5th-order) [Peterson & Hammett, 2008, eq. 3.40]
            #wF_R = 1/60 * (2*w2[:-4] - 13*w[:-2] + 47*wS + 27*w[2:] - 2*w2[4:])

            # Limit interface values [Peterson & Hammett, 2008, eq. 3.33-3.34]
            wF_limit_L, wF_limit_R = limiters.interface_limiter(wF_L, w2[:-4], w[:-2], wS, w[2:]), limiters.interface_limiter(wF_R, w[:-2], wS, w[2:], w2[4:])

            kwargs = {}
        else:
            """Extrapolate the cell averages to face averages (forward/upwind)
            Current convention: |  i-1     ---> |  i       ---> |  i+1     ---> |
                                |       w(i-1/2)|       w(i+1/2)|       w(i+3/2)|
            """
            # Face i+1/2 (4th-order) [McCorquodale & Colella, 2011, eq. 17; Colella et al., 2011, eq. 67]
            wF = 7/12 * (wS + w[2:]) - 1/12 * (w[:-2] + w2[4:])

            if paper.lower() == "c" or paper.lower() == "colella":
                # Limit interface values [Colella et al., 2011, p. 25-26]
                wF = limiters.interface_limiter(wF, w[:-2], wS, w[2:], w2[4:])

            # Define the left and right parabolic interpolants
            wF_pad2 = fv.add_boundary(wF, boundary, 2)
            wF_limit_L, wF_limit_R = np.copy(wF_pad2[1:-3]), np.copy(wF_pad2[2:-2])

            kwargs = {"wF_pad2": wF_pad2, "boundary": boundary}

        """Reconstruct the limited parabolic interpolants from the interface values [McCorquodale & Colella, 2011; Colella et al., 2011; Peterson & Hammett, 2008]
        Current convention: |                        w(i-1/2)                    w(i+1/2)                       |
                            |-->         i-1         <--|-->          i          <--|-->         i+1         <--|
                            |   w_L(i-1)     w_R(i-1)   |   w_L(i)         w_R(i)   |   w_L(i+1)     w_R(i+1)   |
                    OR      |   w+(i-3/2)   w-(i-1/2)   |   w+(i-1/2)   w-(i+1/2)   |  w+(i+1/2)    w-(i+3/2)   |
        """
        wL, wR = limiters.interpolant_limiter(wF_limit_L, wF_limit_R, wS, w, w2, paper.lower(), **kwargs)

        # Get the average solution
        avg_wS = constructors.make_Roe_average(wL, wR)

        # Pad the reconstructed interfaces
        wLs, wRs = fv.add_boundary(wL, boundary)[1:], fv.add_boundary(wR, boundary)[:-1]

        # Convert the primitive variables
        qLs, qRs = fv.convert_primitive(wLs, sim_variables, "face"), fv.convert_primitive(wRs, sim_variables, "face")

        # Compute the fluxes and the Jacobian
        _w = fv.add_boundary(avg_wS, boundary)
        fLs, fRs = constructors.make_flux_term(wLs, gamma, axis), constructors.make_flux_term(wRs, gamma, axis)

        if (paper == "mc" or "mccorquodale" in paper) and dissipate:
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