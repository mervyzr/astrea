from collections import defaultdict

from functions import fv, constructors
from numerics import solvers

##############################################################################
# WENO reconstruction method [Shu, 2009]
##############################################################################

def run(grid, sim_variables):
    gamma, boundary, permutations = sim_variables.gamma, sim_variables.boundary, sim_variables.permutations
    nested_dict = lambda: defaultdict(nested_dict)
    data = nested_dict()

    # Function to generate the WENO interface values
    def extrapolate_face_value(_wS, _boundary):
        # Pad array with boundary
        w2 = fv.add_boundary(_wS, _boundary, 2)

        # Define frequently used terms
        minus_one, minus_two = w2[1:-3], w2[:-4]
        plus_one, plus_two = w2[3:-1], w2[4:]

        # Define the stencils
        u1 = (minus_two/3) - (minus_one*7/6) + (_wS*11/6)
        u2 = -(minus_one/6) + (_wS*5/6) + (plus_one/3)
        u3 = (_wS/3) + (plus_one*5/6) - (plus_two/6)

        # Define the linear weights
        x1, x2, x3 = 1/10, 3/5, 3/10

        # Determine the smoothness indicators
        b1 = (13/12 * (minus_two - 2*minus_one + _wS)**2) + (.25 * (minus_two - 4*minus_one + 3*_wS)**2)
        b2 = (13/12 * (minus_one - 2*_wS + plus_one)**2) + (.25 * (minus_one - plus_one)**2)
        b3 = (13/12 * (_wS - 2*plus_one + plus_two)**2) + (.25 * (3*_wS - 4*plus_one + plus_two)**2)

        # Determine the non-linear weights
        alpha1 = x1/((1e-6 + b1)**2)
        alpha2 = x2/((1e-6 + b2)**2)
        alpha3 = x3/((1e-6 + b3)**2)

        weight1 = fv.divide(alpha1, alpha1+alpha2+alpha3)
        weight2 = fv.divide(alpha2, alpha1+alpha2+alpha3)
        weight3 = fv.divide(alpha3, alpha1+alpha2+alpha3)

        return weight1*u1 + weight2*u2 + weight3*u3

    # Rotate grid and apply algorithm for each axis
    for axis, axes in enumerate(permutations):
        _grid = grid.transpose(axes)

        # Convert to primitive variables
        wS = fv.convert_conservative(_grid, sim_variables)

        # Pad array with boundary
        w = fv.add_boundary(wS, boundary)

        """WENO reconstruction [Shu, 2009]
        Current convention: |               w(i-1/2)                    w(i+1/2)              |
                            | i-1          <-- | -->         i         <-- | -->          i+1 |
                            |        w_R(i-1)  |   w_L(i)          w_R(i)  |  w_L(i+1)        |
        """
        wL, wR = extrapolate_face_value(w[2:], boundary), extrapolate_face_value(w[1:-1], boundary)

        # Pad the reconstructed interfaces
        wLs, wRs = fv.add_boundary(wL, boundary)[1:], fv.add_boundary(wR, boundary)[:-1]

        # Convert the primitive variables
        qLs, qRs = fv.convert_primitive(wLs, sim_variables), fv.convert_primitive(wRs, sim_variables)

        # Compute the fluxes and the Jacobian
        fLs, fRs = constructors.make_flux_term(wLs, gamma, axis), constructors.make_flux_term(wRs, gamma, axis)
        A = constructors.make_Jacobian(w, gamma, axis)

        # Update dict
        data[axes]['wS'] = wS
        data[axes]['wLs'] = wLs
        data[axes]['wRs'] = wRs
        data[axes]['qLs'] = qLs
        data[axes]['qRs'] = qRs
        data[axes]['fLs'] = fLs
        data[axes]['fRs'] = fRs
        data[axes]['jacobian'] = A

    return solvers.calculate_Riemann_flux(sim_variables, data)