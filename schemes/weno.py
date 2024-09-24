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

    # Function to generate the WENO interface values for u_i+1/2
    def reconstruct(_wS, _boundary):
        eps = 1e-6
        w = fv.add_boundary(_wS, _boundary, 2)

        # Define frequently used terms
        minus_one, minus_two = w[1:-3], w[:-4]
        plus_one, plus_two = w[3:-1], w[4:]
        zeroth = w[2:-2]

        # Define the stencils
        u1 = 1/6 * (2*minus_two - 7*minus_one + 11*zeroth)
        u2 = 1/6 * (-minus_one + 5*zeroth + 2*plus_one)
        u3 = 1/6 * (2*zeroth + 5*plus_one - plus_two)

        # Define the linear weights
        g1, g2, g3 = .1, .6, .3

        # Determine the smoothness indicators
        b1 = 1/12 * (13*(minus_two - 2*minus_one + zeroth)**2 + 3*(minus_two - 4*minus_one + 3*zeroth)**2)
        b2 = 1/12 * (13*(minus_one - 2*zeroth + plus_one)**2 + 3*(minus_one - plus_one)**2)
        b3 = 1/12 * (13*(zeroth - 2*plus_one + plus_two)**2 + 3*(3*zeroth - 4*plus_one + plus_two)**2)

        # Define the non-linear weights
        a1 = g1/((eps + b1)**2)
        a2 = g2/((eps + b2)**2)
        a3 = g3/((eps + b3)**2)

        w1 = a1/(a1+a2+a3)
        w2 = a2/(a1+a2+a3)
        w3 = a3/(a1+a2+a3)

        return w1*u1 + w2*u2 + w3*u3

    # Rotate grid and apply algorithm for each axis
    for axis, axes in enumerate(permutations):
        _grid = grid.transpose(axes)

        # Convert to primitive variables
        wS = fv.convert_conservative(_grid, sim_variables)

        """WENO reconstruction [Shu, 2009]
        Current convention: |               w(i-1/2)                    w(i+1/2)              |
                            | i-1          <-- | -->         i         <-- | -->          i+1 |
                            |        w_R(i-1)  |   w_L(i)          w_R(i)  |  w_L(i+1)        |
                    OR      |       w-(i-1/2)  |   w+(i-1/2)    w-(i+1/2)  |  w+(i+1/2)       |
        """
        weno_approx = reconstruct(wS, boundary)

        # Pad arrays with boundary
        w = fv.add_boundary(wS, boundary)
        _w = fv.add_boundary(weno_approx, boundary)

        wLs, wRs = _w[1:], _w[:-1]

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