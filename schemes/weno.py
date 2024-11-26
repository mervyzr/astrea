from collections import defaultdict

from functions import constructor, fv

##############################################################################
# WENO reconstruction method [Shu, 2009]
##############################################################################

def run(grid, sim_variables):
    gamma, subgrid, boundary, permutations = sim_variables.gamma, sim_variables.subgrid, sim_variables.boundary, sim_variables.permutations
    convert_primitive, convert_conservative = sim_variables.convert_primitive, sim_variables.convert_conservative
    nested_dict = lambda: defaultdict(nested_dict)
    data = nested_dict()

    """WENO reconstruction [Shu, 2009; San & Kara, 2015]
    |                        w(i-1/2)                    w(i+1/2)                       |
    |<--         i-1         -->|<--          i          -->|<--         i+1         -->|
    |   w_L(i-1)     w_R(i-1)   |   w_L(i)         w_R(i)   |   w_L(i+1)     w_R(i+1)   |
    |   w+(i-3/2)   w-(i-1/2)   |   w+(i-1/2)   w-(i+1/2)   |   w+(i+1/2)   w-(i+3/2)   |
    """
    def reconstruct(_wS, _boundary, _order=5):
        eps = 1e-6

        if _order == 3:
            w = fv.add_boundary(_wS, _boundary)

            # Define frequently used terms
            zeroth = w[1:-1]
            minus_one, plus_one = w[:-2], w[2:]

            # Define the linear weights
            g0, g1 = 1/3, 2/3

            # Determine the smoothness indicators
            b0 = (zeroth - minus_one)**2
            b1 = (plus_one - zeroth)**2

            # Define the non-linear weights
            a0 = lambda d0: d0/(b0 + eps)**2
            a1 = lambda d1: d1/(b1 + eps)**2

            # Define the stencils
            wR = (a0(g0)/(a0(g0) + a1(g1)))*(1.5*zeroth - .5*minus_one) + (a1(g1)/(a0(g0) + a1(g1)))*(.5*zeroth + .5*plus_one)
            wL = (a1(g0)/(a0(g1) + a1(g0)))*(1.5*zeroth - .5*plus_one) + (a0(g1)/(a0(g1) + a1(g0)))*(.5*zeroth + .5*minus_one)

        elif _order == 7:
            w3 = fv.add_boundary(_wS, _boundary, 3)

            # Define frequently used terms
            w2 = w3[1:-1]
            w = w2[1:-1]
            zeroth = w[1:-1]
            minus_one, minus_two, minus_three = w[:-2], w2[:-4], w3[:-6]
            plus_one, plus_two, plus_three = w[2:], w2[4:], w3[6:]

            # Define the linear weights
            g0, g1, g2, g3 = 1/35, 12/35, 18/35, 4/35

            # Determine the smoothness indicators
            b0 = (
                minus_three * (547*minus_three - 3882*minus_two + 4642*minus_one - 1854*zeroth)
                + minus_two * (7043*minus_two - 17246*minus_one + 7042*zeroth)
                + minus_one * (11003*minus_one - 9402*zeroth)
                + zeroth * (2107*zeroth)
            )
            b1 = (
                minus_two * (267*minus_two - 1642*minus_one + 1602*zeroth - 494*plus_one)
                + minus_one * (2843*minus_one - 5966*zeroth + 1922*plus_one)
                + zeroth * (3443*zeroth - 2522*plus_one)
                + plus_one * (547*plus_one)
            )
            b2 = (
                minus_one * (547*minus_one - 2522*zeroth + 1922*plus_one - 494*plus_two)
                + zeroth * (3443*zeroth - 5966*plus_one + 1602*plus_two)
                + plus_one * (2843*plus_one - 1642*plus_two)
                + plus_two * (267* plus_two)
            )
            b3 = (
                zeroth * (2107*zeroth - 9402*plus_one + 7042*plus_two - 1854*plus_three)
                + plus_one * (11003*plus_one - 17246*plus_two + 4642*plus_three)
                + plus_two * (7043*plus_two - 3882*plus_three)
                + plus_three * (547*plus_three)
            )

            # Define the non-linear weights
            a0 = lambda d0: d0/(b0 + eps)**2
            a1 = lambda d1: d1/(b1 + eps)**2
            a2 = lambda d2: d2/(b2 + eps)**2
            a3 = lambda d3: d3/(b3 + eps)**2

            # Define the stencils
            wR = (
                (a0(g0)/(a0(g0)+a1(g1)+a2(g2)+a3(g3))) * (-1/4*minus_three + 13/12*minus_two - 23/12*minus_one + 25/12*zeroth)
                + (a1(g1)/(a0(g0)+a1(g1)+a2(g2)+a3(g3))) * (1/12*minus_two - 5/12*minus_one + 13/12*zeroth + 1/4*plus_one)
                + (a2(g2)/(a0(g0)+a1(g1)+a2(g2)+a3(g3))) * (-1/12*minus_one + 7/12*zeroth + 7/12*plus_one - 1/12*plus_two)
                + (a3(g3)/(a0(g0)+a1(g1)+a2(g2)+a3(g3))) * (1/4*zeroth + 13/12*plus_one - 5/12*plus_two + 1/12*plus_three)
            )
            wL = (
                (a0(g3)/(a0(g3)+a1(g2)+a2(g1)+a3(g0))) * (1/4*zeroth + 13/12*minus_one - 5/12*minus_two + 1/12*minus_three)
                + (a1(g2)/(a0(g3)+a1(g2)+a2(g1)+a3(g0))) * (-1/12*plus_one + 7/12*zeroth + 7/12*minus_one - 1/12*minus_two)
                + (a2(g1)/(a0(g3)+a1(g2)+a2(g1)+a3(g0))) * (1/12*plus_two - 5/12*plus_one + 13/12*zeroth + 1/4*minus_one)
                + (a3(g0)/(a0(g3)+a1(g2)+a2(g1)+a3(g0))) * (-1/4*plus_three + 13/12*plus_two - 23/12*plus_one + 25/12*zeroth)
            )

        else:
            w2 = fv.add_boundary(_wS, _boundary, 2)

            # Define frequently used terms
            w = w2[1:-1]
            zeroth = w[1:-1]
            minus_one, minus_two = w[:-2], w2[:-4]
            plus_one, plus_two = w[2:], w2[4:]

            # Define the linear weights
            g0, g1, g2 = 1/10, 3/5, 3/10

            # Determine the smoothness indicators
            b0 = (
                13/12 * (minus_two - 2*minus_one + zeroth)**2
                + 1/4 * (minus_two - 4*minus_one + 3*zeroth)**2
            )
            b1 = (
                13/12 * (minus_one - 2*zeroth + plus_one)**2
                + 1/4 * (minus_one - plus_one)**2
            )
            b2 = (
                13/12 * (zeroth - 2*plus_one + plus_two)**2
                + 1/4 * (3*zeroth - 4*plus_one + plus_two)**2
            )

            # Define the non-linear weights
            a0 = lambda d0: d0/(b0 + eps)**2
            a1 = lambda d1: d1/(b1 + eps)**2
            a2 = lambda d2: d2/(b2 + eps)**2

            # Define the stencils
            wR = (
                (a0(g0)/(a0(g0)+a1(g1)+a2(g2))) * (1/3*minus_two - 7/6*minus_one + 11/6*zeroth)
                + (a1(g1)/(a0(g0)+a1(g1)+a2(g2))) * (-1/6*minus_one + 5/6*zeroth + 1/3*plus_one)
                + (a2(g2)/(a0(g0)+a1(g1)+a2(g2))) * (1/3*zeroth + 5/6*plus_one - 1/6*plus_two)
            )
            wL = (
                (a0(g2)/(a0(g2)+a1(g1)+a2(g0))) * (1/3*zeroth + 5/6*minus_one - 1/6*minus_two)
                + (a1(g1)/(a0(g2)+a1(g1)+a2(g0))) * (-1/6*plus_one + 5/6*zeroth + 1/3*minus_one)
                + (a2(g0)/(a0(g2)+a1(g1)+a2(g0))) * (1/3*plus_two - 7/6*plus_one + 11/6*zeroth)
            )

        return wL, wR

    # Rotate grid and apply algorithm for each axis
    for axis, axes in enumerate(permutations):
        _grid = grid.transpose(axes)

        # Convert to primitive variables
        wS = convert_conservative(_grid, sim_variables)

        # Reconstruct the interface states
        if len(subgrid.split("weno")) == 2:
            try:
                wL, wR = reconstruct(wS, boundary, int(subgrid.replace('-','').split("weno")[-1]))
            except Exception as e:
                wL, wR = reconstruct(wS, boundary)
        else:
            wL, wR = reconstruct(wS, boundary)

        # Re-align the interfaces so that cell wall is in between interfaces
        w_plus, w_minus = fv.add_boundary(wL, boundary)[1:], fv.add_boundary(wR, boundary)[:-1]

        # Get the average solution between the interfaces at the boundaries
        intf_avg = constructor.make_Roe_average(w_plus, w_minus)[1:]
        _intf_avg = fv.add_boundary(intf_avg, boundary)

        # Convert the primitive variables
        q_plus, q_minus = convert_primitive(w_plus, sim_variables, "face"), convert_primitive(w_minus, sim_variables, "face")

        # Compute the fluxes and the Jacobian
        flux_plus, flux_minus = constructor.make_flux(w_plus, gamma, axis), constructor.make_flux(w_minus, gamma, axis)
        A = constructor.make_Jacobian(_intf_avg, gamma, axis)

        # Update dict
        data[axes]['wS'] = wS
        data[axes]['wFs'] = w_plus, w_minus
        data[axes]['qFs'] = q_plus, q_minus
        data[axes]['fluxFs'] = flux_plus, flux_minus
        data[axes]['Jacobian'] = A

    return data