from collections import defaultdict

import numpy as np

from functions import constructor, fv

##############################################################################
# WENO reconstruction method [Shu, 2009]
##############################################################################

def run(grids, sim_variables):
    gamma, subgrid, boundary, axes = sim_variables.gamma, sim_variables.subgrid, sim_variables.boundary, sim_variables.axes
    nested_dict = lambda: defaultdict(nested_dict)
    data = nested_dict()

    ax = 1

    # WENO reconstruction [Shu, 2009; San & Kara, 2015]
    """Current convention: |                        w(i-1/2)                    w(i+1/2)                       |
                           |-->         i-1         <--|-->          i          <--|-->         i+1         <--|
                           |   w_L(i-1)     w_R(i-1)   |   w_L(i)         w_R(i)   |   w_L(i+1)     w_R(i+1)   |
                    OR     |   w+(i-3/2)   w-(i-1/2)   |   w+(i-1/2)   w-(i+1/2)   |  w+(i+1/2)    w-(i+3/2)   |
    """
    def reconstruct(_wS, _boundary, _ax, _order=5):
        eps = 1e-6

        if _order == 3:
            w = fv.add_boundary(_wS, _boundary, axis=_ax)

            # Define frequently used terms
            zeroth = np.copy(_wS)
            minus_one, plus_one = w.take(range(0,w.shape[ax]-2), axis=_ax), w.take(range(2,w.shape[ax]), axis=_ax)

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
            w3 = fv.add_boundary(_wS, _boundary, stencil=3, axis=_ax)
            w2 = fv.add_boundary(_wS, _boundary, stencil=2, axis=_ax)
            w = fv.add_boundary(_wS, _boundary, axis=_ax)

            # Define frequently used terms
            zeroth = np.copy(_wS)
            minus_one, minus_two, minus_three = w.take(range(0,w.shape[ax]-2), axis=_ax), w2.take(range(0,w2.shape[ax]-4), axis=_ax), w3.take(range(0,w3.shape[ax]-6), axis=_ax)
            plus_one, plus_two, plus_three = w.take(range(2,w.shape[ax]), axis=_ax), w2.take(range(4,w2.shape[ax]), axis=_ax), w3.take(range(6,w3.shape[ax]), axis=_ax)

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
            w2 = fv.add_boundary(_wS, _boundary, stencil=2, axis=_ax)
            w = fv.add_boundary(_wS, _boundary, axis=_ax)

            # Define frequently used terms
            zeroth = np.copy(_wS)
            minus_one, minus_two = w.take(range(0,w.shape[ax]-2), axis=_ax), w2.take(range(0,w2.shape[ax]-4), axis=_ax)
            plus_one, plus_two = w.take(range(2,w.shape[ax]), axis=_ax), w2.take(range(4,w2.shape[ax]), axis=_ax)

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

    # Convert to primitive variables
    wS = fv.convert_conservative(grids, sim_variables)

    # Reconstruct the interface states
    if len(subgrid.split("weno")) == 2:
        try:
            wL, wR = reconstruct(wS, boundary, ax, int(subgrid.replace('-','').split("weno")[-1]))
        except Exception as e:
            wL, wR = reconstruct(wS, boundary, ax)
    else:
        wL, wR = reconstruct(wS, boundary, ax)

    # Re-align the reconstructed interfaces to correspond with the solvers
    w_minus, w_plus = wR, fv.add_boundary(wL, boundary, axis=ax).take(range(2,wL.shape[ax]+2), axis=ax)

    # Get the average solution between the interfaces at the boundaries
    interface_avg = constructor.make_Roe_average(w_minus, w_plus)

    # Convert the primitive variables
    q_minus, q_plus = fv.convert_primitive(w_minus, sim_variables, "face"), fv.convert_primitive(w_plus, sim_variables, "face")

    # Compute the fluxes and the Jacobian
    flux_minus, flux_plus = constructor.make_flux(w_minus, gamma, axes), constructor.make_flux(w_plus, gamma, axes)

    A = constructor.make_Jacobian(interface_avg, gamma, axes)

    # Update dict
    data['wS'] = wS
    data['w_minus'] = w_minus
    data['w_plus'] = w_plus
    data['q_minus'] = q_minus
    data['q_plus'] = q_plus
    data['flux_minus'] = flux_minus
    data['flux_plus'] = flux_plus
    data['Jacobian'] = A

    return data