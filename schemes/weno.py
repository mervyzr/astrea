from collections import defaultdict

import numpy as np

from functions import constructor, fv
from num_methods import ct

##############################################################################
# WENO reconstruction method [Shu, 2009]
##############################################################################

def run(grid, sim_variables):
    subgrid, boundary, axes, magnetic = sim_variables.subgrid, sim_variables.boundary, sim_variables.axes, sim_variables.magnetic
    convert = sim_variables.convert
    Bx, By, Bz = range(5,8)

    nested_dict = lambda: defaultdict(nested_dict)
    data = nested_dict()

    # Convert to primitive variables
    primitive = convert("conservative", grid, sim_variables, staggered=magnetic)

    """WENO reconstruction [Shu, 2009; San & Kara, 2015]
    |                        w(i-1/2)                    w(i+1/2)                       |
    |<--         i-1         -->|<--          i          -->|<--         i+1         -->|
    |   w_L(i-1)     w_R(i-1)   |   w_L(i)         w_R(i)   |   w_L(i+1)     w_R(i+1)   |
    |   w+(i-3/2)   w-(i-1/2)   |   w+(i-1/2)   w-(i+1/2)   |   w+(i+1/2)   w-(i+3/2)   |
    """
    def reconstruct(_grid, _boundary, _axis, _order=5):
        eps = 1e-6

        if _order == 3:
            padded_grid = fv.add_boundary(_grid, _boundary, axis=_axis)

            # Define frequently used terms
            zeroth = fv.slice_(padded_grid, _axis, *[1,-1])
            minus_one, plus_one = fv.slice_(padded_grid, _axis, end=-2), fv.slice_(padded_grid, _axis, start=2)

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
            padded_grid_3 = fv.add_boundary(_grid, _boundary, stencil=3, axis=_axis)

            # Define frequently used terms
            padded_grid_2 = fv.slice_(padded_grid_3, axis, *[1,-1])
            padded_grid = fv.slice_(padded_grid_2, axis, *[1,-1])

            zeroth = fv.slice_(padded_grid, axis, *[1,-1])
            minus_one, minus_two, minus_three = fv.slice_(padded_grid, axis, end=-2), fv.slice_(padded_grid_2, axis, end=-4), fv.slice_(padded_grid_3, axis, end=-6)
            plus_one, plus_two, plus_three = fv.slice_(padded_grid, axis, start=2), fv.slice_(padded_grid_2, axis, start=4), fv.slice_(padded_grid_3, axis, start=6)

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
            padded_grid_2 = fv.add_boundary(_grid, _boundary, stencil=2, axis=_axis)

            # Define frequently used terms
            padded_grid = fv.slice_(padded_grid_2, axis, *[1,-1])

            zeroth = fv.slice_(padded_grid, axis, *[1,-1])
            minus_one, minus_two = fv.slice_(padded_grid, axis, end=-2), fv.slice_(padded_grid_2, axis, end=-4)
            plus_one, plus_two = fv.slice_(padded_grid, axis, start=2), fv.slice_(padded_grid_2, axis, start=4)

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

    for axis in axes:
        # Reconstruct the interface states
        if len(subgrid.split("weno")) == 2:
            try:
                wL, wR = reconstruct(primitive, boundary, axis, int(subgrid.replace('-','').split("weno")[-1]))
            except Exception as e:
                wL, wR = reconstruct(primitive, boundary, axis)
        else:
            wL, wR = reconstruct(primitive, boundary, axis)

        # Magnetic component after computing to interface
        if magnetic:
            wR[...,(Bx,By)] = grid[...,(Bx,By)]
            data[axis]['ortho_interfaces'] = ct.reconstruct_transverse(wR, sim_variables, axis=axis)

        # Re-align the interfaces so that cell wall is in between interfaces
        prim_plus, prim_minus = fv.slice_(fv.add_boundary(wL, boundary, axis=axis), axis, start=1), fv.slice_(fv.add_boundary(wR, boundary, axis=axis), axis, end=-1)
        if magnetic:
            padded_grid = fv.add_boundary(grid, boundary, axis=axis)
            prim_plus[...,(Bx,By)] = prim_minus[...,(Bx,By)] = fv.slice_(padded_grid, axis, end=-1)[...,(Bx,By)]

        # Get the average solution between the interfaces at the boundaries
        intf_avg = fv.slice_(fv.compute_Roe_average(prim_plus, prim_minus), axis, start=1)
        padded_intf_avg = fv.add_boundary(intf_avg, boundary, axis=axis)

        # Convert the primitive variables
        cons_plus, cons_minus = fv.convert_interface("primitive", prim_plus, axis, sim_variables), fv.convert_interface("primitive", prim_minus, axis, sim_variables)

        # Compute the fluxes and the Jacobian
        flux_plus, flux_minus = constructor.make_flux(prim_plus, sim_variables, axis=axis), constructor.make_flux(prim_minus, sim_variables, axis=axis)

        jacobian = constructor.make_Jacobian(padded_intf_avg, sim_variables, axis=axis)

        # Update dict
        data[axis]['prim_interfaces'] = prim_plus, prim_minus
        data[axis]['cons_interfaces'] = cons_plus, cons_minus
        data[axis]['flux_interfaces'] = flux_plus, flux_minus
        data[axis]['characteristics'] = np.linalg.eigvals(jacobian)

    return data