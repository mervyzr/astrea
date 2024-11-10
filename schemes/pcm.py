from collections import defaultdict

from functions import constructor, fv

##############################################################################
# Piecewise constant reconstruction method (PCM) [Godunov, 1959]
##############################################################################

def run(grids, sim_variables):
    gamma, axes = sim_variables.gamma, sim_variables.axes
    nested_dict = lambda: defaultdict(nested_dict)
    data = nested_dict()

    # Convert to primitive variables
    wS = fv.point_convert_conservative(grids, sim_variables)

    # Compute the fluxes and the Jacobian
    f = constructor.make_flux(wS, gamma, axes)
    A = constructor.make_Jacobian(wS, gamma, axes)

    # Update dict
    data['wS'] = wS
    data['qS'] = grids
    data['flux'] = f
    data['Jacobian'] = A

    return data