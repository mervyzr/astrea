import numpy as np

from functions import constructor, fv
from num_methods import ct, solvers

##############################################################################
# Piecewise constant reconstruction method (PCM) [Godunov, 1959]
##############################################################################

# Reconstruct from averaged cell <w>_{i,j} to averaged interfaces <w>_{i-1/2,j} & <w>_{i+1/2,j} (interface = centre for PCM)
def reconstruct(grid, sim_variables, axis):
    return grid, grid


def run(grid, sim_variables, axis):
    convert, multidimensional, magnetic, ds = sim_variables.convert, sim_variables.multidimensional, sim_variables.magnetic, sim_variables.ds
    data = {}

    Riemann_solver = solvers.get_Riemann_solver(sim_variables)

    # Pad array with boundaries
    padded_primitive = fv.add_boundary(grid, sim_variables, axis=axis)
    padded_conservative = convert("primitive", padded_primitive, sim_variables)

    # Compute the fluxes and the Jacobian
    fluxes = constructor.make_flux(padded_primitive, sim_variables, axis=axis)
    jacobian = constructor.make_Jacobian(padded_primitive, sim_variables, axis=axis)

    # Resolve characteristics at interfaces
    try:
        characteristics = np.linalg.eigvals(jacobian)
    except np.linalg.LinAlgError:
        try:
            characteristics = constructor.make_characteristics(padded_primitive, sim_variables, axis=axis)
        except np.linalg.LinAlgError:
            characteristics = np.full_like(padded_primitive, .1)

    # Compute eigmax for time stepping limits
    data['eigmax'] = ds[axis]/fv.compute_eigmax(characteristics, axis=axis)

    # Compute alphas and save the reconstructed interfaces for CT computation (interface = centre for PCM)
    if magnetic and multidimensional:
        data['alphas'] = ct.compute_alphas(characteristics, axis=axis)
        data['interfaces'] = fv.slice_(fv.add_boundary(grid, sim_variables, axis=axis), axis, start=2), np.copy(grid)

    # Calculate the interface-averaged fluxes (pointwise & averaged values are the same for lower-order schemes)
    intf_fluxes_avgd = intf_fluxes_cntrd = Riemann_solver(axis, sim_variables, **{
        'prim_interfaces': [fv.slice_(padded_primitive, axis, start=1), fv.slice_(padded_primitive, axis, end=-1)],
        'cons_interfaces': [fv.slice_(padded_conservative, axis, start=1), fv.slice_(padded_conservative, axis, end=-1)],
        'flux_interfaces': [fv.slice_(fluxes, axis, start=1), fv.slice_(fluxes, axis, end=-1)],
        'characteristics': characteristics,
        'jacobian': fv.slice_(jacobian, axis, end=-1),
    })

    # Compute flux difference for hydrodynamic components
    data['fluxes'] = np.diff(intf_fluxes_cntrd, axis=axis)/ds[axis]

    return data