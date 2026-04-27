import numpy as np

from functions import grid as gutils
from functions import numeric
from numkit import c_transport as ct
from numkit import solvers

##############################################################################
# Piecewise constant reconstruction method (PCM) [Godunov, 1959]
##############################################################################

# Reconstruct from averaged cell <w>_{i,j} to averaged interfaces <w>_{i-1/2,j} & <w>_{i+1/2,j} (interface = centre for PCM)
def reconstruct(grid, sim_variables, axis):
    return grid, grid


def run(grid, sim_variables, axis):
    multidimensional, magnetic, ds = sim_variables.multidimensional, sim_variables.magnetic, sim_variables.ds
    data = {}

    Riemann_solver = solvers.get_Riemann_solver(sim_variables)

    # Pad array with boundaries
    padded_primitive = gutils.add_boundary(grid, sim_variables, axis=axis)
    padded_conservative = gutils.variable_point_convert("primitive", padded_primitive, sim_variables)

    # Compute the fluxes and the Jacobian
    padded_flux = numeric.compute_flux(padded_primitive, sim_variables, axis=axis)
    jacobian = numeric.compute_jacobian(padded_primitive, sim_variables, axis=axis)

    # Resolve characteristics at interfaces
    try:
        characteristics = np.linalg.eigvals(jacobian)
    except np.linalg.LinAlgError:
        try:
            characteristics = numeric.compute_characteristics(padded_primitive, sim_variables, axis=axis)
        except np.linalg.LinAlgError:
            characteristics = np.full_like(padded_primitive, .1)

    # Compute eigmax for time stepping limits
    data['eigmax'] = ds[axis]/numeric.compute_eigmax(characteristics, axis=axis)

    # Compute alphas and save the reconstructed interfaces for CT computation (interface = centre for PCM)
    if magnetic and multidimensional:
        data['alphas'] = ct.compute_alphas(characteristics, axis=axis)
        data['interfaces'] = np.copy(grid), np.copy(grid)

    # Calculate the interface-averaged fluxes (pointwise & averaged values are the same for lower-order schemes)
    intf_fluxes_avgd = intf_fluxes_cntrd = Riemann_solver(axis, sim_variables, **{
        'prim_interfaces': (gutils.slice_(padded_primitive, axis, start=1), gutils.slice_(padded_primitive, axis, end=-1)),
        'cons_interfaces': (gutils.slice_(padded_conservative, axis, start=1), gutils.slice_(padded_conservative, axis, end=-1)),
        'flux_interfaces': (gutils.slice_(padded_flux, axis, start=1), gutils.slice_(padded_flux, axis, end=-1)),
        'characteristics': characteristics,
        'jacobian': gutils.slice_(jacobian, axis, start=1),
    })

    # Compute flux difference for hydrodynamic components
    data['fluxes'] = np.diff(intf_fluxes_cntrd, axis=axis)/ds[axis]

    return data