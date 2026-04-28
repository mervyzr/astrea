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
    padded_intf = gutils.slice_(gutils.add_boundary(grid, sim_variables, stencil=2, axis=axis), axis, end=-1)
    padded_primitive = gutils.slice_(padded_intf, axis, start=1)

    # Re-align the interfaces so that cell wall is in between interfaces
    prim_plus, prim_minus = gutils.slice_(padded_primitive, axis, start=1), gutils.slice_(padded_primitive, axis, end=-1)

    # Convert the primitive variables at the interface
    cons_plus, cons_minus = gutils.variable_point_convert("primitive", prim_plus, sim_variables), gutils.variable_point_convert("primitive", prim_minus, sim_variables)

    # Compute the fluxes and the Jacobian
    flux_plus, flux_minus = numeric.compute_flux(prim_plus, sim_variables, axis=axis), numeric.compute_flux(prim_minus, sim_variables, axis=axis)
    jacobian = numeric.compute_jacobian(padded_intf, sim_variables, axis=axis)

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
        'prim_interfaces': (prim_plus, prim_minus),
        'cons_interfaces': (cons_plus, cons_minus),
        'flux_interfaces': (flux_plus, flux_minus),
        'characteristics': characteristics,
        'jacobian': gutils.slice_(jacobian, axis, *[1,-1]),
    })

    # Compute flux difference for hydrodynamic components
    data['fluxes'] = np.diff(intf_fluxes_cntrd, axis=axis)/ds[axis]

    return data