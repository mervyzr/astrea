import numpy as np

from functions import constructor, fv
from num_methods import ct, solvers

##############################################################################
# Piecewise constant reconstruction method (PCM) [Godunov, 1959]
##############################################################################

def run(grid, sim_variables, axis):
    boundary, dimension, magnetic, ds = sim_variables.boundary, sim_variables.dimension, sim_variables.magnetic, sim_variables.ds
    data = {}

    Riemann_solver = solvers.get_Riemann_solver(sim_variables)
    ortho_axis = 1 - axis if (magnetic or dimension == 2) else 0

    # Pad array with boundaries
    padded_primitive = fv.add_boundary(grid, boundary, axis=axis)
    padded_conservative = fv.convert_interface("primitive", padded_primitive, axis, sim_variables)

    # Compute the fluxes and the Jacobian
    fluxes = constructor.make_flux(padded_primitive, sim_variables, axis=axis)
    jacobian = constructor.make_Jacobian(padded_primitive, sim_variables, axis=axis)

    # Compute eigmax for time stepping limits
    characteristics = np.linalg.eigvals(jacobian)
    data['eigmax'] = ds[axis]/fv.compute_eigmax(characteristics, axis=axis)

    if magnetic:
        # Magnetic transverse interfaces (interface = centre for PCM) reconstructed orthogonal to the axis
        ortho_plus, ortho_minus = ct.reconstruct_transverse(grid, sim_variables, axis=ortho_axis)
        data['ortho_interfaces'] = fv.slice_(ortho_plus, axis=ortho_axis, start=1), fv.slice_(ortho_minus, axis=ortho_axis, start=1)

        # alphas refers to the maximum(+)/minimum(-) eigenvalues respectively
        local_max, local_min = np.max(characteristics, axis=-1), np.min(characteristics, axis=-1)
        max_eigvals = np.maximum(fv.slice_(local_max, axis, end=-1), fv.slice_(local_max, axis, start=1))
        min_eigvals = np.minimum(fv.slice_(local_min, axis, end=-1), fv.slice_(local_min, axis, start=1))
        data['alphas'] = fv.slice_(np.maximum(0, max_eigvals), axis, start=1), fv.slice_(-np.minimum(0, min_eigvals), axis, start=1)

    # Calculate the interface-averaged fluxes (pointwise & averaged values are the same for lower-order schemes)
    intf_fluxes_avgd = intf_fluxes_cntrd = Riemann_solver(axis, sim_variables, **{
        'prim_interfaces': [fv.slice_(padded_primitive, axis, start=1), fv.slice_(padded_primitive, axis, end=-1)],
        'cons_interfaces': [fv.slice_(padded_conservative, axis, start=1), fv.slice_(padded_conservative, axis, end=-1)],
        'flux_interfaces': [fv.slice_(fluxes, axis, start=1), fv.slice_(fluxes, axis, end=-1)],
        'characteristics': characteristics,
    })

    # Compute flux difference for hydrodynamic components
    data['fluxes'] = np.diff(intf_fluxes_cntrd, axis=axis)/ds[axis]

    return data