import concurrent.futures
from itertools import repeat

import numpy as np

from functions import fv
from num_methods import limiters
from schemes import pcm, plm, ppm, weno, cweno

##############################################################################
# Fourth-order upwind constrained transport algorithm for MHD [Felker & Stone, 2018]
##############################################################################

# Compute the maximum(+) & minimum(-) eigenvalues for alpha+ and alpha- respectively for each axis; used in the compute_emf function
def compute_alphas(characteristics, axis):
    local_max, local_min = np.max(characteristics, axis=-1), np.min(characteristics, axis=-1)
    max_eigvals = np.maximum(fv.slice_(local_max, axis, end=-1), fv.slice_(local_max, axis, start=1))
    min_eigvals = np.minimum(fv.slice_(local_min, axis, end=-1), fv.slice_(local_min, axis, start=1))
    return fv.slice_(np.maximum(0, max_eigvals), axis, start=1), fv.slice_(-np.minimum(0, min_eigvals), axis, start=1)


# Wrapper for reconstruct_transverse; for each orthogonal axis, it will compute for each pair of (+) & (-) interfaces
def reconstruct_wrapper(interfaces, sim_variables, ortho_axes):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        jobs = executor.map(reconstruct_transverse, repeat(interfaces), repeat(sim_variables), ortho_axes)
        return {ortho_axes[idx]:ortho_interfaces for idx, ortho_interfaces in enumerate(jobs)}

# Reconstruct the transverse values for each face average (computation done entirely for orthogonal axis)
# Note that this reconstruction is done at the INTERFACES, NOT CENTRES, and it is for one orthogonal axis
def reconstruct_transverse(interfaces, sim_variables, ortho_axis, method=None, eta=None):
    if not method:
        method = sim_variables.subgrid_category

    """Interpolate the face averages to both corners (upwards & downwards)
    |                w(i-1/2)            w(i+1/2)               |
    |-------------------|-------------------|-------------------|
    |           w_U(i-1/2,j+1/2)    w_U(i+1/2,j+1/2)            |
    |                  ^|                  ^|                  ^|
    |                  ||                  ||                  ||
    |                  ||                  ||                  ||
    |  o (i-1,j)     -->|  o (i,j)       -->|  o (i+1,j)     -->|
    |                  ||                  ||                  ||
    |                  ||                  ||                  ||
    |                  v|                  v|                  v|
    |           w_D(i-1/2,j-1/2)    w_D(i+1/2,j-1/2)            |
    |-------------------|-------------------|-------------------|

        OR

    Reconstruct the limited extrapolants from the interface values. Returns the face averages in the form of w+(y) & w-(y) when considering x-axis, and w+(x) & w-(x) when considering y-axis
    |                w(i-1/2)            w(i+1/2)               |
    |  o (i-1,j+1)      |  o (i,j+1)        |  o (i+1,j+1)      |
    |                   |                   |                   |
    |                   |                   |                   |
    |           w_D(i-1/2,j+1/2)    w_D(i+1/2,j+1/2)            |
    |                 w+(y)               w+(y)               w+(y)
    |                   ^                   ^                   ^
    |-------------------|-------------------|-------------------|
    |                   v                   v                   v
    |                 w-(y)               w-(y)               w-(y)
    |           w_U(i-1/2,j+1/2)    w_U(i+1/2,j+1/2)            |
    |                   |                   |                   |
    |                   |                   |                   |
    |  o (i-1,j)     -->|  o (i,j)       -->|  o (i+1,j)     -->|
    """
    if method == "weno":
        reconstruct = weno.reconstruct
    elif method == "cweno":
        reconstruct = cweno.reconstruct
    elif method == "ppm":
        if sim_variables.ppm_dissipate:
            reconstruct = lambda _grid, _sim_variables, _axis: ppm.reconstruct(_grid, _sim_variables, _axis, eta=eta)
        else:
            reconstruct = ppm.reconstruct
    elif method == "plm":
        reconstruct = plm.reconstruct
    else:
        reconstruct = pcm.reconstruct

    transverse_interfaces = []  # should have 4 interfaces per ortho_axis; 2 transverse interfaces for each longitudinal reconstruction
    with concurrent.futures.ThreadPoolExecutor() as inner_executor:
        jobs = inner_executor.map(reconstruct, interfaces, repeat(sim_variables), repeat(ortho_axis))

        for [wD, wU] in jobs:
            # Re-align the interfaces so that cell wall is in between interfaces
            prim_plus, prim_minus = fv.slice_(fv.add_boundary(wD, sim_variables, axis=ortho_axis), ortho_axis, start=1), fv.slice_(fv.add_boundary(wU, sim_variables, axis=ortho_axis), ortho_axis, end=-1)

            # Remove the 'leftmost' interface since only the upwind corners/lines are needed
            prim_plus, prim_minus = fv.slice_(prim_plus, axis=ortho_axis, start=1), fv.slice_(prim_minus, axis=ortho_axis, start=1)

            transverse_interfaces.append([prim_plus, prim_minus])

    return transverse_interfaces





# CONDITIONALS ARE HANDLING EVERYTHING! SO DON'T NEEDA WORRY ABOUT DIMENSION = 1
# Dimension = 1, axes = [0]
# axis = 0 => ortho_axes = [] => results = {}
# axis = 1 => won't compute
# axis = 2 => won't compute
# So this seems possible; the reconstruct_transverse will not reconstruct anything
#
# Dimension = 2, axes = [0,1]
# axis = 0 => ortho_axes = [1] => results = {1:[[w_y++,w_y+-] , [w_y-+,w_y--]]}
# axis = 1 => ortho_axes = [0] => results = {0:[[w_x++,w_x+-] , [w_x-+,w_x--]]}
# axis = 2 => won't compute
# Seems possible too; the emf will try and compute for all axes, which will only succeed for z-axis bc only x- & y-axis info are available
#
# Dimension = 3, axes = [0,1,2]
# axis = 0 => ortho_axes = [1,2] => results = {1:[[w_y++,w_y+-] , [w_y-+,w_y--]] , 2:[[w_z++,w_z+-] , [w_z-+,w_z--]]}
# axis = 1 => ortho_axes = [0,2] => results = {0:[[w_x++,w_x+-] , [w_x-+,w_x--]] , 2:[[w_z++,w_z+-] , [w_z-+,w_z--]]}
# axis = 2 => ortho_axes = [0,1] => results = {0:[[w_x++,w_x+-] , [w_x-+,w_x--]] , 1:[[w_y++,w_y+-] , [w_y-+,w_y--]]}
# Seems possible; the emf will still try and compute for all axes, but now all axes will succeed. I think?




# Compute the corner/line electric fields wrt to corner/line for each axis [Mignone & Del Zanna, 2020]
def compute_emf(data, axis):
    abscissa, ordinate, applicate = (axis + np.array(range(3)))%3

    # Assume computation for the z-axis EMF here (axis=2), and so x-axis (ordinate) & y-axis (applicate) are needed
    vx, vy = 1+ordinate, 1+applicate
    Bx, By = 5+ordinate, 5+applicate

    # For -v x B in x-axis (axis=0, thumb), y (axis=1, middle finger) becomes (N,S) & z (axis=2, index finger) becomes (E,W)
    # For -v x B in y-axis (axis=1, thumb), z (axis=2, middle finger) becomes (N,S) & x (axis=0, index finger) becomes (E,W)
    # For -v x B in z-axis (axis=2, thumb), x (axis=0, middle finger) becomes (N,S) & y (axis=1, index finger) becomes (E,W)
    try:
        [north, south], [east, west] = data[applicate]['ortho_interfaces'], data[ordinate]['ortho_interfaces']
        [ap_x, am_x], [ap_y, am_y] = data[ordinate]['alphas'], data[applicate]['alphas']
    except KeyError:
        # For 2D cases where the z-axis is missing; make a zeros_like data from the x- or y-axis instead
        emf = np.zeros_like(data[abscissa]['ortho_interfaces'][0][...,vx])
    else:
        # Compute the corner mag. fields wrt to corner/line
        SW = .5*(west[...,vy]+south[...,vy])*south[...,Bx] - .5*(west[...,vx]+south[...,vx])*west[...,By]
        SE = .5*(east[...,vy]+south[...,vy])*south[...,Bx] - .5*(east[...,vx]+south[...,vx])*east[...,By]
        NW = .5*(west[...,vy]+north[...,vy])*north[...,Bx] - .5*(west[...,vx]+north[...,vx])*west[...,By]
        NE = .5*(east[...,vy]+north[...,vy])*north[...,Bx] - .5*(east[...,vx]+north[...,vx])*east[...,By]

        # Averaging procedure for the 4-fold values at each corner/line
        emf = (
            fv.divide(ap_x*ap_y*SW + am_x*ap_y*SE + ap_x*am_y*NW + am_x*am_y*NE, (ap_x+am_x)*(ap_y+am_y)) 
            - fv.divide(ap_y*am_y, ap_y+am_y)*(north[...,Bx]-south[...,Bx]) 
            + fv.divide(ap_x*am_x, ap_x+am_x)*(east[...,By]-west[...,By])
        )

    return emf


# Compute constrained transport flux using corner/line emfs [Mignone & Del Zanna, 2020]
# The hydro fluxes are unaltered. The magnetic fluxes are computed for each axis and automatically allocated,
# while setting the other axes to zero
def compute_ct_flux(flux, emfs, sim_variables, axis):
    _axes = np.array(range(3))

    # Get the orthogonal axes & emfs from the axis being computed
    ortho_axes = _axes[_axes != axis]
    ortho_emfs = emfs[_axes != axis]

    # Get normal axis to the orthogonal axes (for derivatives)
    normal_axes = np.roll(ortho_axes, shift=-1)

    def per_normal_axis(emf, _sim_variables, normal_axis):
        try:
            padded_emf = fv.add_boundary(emf, _sim_variables, axis=normal_axis)
            emf_diff = np.diff(fv.slice_(padded_emf, axis=normal_axis, end=-1), axis=normal_axis)/_sim_variables.ds[normal_axis]
        except:
            emf_diff = emf
        return emf_diff

    # Update CT flux in axis; set the other axes fluxes to zero (for summation later)
    with concurrent.futures.ThreadPoolExecutor() as inner_executor:
        jobs = inner_executor.map(per_normal_axis, ortho_emfs, repeat(sim_variables), normal_axes)

        flux[...,5+axis] = (-1)**axis * np.diff([emf_flux for emf_flux in jobs], axis=0)
        flux[...,5+ortho_axes] = 0

    return flux