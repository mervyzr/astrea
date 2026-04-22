import concurrent.futures
from itertools import repeat

import numpy as np

from functions import fv
from schemes import pcm, plm, ppm, weno, cweno, wenoz

##############################################################################
# Fourth-order upwind constrained transport algorithm for MHD [Felker & Stone, 2018]
##############################################################################

# Re-align the interfaces so that cell wall is in between interfaces and re-assign the staggered magnetic field interfaces (these are never reconstructed) back onto the longitudinally reconstructed grid interfaces
def assign_interfaces(interfaces, grid, sim_variables, axis):
    prim_plus, prim_minus = fv.assign_interfaces(interfaces, grid, sim_variables, axis)

    padded_grid = fv.add_boundary(grid, sim_variables, axis=axis)
    prim_plus[...,5+axis] = fv.slice_(padded_grid, axis, end=-1)[...,5+axis]
    prim_minus[...,5+axis] = fv.slice_(padded_grid, axis, start=1)[...,5+axis]

    return prim_plus, prim_minus


# For MHD simulations, special care is needed for the high-order conversion between primitive and conservative variables with a staggered grid
def convert(variable, grid, sim_variables):
    centred_grid = inverse_reconstruct(grid, sim_variables)
    _grid = fv.convert(variable, centred_grid, sim_variables)
    _grid[...,5+sim_variables.axes] = grid[...,5+sim_variables.axes]
    return _grid


# Compute the maximum(+) & minimum(-) eigenvalues for alpha+ and alpha- respectively for each axis; used in the compute_emf function
def compute_alphas(characteristics, axis):
    local_max, local_min = np.max(characteristics, axis=-1), np.min(characteristics, axis=-1)
    max_eigvals = np.maximum(fv.slice_(local_max, axis, end=-1), fv.slice_(local_max, axis, start=1))
    min_eigvals = np.minimum(fv.slice_(local_min, axis, end=-1), fv.slice_(local_min, axis, start=1))
    return fv.slice_(np.maximum(0, max_eigvals), axis, start=1), fv.slice_(-np.minimum(0, min_eigvals), axis, start=1)


# 'Inverse reconstruct' the magnetic fields' cell-averaged values from the (staggered grid) face-averaged values [Felker & Stone, 2018]
def inverse_reconstruct(grid, sim_variables):
    axes = sim_variables.axes
    new_grid = np.copy(grid)

    def inversion_per_axis(_Bfields, _sim_variables, axis):
        face_cntrd = np.copy(_Bfields)

        if _sim_variables.higher_order:
            # Approximate the face-averaged values to face-centred values with orthogonal axes (eq. 38)
            if _sim_variables.grid_interpolate and _sim_variables.multidimensional:
                ortho_axes = _sim_variables.axes[_sim_variables.axes != axis]
                with concurrent.futures.ThreadPoolExecutor() as inner_executor:
                    jobs = inner_executor.map(fv.laplacian, repeat(_Bfields), repeat(_sim_variables), ortho_axes)
                    for idx, ortho_Bfield in enumerate(jobs):
                        face_cntrd -= (_sim_variables.ds[ortho_axes[idx]]**2)/24 * ortho_Bfield

            # Interpolate the face-centred values to cell-centred values with axis (eq. 39)
            face_cntrd_padded_2 = fv.add_boundary(face_cntrd, _sim_variables, stencil=2, axis=axis)
            face_cntrd_padded = fv.slice_(face_cntrd_padded_2, axis, *[1,-1])
            cell_cntrd = -1/16 * (fv.slice_(face_cntrd_padded, axis, start=2) + fv.slice_(face_cntrd_padded_2, axis, end=-4)) \
                        + 9/16 * (np.copy(face_cntrd) + fv.slice_(face_cntrd_padded, axis, end=-2))

            # Apply Laplacian operator to convert cell-centred values to cell-averaged values (eq. 40)
            cell_avgd = np.copy(cell_cntrd)
            _axes = _sim_variables.axes
            with concurrent.futures.ThreadPoolExecutor() as inner_executor:
                jobs = inner_executor.map(fv.laplacian, repeat(cell_cntrd), repeat(_sim_variables), _axes)
                for idx, _Bfield in enumerate(jobs):
                    cell_avgd += (_sim_variables.ds[_axes[idx]]**2)/24 * _Bfield

        else:
            # Arithmetic averaging for lower-order schemes
            padded_grid = fv.add_boundary(face_cntrd, _sim_variables, axis=axis)
            cell_avgd = .5 * (face_cntrd + fv.slice_(padded_grid, axis, end=-2))

        return cell_avgd

    # Update the grid values with the updated B-field values
    with concurrent.futures.ThreadPoolExecutor() as executor:
        jobs = executor.map(inversion_per_axis, repeat(grid[...,sim_variables.Bfields]), repeat(sim_variables), axes)
        for idx, Bfield in enumerate(jobs):
            new_grid[...,5+axes[idx]] = Bfield[...,axes[idx]]

    return new_grid


# Reconstruct the longitudinal B-fields for each transverse axis
# Note that this reconstruction is done at the cell INTERFACES, NOT cell CENTRES
def reconstruct_transverse(data, sim_variables, axis, method=None, eta=None):
    ortho_interfaces = {}

    # Get the orthogonal axes & emfs from the axis being computed
    ortho_axes = np.delete(np.arange(3), axis)

    # Get normal axis to the orthogonal axes (for derivatives)
    normal_axes = np.roll(ortho_axes, shift=-1)

    # Default to same reconstruction method as longitudinal reconstruction
    if not method:
        method = sim_variables.subgrid_category

    # Check if there are any missing transverse-axis data; skips computation if missing (i.e. data[transverse_axis] = None)
    # 1D: skips for all axes
    # 2D: only computes for z-axis
    # 3D: computes for all axes
    if not None in list(map(data.get, ortho_axes)):
        """Interpolate the face averages to both corners (upwards & downwards)
        |                                   w(i-1/2)                               w(i+1/2)                                  |
        |--------------------------------------|--------------------------------------|--------------------------------------|
        |                    w_U-(i-1/2,j+1/2) | w_U+(i-1/2,j+1/2)  w_U-(i+1/2,j+1/2) | w_U+(i+1/2,j+1/2)  w_U-(i+3/2,j+1/2) |
        |                                  ^   |   ^                              ^   |   ^                              ^   |
        |                                  |   |   |                              |   |   |                              |   |
        |                                  |   |   |                              |   |   |                              |   |
        |              o (i-1,j)          w- <-|-> w+           o (i,j)          w- <-|-> w+          o (i+1,j)         w- <-|
        |                                  |   |   |                              |   |   |                              |   |
        |                                  |   |   |                              |   |   |                              |   |
        |                                  v   |   v                              v   |   v                              v   |
        |                    w_D-(i-1/2,j-1/2) | w_D+(i-1/2,j-1/2)  w_D-(i+1/2,j-1/2) | w_D+(i+1/2,j-1/2)  w_D-(i+3/2,j-1/2) |
        |--------------------------------------|--------------------------------------|--------------------------------------|

            OR

        Reconstruct the limited extrapolants from the interface values. Returns the face averages in the form of w+(y) & w-(y) when considering x-axis, and w+(x) & w-(x) when considering y-axis
        |                                   w(i-1/2)                               w(i+1/2)                                  |
        |           o (i-1,j+1)           w- <-|-> w+          o (i,j+1)         w- <-|-> w+         o (i+1,j+1)        w- <-|
        |                                      |                                      |                                      |
        |                                      |                                      |                                      |
        |                                      |                                      |                                      |
        |                               w-+(y) | w++(y)                        w-+(y) | w++(y)                        w-+(y) |
        |                    w_D-(i-1/2,j+1/2) | w_D+(i-1/2,j+1/2)  w_D-(i+1/2,j+1/2) | w_D+(i+1/2,j+1/2)  w_D-(i+3/2,j+1/2) |
        |                                    ^ | ^                                  ^ | ^                                  ^ |
        |                                    | | |                                  | | |                                  | |
        |--------------------------------------|--------------------------------------|--------------------------------------|
        |                                    | | |                                  | | |                                  | |
        |                                    v | v                                  v | v                                  v |
        |                    w_U-(i-1/2,j+1/2) | w_U+(i-1/2,j+1/2)  w_U-(i+1/2,j+1/2) | w_U+(i+1/2,j+1/2)  w_U-(i+3/2,j+1/2) |
        |                               w--(y) | w+-(y)                        w--(y) | w+-(y)                        w--(y) |
        |                                      |                                      |                                      |
        |                                      |                                      |                                      |
        |                                      |                                      |                                      |
        |            o (i-1,j)            w- <-|-> w+           o (i,j)          w- <-|-> w+         o (i+1,j)          w- <-|
        """
        if method == "weno":
            if "c" in sim_variables.subgrid:
                reconstruct = cweno.reconstruct
            elif "z" in sim_variables.subgrid:
                reconstruct = wenoz.reconstruct
            else:
                reconstruct = weno.reconstruct
        elif method == "ppm":
            if sim_variables.ppm_dissipate:
                reconstruct = lambda _grid, _sim_variables, _axis: ppm.reconstruct(_grid, _sim_variables, _axis, eta=eta)
            else:
                reconstruct = ppm.reconstruct
        elif method == "plm":
            reconstruct = plm.reconstruct
        else:
            reconstruct = pcm.reconstruct

        # Each axis data in the 'data' dict will have a pair of reconstructed interfaces w+ & w-
        # Each interface need to be reconstructed along the same appropriate normal axis (returns intfs=[[w++,w+-],[w-+,w--]])
        # e.g. x-axis interface reconstructed along y-axis --> intfs = [[E+,E-],[W+,W-]]
        # Therefore for each normal axis, there will be 4 reconstructed corners/lines (2D/3D)
        def reconstruct_per_interface_pair(interface_pair, _sim_variables, normal_axis):
            intfs = []
            with concurrent.futures.ThreadPoolExecutor() as inner_executor:
                jobs = inner_executor.map(reconstruct, interface_pair, repeat(_sim_variables), repeat(normal_axis))

                for [wU, wD] in jobs:
                    # Re-align the interfaces so that cell wall is in between interfaces
                    prim_plus, prim_minus = fv.slice_(fv.add_boundary(wD, sim_variables, axis=normal_axis), normal_axis, start=1), fv.slice_(fv.add_boundary(wU, sim_variables, axis=normal_axis), normal_axis, end=-1)

                    # Append only the upwind corners/lines to list
                    intfs.append([fv.slice_(prim_plus, normal_axis, start=1), fv.slice_(prim_minus, normal_axis, start=1)])

            return intfs

        # Collate interfaces based on ortho_axes
        # e.g. computing emf in z-axis: axis = 2 --> ortho_axes = [0,1], normal_axes = [1,0]
        # interfaces = [ 0: (E,W) , 1: (N,S) ]
        interfaces = [data[axis]['interfaces'] for axis in ortho_axes]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            jobs = executor.map(reconstruct_per_interface_pair, interfaces, repeat(sim_variables), normal_axes)
            ortho_interfaces = dict(zip(normal_axes, jobs))

    return ortho_interfaces


# Compute the corner/line electric fields wrt to corner/line for each axis [Verma et al., 2018; Mignone & Del Zanna, 2020]
def compute_emf(ortho_interfaces, alphas, axis, dissipative=False):
    abscissa, ordinate, applicate = (axis + np.array(range(3)))%3

    axis_data = ortho_interfaces[abscissa]

    # DICT KEYS CORRESPOND TO RECONSTRUCTION AXIS
    # For -v x B in x-axis (axis=0, thumb), Bz along y (axis=1, index finger) becomes (E,W) & By along z (axis=2, middle finger) becomes (N,S)
    # For -v x B in y-axis (axis=1, thumb), Bx along z (axis=2, index finger) becomes (E,W) & Bz along x (axis=0, middle finger) becomes (N,S)
    # For -v x B in z-axis (axis=2, thumb), By along x (axis=0, index finger) becomes (E,W) & Bx along y (axis=1, middle finger) becomes (N,S)
    try:
        # Assume computation for the z-axis EMF here (axis=2), and so reconstructions along x-axis (ordinate) & y-axis (applicate) are needed
        """
        By:                                 Bx:
                         |                             |
                         |                      NW (+) | NE (+)
        N --->  (-) WN <-|-> EN (+)                  ^ | ^
        -----------------o-----------       -----------o-----------
        S --->  (-) WS <-|-> ES (+)                  v | v
                         |                      SW (-) | SE (-)
                         |                             |
                                                     ^ | ^
                                                     | | |
                                                     W | E
        """
        [eastnorth, westnorth], [eastsouth, westsouth] = axis_data[ordinate]
        [northeast, southeast], [northwest, southwest] = axis_data[applicate]
        [ap_x, am_x], [ap_y, am_y] = alphas[ordinate], alphas[applicate]
    except (KeyError, TypeError):
        # For 2D cases where the z-axis is missing, make a zeros_like data from the x- or y-axis instead
        emf = np.zeros_like(alphas[0][0])
    else:
        vx, vy = 1+ordinate, 1+applicate
        Bx, By = 5+ordinate, 5+applicate

        # Compute the corner mag. fields wrt to corner/line
        SW = .5*(westsouth[...,vy]+southwest[...,vy])*southwest[...,Bx] - .5*(southwest[...,vx]+westsouth[...,vx])*westsouth[...,By]
        SE = .5*(eastsouth[...,vy]+southeast[...,vy])*southeast[...,Bx] - .5*(southeast[...,vx]+eastsouth[...,vx])*eastsouth[...,By]
        NW = .5*(westnorth[...,vy]+northwest[...,vy])*northwest[...,Bx] - .5*(northwest[...,vx]+westnorth[...,vx])*westnorth[...,By]
        NE = .5*(eastnorth[...,vy]+northeast[...,vy])*northeast[...,Bx] - .5*(northeast[...,vx]+eastnorth[...,vx])*eastnorth[...,By]

        # Averaging procedure for the 4-fold values at each corner/line
        if dissipative:
            # More diffusive/dissipative solver that uses simple averaging [Balsara, 2010]
            S = np.maximum(
                np.maximum(np.abs(ap_x), np.abs(am_x)), 
                np.maximum(np.abs(ap_y), np.abs(am_y))
            )
            emf = (
                .25 * (SW + SE + NW + NE) 
                - S/2 * (.5*(northeast[...,Bx] + northwest[...,Bx]) - .5*(southeast[...,Bx] + southwest[...,Bx])) 
                + S/2 * (.5*(eastnorth[...,By] + eastsouth[...,By]) - .5*(westnorth[...,By] + westsouth[...,By]))
            )
        else:
            # [Londrillo & Del Zanna, 2004]
            emf = (
                fv.divide(ap_x*ap_y*SW + am_x*ap_y*SE + ap_x*am_y*NW + am_x*am_y*NE, (ap_x+am_x)*(ap_y+am_y)) 
                + fv.divide(ap_y*am_y, ap_y+am_y) * .5 * ((southeast[...,Bx] + southwest[...,Bx]) - (northeast[...,Bx] + northwest[...,Bx])) 
                - fv.divide(ap_x*am_x, ap_x+am_x) * .5 * ((westnorth[...,By] + westsouth[...,By]) - (eastnorth[...,By] + eastsouth[...,By]))
            )

    return emf


# Compute constrained transport flux using corner/line emfs [Mignone & Del Zanna, 2020]
# The hydro fluxes are unaltered. The magnetic fluxes are computed for each axis and automatically allocated,
# while setting the other axes to zero
def compute_ct_flux(flux, emfs, sim_variables, axis):
    # Get the orthogonal axes & emfs from the axis being computed
    ortho_axes = np.delete(np.arange(3), axis)
    ortho_emfs = list(map(emfs.get, ortho_axes))

    # Get normal axis to the orthogonal axes (for derivatives)
    normal_axes = np.roll(ortho_axes, shift=-1)

    # Pad emf in normal axis and compute centred difference for emf flux
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