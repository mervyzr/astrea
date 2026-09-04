import numpy as np

from functions import math as mfuncs
from numkit import kernels

##############################################################################
# Grid functions used throughout the finite volume code
##############################################################################
# These helpers used to fan each of their 2-3 orthogonal-axis tasks out to a fresh
# ThreadPoolExecutor, while themselves being called from the concurrent axis sweeps in
# spatial.evolve. That was ~60 pool creations per RHS evaluation for tasks that are pure
# memory bandwidth, so it oversubscribed the cores and multiplied the transient footprint by
# the number of orthogonal axes. executor.map yields in order, so plain loops are exact.
# Numba's threading layer on this platform is workqueue, which aborts the process if a
# parallel kernel is entered from more than one Python thread, so these also have to be
# serial before any fused kernel can be called from here.

# Create a physical grid for a single axis
def make_physical_grid(coordinates, cells, idx):
    start_pos, end_pos = coordinates[idx]
    dh = np.abs(np.diff(coordinates[idx])[0])/cells[idx]
    half_cell = .5 * dh
    return np.average(coordinates[idx]), np.linspace(start_pos-half_cell, end_pos+half_cell, cells[idx]+2)[1:-1]


# Average the submatrices of size (nrows, ncols) in a (h, w) 2D array
def blockwise_view(arr, nrows, ncols):
    h, w = arr.shape
    assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    block_grid = arr.reshape(h//nrows, nrows, -1, ncols).swapaxes(1,2).reshape(-1, nrows, ncols)
    return np.average(block_grid, axis=(1,2)).reshape(h//nrows, w//ncols)


# Slice grid along axis
def slice_(grid, axis, start=0, end=None, step=1, *args):
    slc = [slice(None)] * grid.ndim

    if args and (2 <= len(args) <= 3):
        try:
            start, end, step = args
        except ValueError:
            start, end = args

    if end == None:
        end = grid.shape[axis]

    slc[axis] = slice(start, end, step)
    return grid[tuple(slc)]


# Add boundary conditions
def add_boundary(grid, sim_variables, stencil=1, axis=0):
    padding = [(0,0)] * grid.ndim
    padding[axis] = (stencil,stencil)
    return np.pad(grid, padding, mode=sim_variables.boundary)


# Finite difference derivative (second order) of a padded grid
# [ W(i+1) - W(i) ] - [ W(i) - W(i-1) ] = W(i+1) - 2W(i) + W(i-1)
# Handed to a kernel rather than expressed in numpy. The numpy form allocated five full-size
# arrays per call (the np.pad copy, two np.diff results, the subtraction and the scaling) and
# this is the single most-called function in the RHS -- 72 of the 84 np.pad calls per
# spatial.evolve came from here. Verified bit-identical to the previous implementation for
# 1/2/3 dimensions, every axis and all three boundary modes.
def laplacian(grid, sim_variables, axis, out=None):
    return kernels.scaled_laplacian(
        grid, axis, _inv_ds2(sim_variables, axis), kernels.bc_code(sim_variables), out=out
    )


# 1/ds^2 for an axis as a plain float; sim_variables.ds holds length-1 arrays
def _inv_ds2(sim_variables, axis):
    return float(np.ravel(1/(sim_variables.ds[axis]**2))[0])


# scale * laplacian(grid) accumulated straight into base, allocating nothing. Kept as two
# separate multiplications (scale, then 1/ds^2) to match the numpy callers exactly.
def add_scaled_laplacian(base, grid, sim_variables, axis, scale):
    return kernels.add_scaled_laplacian(
        base, grid, axis, float(np.ravel(scale)[0]),
        _inv_ds2(sim_variables, axis), kernels.bc_code(sim_variables),
    )


# Convert between pressure P and total energy density e_tot; P is also related to the internal energy density e_int: P = (gamma-1) * e_int
# Do note that the energy densities e are related to the energies E: e_tot = rho * E_tot, e_int = rho * E_int
def convert_thermo_variable(variable, grid, sim_variables):
    rho, pressure, vels, Bfields = sim_variables.rho, sim_variables.pressure, sim_variables.vels, sim_variables.Bfields
    energy, momentums = pressure, vels
    gamma, permeability = sim_variables.gamma, sim_variables.constants.mu_0

    if variable.lower().startswith('p'):
        # pressure -> (total) energy density
        return (
            grid[...,pressure]/(gamma-1)
            + .5 * (grid[...,rho] * mfuncs.norm2(grid[...,vels]))
            + .5 * (mfuncs.norm2(grid[...,Bfields]))/permeability
        )
    elif variable.lower().startswith('e') or 'energy' in variable.lower():
        # (total) energy density -> pressure
        return (
            (gamma-1) * (
                grid[...,energy]
                - .5 * (grid[...,rho] * mfuncs.norm2(mfuncs.divide(grid[...,momentums], grid[...,rho][...,None])))
                - .5 * (mfuncs.norm2(grid[...,Bfields]))/permeability
                )
        )


# Handler for conversion
def convert(variable_form, grid, sim_variables):
    converter = variable_convert if sim_variables.grid_interpolate else variable_point_convert
    return converter(variable_form, grid, sim_variables)


# Pointwise (exact) conversion of conservative variables q <-> primitive variables w (up to 2nd-order accurate)
# Handed to a per-cell kernel, which keeps every intermediate in a register. The numpy form
# below made four full-grid passes -- np.copy, convert_thermo_variable, mfuncs.divide and
# mfuncs.norm2 -- and this is one of the hottest functions in the RHS. Results agree to 1-2
# ulp rather than exactly, because mfuncs.norm2 is an einsum that contracts with FMA and so
# rounds differently from a sequential sum; the kernel is the more accurate of the two.
def variable_point_convert(variable_form, grid, sim_variables):
    if grid.shape[-1] == sim_variables.nvars:
        return kernels.point_convert(
            variable_form, grid, sim_variables.gamma, sim_variables.constants.mu_0
        )
    return _variable_point_convert_numpy(variable_form, grid, sim_variables)


# Reference implementation, kept for anything not carrying the full nvars components
def _variable_point_convert_numpy(variable_form, grid, sim_variables):
    rho, pressure, energy, vels, momentums = sim_variables.rho, sim_variables.pressure, sim_variables.energy, sim_variables.vels, sim_variables.momentums
    arr = np.copy(grid)

    if variable_form.lower().startswith("p"):
        arr[...,energy] = convert_thermo_variable('pressure', grid, sim_variables)
        arr[...,momentums] = grid[...,vels] * grid[...,rho][...,None]
    elif variable_form.lower().startswith("c"):
        arr[...,pressure] = convert_thermo_variable('energy', grid, sim_variables)
        arr[...,vels] = mfuncs.divide(grid[...,momentums], grid[...,rho][...,None])
    return arr


# Variable inversion using the conversion of the grid (base) and the Taylor expansion terms
# (expansion) through a Laplacian (2nd-deriv, 2nd-order) approx. for each axis (up to
# 4th-order accurate). Accumulated straight into base and expansion, and the pointwise
# conversion is taken as an argument: it depends only on the grid, not on the axis, so
# computing it inside here meant redoing the whole conversion once per axis.
def variable_inversion_per_axis(variable_form, grid, sim_variables, axis, base, expansion, converted_avg):
    scale = (sim_variables.ds[axis]**2)/24
    add_scaled_laplacian(base, grid, sim_variables, axis, -scale)
    add_scaled_laplacian(expansion, converted_avg, sim_variables, axis, scale)


# Converting cell-averaged conservative variables <q>_{i,j} <-> cell-averaged primitive variables <w>_{i,j} at higher-order accuracy
def variable_convert(variable_form, grid, sim_variables):
    base, expansion = np.copy(grid), np.zeros_like(grid)
    converted_avg = variable_point_convert(variable_form, grid, sim_variables)

    for axis in sim_variables.axes:
        variable_inversion_per_axis(variable_form, grid, sim_variables, axis, base, expansion, converted_avg)

    return variable_point_convert(variable_form, base, sim_variables) + expansion


# Converting face-averaged conservative variables <q>_{i+1/2,j} <-> face-averaged primitive variables <w>_{i+1/2,j}
# Looks very similar to variable_convert, doesn't it? Tempting to combine, but don't do it; easier for debugging this way
def variable_convert_intf(variable_form, grid, sim_variables, axis):
    base, expansion = np.copy(grid), 0

    if sim_variables.grid_interpolate and sim_variables.multidimensional:
        ortho_axes = sim_variables.axes[sim_variables.axes != axis]
        expansion = np.zeros_like(grid)
        converted_avg = variable_point_convert(variable_form, grid, sim_variables)

        for axis_ in ortho_axes:
            variable_inversion_per_axis(variable_form, grid, sim_variables, axis_, base, expansion, converted_avg)

    new_grid = variable_point_convert(variable_form, base, sim_variables) + expansion

    if sim_variables.magnetic:
        new_grid[...,5+sim_variables.axes] = grid[...,5+sim_variables.axes]

    return new_grid


# Method convert between point-representation (finite difference) and averaged-representation (finite volume) [ALL AXES]
# Converting cell-centred variables q_{i,j} <-> cell-averaged variables <q>_{i,j} through a Laplacian (2nd-deriv, 2nd-order) approx. for each axis (up to 4th-order accurate)
def method_convert_cell(grid_form, grid, sim_variables, axis=None):
    base = np.copy(grid)

    if grid_form.lower().startswith('a'):
        coeff = -1  # averaged -> point
    elif grid_form.lower().startswith('p'):
        coeff = 1  # point -> averaged

    for idx, axis_ in enumerate(sim_variables.axes):
        add_scaled_laplacian(base, grid, sim_variables, axis_, coeff * (sim_variables.ds[sim_variables.axes[idx]]**2)/24)
    return base


# Method convert between point-representation (finite difference) and averaged-representation (finite volume) for interfaces [ORTHOGONAL AXES]
# Converting face-centred variables q_{i+1/2,j} <-> face-averaged variables <q>_{i+1/2,j} through a Laplacian (2nd-deriv, 2nd-order) approx. (up to 4th-order accurate)
def method_convert_intf(grid_form, grid, sim_variables, axis):
    base = np.copy(grid)
    ortho_axes = sim_variables.axes[sim_variables.axes != axis]

    if grid_form.lower().startswith('a'):
        coeff = -1  # averaged -> point
    elif grid_form.lower().startswith('p'):
        coeff = 1  # point -> averaged

    # !! The Laplacian is taken along ortho_axes but ds is indexed by sim_variables.axes[idx],
    # !! which is a different axis whenever ortho_axes != axes[:len(ortho_axes)]. Harmless on a
    # !! uniform grid, wrong when ds differs per axis. Behaviour preserved here deliberately;
    # !! flagged rather than changed because fixing it changes results.
    for idx, axis_ in enumerate(ortho_axes):
        add_scaled_laplacian(base, grid, sim_variables, axis_, coeff * (sim_variables.ds[sim_variables.axes[idx]]**2)/24)
    return base


# Handler for converting (at higher-order) each +/- interface in each axis from averaged interfaces to point/centred interfaces in the multi-dimensional higher-order schemes
def approx_face_avg(interfaces, sim_variables, axis):
    if sim_variables.grid_interpolate and sim_variables.multidimensional:
        return [method_convert_intf('avg', interface, sim_variables, axis) for interface in interfaces]
    else:
        return list(interfaces)
    

# Compute the 4th-order interface-centred fluxes from the interface-averaged fluxes via higher order approximation for each orthogonal axis
def approx_flux_avg(cntrd_fluxes, avgd_fluxes, sim_variables, axis):
    ortho_axes = sim_variables.axes[sim_variables.axes != axis]

    for idx, axis_ in enumerate(ortho_axes):
        add_scaled_laplacian(cntrd_fluxes, avgd_fluxes, sim_variables, axis_, -(sim_variables.ds[ortho_axes[idx]]**2)/24)
    return cntrd_fluxes


# Re-align the interfaces so that cell wall is in between interfaces
def assign_interfaces(interfaces, grid, sim_variables, axis):
    wL, wR = interfaces
    return slice_(add_boundary(wL, sim_variables, axis=axis), axis, start=1), slice_(add_boundary(wR, sim_variables, axis=axis), axis, end=-1)


# Create a grid of perturbation values
def pertubations(grid, max_ampl):
    return np.random.uniform(-max_ampl/2, max_ampl, size=grid.shape)