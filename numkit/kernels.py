import numpy as np
from numba import njit, prange

##############################################################################
# Fused numba kernels
##############################################################################
# These replace numpy expressions that were allocating full-size intermediates. The finite
# volume RHS is already fully vectorised -- there are no Python loops over cells anywhere in
# the code -- so the win here is not from removing interpreter overhead but from removing the
# temporaries: a vectorised RHS materialised roughly 85 full-size arrays per timestep, which
# is simultaneously the memory ceiling and the speed ceiling.
#
# Any array handed to a kernel is viewed as (left, n, right) around the axis being worked on,
# which lets one kernel serve any dimensionality and any axis without specialising on either.
# That view only exists for C-contiguous input, hence the ascontiguousarray guard in the
# wrappers.
#
# !! numba's threading layer on arm64 macOS is workqueue, which is not thread safe: entering a
# !! parallel kernel from more than one Python thread aborts the process. Never call these from
# !! inside a ThreadPoolExecutor.

# Boundary condition codes, matching the np.pad modes used by functions.grid.add_boundary
BC_WRAP, BC_EDGE, BC_REFLECT = 0, 1, 2

_BC_CODES = {"wrap": BC_WRAP, "edge": BC_EDGE, "reflect": BC_REFLECT}


def bc_code(sim_variables):
    """Map sim_variables.boundary onto a kernel boundary code."""
    try:
        return _BC_CODES[sim_variables.boundary]
    except KeyError:
        raise NotImplementedError(
            f"boundary mode {sim_variables.boundary!r} has no kernel equivalent; "
            f"known modes are {sorted(_BC_CODES)}"
        ) from None


@njit(cache=True, inline='always')
def _ghost(g, l, r, n, index, code):
    """The value np.pad would have placed one cell outside the domain.

    index is -1 for the low side and n for the high side. wrap takes the opposite edge, edge
    repeats the boundary cell, reflect mirrors about the boundary cell without repeating it.
    """
    if index < 0:
        if code == BC_WRAP:
            return g[l, n-1, r]
        elif code == BC_EDGE:
            return g[l, 0, r]
        else:
            return g[l, 1, r]
    else:
        if code == BC_WRAP:
            return g[l, 0, r]
        elif code == BC_EDGE:
            return g[l, n-1, r]
        else:
            return g[l, n-2, r]


# Work is tiled over both outer dimensions of the (left, n, right) view rather than over
# `left` alone. Parallelising over `left` looks natural but degenerates for axis 0, where
# left == 1 and the whole kernel would run on one thread. Iterating `r` innermost keeps the
# three stencil streams contiguous.
_TILE = 64


@njit(parallel=True, cache=True)
def _scaled_laplacian_into(g, out, inv_ds2, code):
    """out = inv_ds2 * [ (g[i+1] - g[i]) - (g[i] - g[i-1]) ] along the middle axis.

    Operand order matches the numpy original, 1/(ds**2) * (diff(p[1:]) - diff(p[:-1])), so the
    result is bit-identical to it.
    """
    left, n, right = g.shape
    ntiles = (right + _TILE - 1)//_TILE
    for tile in prange(left * ntiles):
        l = tile//ntiles
        r0 = (tile % ntiles) * _TILE
        r1 = min(r0 + _TILE, right)
        for i in range(n):
            below = i - 1
            above = i + 1
            for r in range(r0, r1):
                previous = _ghost(g, l, r, n, -1, code) if i == 0 else g[l, below, r]
                following = _ghost(g, l, r, n, n, code) if i == n-1 else g[l, above, r]
                centre = g[l, i, r]
                out[l, i, r] = inv_ds2 * ((following - centre) - (centre - previous))


@njit(parallel=True, cache=True)
def _add_scaled_laplacian(base, g, scale, inv_ds2, code):
    """base += scale * (inv_ds2 * laplacian(g)), with no intermediate array.

    Keeps scale and inv_ds2 as two separate multiplications because that is what the numpy
    callers did; folding them into one constant would not be bit-identical.
    """
    left, n, right = g.shape
    ntiles = (right + _TILE - 1)//_TILE
    for tile in prange(left * ntiles):
        l = tile//ntiles
        r0 = (tile % ntiles) * _TILE
        r1 = min(r0 + _TILE, right)
        for i in range(n):
            below = i - 1
            above = i + 1
            for r in range(r0, r1):
                previous = _ghost(g, l, r, n, -1, code) if i == 0 else g[l, below, r]
                following = _ghost(g, l, r, n, n, code) if i == n-1 else g[l, above, r]
                centre = g[l, i, r]
                base[l, i, r] += scale * (inv_ds2 * ((following - centre) - (centre - previous)))


def _as_axis_view(arr, axis):
    """View arr as (left, n, right) around axis, copying only if it is not C-contiguous."""
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    shape = arr.shape
    left = int(np.prod(shape[:axis])) if axis else 1
    n = shape[axis]
    right = int(np.prod(shape[axis+1:])) if axis + 1 < len(shape) else 1
    return arr, arr.reshape(left, n, right)


def scaled_laplacian(grid, axis, inv_ds2, code, out=None):
    """inv_ds2 * laplacian(grid) along axis, written into out (allocated if not given)."""
    grid, view = _as_axis_view(grid, axis)
    if out is None:
        out = np.empty_like(grid)
    _, out_view = _as_axis_view(out, axis)
    _scaled_laplacian_into(view, out_view, inv_ds2, code)
    return out


def add_scaled_laplacian(base, grid, axis, scale, inv_ds2, code):
    """base += scale * (inv_ds2 * laplacian(grid)) along axis, in place, no temporaries."""
    if not base.flags.c_contiguous:
        raise ValueError("base must be C-contiguous to be updated in place")
    grid, view = _as_axis_view(grid, axis)
    _, base_view = _as_axis_view(base, axis)
    _add_scaled_laplacian(base_view, view, scale, inv_ds2, code)
    return base


##############################################################################
# Per-cell kernels
##############################################################################
# Nothing below needs a stencil, so each works on the grid flattened to (ncells, nvars) and
# every intermediate stays in a register. The numpy originals allocated one full-size array per
# sub-expression -- the prim/cons conversion alone went through np.copy, convert_thermo_variable,
# mfuncs.divide and mfuncs.norm2, each of which is a separate pass over the whole grid.

EPS_SENTINEL = 1e16  # what mfuncs.divide yields for a zero divisor, i.e. 1/eps with eps=1e-16


@njit(cache=True, inline='always')
def _guarded_divide(dividend, divisor):
    """mfuncs.divide for a single value: the 1/eps sentinel when the divisor is exactly zero."""
    return dividend/divisor if divisor != 0 else EPS_SENTINEL


def _flat(arr):
    """View arr as (ncells, nvars), copying only if it is not C-contiguous."""
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return arr, arr.reshape(-1, arr.shape[-1])


@njit(parallel=True, cache=True)
def _prim_to_cons(grid, out, gamma, permeability):
    """Primitive -> conservative, matching variable_point_convert('primitive', ...)."""
    ncells = grid.shape[0]
    for c in prange(ncells):
        rho = grid[c, 0]
        vx, vy, vz = grid[c, 1], grid[c, 2], grid[c, 3]
        pressure = grid[c, 4]
        bx, by, bz = grid[c, 5], grid[c, 6], grid[c, 7]

        v2 = vx*vx + vy*vy + vz*vz
        b2 = bx*bx + by*by + bz*bz

        out[c, 0] = rho
        out[c, 1] = vx*rho
        out[c, 2] = vy*rho
        out[c, 3] = vz*rho
        out[c, 4] = pressure/(gamma-1) + .5 * (rho*v2) + .5 * b2/permeability
        out[c, 5] = bx
        out[c, 6] = by
        out[c, 7] = bz


@njit(parallel=True, cache=True)
def _cons_to_prim(grid, out, gamma, permeability):
    """Conservative -> primitive, matching variable_point_convert('conservative', ...)."""
    ncells = grid.shape[0]
    for c in prange(ncells):
        rho = grid[c, 0]
        energy = grid[c, 4]
        bx, by, bz = grid[c, 5], grid[c, 6], grid[c, 7]

        vx = _guarded_divide(grid[c, 1], rho)
        vy = _guarded_divide(grid[c, 2], rho)
        vz = _guarded_divide(grid[c, 3], rho)

        v2 = vx*vx + vy*vy + vz*vz
        b2 = bx*bx + by*by + bz*bz

        out[c, 0] = rho
        out[c, 1] = vx
        out[c, 2] = vy
        out[c, 3] = vz
        out[c, 4] = (gamma-1) * (energy - .5 * (rho*v2) - .5 * b2/permeability)
        out[c, 5] = bx
        out[c, 6] = by
        out[c, 7] = bz


def point_convert(variable_form, grid, gamma, permeability, out=None):
    """Pointwise prim <-> cons conversion. variable_form is the form of the *input*."""
    grid, view = _flat(grid)
    if out is None:
        out = np.empty_like(grid)
    _, out_view = _flat(out)

    if variable_form.lower().startswith("p"):
        _prim_to_cons(view, out_view, gamma, permeability)
    elif variable_form.lower().startswith("c"):
        _cons_to_prim(view, out_view, gamma, permeability)
    else:
        raise ValueError(f"unknown variable form {variable_form!r}")
    return out


@njit(parallel=True, cache=True)
def _flux(grid, out, gamma, permeability, abscissa, ordinate, applicate):
    """Ideal MHD flux along one axis, matching functions.numeric.compute_flux."""
    ncells = grid.shape[0]
    for c in prange(ncells):
        rho = grid[c, 0]
        pressure = grid[c, 4]

        # The coordinate rotation is done by permuting which component is the normal one
        un = grid[c, 1+abscissa]
        ut = grid[c, 1+ordinate]
        us = grid[c, 1+applicate]
        bn = grid[c, 5+abscissa]
        bt = grid[c, 5+ordinate]
        bs = grid[c, 5+applicate]

        v2 = grid[c, 1]*grid[c, 1] + grid[c, 2]*grid[c, 2] + grid[c, 3]*grid[c, 3]
        b2 = grid[c, 5]*grid[c, 5] + grid[c, 6]*grid[c, 6] + grid[c, 7]*grid[c, 7]
        vdotb = grid[c, 1]*grid[c, 5] + grid[c, 2]*grid[c, 6] + grid[c, 3]*grid[c, 7]

        out[c, 0] = rho*un
        out[c, 1+abscissa] = rho*un**2 + pressure + .5*b2 - (bn**2)/permeability
        out[c, 1+ordinate] = rho*un*ut - (bn*bt)/permeability
        out[c, 1+applicate] = rho*un*us - (bn*bs)/permeability
        out[c, 4] = un*(.5*rho*v2 + (gamma*pressure)/(gamma-1) + b2) - (bn*vdotb)/permeability
        out[c, 5+abscissa] = 0.
        out[c, 5+ordinate] = bt*un - bn*ut
        out[c, 5+applicate] = bs*un - bn*us


def flux(grid, gamma, permeability, axis, out=None):
    """Ideal MHD flux along axis, into out (allocated if not given)."""
    abscissa, ordinate, applicate = (axis + np.arange(3)) % 3
    grid, view = _flat(grid)
    if out is None:
        out = np.empty_like(grid)
    _, out_view = _flat(out)
    _flux(view, out_view, gamma, permeability, abscissa, ordinate, applicate)
    return out


@njit(parallel=True, cache=True)
def _roe_average(plus, minus, out):
    """Roe-averaged interface state, matching functions.numeric.compute_Roe_average.

    Note the weighting is not symmetric between the velocity and magnetic components: the
    velocities weight the plus state by sqrt(rho_plus) while the fields weight it by
    sqrt(rho_minus). That is what the numpy original does and it is reproduced here as-is.
    """
    ncells = plus.shape[0]
    for c in prange(ncells):
        # mfuncs.sqrt clamps at zero
        rp = plus[c, 0]
        rm = minus[c, 0]
        rho_plus = np.sqrt(rp) if rp > 0. else 0.
        rho_minus = np.sqrt(rm) if rm > 0. else 0.
        total = rho_minus + rho_plus

        out[c, 0] = rho_minus * rho_plus
        for v in range(1, 4):
            out[c, v] = _guarded_divide(plus[c, v]*rho_plus + minus[c, v]*rho_minus, total)
        out[c, 4] = _guarded_divide(rho_plus*plus[c, 4] + rho_minus*minus[c, 4], total)
        for v in range(5, 8):
            out[c, v] = _guarded_divide(plus[c, v]*rho_minus + minus[c, v]*rho_plus, total)


def roe_average(plus, minus, out=None):
    """Roe-averaged state between the plus- and minus-interface values."""
    plus, plus_view = _flat(plus)
    minus, minus_view = _flat(minus)
    if out is None:
        out = np.empty_like(plus)
    _, out_view = _flat(out)
    _roe_average(plus_view, minus_view, out_view)
    return out
