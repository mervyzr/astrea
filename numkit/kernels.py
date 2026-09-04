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
