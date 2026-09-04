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


##############################################################################
# CWENO(Z) reconstruction
##############################################################################

@njit(cache=True, inline='always')
def _stencil_index(i, n, code):
    """Logical index i, possibly outside [0,n), mapped onto the cell np.pad would have used.

    Covers |i| up to the two-cell halo the five-point stencil needs. wrap is periodic, edge
    clamps to the boundary cell, reflect mirrors about the boundary cell without repeating it
    (np.pad's 'reflect', not 'symmetric').
    """
    if 0 <= i < n:
        return i
    if code == BC_WRAP:
        return i % n
    elif code == BC_EDGE:
        return 0 if i < 0 else n-1
    else:
        return -i if i < 0 else 2*(n-1) - i


@njit(parallel=True, cache=True)
def _cweno_reconstruct(g, wl, wr, eps, power, code, wenoz):
    """CWENO/CWENOZ reconstruction from cell averages to both face averages.

    Levy et al. 1999 eq. 3.11-3.14, with the CWENOZ tau of Cravero et al. 2019. Operand
    grouping follows the numpy original expression by expression so the result matches it.
    """
    left, n, right = g.shape
    ntiles = (right + _TILE - 1)//_TILE

    # Linear weights dC_k [tbl. 3.1]
    dc0, dc1, dc2 = 1/6, 2/3, 1/6

    for tile in prange(left * ntiles):
        l = tile//ntiles
        r0 = (tile % ntiles) * _TILE
        r1 = min(r0 + _TILE, right)
        for i in range(n):
            im2 = _stencil_index(i-2, n, code)
            im1 = _stencil_index(i-1, n, code)
            ip1 = _stencil_index(i+1, n, code)
            ip2 = _stencil_index(i+2, n, code)
            for r in range(r0, r1):
                m2 = g[l, im2, r]
                m1 = g[l, im1, r]
                z = g[l, i, r]
                p1 = g[l, ip1, r]
                p2 = g[l, ip2, r]

                # Smoothness indicators [eq. 3.14]
                si0 = 13/12 * (m2 - 2*m1 + z)**2 + 1/4 * (m2 - 4*m1 + 3*z)**2
                si1 = 13/12 * (m1 - 2*z + p1)**2 + 1/4 * (m1 - p1)**2
                si2 = 13/12 * (z - 2*p1 + p2)**2 + 1/4 * (3*z - 4*p1 + p2)**2

                if wenoz:
                    d = m2 - 2*m1 + 2*p1 - p2
                    si_opt = (
                        1/4 * (p1 - m1 + 1/3*d)**2
                        + 13/12 * (m1 - 2*z + p1 + 1/12*d)**2
                        + 7/240 * (m2 - 4*m1 + 6*z - 4*p1 + p2)**2
                        + 9/80 * (-m2 + 2*m1 - 2*p1 + p2)**2
                    )
                    tau = abs(si_opt - (si0 + si1 + si2)/3)
                    a0 = dc0 * (1 + (tau/(si0 + eps))**power)
                    a1 = dc1 * (1 + (tau/(si1 + eps))**power)
                    a2 = dc2 * (1 + (tau/(si2 + eps))**power)
                else:
                    a0 = dc0/(si0 + eps)**power
                    a1 = dc1/(si1 + eps)**power
                    a2 = dc2/(si2 + eps)**power

                # Non-linear weights [eq. 3.11]
                total = a0 + a1 + a2
                o0 = _guarded_divide(a0, total)
                o1 = _guarded_divide(a1, total)
                o2 = _guarded_divide(a2, total)

                # No need to flip the linear weights since dC_k is symmetrical
                wr[l, i, r] = 1/6 * (
                    o0 * (2*m2 - 7*m1 + 11*z)
                    + o1 * (-m1 + 5*z + 2*p1)
                    + o2 * (2*z + 5*p1 - p2)
                )
                wl[l, i, r] = 1/6 * (
                    o0 * (2*z + 5*m1 - m2)
                    + o1 * (-p1 + 5*z + 2*m1)
                    + o2 * (2*p2 - 7*p1 + 11*z)
                )


def cweno_reconstruct(grid, axis, eps, code, power=2, wenoz=False):
    """Return (wL, wR), the reconstructed left and right face averages along axis."""
    grid, view = _as_axis_view(grid, axis)
    wl, wr = np.empty_like(grid), np.empty_like(grid)
    _, wl_view = _as_axis_view(wl, axis)
    _, wr_view = _as_axis_view(wr, axis)
    _cweno_reconstruct(view, wl_view, wr_view, eps, power, code, wenoz)
    return wl, wr


##############################################################################
# Riemann solver
##############################################################################

def _as_axis_view_nv(arr, axis, nvars):
    """View arr as (left, n, right, nvars) around axis, with the components kept trailing."""
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    shape = arr.shape
    left = int(np.prod(shape[:axis])) if axis else 1
    n = shape[axis]
    right = int(np.prod(shape[axis+1:-1])) if axis + 1 < len(shape) - 1 else 1
    return arr, arr.reshape(left, n, right, nvars)


@njit(parallel=True, cache=True)
def _lax_friedrich(eigvals, cons_plus, cons_minus, flux_plus, flux_minus, out):
    """Local Lax-Friedrich intercell flux.

    eigvals is max|lambda| per cell and is two cells longer than the interface arrays along
    the sweep axis, so the dissipation coefficient at interface i averages the pairwise maxima
    on either side of it. Grouping follows the numpy original,
    .5*(F- + F+) - (q+ - q-)*(.5*a), so the result is bit-identical to it.
    """
    left, n, right, nvars = cons_plus.shape
    for l in prange(left):
        for i in range(n):
            for r in range(right):
                # Local max eigenvalue between consecutive pairs of cells
                centre = eigvals[l, i+1, r]
                above = eigvals[l, i+2, r]
                below = eigvals[l, i, r]
                plus = above if above > centre else centre
                minus = centre if centre > below else below

                # Averaged maximum localised eigenvalue at this interface
                half_max_eigval = .5 * (.5 * (plus + minus))
                for v in range(nvars):
                    out[l, i, r, v] = (
                        (flux_minus[l, i, r, v] + flux_plus[l, i, r, v]) * .5
                        - (cons_plus[l, i, r, v] - cons_minus[l, i, r, v]) * half_max_eigval
                    )


def lax_friedrich(local_max_eigvals, cons_plus, cons_minus, flux_plus, flux_minus, axis, out=None):
    """Local Lax-Friedrich flux; local_max_eigvals is max|lambda| per cell, without a component axis."""
    nvars = cons_plus.shape[-1]
    cons_plus, cp = _as_axis_view_nv(cons_plus, axis, nvars)
    _, cm = _as_axis_view_nv(cons_minus, axis, nvars)
    _, fp = _as_axis_view_nv(flux_plus, axis, nvars)
    _, fm = _as_axis_view_nv(flux_minus, axis, nvars)

    if not local_max_eigvals.flags.c_contiguous:
        local_max_eigvals = np.ascontiguousarray(local_max_eigvals)
    shape = local_max_eigvals.shape
    left = int(np.prod(shape[:axis])) if axis else 1
    eig = local_max_eigvals.reshape(left, shape[axis], -1)

    if out is None:
        out = np.empty_like(cons_plus)
    _, out_view = _as_axis_view_nv(out, axis, nvars)
    _lax_friedrich(eig, cp, cm, fp, fm, out_view)
    return out


##############################################################################
# Multi-axis Laplacian accumulation
##############################################################################
# The Taylor-expansion corrections sum a Laplacian over two or three axes into the same
# destination. Done one axis at a time that is two or three separate streaming passes over the
# whole grid, and at these sizes each pass already runs at close to memory bandwidth -- so the
# only way to go faster is to make fewer passes. This visits each cell once and accumulates
# every requested axis into it.
#
# The accumulation stays sequential within a cell (val += term for each axis in turn, in the
# caller's axis order) rather than summing the terms first, because that is what a sequence of
# separate passes did and it keeps the result bit-identical.


@njit(parallel=True, cache=True)
def _add_laplacians_4d(base, g, order, scales, invs, count, code):
    """base += sum over the requested axes of scale * (inv_ds2 * laplacian(g, axis)).

    Arrays are canonical 4D (n0, n1, n2, nv); absent spatial dimensions are length 1.

    The axis dispatch is hoisted above the component loop, so the innermost loop is a clean
    fixed-stride run over nv that the compiler can vectorise. Putting the dispatch inside it
    instead measured no faster than one separate pass per axis, which is the whole point of
    this kernel. base[i,j,k,:] stays in L1 across the axes, so the repeated read-modify-write
    of it is free relative to the streaming reads of g.
    """
    n0, n1, n2, nv = g.shape
    for i in prange(n0):
        for j in range(n1):
            for k in range(n2):
                for t in range(count):
                    axis = order[t]
                    scale = scales[t]
                    inv = invs[t]

                    if axis == 0:
                        below, above = _stencil_index(i-1, n0, code), _stencil_index(i+1, n0, code)
                        for v in range(nv):
                            centre = g[i, j, k, v]
                            base[i, j, k, v] += scale * (inv * ((g[above, j, k, v] - centre) - (centre - g[below, j, k, v])))
                    elif axis == 1:
                        below, above = _stencil_index(j-1, n1, code), _stencil_index(j+1, n1, code)
                        for v in range(nv):
                            centre = g[i, j, k, v]
                            base[i, j, k, v] += scale * (inv * ((g[i, above, k, v] - centre) - (centre - g[i, below, k, v])))
                    else:
                        below, above = _stencil_index(k-1, n2, code), _stencil_index(k+1, n2, code)
                        for v in range(nv):
                            centre = g[i, j, k, v]
                            base[i, j, k, v] += scale * (inv * ((g[i, j, above, v] - centre) - (centre - g[i, j, below, v])))


def _as_canonical_4d(arr, ndim_spatial):
    """View arr as (n0, n1, n2, nv), padding absent spatial dimensions with length 1."""
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    shape = list(arr.shape[:ndim_spatial]) + [1]*(3 - ndim_spatial) + [arr.shape[-1]]
    return arr, arr.reshape(shape)


def add_laplacians(base, grid, axes, scales, invs, code, ndim_spatial):
    """base += sum_a scales[a] * (invs[a] * laplacian(grid, axes[a])), in one pass.

    axes, scales and invs are parallel sequences in the order the caller would have applied
    them. Returns base.
    """
    if not base.flags.c_contiguous:
        raise ValueError("base must be C-contiguous to be updated in place")
    grid, grid_view = _as_canonical_4d(grid, ndim_spatial)
    _, base_view = _as_canonical_4d(base, ndim_spatial)
    order = np.asarray(axes, dtype=np.int64)
    _add_laplacians_4d(
        base_view, grid_view, order,
        np.asarray(scales, dtype=np.float64), np.asarray(invs, dtype=np.float64),
        len(order), code,
    )
    return base


##############################################################################
# Wave speed bounds
##############################################################################
# Every consumer of the characteristic spectrum only ever reduces it to the signed extremes
# along the last axis -- compute_eigmax, the Lax-Friedrich and HLL solvers, and the CT alphas.
# Those extremes are available in closed form: the fast magnetosonic speed dominates the whole
# seven-wave set (cFF^2 = (X + sqrt(X^2 - Y))/2 with X = cs^2 + cA^2 and Y = (2 cs cAx)^2, and
# cAx <= cA gives sqrt(X^2 - Y) >= |cs^2 - cA^2|, hence cFF >= max(cs, cA)), so
#
#     max = uN + c,   min = uN - c,   max|.| = |uN| + c
#
# and the last of those is exact in IEEE because negation is. Carrying two scalar fields
# instead of five or seven avoids building the spectrum at all -- it was assembled with
# np.array(...).transpose(...), whose last axis then had stride N^3, making the reduction over
# it maximally cache-hostile.

WAVESPEED_NORMAL, WAVESPEED_DOMINANT = 0, 1  # component layout of the returned array


@njit(parallel=True, cache=True)
def _wavespeed_bounds(grid, out, gamma, permeability, axis, magnetic):
    """out[...,0] = normal velocity, out[...,1] = dominant wave speed along axis."""
    ncells = grid.shape[0]
    for c in prange(ncells):
        rho = grid[c, 0]
        pressure = grid[c, 4]

        # mfuncs.sqrt clamps at zero; mfuncs.divide yields the 1/eps sentinel on a zero divisor
        cs2 = _guarded_divide(gamma * pressure, rho)
        sound_speed = np.sqrt(cs2) if cs2 > 0. else 0.

        out[c, WAVESPEED_NORMAL] = grid[c, 1+axis]

        if not magnetic:
            out[c, WAVESPEED_DOMINANT] = sound_speed
            continue

        bx, by, bz = grid[c, 5], grid[c, 6], grid[c, 7]
        rho_mu = rho * permeability
        root_rho_mu = np.sqrt(rho_mu) if rho_mu > 0. else 0.

        b2 = bx*bx + by*by + bz*bz
        b_norm = np.sqrt(b2) if b2 > 0. else 0.
        alfven = _guarded_divide(b_norm, root_rho_mu)
        alfven_x = _guarded_divide(grid[c, 5+axis], root_rho_mu)

        total = sound_speed*sound_speed + alfven*alfven
        discriminant = total*total - (2 * sound_speed * alfven_x)**2
        root = np.sqrt(discriminant) if discriminant > 0. else 0.
        fast = .5 * (total + root)
        out[c, WAVESPEED_DOMINANT] = np.sqrt(fast) if fast > 0. else 0.


def wavespeed_bounds(grid, gamma, permeability, axis, magnetic):
    """(..., 2) array of [normal velocity, dominant wave speed] for each cell."""
    grid, view = _flat(grid)
    out = np.empty(grid.shape[:-1] + (2,), dtype=grid.dtype)
    _wavespeed_bounds(view, out.reshape(-1, 2), gamma, permeability, axis, magnetic)
    return out
