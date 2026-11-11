import numpy as np
from numba import njit, prange

##############################################################################
# Fused numba kernels; arrays are viewed as (left, n, right) around the working axis
#
# Never call the parallel kernels from more than one Python thread (e.g. ThreadPoolExecutor):
# numba's workqueue threading layer on arm64 macOS is not thread safe and aborts the process.
##############################################################################

# Tile width along `right`; tiling lets the stencil kernels parallelise even when left == 1
_TILE = 64
EPS_SENTINEL = 1e16


##############################################################################
# Boundary conditions
##############################################################################

# Get the boundary condition index (bci); integers are more safe than whole strings
def get_bci(sim_variables):
    try:
        return {"wrap": 0, "edge": 1, "reflect": 2}[sim_variables.boundary]
    except KeyError:
        raise NotImplementedError(
            f"boundary mode {sim_variables.boundary!r} has no kernel equivalent"
        ) from None


# Value of g one cell outside the domain, at index -1 (low side) or n (high side)
@njit(cache=True, inline='always')
def _ghost(g, l, r, n, index, bci):
    if index < 0:
        if bci == 0:
            # wrap
            return g[l, n-1, r]
        elif bci == 1:
            # edge
            return g[l, 0, r]
        else:
            # reflect
            return g[l, 1, r]
    else:
        if bci == 0:
            return g[l, 0, r]
        elif bci == 1:
            return g[l, n-1, r]
        else:
            return g[l, n-2, r]


# Map an index i with |i| < n onto [0, n): wrap is periodic, edge clamps to the boundary cell, reflect mirrors about the boundary cell without repeating it
@njit(cache=True, inline='always')
def _stencil_index(i, n, bci):
    if 0 <= i < n:
        return i
    if bci == 0:
        # wrap
        return i % n
    elif bci == 1:
        # edge
        return 0 if i < 0 else n-1
    else:
        # reflect
        return -i if i < 0 else 2*(n-1) - i


# _stencil_index for an array of n faces that includes both boundary faces; under wrap those two faces coincide, so the period is n-1
@njit(cache=True, inline='always')
def _face_index(i, n, bci):
    if bci == 0 and not 0 <= i < n:
        return i % (n - 1)
    return _stencil_index(i, n, bci)


##############################################################################
# Array views
##############################################################################

# Return arr and its (left, n, right) view around axis
def _as_axis_view(arr, axis):
    shape = arr.shape
    left = int(np.prod(shape[:axis])) if axis else 1
    n = shape[axis]
    right = int(np.prod(shape[axis+1:])) if axis + 1 < len(shape) else 1
    return arr, arr.reshape(left, n, right)


# View arr as (left, n, right, trailing) around axis, trailing being its last dimension
def _axis_view(arr, axis, trailing):
    shape = arr.shape
    left = int(np.prod(shape[:axis])) if axis else 1
    right = int(np.prod(shape[axis+1:-1])) if axis + 1 < len(shape) - 1 else 1
    return arr.reshape(left, shape[axis], right, trailing)


# Return (C-contiguous) arr and its (ncells, nvars) view
def _flat(arr):
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return arr, arr.reshape(-1, arr.shape[-1])


# Return (C-contiguous) arr and its (n0, n1, n2, nvars) view with absent spatial dimensions of length 1
def _as_canonical_4d(arr, ndim_spatial):
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    shape = list(arr.shape[:ndim_spatial]) + [1]*(3 - ndim_spatial) + [arr.shape[-1]]
    return arr, arr.reshape(shape)


##############################################################################
# Scalar helpers
##############################################################################

@njit(cache=True, inline='always')
def _guarded_divide(dividend, divisor):
    return dividend/divisor if divisor != 0 else EPS_SENTINEL


@njit(cache=True, inline='always')
def _npmin(a, b):
    return a if (a <= b or a != a) else b


@njit(cache=True, inline='always')
def _npmax(a, b):
    return a if (a >= b or a != a) else b


@njit(cache=True, inline='always')
def _sign(x):
    if x > 0.:
        return 1.
    elif x < 0.:
        return -1.
    return x * 0.


@njit(cache=True, inline='always')
def _clamped_sqrt(x):
    if x > 0.:
        return np.sqrt(x)
    return 0. if x == x else x


##############################################################################
# Single-state helpers. A state is an 8-tuple in the grid's component order, 
# kept in registers rather than an array
##############################################################################

@njit(cache=True, inline='always')
def _prim_to_cons8(rho, vx, vy, vz, pressure, bx, by, bz, gamma, permeability):
    v2 = vx*vx + vy*vy + vz*vz
    b2 = bx*bx + by*by + bz*bz
    return (rho, rho*vx, rho*vy, rho*vz, pressure/(gamma-1) + .5*(rho*v2) + .5*b2/permeability, bx, by, bz)


@njit(cache=True, inline='always')
def _cons_to_prim8(rho, mx, my, mz, energy, bx, by, bz, gamma, permeability):
    vx = _guarded_divide(mx, rho)
    vy = _guarded_divide(my, rho)
    vz = _guarded_divide(mz, rho)
    v2 = vx*vx + vy*vy + vz*vz
    b2 = bx*bx + by*by + bz*bz
    return (rho, vx, vy, vz, (gamma-1) * (energy - .5*(rho*v2) - .5*b2/permeability), bx, by, bz)


# Ideal MHD flux of one primitive state along axis
@njit(cache=True, inline='always')
def _flux8(rho, vx, vy, vz, pressure, bx, by, bz, gamma, permeability, axis):
    if axis == 0:
        un, ut, us, bn, bt, bs = vx, vy, vz, bx, by, bz
    elif axis == 1:
        un, ut, us, bn, bt, bs = vy, vz, vx, by, bz, bx
    else:
        un, ut, us, bn, bt, bs = vz, vx, vy, bz, bx, by

    v2 = vx*vx + vy*vy + vz*vz
    b2 = bx*bx + by*by + bz*bz
    vdotb = vx*bx + vy*by + vz*bz

    mass = rho*un
    normal = rho*un**2 + pressure + .5*b2 - (bn**2)/permeability
    transverse = rho*un*ut - (bn*bt)/permeability
    secondary = rho*un*us - (bn*bs)/permeability
    energy = un*(.5*rho*v2 + (gamma*pressure)/(gamma-1) + b2) - (bn*vdotb)/permeability
    induction_t = bt*un - bn*ut
    induction_s = bs*un - bn*us

    if axis == 0:
        return (mass, normal, transverse, secondary, energy, 0., induction_t, induction_s)
    elif axis == 1:
        return (mass, secondary, normal, transverse, energy, induction_s, 0., induction_t)
    return (mass, transverse, secondary, normal, energy, induction_t, induction_s, 0.)


# _flux8 of a primitive state given as an 8-tuple
@njit(cache=True, inline='always')
def _state_flux(w, gamma, permeability, axis):
    return _flux8(w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], gamma, permeability, axis)


# Reorder a state so the axis components lead: (rho, n, t, s, p|E, Bn, Bt, Bs)
@njit(cache=True, inline='always')
def _rotate8(q, axis):
    if axis == 0:
        return q
    elif axis == 1:
        return (q[0], q[2], q[3], q[1], q[4], q[6], q[7], q[5])
    return (q[0], q[3], q[1], q[2], q[4], q[7], q[5], q[6])


# Inverse of _rotate8
@njit(cache=True, inline='always')
def _unrotate8(q, axis):
    if axis == 0:
        return q
    elif axis == 1:
        return (q[0], q[3], q[1], q[2], q[4], q[7], q[5], q[6])
    return (q[0], q[2], q[3], q[1], q[4], q[6], q[7], q[5])


# The 8 components of a at (l, i, r) as a tuple
@njit(cache=True, inline='always')
def _load8(a, l, i, r):
    return (a[l, i, r, 0], a[l, i, r, 1], a[l, i, r, 2], a[l, i, r, 3],
            a[l, i, r, 4], a[l, i, r, 5], a[l, i, r, 6], a[l, i, r, 7])


# Write the 8-tuple q into a at (l, i, r)
@njit(cache=True, inline='always')
def _store8(a, l, i, r, q):
    a[l, i, r, 0] = q[0]
    a[l, i, r, 1] = q[1]
    a[l, i, r, 2] = q[2]
    a[l, i, r, 3] = q[3]
    a[l, i, r, 4] = q[4]
    a[l, i, r, 5] = q[5]
    a[l, i, r, 6] = q[6]
    a[l, i, r, 7] = q[7]


##############################################################################
# Variable conversion, physical flux and Roe average. Pointwise kernels over the grid flattened to (ncells, nvars)
##############################################################################

@njit(parallel=True, cache=True)
def _prim_to_cons(grid, out, gamma, permeability):
    ncells = grid.shape[0]
    for c in prange(ncells):
        q = _prim_to_cons8(grid[c, 0], grid[c, 1], grid[c, 2], grid[c, 3],
                           grid[c, 4], grid[c, 5], grid[c, 6], grid[c, 7], gamma, permeability)
        for v in range(8):
            out[c, v] = q[v]


@njit(parallel=True, cache=True)
def _cons_to_prim(grid, out, gamma, permeability):
    ncells = grid.shape[0]
    for c in prange(ncells):
        w = _cons_to_prim8(grid[c, 0], grid[c, 1], grid[c, 2], grid[c, 3],
                           grid[c, 4], grid[c, 5], grid[c, 6], grid[c, 7], gamma, permeability)
        for v in range(8):
            out[c, v] = w[v]


# Pointwise primitive <-> conservative conversion; variable_form is the form of the input
def point_convert(variable_form, grid, gamma, permeability, out=None):
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


# Ideal MHD flux along axis in every cell
@njit(parallel=True, cache=True)
def _flux(grid, out, gamma, permeability, axis):
    ncells = grid.shape[0]
    for c in prange(ncells):
        f = _flux8(grid[c, 0], grid[c, 1], grid[c, 2], grid[c, 3],
                   grid[c, 4], grid[c, 5], grid[c, 6], grid[c, 7], gamma, permeability, axis)
        for v in range(8):
            out[c, v] = f[v]


# Ideal MHD flux along axis, written into out (allocated if not given)
def flux(grid, gamma, permeability, axis, out=None):
    grid, view = _flat(grid)
    if out is None:
        out = np.empty_like(grid)
    _, out_view = _flat(out)
    _flux(view, out_view, gamma, permeability, int(axis))
    return out


# Roe-averaged state of each (plus, minus) pair
@njit(parallel=True, cache=True)
def _roe_average(plus, minus, out):
    ncells = plus.shape[0]
    for c in prange(ncells):
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


# Roe-averaged state between the plus and minus states
def roe_average(plus, minus, out=None):
    plus, plus_view = _flat(plus)
    minus, minus_view = _flat(minus)
    if out is None:
        out = np.empty_like(plus)
    _, out_view = _flat(out)
    _roe_average(plus_view, minus_view, out_view)
    return out


##############################################################################
# Wavespeed bounds. Stored as [normal velocity uN, dominant wave speed c] in the last dimension: 
# every signal speed lies within uN -/+ c, so the largest in magnitude is |uN| + c
##############################################################################

# Sound speed or fast magnetosonic speed; bn is the normal field, b2 = |B|^2
@njit(cache=True, inline='always')
def _dominant_speed(rho, pressure, bn, b2, gamma, permeability, magnetic):
    cs2 = _guarded_divide(gamma * pressure, rho)
    sound_speed = np.sqrt(cs2) if cs2 > 0. else 0.
    if not magnetic:
        return sound_speed

    rho_mu = rho * permeability
    root_rho_mu = np.sqrt(rho_mu) if rho_mu > 0. else 0.
    b_norm = np.sqrt(b2) if b2 > 0. else 0.
    alfven = _guarded_divide(b_norm, root_rho_mu)
    alfven_x = _guarded_divide(bn, root_rho_mu)

    total = sound_speed*sound_speed + alfven*alfven
    discriminant = total*total - (2 * sound_speed * alfven_x)**2
    root = np.sqrt(discriminant) if discriminant > 0. else 0.
    fast = .5 * (total + root)
    return np.sqrt(fast) if fast > 0. else 0.


# Normal velocity and dominant wave speed along axis in every cell
@njit(parallel=True, cache=True)
def _wavespeed_bounds(grid, out, gamma, permeability, axis, magnetic):
    ncells = grid.shape[0]
    for c in prange(ncells):
        out[c, 0] = grid[c, 1+axis]
        bn = b2 = 0.
        if magnetic:
            bx, by, bz = grid[c, 5], grid[c, 6], grid[c, 7]
            b2 = bx*bx + by*by + bz*bz
            bn = grid[c, 5+axis]
        out[c, 1] = _dominant_speed(grid[c, 0], grid[c, 4], bn, b2, gamma, permeability, magnetic)


# (..., 2) array of [normal velocity, dominant wave speed] for each cell
def wavespeed_bounds(grid, gamma, permeability, axis, magnetic):
    grid, view = _flat(grid)
    out = np.empty(grid.shape[:-1] + (2,), dtype=grid.dtype)
    _wavespeed_bounds(view, out.reshape(-1, 2), gamma, permeability, axis, magnetic)
    return out


# Wavespeed bounds of the Roe or arithmetic average of the two states at each interface
# out has one extra face at each end, resolved through the boundary condition
@njit(parallel=True, cache=True)
def _interface_wavespeed_bounds(plus, minus, out, gamma, permeability, axis, magnetic, bci, roe):
    left, n, right, _ = plus.shape
    m = n + 2
    for t in prange(left * m):
        l = t//m
        i = t - l*m
        src = _face_index(i - 1, n, bci)
        for r in range(right):
            if roe:
                rp = plus[l, src, r, 0]
                rm = minus[l, src, r, 0]
                root_plus = np.sqrt(rp) if rp > 0. else 0.
                root_minus = np.sqrt(rm) if rm > 0. else 0.
                total_root = root_minus + root_plus

                rho = root_minus * root_plus
                pressure = _guarded_divide(
                    root_plus*plus[l, src, r, 4] + root_minus*minus[l, src, r, 4], total_root
                )
                normal = _guarded_divide(
                    plus[l, src, r, 1+axis]*root_plus + minus[l, src, r, 1+axis]*root_minus,
                    total_root,
                )
            else:
                rho = .5 * (plus[l, src, r, 0] + minus[l, src, r, 0])
                pressure = .5 * (plus[l, src, r, 4] + minus[l, src, r, 4])
                normal = .5 * (plus[l, src, r, 1+axis] + minus[l, src, r, 1+axis])

            bn = b2 = 0.
            if magnetic:
                for v in range(5, 8):
                    if roe:
                        b = _guarded_divide(
                            plus[l, src, r, v]*root_minus + minus[l, src, r, v]*root_plus, total_root
                        )
                    else:
                        b = .5 * (plus[l, src, r, v] + minus[l, src, r, v])
                    b2 += b*b
                    if v == 5+axis:
                        bn = b

            out[l, i, r, 0] = normal
            out[l, i, r, 1] = _dominant_speed(rho, pressure, bn, b2, gamma, permeability, magnetic)


# Wavespeed bounds of the averaged interface states, with one ghost face at each end along axis
def interface_wavespeed_bounds(plus, minus, gamma, permeability, axis, magnetic, bci, roe=True):
    nvars = plus.shape[-1]
    shape = list(plus.shape)
    shape[axis] += 2
    shape[-1] = 2
    out = np.empty(shape, dtype=np.float64)
    _interface_wavespeed_bounds(
        _axis_view(plus, axis, nvars), _axis_view(minus, axis, nvars), _axis_view(out, axis, 2),
        gamma, permeability, int(axis), magnetic, bci, roe,
    )
    return out


# Wave speed bounds of the arithmetic mean of neighbouring cells; out[k] is the face between
# cells k-2 and k-1, resolved through the boundary condition
@njit(parallel=True, cache=True)
def _cell_wavespeed_bounds(grid, out, gamma, permeability, axis, magnetic, bci):
    left, n, right, _ = grid.shape
    m = n + 3
    for t in prange(left * m):
        l = t//m
        k = t - l*m
        below = _stencil_index(k - 2, n, bci)
        above = _stencil_index(k - 1, n, bci)
        for r in range(right):
            rho = .5 * (grid[l, below, r, 0] + grid[l, above, r, 0])
            pressure = .5 * (grid[l, below, r, 4] + grid[l, above, r, 4])
            bn = b2 = 0.
            if magnetic:
                for v in range(5, 8):
                    b = .5 * (grid[l, below, r, v] + grid[l, above, r, v])
                    b2 += b*b
                    if v == 5+axis:
                        bn = b
            out[l, k, r, 0] = .5 * (grid[l, below, r, 1+axis] + grid[l, above, r, 1+axis])
            out[l, k, r, 1] = _dominant_speed(rho, pressure, bn, b2, gamma, permeability, magnetic)


# Wave speed bounds at every face of the grid along axis, plus one ghost face at each end
def cell_wavespeed_bounds(grid, gamma, permeability, axis, magnetic, bci):
    nvars = grid.shape[-1]
    shape = list(grid.shape)
    shape[axis] += 3
    shape[-1] = 2
    out = np.empty(shape, dtype=np.float64)
    _cell_wavespeed_bounds(
        _axis_view(grid, axis, nvars), _axis_view(out, axis, 2),
        gamma, permeability, int(axis), magnetic, bci,
    )
    return out


##############################################################################
# Laplacians. Scaled second-order differences, used by the corrections 
# between point values and cell averages
##############################################################################

# out = inv_ds2 * ((g[i+1] - g[i]) - (g[i] - g[i-1])) along the middle axis
@njit(parallel=True, cache=True)
def _scaled_laplacian_into(g, out, inv_ds2, bci):
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
                previous = _ghost(g, l, r, n, -1, bci) if i == 0 else g[l, below, r]
                following = _ghost(g, l, r, n, n, bci) if i == n-1 else g[l, above, r]
                centre = g[l, i, r]
                out[l, i, r] = inv_ds2 * ((following - centre) - (centre - previous))


# inv_ds2 * laplacian(grid) along axis, written into out (allocated if not given)
def scaled_laplacian(grid, axis, inv_ds2, bci, out=None):
    grid, view = _as_axis_view(grid, axis)
    if out is None:
        out = np.empty(grid.shape, dtype=grid.dtype)
    elif not out.flags.c_contiguous:
        raise ValueError("out must be C-contiguous to be written in place")
    _, out_view = _as_axis_view(out, axis)
    _scaled_laplacian_into(view, out_view, inv_ds2, bci)
    return out


# base += scale * (inv_ds2 * laplacian(g)) along the middle axis
@njit(parallel=True, cache=True)
def _add_scaled_laplacian(base, g, scale, inv_ds2, bci):
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
                previous = _ghost(g, l, r, n, -1, bci) if i == 0 else g[l, below, r]
                following = _ghost(g, l, r, n, n, bci) if i == n-1 else g[l, above, r]
                centre = g[l, i, r]
                base[l, i, r] += scale * (inv_ds2 * ((following - centre) - (centre - previous)))


# base += scale * (inv_ds2 * laplacian(grid)) along axis, in place
def add_scaled_laplacian(base, grid, axis, scale, inv_ds2, bci):
    if not base.flags.c_contiguous:
        raise ValueError("base must be C-contiguous to be updated in place")
    grid, view = _as_axis_view(grid, axis)
    _, base_view = _as_axis_view(base, axis)
    _add_scaled_laplacian(base_view, view, scale, inv_ds2, bci)
    return base


# base += scales[t] * (invs[t] * laplacian(g, order[t])) for each t < count, accumulated in
# that order in a single pass over the (n0, n1, n2, nv) grid
@njit(parallel=True, cache=True)
def _add_laplacians_4d(base, g, order, scales, invs, count, bci):
    n0, n1, n2, nv = g.shape
    for i in prange(n0):
        for j in range(n1):
            for k in range(n2):
                for t in range(count):
                    axis = order[t]
                    scale = scales[t]
                    inv = invs[t]

                    if axis == 0:
                        below, above = _stencil_index(i-1, n0, bci), _stencil_index(i+1, n0, bci)
                        for v in range(nv):
                            centre = g[i, j, k, v]
                            base[i, j, k, v] += scale * (inv * ((g[above, j, k, v] - centre) - (centre - g[below, j, k, v])))
                    elif axis == 1:
                        below, above = _stencil_index(j-1, n1, bci), _stencil_index(j+1, n1, bci)
                        for v in range(nv):
                            centre = g[i, j, k, v]
                            base[i, j, k, v] += scale * (inv * ((g[i, above, k, v] - centre) - (centre - g[i, below, k, v])))
                    else:
                        below, above = _stencil_index(k-1, n2, bci), _stencil_index(k+1, n2, bci)
                        for v in range(nv):
                            centre = g[i, j, k, v]
                            base[i, j, k, v] += scale * (inv * ((g[i, j, above, v] - centre) - (centre - g[i, j, below, v])))


# base += scales[a] * (invs[a] * laplacian(grid, axes[a])) for each a, in one pass; 
# axes, scales and invs are parallel sequences. Returns base
def add_laplacians(base, grid, axes, scales, invs, bci, ndim_spatial):
    if not base.flags.c_contiguous:
        raise ValueError("base must be C-contiguous to be updated in place")
    grid, grid_view = _as_canonical_4d(grid, ndim_spatial)
    _, base_view = _as_canonical_4d(base, ndim_spatial)
    order = np.asarray(axes, dtype=np.int64)
    _add_laplacians_4d(
        base_view, grid_view, order,
        np.asarray(scales, dtype=np.float64), np.asarray(invs, dtype=np.float64),
        len(order), bci,
    )
    return base


##############################################################################
# Spatial reconstruction
#
# Cell averages to the values at the left (wL) and right (wR) face of each cell along axis;
# neighbours outside the domain are resolved through _stencil_index
##############################################################################

# --- PLM --------------------------------------------------------------------

# Piecewise linear reconstruction with the minmod-limited slope [Derigs et al., 2017]
@njit(parallel=True, cache=True)
def _plm_reconstruct(g, wl, wr, bci):
    left, n, right = g.shape
    ntiles = (right + _TILE - 1)//_TILE
    for tile in prange(left * ntiles):
        l = tile//ntiles
        r0 = (tile % ntiles) * _TILE
        r1 = min(r0 + _TILE, right)
        for i in range(n):
            im1 = _stencil_index(i-1, n, bci)
            ip1 = _stencil_index(i+1, n, bci)
            for r in range(r0, r1):
                z = g[l, i, r]
                a = z - g[l, im1, r]
                b = g[l, ip1, r] - z

                # minmod [eq. 4.17]
                if a*b > 0.:
                    limited = a if abs(a) < abs(b) else b
                else:
                    limited = 0.

                gradient = .5 * limited
                wl[l, i, r] = z - gradient
                wr[l, i, r] = z + gradient


# Return (wL, wR), the minmod-limited linear reconstruction at both faces of each cell
def plm_reconstruct(grid, axis, bci):
    _, view = _as_axis_view(grid, axis)
    wl, wr = np.empty(grid.shape), np.empty(grid.shape)
    _plm_reconstruct(view, _as_axis_view(wl, axis)[1], _as_axis_view(wr, axis)[1], bci)
    return wl, wr


# --- PPM --------------------------------------------------------------------

# 4th-order value at face j+1/2 of cell j [Felker & Stone, 2018, eq. 10]; 
# staggered (face-centred) components take the cell value
@njit(cache=True, inline='always')
def _ppm_interface(g, l, j, r, n, bci, staggered):
    if staggered:
        return g[l, j, r]
    return (
        7/12 * (g[l, j, r] + g[l, _stencil_index(j+1, n, bci), r])
        - 1/12 * (g[l, _stencil_index(j-1, n, bci), r] + g[l, _stencil_index(j+2, n, bci), r])
    )


# Centred second difference d2w_C at cell j
@njit(cache=True, inline='always')
def _ppm_curvature(g, l, j, r, n, bci):
    return g[l, _stencil_index(j-1, n, bci), r] - 2*g[l, j, r] + g[l, _stencil_index(j+1, n, bci), r]


# Third difference d2w_C(j+1) - d2w_C(j) at cell j [McCorquodale & Colella, 2011, eq. 23]
@njit(cache=True, inline='always')
def _ppm_third(g, l, j, r, n, bci):
    return _ppm_curvature(g, l, _stencil_index(j+1, n, bci), r, n, bci) - _ppm_curvature(g, l, j, r, n, bci)


# PPM face values with the extrapolant limiter of McCorquodale & Colella (2011): cells without
# an extremum only get the overshoot correction, the others the curvature limiter
@njit(parallel=True, cache=True)
def _ppm_reconstruct(g, wl, wr, bci, nvars, staggered_mask):
    left, n, right = g.shape
    ntiles = (right + _TILE - 1)//_TILE
    C = 5/4
    for tile in prange(left * ntiles):
        l = tile//ntiles
        r0 = (tile % ntiles) * _TILE
        r1 = min(r0 + _TILE, right)
        for i in range(n):
            im1 = _stencil_index(i-1, n, bci)
            ip1 = _stencil_index(i+1, n, bci)
            im2 = _stencil_index(i-2, n, bci)
            ip2 = _stencil_index(i+2, n, bci)
            for r in range(r0, r1):
                staggered = (staggered_mask >> (r % nvars)) & 1 == 1
                z = g[l, i, r]
                left_of_centre = _ppm_interface(g, l, im1, r, n, bci, staggered)
                right_of_centre = _ppm_interface(g, l, i, r, n, bci, staggered)

                dw_minus = z - left_of_centre
                dw_plus = right_of_centre - z
                m1, p1 = g[l, im1, r], g[l, ip1, r]
                m2, p2 = g[l, im2, r], g[l, ip2, r]

                # No extremum [eq. 24-25]: overshoot correction only
                if not ((dw_minus*dw_plus <= 0) or ((z - m2)*(p2 - z) <= 0)):
                    left_value, right_value = left_of_centre, right_of_centre
                    if abs(dw_minus) >= 2*abs(dw_plus):
                        left_value = z - 2*dw_plus
                    if abs(dw_plus) >= 2*abs(dw_minus):
                        right_value = z + 2*dw_minus
                    wl[l, i, r] = left_value
                    wr[l, i, r] = right_value
                    continue

                d2w = 6 * (left_of_centre - 2*z + right_of_centre)
                d2w_C = m1 - 2*z + p1
                d2w_C_minus = _ppm_curvature(g, l, im1, r, n, bci)
                d2w_C_plus = _ppm_curvature(g, l, ip1, r, n, bci)

                # Limited curvature [eq. 26]
                limited_curvature = _sign(d2w_C) * _npmin(
                    _npmin(abs(d2w), C*abs(d2w_C)),
                    _npmin(C*abs(d2w_C_plus), C*abs(d2w_C_minus)),
                )

                # Limiter sensitive to roundoff [eq. 27]
                rho_limiter = 0.
                scale = _npmax(abs(z), _npmax(_npmax(abs(m1), abs(p1)), _npmax(abs(m2), abs(p2))))
                if abs(d2w) > 1e-12 * scale:
                    rho_limiter = _guarded_divide(limited_curvature, d2w)

                d3w = _ppm_third(g, l, i, r, n, bci)
                d3w_m1 = _ppm_third(g, l, im1, r, n, bci)
                d3w_m2 = _ppm_third(g, l, im2, r, n, bci)
                d3w_p2 = _ppm_third(g, l, ip2, r, n, bci)
                d3w_min = _npmin(_npmin(d3w_m1, d3w), _npmin(d3w_m2, d3w_p2))
                d3w_max = _npmax(_npmax(d3w_m1, d3w), _npmax(d3w_m2, d3w_p2))

                # [eq. 28]
                roundoff = (rho_limiter < (1-1e-12)) or (.1*_npmax(abs(d3w_max), abs(d3w_min)) <= (d3w_max-d3w_min))

                left_value, right_value = left_of_centre, right_of_centre
                if roundoff:
                    # [eq. 29-30]
                    if dw_minus*dw_plus < 0:
                        left_value = z - rho_limiter*dw_minus
                        right_value = z + rho_limiter*dw_plus
                    # [eq. 31-32]
                    if abs(dw_minus) >= 2*abs(dw_plus):
                        left_value = z - 2*(1-rho_limiter)*dw_plus - rho_limiter*dw_minus
                    if abs(dw_plus) >= 2*abs(dw_minus):
                        right_value = z + 2*(1-rho_limiter)*dw_minus + rho_limiter*dw_plus
                wl[l, i, r] = left_value
                wr[l, i, r] = right_value


# Return (wL, wR) from PPM with the McCorquodale & Colella (2011) limiter; the components in
# staggered (the face-centred field components) take the cell value as interface value
def ppm_reconstruct(grid, axis, bci, staggered=()):
    _, view = _as_axis_view(grid, axis)
    wl, wr = np.empty(grid.shape), np.empty(grid.shape)
    mask = 0
    for component in staggered:
        mask |= 1 << int(component)
    _ppm_reconstruct(view, _as_axis_view(wl, axis)[1], _as_axis_view(wr, axis)[1], bci, grid.shape[-1], mask)
    return wl, wr


# --- WENO and WENO-Z --------------------------------------------------------

# 3rd-order WENO [Jiang & Shu, 1996]
@njit(parallel=True, cache=True)
def _weno3_reconstruct(g, wl, wr, eps, bci):
    left, n, right = g.shape
    ntiles = (right + _TILE - 1)//_TILE
    for tile in prange(left * ntiles):
        l = tile//ntiles
        r0 = (tile % ntiles) * _TILE
        r1 = min(r0 + _TILE, right)
        for i in range(n):
            im1 = _stencil_index(i-1, n, bci)
            ip1 = _stencil_index(i+1, n, bci)
            for r in range(r0, r1):
                m1 = g[l, im1, r]
                z = g[l, i, r]
                p1 = g[l, ip1, r]

                d0, d1 = ((z - m1)**2 + eps)**2, ((p1 - z)**2 + eps)**2
                a0, a1 = (1/3)/d0, (2/3)/d1
                ia0, ia1 = (2/3)/d0, (1/3)/d1

                total = a0 + a1
                inv_total = ia0 + ia1
                wr[l, i, r] = .5 * (
                    _guarded_divide(a0, total) * (3*z - m1)
                    + _guarded_divide(a1, total) * (z + p1)
                )
                wl[l, i, r] = .5 * (
                    _guarded_divide(ia0, inv_total) * (3*z - p1)
                    + _guarded_divide(ia1, inv_total) * (z + m1)
                )


# 5th-order WENO [Jiang & Shu, 1996], or WENO-Z [Borges et al., 2008]
@njit(parallel=True, cache=True)
def _weno5_reconstruct(g, wl, wr, eps, bci, wenoz, power):
    left, n, right = g.shape
    ntiles = (right + _TILE - 1)//_TILE
    for tile in prange(left * ntiles):
        l = tile//ntiles
        r0 = (tile % ntiles) * _TILE
        r1 = min(r0 + _TILE, right)
        for i in range(n):
            im2 = _stencil_index(i-2, n, bci)
            im1 = _stencil_index(i-1, n, bci)
            ip1 = _stencil_index(i+1, n, bci)
            ip2 = _stencil_index(i+2, n, bci)
            for r in range(r0, r1):
                m2 = g[l, im2, r]
                m1 = g[l, im1, r]
                z = g[l, i, r]
                p1 = g[l, ip1, r]
                p2 = g[l, ip2, r]

                # Smoothness indicators
                b0 = 13/12 * (m2 - 2*m1 + z)**2 + 1/4 * (m2 - 4*m1 + 3*z)**2
                b1 = 13/12 * (m1 - 2*z + p1)**2 + 1/4 * (m1 - p1)**2
                b2 = 13/12 * (z - 2*p1 + p2)**2 + 1/4 * (3*z - 4*p1 + p2)**2

                if wenoz:
                    tau = abs(b0 - b2)
                    f0 = 1 + (tau/(b0 + eps))**power
                    f1 = 1 + (tau/(b1 + eps))**power
                    f2 = 1 + (tau/(b2 + eps))**power
                    a0, a1, a2 = 1/10 * f0, 3/5 * f1, 3/10 * f2
                    ia0, ia1, ia2 = 3/10 * f0, 3/5 * f1, 1/10 * f2
                else:
                    d0, d1, d2 = (b0 + eps)**2, (b1 + eps)**2, (b2 + eps)**2
                    a0, a1, a2 = (1/10)/d0, (3/5)/d1, (3/10)/d2
                    ia0, ia1, ia2 = (3/10)/d0, (3/5)/d1, (1/10)/d2

                # Non-linear weights
                total = a0 + a1 + a2
                inv_total = ia0 + ia1 + ia2
                o0, o1, o2 = _guarded_divide(a0, total), _guarded_divide(a1, total), _guarded_divide(a2, total)
                io0, io1, io2 = _guarded_divide(ia0, inv_total), _guarded_divide(ia1, inv_total), _guarded_divide(ia2, inv_total)

                wr[l, i, r] = 1/6 * (
                    o0 * (2*m2 - 7*m1 + 11*z)
                    + o1 * (-m1 + 5*z + 2*p1)
                    + o2 * (2*z + 5*p1 - p2)
                )
                wl[l, i, r] = 1/6 * (
                    io0 * (2*z + 5*m1 - m2)
                    + io1 * (-p1 + 5*z + 2*m1)
                    + io2 * (2*p2 - 7*p1 + 11*z)
                )


# 7th-order WENO [Balsara & Shu, 2000]
@njit(parallel=True, cache=True)
def _weno7_reconstruct(g, wl, wr, eps, bci):
    left, n, right = g.shape
    ntiles = (right + _TILE - 1)//_TILE
    for tile in prange(left * ntiles):
        l = tile//ntiles
        r0 = (tile % ntiles) * _TILE
        r1 = min(r0 + _TILE, right)
        for i in range(n):
            im3 = _stencil_index(i-3, n, bci)
            im2 = _stencil_index(i-2, n, bci)
            im1 = _stencil_index(i-1, n, bci)
            ip1 = _stencil_index(i+1, n, bci)
            ip2 = _stencil_index(i+2, n, bci)
            ip3 = _stencil_index(i+3, n, bci)
            for r in range(r0, r1):
                m3 = g[l, im3, r]
                m2 = g[l, im2, r]
                m1 = g[l, im1, r]
                z = g[l, i, r]
                p1 = g[l, ip1, r]
                p2 = g[l, ip2, r]
                p3 = g[l, ip3, r]

                # Smoothness indicators
                b0 = (
                    m3 * (547*m3 - 3882*m2 + 4642*m1 - 1854*z)
                    + m2 * (7043*m2 - 17246*m1 + 7042*z)
                    + m1 * (11003*m1 - 9402*z)
                    + z * (2107*z)
                )
                b1 = (
                    m2 * (267*m2 - 1642*m1 + 1602*z - 494*p1)
                    + m1 * (2843*m1 - 5966*z + 1922*p1)
                    + z * (3443*z - 2522*p1)
                    + p1 * (547*p1)
                )
                b2 = (
                    m1 * (547*m1 - 2522*z + 1922*p1 - 494*p2)
                    + z * (3443*z - 5966*p1 + 1602*p2)
                    + p1 * (2843*p1 - 1642*p2)
                    + p2 * (267*p2)
                )
                b3 = (
                    z * (2107*z - 9402*p1 + 7042*p2 - 1854*p3)
                    + p1 * (11003*p1 - 17246*p2 + 4642*p3)
                    + p2 * (7043*p2 - 3882*p3)
                    + p3 * (547*p3)
                )

                d0, d1, d2, d3 = (b0 + eps)**2, (b1 + eps)**2, (b2 + eps)**2, (b3 + eps)**2
                a0, a1, a2, a3 = (1/35)/d0, (12/35)/d1, (18/35)/d2, (4/35)/d3
                ia0, ia1, ia2, ia3 = (4/35)/d0, (18/35)/d1, (12/35)/d2, (1/35)/d3
                total = a0 + a1 + a2 + a3
                inv_total = ia0 + ia1 + ia2 + ia3

                wr[l, i, r] = 1/12 * (
                    _guarded_divide(a0, total) * (-3*m3 + 13*m2 - 23*m1 + 25*z)
                    + _guarded_divide(a1, total) * (m2 - 5*m1 + 13*z + 3*p1)
                    + _guarded_divide(a2, total) * (-m1 + 7*z + 7*p1 - p2)
                    + _guarded_divide(a3, total) * (3*z + 13*p1 - 5*p2 + p3)
                )
                wl[l, i, r] = 1/12 * (
                    _guarded_divide(ia0, inv_total) * (3*z + 13*m1 - 5*m2 + m3)
                    + _guarded_divide(ia1, inv_total) * (-p1 + 7*z + 7*m1 - m2)
                    + _guarded_divide(ia2, inv_total) * (p2 - 5*p1 + 13*z + 3*m1)
                    + _guarded_divide(ia3, inv_total) * (-3*p3 + 13*p2 - 23*p1 + 25*z)
                )


# Return (wL, wR) from WENO of order 3, 5 or 7, or from 5th-order WENO-Z
def weno_reconstruct(grid, axis, eps, bci, order=5, wenoz=False, power=1):
    grid, view = _as_axis_view(grid, axis)
    wl, wr = np.empty(grid.shape, dtype=grid.dtype), np.empty(grid.shape, dtype=grid.dtype)
    _, wl_view = _as_axis_view(wl, axis)
    _, wr_view = _as_axis_view(wr, axis)
    if wenoz or order == 5:
        _weno5_reconstruct(view, wl_view, wr_view, eps, bci, wenoz, float(power))
    elif order == 3:
        _weno3_reconstruct(view, wl_view, wr_view, eps, bci)
    elif order == 7:
        _weno7_reconstruct(view, wl_view, wr_view, eps, bci)
    else:
        raise ValueError(f"no WENO kernel of order {order}")
    return wl, wr


# --- CWENO ------------------------------------------------------------------

# CWENO [Levy et al., 1999, eq. 3.11-3.14], or CWENO-Z with the tau of Cravero et al. (2019).
# With gauss, the values at the Gauss-Legendre points x = ∓1/(2√3) instead, with 
# components positive0/positive1 (-1 for none) kept positive there [Zhang & Shu, 2010]
@njit(parallel=True, cache=True)
def _cweno_reconstruct(g, wl, wr, eps, power, bci, wenoz, gauss, nvars, positive0, positive1):
    left, n, right = g.shape
    ntiles = (right + _TILE - 1)//_TILE

    # Linear weights dC_k [tbl. 3.1], exact for cubics at the faces, or at the Gauss points
    if gauss:
        dc0, dc1, dc2 = 7/36, 11/18, 7/36
    else:
        dc0, dc1, dc2 = 1/6, 2/3, 1/6
    gauss_divisor = 2*np.sqrt(3.)

    for tile in prange(left * ntiles):
        l = tile//ntiles
        r0 = (tile % ntiles) * _TILE
        r1 = min(r0 + _TILE, right)
        for i in range(n):
            im2 = _stencil_index(i-2, n, bci)
            im1 = _stencil_index(i-1, n, bci)
            ip1 = _stencil_index(i+1, n, bci)
            ip2 = _stencil_index(i+2, n, bci)
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

                if gauss:
                    offset = .5 * (
                        o0 * (m2 - 4*m1 + 3*z)
                        + o1 * (p1 - m1)
                        + o2 * (-3*z + 4*p1 - p2)
                    ) / gauss_divisor

                    # Component index; the variables are the innermost dimension
                    component = r % nvars
                    if component == positive0 or component == positive1:
                        offset *= min(1., _guarded_divide(max(z - eps, 0.), abs(offset)))

                    wl[l, i, r] = z - offset
                    wr[l, i, r] = z + offset
                    continue

                # dC_k is symmetric, so both faces use the same linear weights
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


# Return (wL, wR) from CWENO along axis; with gauss, the lower and upper Gauss-point values,
# with up to two `positive` components kept positive
def cweno_reconstruct(grid, axis, eps, bci, power=2, wenoz=False, gauss=False, positive=()):
    grid, view = _as_axis_view(grid, axis)
    wl, wr = np.empty(grid.shape, dtype=grid.dtype), np.empty(grid.shape, dtype=grid.dtype)
    _, wl_view = _as_axis_view(wl, axis)
    _, wr_view = _as_axis_view(wr, axis)
    positive0, positive1 = (tuple(positive) + (-1, -1))[:2]
    _cweno_reconstruct(view, wl_view, wr_view, eps, power, bci, wenoz, gauss, grid.shape[-1], positive0, positive1)
    return wl, wr


# --- TENO -------------------------------------------------------------------

# Local smoothness measure of the adaptive cutoff [Fu, 2021]
@njit(cache=True, inline='always')
def _teno_eta(s0, s1, s2, eps):
    return (abs(2 * (s2-s1) * (s1-s0)) + eps)/((s2-s1)**2 + (s1-s0)**2 + eps)


# 5-point TENO [Fu et al., 2016], or TENO-AA with the adaptive cutoff [Fu, 2021].
# Candidate stencils are ordered i, i+1, i-1 with linear weights g0..g2, and 
# a stencil whose normalised scale separation falls below the cutoff gets no weight
@njit(parallel=True, cache=True)
def _teno5_reconstruct(g, wl, wr, eps, bci, g0, g1, g2, q, C_T, adaptive):
    left, n, right = g.shape
    ntiles = (right + _TILE - 1)//_TILE
    C_r, alpha_1, alpha_2 = .265, 14, 6.4
    for tile in prange(left * ntiles):
        l = tile//ntiles
        r0 = (tile % ntiles) * _TILE
        r1 = min(r0 + _TILE, right)
        for i in range(n):
            im2 = _stencil_index(i-2, n, bci)
            im1 = _stencil_index(i-1, n, bci)
            ip1 = _stencil_index(i+1, n, bci)
            ip2 = _stencil_index(i+2, n, bci)
            ip3 = _stencil_index(i+3, n, bci)
            for r in range(r0, r1):
                m2 = g[l, im2, r]
                m1 = g[l, im1, r]
                z = g[l, i, r]
                p1 = g[l, ip1, r]
                p2 = g[l, ip2, r]

                # Smoothness indicators of the stencils i, i+1, i-1
                b0 = 13/12 * (m1 - 2*z + p1)**2 + 1/4 * (m1 - p1)**2
                b1 = 13/12 * (z - 2*p1 + p2)**2 + 1/4 * (3*z - 4*p1 + p2)**2
                b2 = 13/12 * (m2 - 2*m1 + z)**2 + 1/4 * (m2 - 4*m1 + 3*z)**2

                # Scale separation, and the cutoff it is tested against
                cutoff = C_T
                if adaptive:
                    gamma0 = (1/(b0+eps))**7
                    gamma1 = (1/(b1+eps))**7
                    gamma2 = (1/(b2+eps))**7

                    # Minimum of eta over the four three-point stencils (i-2, i-1, i) ... (i+1, i+2, i+3)
                    p3 = g[l, ip3, r]
                    eta = _npmin(
                        _npmin(_teno_eta(m2, m1, z, eps), _teno_eta(m1, z, p1, eps)),
                        _npmin(_teno_eta(z, p1, p2, eps), _teno_eta(p1, p2, p3, eps)),
                    )
                    m = 1 - _npmin(1., eta/C_r)
                    g_m = (1 + 4*m) * (1 - m)**4
                    beta = alpha_1 - alpha_2*(1-g_m)
                    cutoff = 10.**-np.floor(beta)
                else:
                    separation = abs(b2 - b1)
                    gamma0 = (1 + separation/(b0+eps))**q
                    gamma1 = (1 + separation/(b1+eps))**q
                    gamma2 = (1 + separation/(b2+eps))**q

                # Sharp cutoff: a stencil below it gets no weight at all
                gamma_sum = gamma0 + gamma1 + gamma2
                keep0 = not (_guarded_divide(gamma0, gamma_sum) < cutoff)
                keep1 = not (_guarded_divide(gamma1, gamma_sum) < cutoff)
                keep2 = not (_guarded_divide(gamma2, gamma_sum) < cutoff)
                d0, d1, d2 = (g0 if keep0 else 0.), (g1 if keep1 else 0.), (g2 if keep2 else 0.)
                id0, id1, id2 = (g2 if keep0 else 0.), (g1 if keep1 else 0.), (g0 if keep2 else 0.)

                # Non-linear weights
                total = d0 + d1 + d2
                inv_total = id0 + id1 + id2
                wr[l, i, r] = 1/6 * (
                    _guarded_divide(d0, total) * (-m1 + 5*z + 2*p1)
                    + _guarded_divide(d1, total) * (2*z + 5*p1 - p2)
                    + _guarded_divide(d2, total) * (2*m2 - 7*m1 + 11*z)
                )
                wl[l, i, r] = 1/6 * (
                    _guarded_divide(id0, inv_total) * (-p1 + 5*z + 2*m1)
                    + _guarded_divide(id1, inv_total) * (2*p2 - 7*p1 + 11*z)
                    + _guarded_divide(id2, inv_total) * (2*z + 5*m1 - m2)
                )


# Return (wL, wR) from 5-point TENO with linear weights (g0, g1, g2), or from TENO-AA
def teno_reconstruct(grid, axis, bci, weights, q=6, C_T=1e-7, adaptive=False, eps=1e-40):
    grid, view = _as_axis_view(grid, axis)
    wl, wr = np.empty(grid.shape, dtype=grid.dtype), np.empty(grid.shape, dtype=grid.dtype)
    _, wl_view = _as_axis_view(wl, axis)
    _, wr_view = _as_axis_view(wr, axis)
    g0, g1, g2 = (float(w) for w in weights)
    _teno5_reconstruct(view, wl_view, wr_view, eps, bci, g0, g1, g2, float(q), float(C_T), bool(adaptive))
    return wl, wr


##############################################################################
# Riemann solvers
#
# Interface fluxes from the (plus, minus) primitive and conservative states, 
# with the minus (w-) and plus (w+) states given as the left (L) and right (R)
# states in literature
##############################################################################

# Allocate the flux and return it with the (left, n, right, 8) views of the plus, minus,
# conservative plus and minus states and the flux
def _riemann_views(prim, cons, axis):
    (plus, minus), (cons_plus, cons_minus) = prim, cons
    nvars = plus.shape[-1]
    if nvars != 8:
        raise ValueError(f"the Riemann-solver kernels expect 8 components, got {nvars}")
    out = np.empty(plus.shape, dtype=np.float64)
    views = tuple(_axis_view(arr, axis, nvars) for arr in (plus, minus, cons_plus, cons_minus, out))
    return out, views


# Local maximum signal speed at interface i: the mean of the pairwise maxima of |uN| + c over its neighbouring faces
@njit(cache=True, inline='always')
def _local_max_speed(ws, l, i, r):
    below = abs(ws[l, i, r, 0]) + ws[l, i, r, 1]
    centre = abs(ws[l, i+1, r, 0]) + ws[l, i+1, r, 1]
    above = abs(ws[l, i+2, r, 0]) + ws[l, i+2, r, 1]
    plus = above if above > centre else centre
    minus = centre if centre > below else below
    return .5 * (plus + minus)


# --- Lax-Friedrich ----------------------------------------------------------

# Local Lax-Friedrich (Rusanov) flux .5*(F- + F+) - (q+ - q-)*(.5*a), for the local maximum signal speed a
@njit(parallel=True, cache=True, error_model='numpy')
def _lax_friedrich(pp, pm, cp, cm, ws, out, gamma, permeability, axis):
    left, n, right, _ = out.shape
    for t in prange(left * n):
        l = t//n
        i = t - l*n
        for r in range(right):
            half_speed = .5 * _local_max_speed(ws, l, i, r)
            flux_plus = _state_flux(_load8(pp, l, i, r), gamma, permeability, axis)
            flux_minus = _state_flux(_load8(pm, l, i, r), gamma, permeability, axis)
            for v in range(8):
                out[l, i, r, v] = (
                    (flux_minus[v] + flux_plus[v]) * .5
                    - (cp[l, i, r, v] - cm[l, i, r, v]) * half_speed
                )


# Local Lax-Friedrich flux from the (plus, minus) primitive and conservative interface states
def lax_friedrich(prim, cons, wavespeeds, gamma, permeability, axis):
    out, (pp, pm, cp, cm, ov) = _riemann_views(prim, cons, axis)
    _lax_friedrich(pp, pm, cp, cm, _axis_view(wavespeeds, axis, 2), ov, gamma, permeability, int(axis))
    return out


# --- Lax-Wendroff and GFORCE ------------------------------------------------

# U^LW = (U- + U+)/2 - (tau/2)(F+ - F-) [Mignone & Del Zanna, 2021, eq. 48]
@njit(cache=True, inline='always')
def _richtmyer_state(q_minus, q_plus, flux_minus, flux_plus, half_tau):
    return (
        .5*(q_minus[0] + q_plus[0]) - half_tau*(flux_plus[0] - flux_minus[0]),
        .5*(q_minus[1] + q_plus[1]) - half_tau*(flux_plus[1] - flux_minus[1]),
        .5*(q_minus[2] + q_plus[2]) - half_tau*(flux_plus[2] - flux_minus[2]),
        .5*(q_minus[3] + q_plus[3]) - half_tau*(flux_plus[3] - flux_minus[3]),
        .5*(q_minus[4] + q_plus[4]) - half_tau*(flux_plus[4] - flux_minus[4]),
        .5*(q_minus[5] + q_plus[5]) - half_tau*(flux_plus[5] - flux_minus[5]),
        .5*(q_minus[6] + q_plus[6]) - half_tau*(flux_plus[6] - flux_minus[6]),
        .5*(q_minus[7] + q_plus[7]) - half_tau*(flux_plus[7] - flux_minus[7]),
    )


# GFORCE flux weight*F(U^LW) + (1-weight)*F^LF [Toro & Titarev, 2006; Mignone & Del Zanna, 2021], 
# with tau = 1/a for the local maximum signal speed a; weight = 1 gives the Lax-Wendroff flux.
# U^LW is the plain average where a = 0, and a U^LW with rho <= 0 or p < 0 takes F^LF instead
@njit(parallel=True, cache=True, error_model='numpy')
def _gforce(pp, pm, cp, cm, ws, out, gamma, permeability, axis, weight):
    left, n, right, _ = out.shape
    for t in prange(left * n):
        l = t//n
        i = t - l*n
        for r in range(right):
            speed = _local_max_speed(ws, l, i, r)
            half_speed = .5 * speed
            half_tau = .5/speed if speed > 0. else 0.

            flux_plus = _state_flux(_load8(pp, l, i, r), gamma, permeability, axis)
            flux_minus = _state_flux(_load8(pm, l, i, r), gamma, permeability, axis)
            q_plus, q_minus = _load8(cp, l, i, r), _load8(cm, l, i, r)

            u = _richtmyer_state(q_minus, q_plus, flux_minus, flux_plus, half_tau)
            w = _cons_to_prim8(u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], gamma, permeability)
            hyperbolic = w[0] > 0. and w[4] >= 0.
            flux_lw = _state_flux(w, gamma, permeability, axis)

            for v in range(8):
                flux_lf = (flux_minus[v] + flux_plus[v]) * .5 - (q_plus[v] - q_minus[v]) * half_speed
                if not hyperbolic:
                    out[l, i, r, v] = flux_lf
                elif weight == 1.:
                    out[l, i, r, v] = flux_lw[v]
                else:
                    out[l, i, r, v] = weight*flux_lw[v] + (1-weight)*flux_lf


# GFORCE flux; weight = 1 is the Lax-Wendroff (Richtmyer) flux alone
def gforce(prim, cons, wavespeeds, gamma, permeability, axis, weight):
    out, (pp, pm, cp, cm, ov) = _riemann_views(prim, cons, axis)
    _gforce(pp, pm, cp, cm, _axis_view(wavespeeds, axis, 2), ov, gamma, permeability, int(axis), float(weight))
    return out


# --- HLL --------------------------------------------------------------------

# HLL flux [Harten, Lax & van Leer, 1983] with the signal speeds uN -/+ c of the interface wave speed bounds
@njit(parallel=True, cache=True, error_model='numpy')
def _hll(pp, pm, cp, cm, ws, out, gamma, permeability, axis):
    left, n, right, _ = out.shape
    for t in prange(left * n):
        l = t//n
        i = t - l*n
        for r in range(right):
            normal, dominant = ws[l, i+1, r, 0], ws[l, i+1, r, 1]
            alpha_plus = _npmax(0., normal + dominant)
            alpha_minus = -_npmin(0., normal - dominant)
            product, total = alpha_plus*alpha_minus, alpha_plus + alpha_minus

            flux_plus = _state_flux(_load8(pp, l, i, r), gamma, permeability, axis)
            flux_minus = _state_flux(_load8(pm, l, i, r), gamma, permeability, axis)
            for v in range(8):
                out[l, i, r, v] = _guarded_divide(
                    (flux_minus[v]*alpha_plus + flux_plus[v]*alpha_minus)
                    - (product*(cp[l, i, r, v] - cm[l, i, r, v])),
                    total,
                )


# HLL flux, the signal speeds bounded by the given interface wave speeds
def hll(prim, cons, wavespeeds, gamma, permeability, axis):
    out, (pp, pm, cp, cm, ov) = _riemann_views(prim, cons, axis)
    _hll(pp, pm, cp, cm, _axis_view(wavespeeds, axis, 2), ov, gamma, permeability, int(axis))
    return out


# --- HLLC -------------------------------------------------------------------

# HLLC flux [Toro et al., 1994]; low_mach scales the outer wave speeds used to select the state [Fleischmann et al., 2020]
@njit(parallel=True, cache=True, error_model='numpy')
def _hllc(pp, pm, cp, cm, out, gamma, permeability, axis, low_mach):
    left, n, right, _ = out.shape
    Ma_limit = .1
    for t in prange(left * n):
        l = t//n
        i = t - l*n
        for r in range(right):
            rhoL, uL, pL = pm[l, i, r, 0], pm[l, i, r, 1+axis], pm[l, i, r, 4]
            rhoR, uR, pR = pp[l, i, r, 0], pp[l, i, r, 1+axis], pp[l, i, r, 4]

            # Signal speeds [Davis, 1988]
            cs_L = _clamped_sqrt(_guarded_divide(gamma*pL, rhoL))
            cs_R = _clamped_sqrt(_guarded_divide(gamma*pR, rhoR))
            sL = _npmin(0., _npmin(uL, uR) - _npmax(cs_L, cs_R))
            sR = _npmax(0., _npmax(uL, uR) + _npmax(cs_L, cs_R))
            sM = _guarded_divide(pL - pR + rhoR*uR*(sR-uR) - rhoL*uL*(sL-uL), rhoR*(sR-uR) - rhoL*(sL-uL))

            coeffL, coeffR = _guarded_divide(sL-uL, sL-sM), _guarded_divide(sR-uR, sR-sM)

            # Modification for low Mach shocks; only changes which state is picked
            select_L, select_R = sL, sR
            if low_mach:
                cL = np.sqrt(_guarded_divide(gamma*pL, rhoL))
                cR = np.sqrt(_guarded_divide(gamma*pR, rhoR))
                Ma_local = _npmax(abs(_guarded_divide(uL, cL)), abs(_guarded_divide(uR, cR)))
                phi = np.sin(.5 * np.pi * _npmin(1., Ma_local/Ma_limit))
                select_L, select_R = phi * sL, phi * sR

            if select_R < 0:
                _store8(out, l, i, r, _state_flux(_load8(pp, l, i, r), gamma, permeability, axis))
            elif (sM <= 0) and (0 <= select_R):
                flux_plus = _state_flux(_load8(pp, l, i, r), gamma, permeability, axis)
                energy = coeffR*(sM-uR)*(rhoR*sM + _guarded_divide(pR, sR-uR))
                for v in range(8):
                    q = cp[l, i, r, v]
                    q_star = q * coeffR
                    if v == 1+axis:
                        q_star = rhoR * coeffR * sM
                    elif v == 4:
                        q_star = q_star + energy
                    out[l, i, r, v] = flux_plus[v] + (q_star - q) * sR
            elif (select_L <= 0) and (0 < sM):
                flux_minus = _state_flux(_load8(pm, l, i, r), gamma, permeability, axis)
                energy = coeffL*(sM-uL)*(rhoL*sM + _guarded_divide(pL, sL-uL))
                for v in range(8):
                    q = cm[l, i, r, v]
                    q_star = q * coeffL
                    if v == 1+axis:
                        q_star = rhoL * coeffL * sM
                    elif v == 4:
                        q_star = q_star + energy
                    out[l, i, r, v] = flux_minus[v] + (q_star - q) * sL
            else:
                _store8(out, l, i, r, _state_flux(_load8(pp, l, i, r), gamma, permeability, axis))


def hllc(prim, cons, gamma, permeability, axis, low_mach=False):
    out, (pp, pm, cp, cm, ov) = _riemann_views(prim, cons, axis)
    _hllc(pp, pm, cp, cm, ov, gamma, permeability, int(axis), bool(low_mach))
    return out


# --- HLLD -------------------------------------------------------------------

# Fast magnetosonic speed along the normal (with mu = 1)
@njit(cache=True, inline='always')
def _fast_speed(rho, pressure, bn, bt, bs, gamma):
    sound2 = _guarded_divide(gamma*pressure, rho)
    alfven2 = _guarded_divide(bn*bn + bt*bt + bs*bs, rho)
    total = sound2 + alfven2
    discriminant = total*total - 4.*sound2*_guarded_divide(bn*bn, rho)
    root = np.sqrt(discriminant) if discriminant > 0. else 0.
    fast2 = .5*(total + root)
    return np.sqrt(fast2) if fast2 > 0. else 0.


# HLL flux (sR*F_L - sL*F_R + sL*sR*(U_R - U_L))/(sR - sL), for sL < 0 < sR
@njit(cache=True, inline='always')
def _hll8(flux_L, flux_R, qL, qR, sL, sR):
    inverse = 1./(sR - sL)
    return (
        (sR*flux_L[0] - sL*flux_R[0] + sL*sR*(qR[0] - qL[0]))*inverse,
        (sR*flux_L[1] - sL*flux_R[1] + sL*sR*(qR[1] - qL[1]))*inverse,
        (sR*flux_L[2] - sL*flux_R[2] + sL*sR*(qR[2] - qL[2]))*inverse,
        (sR*flux_L[3] - sL*flux_R[3] + sL*sR*(qR[3] - qL[3]))*inverse,
        (sR*flux_L[4] - sL*flux_R[4] + sL*sR*(qR[4] - qL[4]))*inverse,
        (sR*flux_L[5] - sL*flux_R[5] + sL*sR*(qR[5] - qL[5]))*inverse,
        (sR*flux_L[6] - sL*flux_R[6] + sL*sR*(qR[6] - qL[6]))*inverse,
        (sR*flux_L[7] - sL*flux_R[7] + sL*sR*(qR[7] - qL[7]))*inverse,
    )


# Flux f + s*(q_star - q) beyond a wave of speed s that takes the state q to q_star (Rankine-Hugoniot condition)
@njit(cache=True, inline='always')
def _across_wave(f, s, q_star, q):
    return (
        f[0] + s*(q_star[0] - q[0]), f[1] + s*(q_star[1] - q[1]),
        f[2] + s*(q_star[2] - q[2]), f[3] + s*(q_star[3] - q[3]),
        f[4] + s*(q_star[4] - q[4]), f[5] + s*(q_star[5] - q[5]),
        f[6] + s*(q_star[6] - q[6]), f[7] + s*(q_star[7] - q[7]),
    )


# HLLD flux [Miyoshi & Kusano, 2005]
@njit(parallel=True, cache=True, error_model='numpy')
def _hlld(pp, pm, cp, cm, out, gamma, permeability, axis):
    left, n, right, _ = out.shape
    for t in prange(left * n):
        l = t//n
        i = t - l*n
        for r in range(right):
            wL, wR = _load8(pm, l, i, r), _load8(pp, l, i, r)
            qL, qR = _load8(cm, l, i, r), _load8(cp, l, i, r)
            flux_L = _state_flux(wL, gamma, permeability, axis)
            flux_R = _state_flux(wR, gamma, permeability, axis)

            dL, uL, vL, zL, pL, bnL, byL, bzL = _rotate8(wL, axis)
            dR, uR, vR, zR, pR, bnR, byR, bzR = _rotate8(wR, axis)
            eL, eR = qL[4], qR[4]

            bx = .5*(bnL + bnR)
            bx2 = bx*bx
            ptL = pL + .5*(bx2 + byL*byL + bzL*bzL)
            ptR = pR + .5*(bx2 + byR*byR + bzR*bzR)

            # Outer signal speeds [eq. 67]
            cf = _npmax(_fast_speed(dL, pL, bx, byL, bzL, gamma), _fast_speed(dR, pR, bx, byR, bzR, gamma))
            sL = _npmin(uL, uR) - cf
            sR = _npmax(uL, uR) + cf

            if sL >= 0.:
                _store8(out, l, i, r, flux_L)
                continue
            if sR <= 0.:
                _store8(out, l, i, r, flux_R)
                continue

            # Contact speed [eq. 38]; HLL where the star states are undefined (e.g. p <= 0)
            sdL, sdR = sL - uL, sR - uR
            denominator = sdR*dR - sdL*dL
            sM = (sdR*dR*uR - sdL*dL*uL - ptR + ptL)/denominator if denominator > 0. else 0.
            if not (sdL < 0. and sdR > 0. and denominator > 0. and sL < sM < sR and dL > 0. and dR > 0.):
                _store8(out, l, i, r, _hll8(flux_L, flux_R, qL, qR, sL, sR))
                continue

            # Star densities and star total pressure [eq. 43, 41]
            sdmL, sdmR = sL - sM, sR - sM
            dLs = dL*sdL/sdmL
            dRs = dR*sdR/sdmR
            pts = (sdR*dR*ptL - sdL*dL*ptR + dL*dR*sdR*sdL*(uR - uL))/denominator

            # Left star state [eq. 44-48]; transverse components unchanged where the denominator vanishes
            denom = dL*sdL*sdmL - bx2
            if abs(denom) < 1e-8*pts:
                vLs, zLs, byLs, bzLs = vL, zL, byL, bzL
            else:
                factor = _guarded_divide(bx*(sM - uL), denom)
                vLs, zLs = vL - byL*factor, zL - bzL*factor
                factor = _guarded_divide(dL*sdL*sdL - bx2, denom)
                byLs, bzLs = byL*factor, bzL*factor
            vbL = uL*bx + vL*byL + zL*bzL
            vbLs = sM*bx + vLs*byLs + zLs*bzLs
            eLs = _guarded_divide(sdL*eL - ptL*uL + pts*sM + bx*(vbL - vbLs), sdmL)

            # Right star state
            denom = dR*sdR*sdmR - bx2
            if abs(denom) < 1e-8*pts:
                vRs, zRs, byRs, bzRs = vR, zR, byR, bzR
            else:
                factor = _guarded_divide(bx*(sM - uR), denom)
                vRs, zRs = vR - byR*factor, zR - bzR*factor
                factor = _guarded_divide(dR*sdR*sdR - bx2, denom)
                byRs, bzRs = byR*factor, bzR*factor
            vbR = uR*bx + vR*byR + zR*bzR
            vbRs = sM*bx + vRs*byRs + zRs*bzRs
            eRs = _guarded_divide(sdR*eR - ptR*uR + pts*sM + bx*(vbR - vbRs), sdmR)

            # Alfven speeds [eq. 51]
            rootL, rootR = _clamped_sqrt(dLs), _clamped_sqrt(dRs)
            sLs = sM - (_guarded_divide(abs(bx), rootL) if bx != 0. else 0.)
            sRs = sM + (_guarded_divide(abs(bx), rootR) if bx != 0. else 0.)

            qLs = _unrotate8((dLs, dLs*sM, dLs*vLs, dLs*zLs, eLs, bx, byLs, bzLs), axis)
            qRs = _unrotate8((dRs, dRs*sM, dRs*vRs, dRs*zRs, eRs, bx, byRs, bzRs), axis)

            if sLs >= 0.:
                _store8(out, l, i, r, _across_wave(flux_L, sL, qLs, qL))
                continue
            if sRs <= 0. and sM < 0.:
                _store8(out, l, i, r, _across_wave(flux_R, sR, qRs, qR))
                continue

            # Double-star state between the Alfven waves [eq. 59-63]; only reached when Bx != 0
            sign = 1. if bx > 0. else -1.
            inverse = _guarded_divide(1., rootL + rootR)
            vss = inverse*(rootL*vLs + rootR*vRs + sign*(byRs - byLs))
            zss = inverse*(rootL*zLs + rootR*zRs + sign*(bzRs - bzLs))
            byss = inverse*(rootL*byRs + rootR*byLs + sign*rootL*rootR*(vRs - vLs))
            bzss = inverse*(rootL*bzRs + rootR*bzLs + sign*rootL*rootR*(zRs - zLs))
            vbss = sM*bx + vss*byss + zss*bzss

            if sM >= 0.:
                eLss = eLs - rootL*sign*(vbLs - vbss)
                qLss = _unrotate8((dLs, dLs*sM, dLs*vss, dLs*zss, eLss, bx, byss, bzss), axis)
                _store8(out, l, i, r, _across_wave(_across_wave(flux_L, sL, qLs, qL), sLs, qLss, qLs))
            else:
                eRss = eRs + rootR*sign*(vbRs - vbss)
                qRss = _unrotate8((dRs, dRs*sM, dRs*vss, dRs*zss, eRss, bx, byss, bzss), axis)
                _store8(out, l, i, r, _across_wave(_across_wave(flux_R, sR, qRs, qR), sRs, qRss, qRs))


def hlld(prim, cons, gamma, permeability, axis):
    out, (pp, pm, cp, cm, ov) = _riemann_views(prim, cons, axis)
    _hlld(pp, pm, cp, cm, ov, gamma, permeability, int(axis))
    return out