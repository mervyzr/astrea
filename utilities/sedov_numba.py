import numpy as np
import numba as nb
from time import perf_counter

# ============================================================
# 2D Sedov Blast Test
# ============================================================
# Variables:
#   0 : density
#   1 : vx
#   2 : vy
#   3 : vz
#   4 : pressure
#   5 : bx
#   6 : by
#   7 : bz
#
# Finite-volume:
#   - Piecewise constant reconstruction
#   - Local Lax-Friedrichs (Rusanov) flux
#   - Forward Euler time stepping
#
# Two implementations:
#   1. Pure NumPy
#   2. Optimized Numba
#
# Grid: 128 x 128
# ============================================================

GAMMA = 1.4

IDN = 0
IVX = 1
IVY = 2
IVZ = 3
IPR = 4
IBX = 5
IBY = 6
IBZ = 7

NVAR = 8

NX = 128
NY = 128
NZ = 128

CFL = 0.5
TFINAL = 2

# ------------------------------------------------------------
# Primitive <-> Conserved
# ------------------------------------------------------------

def prim_to_cons_numpy(P):
    U = np.zeros_like(P)

    rho = P[..., IDN]
    vx  = P[..., IVX]
    vy  = P[..., IVY]
    vz  = P[..., IVZ]
    p   = P[..., IPR]
    bx  = P[..., IBX]
    by  = P[..., IBY]
    bz  = P[..., IBZ]

    kinetic = 0.5 * rho * (vx**2 + vy**2 + vz**2)
    magnetic = 0.5 * (bx**2 + by**2 + bz**2)

    E = p / (GAMMA - 1.0) + kinetic + magnetic

    U[..., IDN] = rho
    U[..., IVX] = rho * vx
    U[..., IVY] = rho * vy
    U[..., IVZ] = rho * vz
    U[..., IPR] = E
    U[..., IBX] = bx
    U[..., IBY] = by
    U[..., IBZ] = bz

    return U


@nb.njit(parallel=True, fastmath=True)
def prim_to_cons_numba(P, U):
    nx, ny, nz, _ = P.shape

    for i in nb.prange(nx):
        for j in nb.prange(ny):
            for k in nb.prange(nz):

                rho = P[i, j, k, IDN]
                vx  = P[i, j, k, IVX]
                vy  = P[i, j, k, IVY]
                vz  = P[i, j, k, IVZ]
                p   = P[i, j, k, IPR]
                bx  = P[i, j, k, IBX]
                by  = P[i, j, k, IBY]
                bz  = P[i, j, k, IBZ]

                kinetic = 0.5 * rho * (vx*vx + vy*vy + vz*vz)
                magnetic = 0.5 * (bx*bx + by*by + bz*bz)

                E = p / (GAMMA - 1.0) + kinetic + magnetic

                U[i, j, k, IDN] = rho
                U[i, j, k, IVX] = rho * vx
                U[i, j, k, IVY] = rho * vy
                U[i, j, k, IVZ] = rho * vz
                U[i, j, k, IPR] = E
                U[i, j, k, IBX] = bx
                U[i, j, k, IBY] = by
                U[i, j, k, IBZ] = bz


# ------------------------------------------------------------
# Fluxes
# ------------------------------------------------------------

def flux_x_numpy(P):

    F = np.zeros_like(P)

    rho = P[..., IDN]
    vx  = P[..., IVX]
    vy  = P[..., IVY]
    vz  = P[..., IVZ]
    p   = P[..., IPR]
    bx  = P[..., IBX]
    by  = P[..., IBY]
    bz  = P[..., IBZ]

    magnetic = 0.5 * (bx**2 + by**2 + bz**2)
    ptot = p + magnetic

    E = (
        p / (GAMMA - 1.0)
        + 0.5 * rho * (vx**2 + vy**2 + vz**2)
        + magnetic
    )

    vdotb = vx*bx + vy*by + vz*bz

    F[..., IDN] = rho * vx
    F[..., IVX] = rho*vx*vx + ptot - bx*bx
    F[..., IVY] = rho*vx*vy - bx*by
    F[..., IVZ] = rho*vx*vz - bx*bz
    F[..., IPR] = (E + ptot)*vx - bx*vdotb
    F[..., IBX] = 0.0
    F[..., IBY] = vy*bx - vx*by
    F[..., IBZ] = vz*bx - vx*bz

    return F


def flux_y_numpy(P):

    F = np.zeros_like(P)

    rho = P[..., IDN]
    vx  = P[..., IVX]
    vy  = P[..., IVY]
    vz  = P[..., IVZ]
    p   = P[..., IPR]
    bx  = P[..., IBX]
    by  = P[..., IBY]
    bz  = P[..., IBZ]

    magnetic = 0.5 * (bx**2 + by**2 + bz**2)
    ptot = p + magnetic

    E = (
        p / (GAMMA - 1.0)
        + 0.5 * rho * (vx**2 + vy**2 + vz**2)
        + magnetic
    )

    vdotb = vx*bx + vy*by + vz*bz

    F[..., IDN] = rho * vy
    F[..., IVX] = rho*vy*vx - by*bx
    F[..., IVY] = rho*vy*vy + ptot - by*by
    F[..., IVZ] = rho*vy*vz - by*bz
    F[..., IPR] = (E + ptot)*vy - by*vdotb
    F[..., IBX] = vx*by - vy*bx
    F[..., IBY] = 0.0
    F[..., IBZ] = vz*by - vy*bz

    return F


def flux_z_numpy(P):

    F = np.zeros_like(P)

    rho = P[..., IDN]
    vx  = P[..., IVX]
    vy  = P[..., IVY]
    vz  = P[..., IVZ]
    p   = P[..., IPR]
    bx  = P[..., IBX]
    by  = P[..., IBY]
    bz  = P[..., IBZ]

    magnetic = 0.5 * (bx**2 + by**2 + bz**2)
    ptot = p + magnetic

    E = (
        p / (GAMMA - 1.0)
        + 0.5 * rho * (vx**2 + vy**2 + vz**2)
        + magnetic
    )

    vdotb = vx*bx + vy*by + vz*bz

    F[..., IDN] = rho * vz
    F[..., IVX] = rho*vz*vx - bz*bx
    F[..., IVY] = rho*vz*vy - bz*by
    F[..., IVZ] = rho*vz*vz + ptot - bz*bz
    F[..., IPR] = (E + ptot)*vz - bz*vdotb
    F[..., IBX] = vx*bz - vz*bx
    F[..., IBY] = vy*bz - vz*by
    F[..., IBZ] = 0.0

    return F


# ------------------------------------------------------------
# CFL timestep
# ------------------------------------------------------------

def compute_dt_numpy(P, dx):

    rho = P[..., IDN]
    vx = P[..., IVX]
    vy = P[..., IVY]
    vz = P[..., IVZ]
    p = P[..., IPR]

    cs = np.sqrt(GAMMA * p / rho)

    smax = np.max(np.abs(vx) + cs)
    smax = max(smax, np.max(np.abs(vy) + cs))
    smax = max(smax, np.max(np.abs(vz) + cs))

    return CFL * dx / smax


# ------------------------------------------------------------
# NumPy Rusanov update
# ------------------------------------------------------------

def step_numpy(P, dx, dt):

    U = prim_to_cons_numpy(P)

    Fx = flux_x_numpy(P)
    Fy = flux_y_numpy(P)
    Fz = flux_z_numpy(P)

    Unew = U.copy()

    # x-direction
    for i in range(1, NX-1):

        UL = U[i-1]
        UR = U[i]

        PL = P[i-1]
        PR = P[i]

        csL = np.sqrt(GAMMA * PL[:, IPR] / PL[:, IDN])
        csR = np.sqrt(GAMMA * PR[:, IPR] / PR[:, IDN])

        a = np.maximum(
            np.abs(PL[:, IVX]) + csL,
            np.abs(PR[:, IVX]) + csR
        )

        Fh = 0.5*(Fx[i-1] + Fx[i]) - 0.5*a[:, None]*(UR - UL)

        Unew[i-1] -= dt/dx * Fh
        Unew[i]   += dt/dx * Fh

    # y-direction
    for j in range(1, NY-1):

        UL = U[:, j-1]
        UR = U[:, j]

        PL = P[:, j-1]
        PR = P[:, j]

        csL = np.sqrt(GAMMA * PL[:, IPR] / PL[:, IDN])
        csR = np.sqrt(GAMMA * PR[:, IPR] / PR[:, IDN])

        a = np.maximum(
            np.abs(PL[:, IVY]) + csL,
            np.abs(PR[:, IVY]) + csR
        )

        Fh = 0.5*(Fy[:, j-1] + Fy[:, j]) - 0.5*a[:, None]*(UR - UL)

        Unew[:, j-1] -= dt/dx * Fh
        Unew[:, j]   += dt/dx * Fh

    # z-direction
    for k in range(1, NZ-1):

        UL = U[:, k-1]
        UR = U[:, k]

        PL = P[:, k-1]
        PR = P[:, k]

        csL = np.sqrt(GAMMA * PL[:, IPR] / PL[:, IDN])
        csR = np.sqrt(GAMMA * PR[:, IPR] / PR[:, IDN])

        a = np.maximum(
            np.abs(PL[:, IVZ]) + csL,
            np.abs(PR[:, IVZ]) + csR
        )

        Fh = 0.5*(Fz[:, k-1] + Fz[:, k]) - 0.5*a[:, None]*(UR - UL)

        Unew[:, k-1] -= dt/dx * Fh
        Unew[:, k]   += dt/dx * Fh

    return cons_to_prim_numpy(Unew)


# ------------------------------------------------------------
# Conserved -> Primitive
# ------------------------------------------------------------

def cons_to_prim_numpy(U):

    P = np.zeros_like(U)

    rho = U[..., IDN]

    vx = U[..., IVX] / rho
    vy = U[..., IVY] / rho
    vz = U[..., IVZ] / rho

    bx = U[..., IBX]
    by = U[..., IBY]
    bz = U[..., IBZ]

    kinetic = 0.5 * rho * (vx**2 + vy**2 + vz**2)
    magnetic = 0.5 * (bx**2 + by**2 + bz**2)

    E = U[..., IPR]

    p = (GAMMA - 1.0) * (E - kinetic - magnetic)

    P[..., IDN] = rho
    P[..., IVX] = vx
    P[..., IVY] = vy
    P[..., IVZ] = vz
    P[..., IPR] = p
    P[..., IBX] = bx
    P[..., IBY] = by
    P[..., IBZ] = bz

    return P


# ------------------------------------------------------------
# NUMBA IMPLEMENTATION
# ------------------------------------------------------------

@nb.njit(parallel=True, fastmath=True)
def cons_to_prim_numba(U, P):

    nx, ny, nz, _ = U.shape

    for i in nb.prange(nx):
        for j in nb.prange(ny):
            for k in nb.prange(nz):

                rho = U[i, j, k, IDN]

                vx = U[i, j, k, IVX] / rho
                vy = U[i, j, k, IVY] / rho
                vz = U[i, j, k, IVZ] / rho

                bx = U[i, j, k, IBX]
                by = U[i, j, k, IBY]
                bz = U[i, j, k, IBZ]

                kinetic = 0.5 * rho * (vx*vx + vy*vy + vz*vz)
                magnetic = 0.5 * (bx*bx + by*by + bz*bz)

                E = U[i, j, k, IPR]

                p = (GAMMA - 1.0) * (E - kinetic - magnetic)

                P[i, j, k, IDN] = rho
                P[i, j, k, IVX] = vx
                P[i, j, k, IVY] = vy
                P[i, j, k, IVZ] = vz
                P[i, j, k, IPR] = p
                P[i, j, k, IBX] = bx
                P[i, j, k, IBY] = by
                P[i, j, k, IBZ] = bz


@nb.njit(parallel=True, fastmath=True)
def compute_dt_numba(P, dx):

    nx, ny, nz, _ = P.shape

    smax = 0.0

    for i in nb.prange(nx):
        for j in nb.prange(ny):
            for k in nb.prange(nz):

                rho = P[i, j, k, IDN]
                p   = P[i, j, k, IPR]

                cs = np.sqrt(GAMMA * p / rho)

                sx = abs(P[i, j, k, IVX]) + cs
                sy = abs(P[i, j, k, IVY]) + cs
                sz = abs(P[i, j, k, IVZ]) + cs

                if sx > smax:
                    smax = sx

                if sy > smax:
                    smax = sy

                if sz > smax:
                    smax = sz

    return CFL * dx / smax


@nb.njit(parallel=True, fastmath=True)
def step_numba(P, dx, dt):

    nx, ny, nz, _ = P.shape

    U = np.zeros_like(P)
    Unew = np.zeros_like(P)

    prim_to_cons_numba(P, U)

    Unew[:] = U[:]

    # x-fluxes
    for i in nb.prange(1, nx):
        for j in nb.prange(ny):
            for k in nb.prange(nz):

                rhoL = P[i-1, j, k, IDN]
                rhoR = P[i, j, k, IDN]

                vxL = P[i-1, j, k, IVX]
                vxR = P[i, j, k, IVX]

                pL = P[i-1, j, k, IPR]
                pR = P[i, j, k, IPR]

                csL = np.sqrt(GAMMA * pL / rhoL)
                csR = np.sqrt(GAMMA * pR / rhoR)

                a = max(abs(vxL)+csL, abs(vxR)+csR)

                for n in nb.prange(NVAR):

                    FL = U[i-1, j, k, n] * vxL
                    FR = U[i, j, k, n] * vxR

                    flux = 0.5*(FL + FR) - 0.5*a*(U[i, j, k, n] - U[i-1, j, k, n])

                    Unew[i-1, j, k, n] -= dt/dx * flux
                    Unew[i,   j, k, n] += dt/dx * flux

    # y-fluxes
    for i in nb.prange(nx):
        for j in nb.prange(1, ny):
            for k in nb.prange(nz):

                rhoL = P[i, j-1, k, IDN]
                rhoR = P[i, j, k, IDN]

                vyL = P[i, j-1, k, IVY]
                vyR = P[i, j, k, IVY]

                pL = P[i, j-1, k, IPR]
                pR = P[i, j, k, IPR]

                csL = np.sqrt(GAMMA * pL / rhoL)
                csR = np.sqrt(GAMMA * pR / rhoR)

                a = max(abs(vyL)+csL, abs(vyR)+csR)

                for n in nb.prange(NVAR):

                    FL = U[i, j-1, k, n] * vyL
                    FR = U[i, j, k, n] * vyR

                    flux = 0.5*(FL + FR) - 0.5*a*(U[i, j, k, n] - U[i, j-1, k, n])

                    Unew[i, j-1, k, n] -= dt/dx * flux
                    Unew[i, j,   k, n] += dt/dx * flux

    # z-fluxes
    for i in nb.prange(nx):
        for j in nb.prange(ny):
            for k in nb.prange(1, nz):

                rhoL = P[i, j, k-1, IDN]
                rhoR = P[i, j, k, IDN]

                vzL = P[i, j, k-1, IVZ]
                vzR = P[i, j, k, IVZ]

                pL = P[i, j, k-1, IPR]
                pR = P[i, j, k, IPR]

                csL = np.sqrt(GAMMA * pL / rhoL)
                csR = np.sqrt(GAMMA * pR / rhoR)

                a = max(abs(vzL)+csL, abs(vzR)+csR)

                for n in nb.prange(NVAR):

                    FL = U[i, j, k-1, n] * vzL
                    FR = U[i, j, k, n] * vzR

                    flux = 0.5*(FL + FR) - 0.5*a*(U[i, j, k, n] - U[i, j, k-1, n])

                    Unew[i, j, k-1, n] -= dt/dx * flux
                    Unew[i, j,   k, n] += dt/dx * flux

    Pnew = np.zeros_like(P)

    cons_to_prim_numba(Unew, Pnew)

    return Pnew


# ------------------------------------------------------------
# Initial Condition
# ------------------------------------------------------------

def initialize():

    P = np.zeros((NX, NY, NZ, NVAR))

    P[..., IDN] = 1.0
    P[..., IPR] = 1e-12

    x = np.linspace(-10.0, 10.0, NX)
    y = np.linspace(-10.0, 10.0, NY)
    z = np.linspace(-10.0, 10.0, NZ)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    r2 = (X)**2 + (Y)**2 + (Z)**2

    blast = r2 < 0.5**2

    P[blast, IPR] = 100.0

    return P


# ------------------------------------------------------------
# Driver
# ------------------------------------------------------------

dx = 10.0 / NX

# ---------------- NumPy ----------------

P_numpy = initialize()

t = 0.0

start = perf_counter()

while t < TFINAL:

    dt = compute_dt_numpy(P_numpy, dx)

    if t + dt > TFINAL:
        dt = TFINAL - t

    P_numpy = step_numpy(P_numpy, dx, dt)

    t += dt

numpy_time = perf_counter() - start

print(f"NumPy runtime : {numpy_time:.4f} s")

# ---------------- Numba ----------------

P_numba = initialize()

# warmup compile
dt0 = compute_dt_numba(P_numba, dx)
P_numba = step_numba(P_numba, dx, dt0)

t = 0.0

start = perf_counter()

while t < TFINAL:

    dt = compute_dt_numba(P_numba, dx)

    if t + dt > TFINAL:
        dt = TFINAL - t

    P_numba = step_numba(P_numba, dx, dt)

    t += dt

numba_time = perf_counter() - start

print(f"Numba runtime : {numba_time:.4f} s")

print(f"Speedup       : {numpy_time / numba_time:.2f}x")
