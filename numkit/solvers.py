import numpy as np

from functions import grid as gutils
from functions import math as mfuncs
from functions import numeric
from numkit import kernels

##############################################################################
# Approximate linearised and non-linearised Riemann solvers
##############################################################################

# Intercell numerical fluxes between L and R interfaces based on Riemann solver
def get_Riemann_solver(sim_variables):
    # HLL-type solvers
    if sim_variables.solver_category == "hll":
        if sim_variables.solver.endswith("d"):
            return calculate_HLLD_flux
        elif sim_variables.solver.endswith("c"):
            return calculate_HLLC_flux
        else:
            return calculate_HLL_flux
    # 'Complete Riemann' solvers
    elif sim_variables.solver_category == "complete":
        if sim_variables.solver.startswith("e"):
            return calculate_ES_flux
        else:
            return calculate_DOTS_flux
    # Roe-type/Lax-type solvers
    else:
        if sim_variables.solver.endswith("w"):
            return calculate_LaxWendroff_flux
        elif "g" in sim_variables.solver:
            return calculate_gForce_flux
        else:
            return calculate_LaxFriedrich_flux


# Compute fluxes from the (plus, minus) interface states
def riemann_flux(axis, sim_variables, wavespeeds, prim, cons=None, convert=None):
    Riemann_solver = get_Riemann_solver(sim_variables)

    if cons is None:
        if convert is None:
            convert = lambda state: gutils.variable_convert_intf("primitive", state, sim_variables, axis=axis)
        cons = [convert(state) for state in prim]

    return Riemann_solver(axis, sim_variables, wavespeeds=wavespeeds, prim_interfaces=prim, cons_interfaces=cons)


# Interface fluxes of the higher-order schemes: the solver on the face-averaged states, with the transverse face-centred states combined into the higher-order face-averaged flux
def higher_order_flux(axis, sim_variables, wavespeeds, prim):
    cons = [gutils.variable_convert_intf("primitive", state, sim_variables, axis=axis) for state in prim]

    # Calculate the interface-averaged fluxes
    intf_fluxes_avgd = riemann_flux(axis, sim_variables, wavespeeds, prim, cons=cons)
    if not sim_variables.multidimensional:
        # Orthogonal Laplacian in 1d is zero
        return intf_fluxes_avgd

    # Fluxes are evaluated from the interface-centred states; converting F(<w>) is only 2nd-order accurate
    prim = gutils.approx_face_avg(prim, sim_variables, axis)
    cons = gutils.approx_face_avg(cons, sim_variables, axis)
    intf_fluxes_cntrd = riemann_flux(axis, sim_variables, wavespeeds, prim, cons=cons)

    # Compute the higher-order fluxes
    return gutils.approx_flux_avg(intf_fluxes_cntrd, intf_fluxes_avgd, sim_variables, axis)


# Arguments shared by the kernel-backed solvers
def _kernel_args(sim_variables, kwargs):
    return kwargs["prim_interfaces"], kwargs["cons_interfaces"], sim_variables.gamma, sim_variables.constants.mu_0


# (Local) Lax-Friedrich solver (1st-order; highly diffusive) [Lax & Friedrichs, ?; Mignone & Del Zanna, 2021]
def calculate_LaxFriedrich_flux(axis, sim_variables, **kwargs):
    prim, cons, gamma, permeability = _kernel_args(sim_variables, kwargs)
    return kernels.lax_friedrich(prim, cons, kwargs["wavespeeds"], gamma, permeability, axis)


# Lax-Wendroff (Richtmyer two-step) solver (2nd-order; contains oscillations) [Lax & Wendroff, 1960; Mignone & Del Zanna, 2021, eq. 48]
# F = F(U^LW) with U^LW = (U- + U+)/2 - (tau/2)(F+ - F-), and tau = 1/a for the local maximum signal speed a
def calculate_LaxWendroff_flux(axis, sim_variables, **kwargs):
    prim, cons, gamma, permeability = _kernel_args(sim_variables, kwargs)
    return kernels.gforce(prim, cons, kwargs["wavespeeds"], gamma, permeability, axis, weight=1.)


# GFORCE solver [Toro & Titarev, 2006; Mignone & Del Zanna, 2021, eq. 47-49]
# omega*F^LW + (1-omega)*F^LF with omega = 1/(1+c) for the Courant number c (to keep it monotone)
def calculate_gForce_flux(axis, sim_variables, **kwargs):
    prim, cons, gamma, permeability = _kernel_args(sim_variables, kwargs)
    return kernels.gforce(prim, cons, kwargs["wavespeeds"], gamma, permeability, axis, weight=1/(1+sim_variables.cfl))


# HLL Riemann solver [Harten, Lax & van Leer, 1983]
def calculate_HLL_flux(axis, sim_variables, **kwargs):
    prim, cons, gamma, permeability = _kernel_args(sim_variables, kwargs)
    return kernels.hll(prim, cons, kwargs["wavespeeds"], gamma, permeability, axis)


# HLLC Riemann solver [Fleischmann et al., 2020]
def calculate_HLLC_flux(axis, sim_variables, low_mach=False, **kwargs):
    prim, cons, gamma, permeability = _kernel_args(sim_variables, kwargs)
    return kernels.hllc(prim, cons, gamma, permeability, axis, low_mach=low_mach)


# HLLD Riemann solver [Miyoshi & Kusano, 2005]
def calculate_HLLD_flux(axis, sim_variables, **kwargs):
    prim, cons, gamma, permeability = _kernel_args(sim_variables, kwargs)
    return kernels.hlld(prim, cons, gamma, permeability, axis)


# Osher-Solomon(-Dumbser-Toro) Riemann solver [Dumbser & Toro, 2011]
def calculate_DOTS_flux(axis, sim_variables, **kwargs):
    cons_plus, cons_minus = kwargs["cons_interfaces"]
    flux_plus, flux_minus = [numeric.compute_flux(state, sim_variables, axis=axis) for state in kwargs["prim_interfaces"]]
    roots, weights = sim_variables.roots, sim_variables.weights

    # Define the path integral for the Osher-Solomon dissipation term
    arr_plus, arr_minus = np.repeat(cons_plus[None,:], len(roots), axis=0), np.repeat(cons_minus[None,:], len(roots), axis=0)
    psi = arr_minus + (roots*(arr_plus - arr_minus).T).T

    # Define the left & right eigenvectors
    left_eigenvectors, right_eigenvectors = numeric.compute_eigenvectors(psi, sim_variables, axis)

    # Generate the diagonal matrix of eigenvalues
    eigenvalues = np.zeros_like(right_eigenvectors)

    # Compute wavespeeds
    _, _, cAx, cFF, cSS = numeric.compute_wavespeeds(psi, sim_variables, axis)
    vxs = psi[...,1+axis]

    # Compute the diagonal matrix of eigenvalues
    eigenvalues[...,0,0] = vxs - cFF
    eigenvalues[...,1,1] = vxs - cAx
    eigenvalues[...,2,2] = vxs - cSS
    eigenvalues[...,3,3] = vxs
    eigenvalues[...,4,4] = vxs
    eigenvalues[...,5,5] = vxs + cSS
    eigenvalues[...,6,6] = vxs + cAx
    eigenvalues[...,7,7] = vxs + cFF

    # Compute the absolute value of the Jacobian
    abs_A = right_eigenvectors @ np.abs(eigenvalues) @ left_eigenvectors

    # Compute the Dumbser-Toro Jacobian with the Gauss-Legendre quadrature
    jacobian = np.sum((weights * abs_A.T).T, axis=0)

    # Compute the Osher-Solomon dissipation term
    q_plus = (jacobian @ cons_plus[...,None]).squeeze()
    q_minus = (jacobian @ cons_minus[...,None]).squeeze()

    return .5*(flux_plus+flux_minus) - .5*(q_plus-q_minus)


# Entropy-stable flux calculation based on left and right interpolated primitive variables [Winters & Gassner, 2015; Derigs et al., 2016]
def calculate_ES_flux(axis, sim_variables, **kwargs):
    prim_plus, prim_minus = kwargs["prim_interfaces"]
    abscissa, ordinate, applicate = (axis + np.array(range(3)))%3
    rho, vels, pressure, Bfields = sim_variables.rho, sim_variables.vels, sim_variables.pressure, sim_variables.Bfields
    gamma = sim_variables.gamma

    version = 'hybrid'

    """The convention here uses L & R states, i.e. L state = w-, R state = w+
        |                        w(i-1/2)                    w(i+1/2)                       |
        |-->         i-1         <--|-->          i          <--|-->         i+1         <--|
        |   w_R(i-1)     w_L(i-1)   |   w_R(i)         w_L(i)   |   w_R(i+1)     w_L(i+1)   |
    --> |   w+(i-3/2)   w-(i-1/2)   |   w+(i-1/2)   w-(i+1/2)   |  w+(i+1/2)    w-(i+3/2)   |
    """
    rhoL, vecL, pL, BfieldsL = prim_minus[...,rho], prim_minus[...,vels], prim_minus[...,pressure], prim_minus[...,Bfields]
    rhoR, vecR, pR, BfieldsR = prim_plus[...,rho], prim_plus[...,vels], prim_plus[...,pressure], prim_plus[...,Bfields]

    uL, vL, wL, bxL, byL, bzL = vecL[...,abscissa], vecL[...,ordinate], vecL[...,applicate], BfieldsL[...,abscissa], BfieldsL[...,ordinate], BfieldsL[...,applicate]
    uR, vR, wR, bxR, byR, bzR = vecR[...,abscissa], vecR[...,ordinate], vecR[...,applicate], BfieldsR[...,abscissa], BfieldsR[...,ordinate], BfieldsR[...,applicate]

    amean = lambda L,R: .5 * (L-R)
    lon = lambda L,R: mfuncs.divide(L-R, np.log(L)-np.log(R))  # Stable numerical procedure for computing logarithmic mean [Ismail & Roe, 2009]


    # To construct the entropy-stable flux, 2 components are needed:
    # the entropy-conserving component, and the dissipation term to make the flux entropy-stable

    # Entropy-conserving flux section [Winters & Gassner, 2015]
    ec_flux = np.zeros_like(prim_plus)

    z1L, z1R = np.sqrt(mfuncs.divide(rhoL, pL)), np.sqrt(mfuncs.divide(rhoR, pR))
    z5L, z5R = np.sqrt(rhoL*pL), np.sqrt(rhoR*pR)

    # Compute the averages
    rho_hat = amean(z1L,z1R) * lon(z5L,z5R)
    p1_hat = mfuncs.divide(amean(z5L,z5R),amean(z1L,z1R))
    p2_hat = .5 * ((gamma+1)/gamma * mfuncs.divide(lon(z5L,z5R), lon(z1L,z1R)) + (gamma-1)/gamma * mfuncs.divide(amean(z5L,z5R), amean(z1L,z1R)))
    u1_hat = mfuncs.divide(amean(z1L*uL,z1R*uR), amean(z1L,z1R))
    v1_hat = mfuncs.divide(amean(z1L*vL,z1R*vR), amean(z1L,z1R))
    w1_hat = mfuncs.divide(amean(z1L*wL,z1R*wR), amean(z1L,z1R))
    u2_hat = mfuncs.divide(amean(uL*z1L**2,uR*z1R**2), amean(z1L**2,z1R**2))
    v2_hat = mfuncs.divide(amean(vL*z1L**2,vR*z1R**2), amean(z1L**2,z1R**2))
    w2_hat = mfuncs.divide(amean(wL*z1L**2,wR*z1R**2), amean(z1L**2,z1R**2))
    b1_hat = amean(bxL,bxR)
    b2_hat = amean(byL,byR)
    b3_hat = amean(bzL,bzR)
    b1_dot = amean(bxL**2,bxR**2)
    b2_dot = amean(byL**2,byR**2)
    b3_dot = amean(bzL**2,bzR**2)
    b1b2 = amean(bxL*byL, bxR*byR)
    b1b3 = amean(bxL*bzL, bxR*bzR)

    # Update the entropy-conserving flux vector; suitable for smooth solutions
    ec_flux[...,rho] = rho_hat * u1_hat
    ec_flux[...,1+abscissa] = p1_hat + rho_hat*u1_hat**2 + .5*(b1_dot+b2_dot+b3_dot) - b1_dot
    ec_flux[...,1+ordinate] = rho_hat*u1_hat*v1_hat - b1b2
    ec_flux[...,1+applicate] = rho_hat*u1_hat*w1_hat - b1b3
    ec_flux[...,pressure] = (gamma*u1_hat*p2_hat)/(gamma-1) + .5*rho_hat*u1_hat*(u1_hat**2 + v1_hat**2 + w1_hat**2) + u2_hat*(b2_hat**2 + b3_hat**2) - b1_hat*(v2_hat*b2_hat + w2_hat*b3_hat)
    ec_flux[...,5+ordinate] = u2_hat*b2_hat - v2_hat*b1_hat
    ec_flux[...,5+applicate] = u2_hat*b3_hat - w2_hat*b1_hat


    # Entropy-stable flux with dissipation term section [Derigs et al., 2016]
    # Make the right eigenvectors for each cell in each grid using the averaged primitive variables
    es_right_eigenvectors = numeric.compute_right_eigenvectors(np.array([rho_hat.T, u1_hat.T, v1_hat.T, w1_hat.T, p1_hat.T, b1_hat.T, b2_hat.T, b3_hat.T]).T, sim_variables, axis)

    # Define the jump in the entropy vector
    entropy = np.log(p1_hat * rho_hat**-gamma)
    entropy_vector = np.zeros_like(prim_plus)
    entropy_vector[...,rho] = ((gamma-entropy)/(gamma-1) - mfuncs.divide(rho_hat*mfuncs.norm2(np.array([u1_hat.T, v1_hat.T, w1_hat.T]).T), 2*p1_hat))
    entropy_vector[...,1+abscissa] = mfuncs.divide(rho_hat*u1_hat, p1_hat)
    entropy_vector[...,1+ordinate] = mfuncs.divide(rho_hat*v1_hat, p1_hat)
    entropy_vector[...,1+applicate] = mfuncs.divide(rho_hat*w1_hat, p1_hat)
    entropy_vector[...,pressure] = -mfuncs.divide(rho_hat, p1_hat)
    entropy_vector[...,5+abscissa] = mfuncs.divide(rho_hat*b1_hat, p1_hat)
    entropy_vector[...,5+ordinate] = mfuncs.divide(rho_hat*b2_hat, p1_hat)
    entropy_vector[...,5+applicate] = mfuncs.divide(rho_hat*b3_hat, p1_hat)
    entropy_vector *= -1

    # Define speeds
    sound_speed = np.sqrt(gamma * mfuncs.divide(p1_hat, rho_hat))
    alfven_speed = mfuncs.divide(mfuncs.norm(np.array([b1_hat.T, b2_hat.T, b3_hat.T]).T), np.sqrt(rho_hat))
    alfven_speed_x = mfuncs.divide(b1_hat, np.sqrt(rho_hat))
    fast_magnetosonic_wave = np.sqrt(.5 * (sound_speed**2 + alfven_speed**2 + np.sqrt(((sound_speed**2 + alfven_speed**2)**2) - (4*(sound_speed**2)*(alfven_speed_x**2)))))
    slow_magnetosonic_wave = np.sqrt(.5 * (sound_speed**2 + alfven_speed**2 - np.sqrt(((sound_speed**2 + alfven_speed**2)**2) - (4*(sound_speed**2)*(alfven_speed_x**2)))))

    # Compute the diagonal matrix of eigenvalues for Roe
    if version.lower().startswith(('r','h')):
        roe_eigenvalues = np.zeros_like(es_right_eigenvectors)
        roe_eigenvalues[...,0,0] = u1_hat - fast_magnetosonic_wave
        roe_eigenvalues[...,1,1] = u1_hat - alfven_speed_x
        roe_eigenvalues[...,2,2] = u1_hat - slow_magnetosonic_wave
        roe_eigenvalues[...,3,3] = u1_hat
        roe_eigenvalues[...,4,4] = u1_hat
        roe_eigenvalues[...,5,5] = u1_hat + slow_magnetosonic_wave
        roe_eigenvalues[...,6,6] = u1_hat + alfven_speed_x
        roe_eigenvalues[...,7,7] = u1_hat + fast_magnetosonic_wave
        roe_eigenvalues = np.abs(roe_eigenvalues)

    # Compute the diagonal matrix of eigenvalues for Local Lax-Friedrich
    if version.lower().startswith(('l','h')):
        llf_eigenvalues = np.zeros_like(es_right_eigenvectors)
        i, j = np.diag_indices(llf_eigenvalues.shape[-1])
        max_values = np.maximum.reduce([
            np.abs(u1_hat - fast_magnetosonic_wave),
            np.abs(u1_hat - alfven_speed_x),
            np.abs(u1_hat - slow_magnetosonic_wave),
            np.abs(u1_hat),
            np.abs(u1_hat + slow_magnetosonic_wave),
            np.abs(u1_hat + alfven_speed_x),
            np.abs(u1_hat + fast_magnetosonic_wave)])
        llf_eigenvalues[...,i,j] = max_values[..., None]

    # Compute the hydrid entropy stabilisation diagonal matrix
    if version.lower().startswith('h'):
        Epsilon = np.sqrt(np.abs(mfuncs.divide(pL-pR, pL+pR)))
        hybrid_eigenvalues = Epsilon[...,None,None] * llf_eigenvalues + (1-Epsilon)[...,None,None] * roe_eigenvalues

    # Calculate the dissipation term
    if version.lower().startswith('h'):
        eigenvalues = hybrid_eigenvalues
    elif version.lower().startswith('r'):
        eigenvalues = roe_eigenvalues
    elif version.lower().startswith('l'):
        eigenvalues = llf_eigenvalues
    abs_A = es_right_eigenvectors @ eigenvalues @ np.linalg.pinv(es_right_eigenvectors)
    dissipation = abs_A @ entropy_vector[...,None]

    return ec_flux + .5 * dissipation.reshape(len(entropy_vector), len(entropy_vector[0]))
