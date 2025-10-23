import numpy as np

from functions import constructor, fv

##############################################################################
# Approximate linearised and non-linearised Riemann solvers
##############################################################################

# Intercell numerical fluxes between L and R interfaces based on Riemann solver
def get_Riemann_solver(sim_variables):
    # HLL-type solvers
    if sim_variables.solver_category == "hll":
        if sim_variables.solver.endswith("d"):
            return calculate_HLLD_flux
        else:
            return calculate_HLLC_flux
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


# (Local) Lax-Friedrich solver (1st-order; highly diffusive) [Lax & Friedrichs, ?; Mignone & Del Zanna, 2021]
def calculate_LaxFriedrich_flux(axis, sim_variables, **kwargs):
    cons_plus, cons_minus = kwargs["cons_interfaces"]
    flux_plus, flux_minus = kwargs["flux_interfaces"]
    characteristics = kwargs["characteristics"]

    local_max_eigvals = np.max(np.abs(np.real(characteristics)), axis=-1)  # Get maximum eigenvalues in each (localised) cell
    max_eigvals = np.maximum(fv.slice_(local_max_eigvals, axis, end=-1), fv.slice_(local_max_eigvals, axis, start=1))  # Get the maximum eigenvalue between each consecutive pair of cells
    return .5*(flux_minus+flux_plus) - .5*((cons_plus-cons_minus) * max_eigvals[...,None])


# Lax-Wendroff (Richtmyer) solver (2nd-order, Jacobian method; contains overshoots) [Lax & Wendroff, 1960; Mignone & Del Zanna, 2021]
def calculate_LaxWendroff_flux(axis, sim_variables, **kwargs):
    cons_plus, cons_minus = kwargs["cons_interfaces"]
    flux_plus, flux_minus = kwargs["flux_interfaces"]
    characteristics = kwargs["characteristics"]

    local_max_eigvals = np.max(np.abs(np.real(characteristics)), axis=-1)  # Get maximum eigenvalues in each (localised) cell
    max_eigvals = np.maximum(fv.slice_(local_max_eigvals, axis, end=-1), fv.slice_(local_max_eigvals, axis, start=1))  # Get the maximum eigenvalue between each consecutive pair of cells

    intermediate_cons = .5*(cons_minus+cons_plus) - .5*fv.divide(flux_plus-flux_minus, max_eigvals[...,None])

    # Convert to primitive grid again for flux computation
    centred_grid = fv.inverse_reconstruct(intermediate_cons, sim_variables) if sim_variables.magnetic else intermediate_cons
    intermediate_prim = sim_variables.convert("conservative", centred_grid, sim_variables)
    intermediate_prim[...,5+sim_variables.axes] = cons_plus[...,5+sim_variables.axes]

    return constructor.make_flux(intermediate_prim, sim_variables, axis)


# GFORCE solver [Toro & Titarev, 2006; Mignone & Del Zanna, 2021]
def calculate_gForce_flux(axis, sim_variables, **kwargs):
    wg = 1/(1+sim_variables.cfl)
    return wg*calculate_LaxWendroff_flux(axis, sim_variables, **kwargs) + (1-wg)*calculate_LaxFriedrich_flux(axis, sim_variables, **kwargs)


# HLLC Riemann solver [Fleischmann et al., 2020]
def calculate_HLLC_flux(axis, sim_variables, low_mach=False, **kwargs):
    Ma_limit = .1
    rho, pressure, gamma = sim_variables.rho, sim_variables.pressure, sim_variables.gamma
    energy = pressure

    prim_plus, prim_minus = kwargs["prim_interfaces"]
    cons_plus, cons_minus = kwargs["cons_interfaces"]
    flux_plus, flux_minus = kwargs["flux_interfaces"]

    """The convention here uses L & R states, i.e. L state = w-, R state = w+
        |                        w(i-1/2)                    w(i+1/2)                       |
        |-->         i-1         <--|-->          i          <--|-->         i+1         <--|
        |   w_R(i-1)     w_L(i-1)   |   w_R(i)         w_L(i)   |   w_R(i+1)     w_L(i+1)   |
    --> |   w+(i-3/2)   w-(i-1/2)   |   w+(i-1/2)   w-(i+1/2)   |  w+(i+1/2)    w-(i+3/2)   |
    """
    rhoL, uL, pL, qL = prim_minus[...,rho], prim_minus[...,1+axis], prim_minus[...,pressure], cons_minus
    rhoR, uR, pR, qR = prim_plus[...,rho], prim_plus[...,1+axis], prim_plus[...,pressure], cons_plus

    # Compute the wavespeeds
    cs_L, cAx_L, caf_L, cas_L = constructor.make_wavespeeds(prim_minus, sim_variables, axis)
    cs_R, cAx_R, caf_R, cas_R = constructor.make_wavespeeds(prim_plus, sim_variables, axis)
    sL = np.minimum(0, np.minimum(uL, uR) - np.maximum(cs_L, cs_R))
    sR = np.maximum(0, np.maximum(uL, uR) + np.maximum(cs_L, cs_R))
    sM = fv.divide(pL - pR + rhoR*uR*(sR-uR) - rhoL*uL*(sL-uL), rhoR*(sR-uR) - rhoL*(sL-uL))

    # Calculate the intermediate states
    coeffL, coeffR = fv.divide(sL-uL, sL-sM), fv.divide(sR-uR, sR-sM)
    qL_star, qR_star = qL * coeffL[...,None], qR * coeffR[...,None]
    qL_star[...,1+axis], qR_star[...,1+axis] = rhoL * coeffL * sM, rhoR * coeffR * sM
    qL_star[...,energy] = qL_star[...,energy] + coeffL*(sM-uL)*(rhoL*sM + fv.divide(pL, sL-uL))
    qR_star[...,energy] = qR_star[...,energy] + coeffR*(sM-uR)*(rhoR*sM + fv.divide(pR, sR-uR))

    fLs_star, fRs_star = np.copy(flux_minus), np.copy(flux_plus)
    fLs_star, fRs_star = fLs_star + (qL_star-qL) * sL[...,None], fRs_star + (qR_star-qR) * sR[...,None]

    # Modification to HLLC solver for low Mach shocks [Fleischmann et al., 2020]
    if low_mach:
        cL, cR = np.sqrt(fv.divide(gamma*pL, rhoL)), np.sqrt(fv.divide(gamma*pR, rhoR))
        Ma_local = np.maximum(np.abs(fv.divide(uL,cL)), np.abs(fv.divide(uR,cR)))
        phi = np.sin(.5 * np.pi * np.minimum(1, Ma_local/Ma_limit))
        sL = phi * sL
        sR = phi * sR

    # Calculate the flux
    flux = np.copy(flux_plus)
    flux[(sL <= 0) & (0 < sM)] = fLs_star[(sL <= 0) & (0 < sM)]
    flux[(sM <= 0) & (0 <= sR)] = fRs_star[(sM <= 0) & (0 <= sR)]
    flux[sR < 0] = flux_plus[sR < 0]
    return flux


# HLLD Riemann solver [Miyoshi & Kusano, 2005]
def calculate_HLLD_flux(axis, sim_variables, **kwargs):
    axes = (axis + np.array(range(3)))%3
    abscissa, ordinate, applicate = axes
    Bx, By, Bz = 5 + axes
    momx, momy, momz = 1 + axes

    rho, pressure, vels, Bfields, energy, momentums = sim_variables.rho, sim_variables.pressure, sim_variables.vels, sim_variables.Bfields, sim_variables.energy, sim_variables.momentums

    prim_plus, prim_minus = kwargs["prim_interfaces"]
    cons_plus, cons_minus = kwargs["cons_interfaces"]
    flux_plus, flux_minus = kwargs["flux_interfaces"]

    """The convention here uses L & R states, i.e. L state = w-, R state = w+
        |                        w(i-1/2)                    w(i+1/2)                       |
        |-->         i-1         <--|-->          i          <--|-->         i+1         <--|
        |   w_R(i-1)     w_L(i-1)   |   w_R(i)         w_L(i)   |   w_R(i+1)     w_L(i+1)   |
    --> |   w+(i-3/2)   w-(i-1/2)   |   w+(i-1/2)   w-(i+1/2)   |  w+(i+1/2)    w-(i+3/2)   |
    """
    rhoL, vecL, pL, bL, qL = prim_minus[...,rho], prim_minus[...,vels], prim_minus[...,pressure], prim_minus[...,Bfields], cons_minus
    rhoR, vecR, pR, bR, qR = prim_plus[...,rho], prim_plus[...,vels], prim_plus[...,pressure], prim_plus[...,Bfields], cons_plus
    pTL, pTR = pL + .5*fv.norm(bL)**2, pR + .5*fv.norm(bR)**2

    # Compute the wavespeeds
    cs_L, cAx_L, caf_L, cas_L = constructor.make_wavespeeds(prim_minus, sim_variables, axis)
    cs_R, cAx_R, caf_R, cas_R = constructor.make_wavespeeds(prim_plus, sim_variables, axis)
    sL = np.minimum(0, np.minimum(vecL[...,abscissa], vecR[...,abscissa]) - np.maximum(caf_L, caf_R))
    sR = np.maximum(0, np.maximum(vecL[...,abscissa], vecR[...,abscissa]) + np.maximum(caf_L, caf_R))
    sM = fv.divide(pTL - pTR + rhoR*vecR[...,axis]*(sR-vecR[...,axis]) - rhoL*vecL[...,axis]*(sL-vecL[...,axis]), rhoR*(sR-vecR[...,axis]) - rhoL*(sL-vecL[...,axis]))

    # Calculate the star states
    rhoL_star, rhoR_star = rhoL * fv.divide(sL-vecL[...,axis], sL-sM), rhoR * fv.divide(sR-vecR[...,axis], sR-sM)
    sL_star, sR_star = sM - fv.divide(np.abs(bL[...,axis]), np.sqrt(rhoL_star)), sM + fv.divide(np.abs(bR[...,axis]), np.sqrt(rhoR_star))
    pT_star = fv.divide(rhoR*pTL*(sR-vecR[...,axis]) - rhoL*pTR*(sL-vecL[...,axis]) + rhoL*rhoR*(sR-vecR[...,axis])*(sL-vecL[...,axis])*(vecR[...,axis]-vecL[...,axis]), rhoR*(sR-vecR[...,axis]) - rhoL*(sL-vecL[...,axis]))

    vyL_star = vecL[...,ordinate] - prim_minus[...,Bx]*bL[...,ordinate]*fv.divide(sM-vecL[...,axis], rhoL*(sL-vecL[...,axis])*(sL-sM) - prim_minus[...,Bx]**2)
    vyR_star = vecR[...,ordinate] - prim_plus[...,Bx]*bR[...,ordinate]*fv.divide(sM-vecR[...,axis], rhoR*(sR-vecR[...,axis])*(sR-sM) - prim_plus[...,Bx]**2)
    vzL_star = vecL[...,applicate] - prim_minus[...,Bx]*bL[...,applicate]*fv.divide(sM-vecL[...,axis], rhoL*(sL-vecL[...,axis])*(sL-sM) - prim_minus[...,Bx]**2)
    vzR_star = vecR[...,applicate] - prim_plus[...,Bx]*bR[...,applicate]*fv.divide(sM-vecR[...,axis], rhoR*(sR-vecR[...,axis])*(sR-sM) - prim_plus[...,Bx]**2)
    ByL_star = bL[...,ordinate] * fv.divide(rhoL*(sL-vecL[...,axis])**2 - prim_minus[...,Bx]**2, rhoL*(sL-vecL[...,axis])*(sL-sM) - prim_minus[...,Bx]**2)
    ByR_star = bR[...,ordinate] * fv.divide(rhoR*(sR-vecR[...,axis])**2 - prim_plus[...,Bx]**2, rhoR*(sR-vecR[...,axis])*(sR-sM) - prim_plus[...,Bx]**2)
    BzL_star = bL[...,applicate] * fv.divide(rhoL*(sL-vecL[...,axis])**2 - prim_minus[...,Bx]**2, rhoL*(sL-vecL[...,axis])*(sL-sM) - prim_minus[...,Bx]**2)
    BzR_star = bR[...,applicate] * fv.divide(rhoR*(sR-vecR[...,axis])**2 - prim_plus[...,Bx]**2, rhoR*(sR-vecR[...,axis])*(sR-sM) - prim_plus[...,Bx]**2)

    qL_star, qR_star = np.zeros_like(qL), np.zeros_like(qR)
    qL_star[...,rho], qR_star[...,rho] = rhoL_star, rhoR_star
    qL_star[...,momx], qR_star[...,momx] = rhoL * sM, rhoR * sM
    qL_star[...,momy], qR_star[...,momy] = rhoL * vyL_star, rhoR * vyR_star
    qL_star[...,momz], qR_star[...,momz] = rhoL * vzL_star, rhoR * vzR_star
    qL_star[...,Bx], qR_star[...,Bx] = np.copy(qL[...,Bx]), np.copy(qR[...,Bx])
    qL_star[...,By], qR_star[...,By] = ByL_star, ByR_star
    qL_star[...,Bz], qR_star[...,Bz] = BzL_star, BzR_star
    qL_star[...,energy] = fv.divide(qL[...,energy]*(sL-vecL[...,axis]) - pTL*vecL[...,axis] + pT_star*sM + prim_minus[...,Bx]*(np.sum(vecL*bL, axis=-1) - np.sum(fv.divide(qL_star[...,momentums], rhoL[...,None])*qL_star[...,Bfields], axis=-1)), sL-sM)
    qR_star[...,energy] = fv.divide(qR[...,energy]*(sR-vecR[...,axis]) - pTR*vecR[...,axis] + pT_star*sM + prim_plus[...,Bx]*(np.sum(vecR*bR, axis=-1) - np.sum(fv.divide(qR_star[...,momentums], rhoR[...,None])*qR_star[...,Bfields], axis=-1)), sR-sM)

    fLs_star, fRs_star = np.copy(flux_minus), np.copy(flux_plus)
    fLs_star, fRs_star = fLs_star + (qL_star - qL) * sL[...,None], fRs_star + (qR_star - qR) * sR[...,None]

    # Calculate the double-star states
    vy_starstar = fv.divide(vyL_star*np.sqrt(rhoL_star) + vyR_star*np.sqrt(rhoR_star) + np.sign(prim_plus[...,Bx])*(ByR_star-ByL_star), np.sqrt(rhoL_star) + np.sqrt(rhoR_star))
    vz_starstar = fv.divide(vzL_star*np.sqrt(rhoL_star) + vzR_star*np.sqrt(rhoR_star) + np.sign(prim_plus[...,Bx])*(BzR_star-BzL_star), np.sqrt(rhoL_star) + np.sqrt(rhoR_star))
    By_starstar = fv.divide(ByR_star*np.sqrt(rhoL_star) + ByL_star*np.sqrt(rhoR_star) + np.sign(prim_plus[...,Bx])*(vyR_star-vyL_star)*np.sqrt(rhoL_star*rhoR_star), np.sqrt(rhoL_star) + np.sqrt(rhoR_star))
    Bz_starstar = fv.divide(BzR_star*np.sqrt(rhoL_star) + BzL_star*np.sqrt(rhoR_star) + np.sign(prim_plus[...,Bx])*(vzR_star-vzL_star)*np.sqrt(rhoL_star*rhoR_star), np.sqrt(rhoL_star) + np.sqrt(rhoR_star))

    qL_starstar, qR_starstar = np.zeros_like(qL), np.zeros_like(qR)
    qL_starstar[...,rho], qR_starstar[...,rho] = rhoL_star, rhoR_star
    qL_starstar[...,momx], qR_starstar[...,momx] = rhoL_star * sM, rhoR_star * sM
    qL_starstar[...,momy], qR_starstar[...,momy] = rhoL_star * vy_starstar, rhoR_star * vy_starstar
    qL_starstar[...,momz], qR_starstar[...,momz] = rhoL_star * vz_starstar, rhoR_star * vz_starstar
    qL_starstar[...,Bx], qR_starstar[...,Bx] = np.copy(qL_star[...,Bx]), np.copy(qR_star[...,Bx])
    qL_starstar[...,By], qR_starstar[...,By] = By_starstar, By_starstar
    qL_starstar[...,Bz], qR_starstar[...,Bz] = Bz_starstar, Bz_starstar
    qL_starstar[...,energy] = np.copy(qL_star[...,energy] - np.sqrt(rhoL_star)*np.sign(prim_plus[...,Bx])*(np.sum(fv.divide(qL_star[...,momentums], rhoL[...,None])*qL_star[...,Bfields], axis=-1) - np.sum(fv.divide(qL_starstar[...,momentums], rhoL_star[...,None])*qL_starstar[...,Bfields], axis=-1)))
    qR_starstar[...,energy] = np.copy(qR_star[...,energy] + np.sqrt(rhoR_star)*np.sign(prim_plus[...,Bx])*(np.sum(fv.divide(qR_star[...,momentums], rhoR[...,None])*qR_star[...,Bfields], axis=-1) - np.sum(fv.divide(qR_starstar[...,momentums], rhoR_star[...,None])*qR_starstar[...,Bfields], axis=-1)))

    fLs_starstar, fRs_starstar = np.copy(fLs_star), np.copy(fRs_star)
    fLs_starstar, fRs_starstar = fLs_starstar + (qL_starstar - qL_star) * sL_star[...,None], fRs_starstar + (qR_starstar - qR_star) * sR_star[...,None]

    flux = np.copy(flux_minus)
    flux[(sL <= 0) & (0 < sL_star)] = fLs_star[(sL <= 0) & (0 < sL_star)]
    flux[(sL_star <= 0) & (0 < sM)] = fLs_starstar[(sL_star <= 0) & (0 < sM)]
    flux[(sM <= 0) & (0 < sR_star)] = fRs_starstar[(sM <= 0) & (0 < sR_star)]
    flux[(sR_star <= 0) & (0 <= sR)] = fRs_star[(sR_star <= 0) & (0 <= sR)]
    flux[sR < 0] = flux_plus[sR < 0]
    return flux


# Osher-Solomon(-Dumbser-Toro) Riemann solver [Dumbser & Toro, 2011]
def calculate_DOTS_flux(axis, sim_variables, **kwargs):
    cons_plus, cons_minus = kwargs["cons_interfaces"]
    flux_plus, flux_minus = kwargs["flux_interfaces"]
    roots, weights = sim_variables.roots, sim_variables.weights

    # Define the path integral for the Osher-Solomon dissipation term
    arr_plus, arr_minus = np.repeat(cons_plus[None,:], len(roots), axis=0), np.repeat(cons_minus[None,:], len(roots), axis=0)
    psi = arr_minus + (roots*(arr_plus - arr_minus).T).T

    # Define the left & right eigenvectors
    left_eigenvectors, right_eigenvectors = constructor.make_eigenvectors(psi, sim_variables, axis)

    # Generate the diagonal matrix of eigenvalues
    eigenvalues = np.zeros_like(right_eigenvectors)

    # Compute wavespeeds
    sound_speed, alfven_speed_x, fast_magnetosonic_wave, slow_magnetosonic_wave = constructor.make_wavespeeds(psi, sim_variables, axis)
    vxs = psi[...,1+axis]

    # Compute the diagonal matrix of eigenvalues
    eigenvalues[...,0,0] = vxs - fast_magnetosonic_wave
    eigenvalues[...,1,1] = vxs - alfven_speed_x
    eigenvalues[...,2,2] = vxs - slow_magnetosonic_wave
    eigenvalues[...,3,3] = vxs
    eigenvalues[...,4,4] = vxs
    eigenvalues[...,5,5] = vxs + slow_magnetosonic_wave
    eigenvalues[...,6,6] = vxs + alfven_speed_x
    eigenvalues[...,7,7] = vxs + fast_magnetosonic_wave

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
    lon = lambda L,R: fv.divide(L-R, np.log(L)-np.log(R))  # Stable numerical procedure for computing logarithmic mean [Ismail & Roe, 2009]


    # To construct the entropy-stable flux, 2 components are needed:
    # the entropy-conserving component, and the dissipation term to make the flux entropy-stable

    # Entropy-conserving flux section [Winters & Gassner, 2015]
    ec_flux = np.zeros_like(prim_plus)

    z1L, z1R = np.sqrt(fv.divide(rhoL, pL)), np.sqrt(fv.divide(rhoR, pR))
    z5L, z5R = np.sqrt(rhoL*pL), np.sqrt(rhoR*pR)

    # Compute the averages
    rho_hat = amean(z1L,z1R) * lon(z5L,z5R)
    p1_hat = fv.divide(amean(z5L,z5R),amean(z1L,z1R))
    p2_hat = .5 * ((gamma+1)/gamma * fv.divide(lon(z5L,z5R), lon(z1L,z1R)) + (gamma-1)/gamma * fv.divide(amean(z5L,z5R), amean(z1L,z1R)))
    u1_hat = fv.divide(amean(z1L*uL,z1R*uR), amean(z1L,z1R))
    v1_hat = fv.divide(amean(z1L*vL,z1R*vR), amean(z1L,z1R))
    w1_hat = fv.divide(amean(z1L*wL,z1R*wR), amean(z1L,z1R))
    u2_hat = fv.divide(amean(uL*z1L**2,uR*z1R**2), amean(z1L**2,z1R**2))
    v2_hat = fv.divide(amean(vL*z1L**2,vR*z1R**2), amean(z1L**2,z1R**2))
    w2_hat = fv.divide(amean(wL*z1L**2,wR*z1R**2), amean(z1L**2,z1R**2))
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
    es_right_eigenvectors = constructor.make_right_eigenvectors(np.array([rho_hat.T, u1_hat.T, v1_hat.T, w1_hat.T, p1_hat.T, b1_hat.T, b2_hat.T, b3_hat.T]).T, sim_variables, axis)

    # Define the jump in the entropy vector
    entropy = np.log(p1_hat * rho_hat**-gamma)
    entropy_vector = np.zeros_like(prim_plus)
    entropy_vector[...,rho] = ((gamma-entropy)/(gamma-1) - fv.divide(rho_hat*fv.norm(np.array([u1_hat.T, v1_hat.T, w1_hat.T]).T)**2, 2*p1_hat))
    entropy_vector[...,1+abscissa] = fv.divide(rho_hat*u1_hat, p1_hat)
    entropy_vector[...,1+ordinate] = fv.divide(rho_hat*v1_hat, p1_hat)
    entropy_vector[...,1+applicate] = fv.divide(rho_hat*w1_hat, p1_hat)
    entropy_vector[...,pressure] = -fv.divide(rho_hat, p1_hat)
    entropy_vector[...,5+abscissa] = fv.divide(rho_hat*b1_hat, p1_hat)
    entropy_vector[...,5+ordinate] = fv.divide(rho_hat*b2_hat, p1_hat)
    entropy_vector[...,5+applicate] = fv.divide(rho_hat*b3_hat, p1_hat)
    entropy_vector *= -1

    # Define speeds
    sound_speed = np.sqrt(gamma * fv.divide(p1_hat, rho_hat))
    alfven_speed = fv.divide(fv.norm(np.array([b1_hat.T, b2_hat.T, b3_hat.T]).T), np.sqrt(rho_hat))
    alfven_speed_x = fv.divide(b1_hat, np.sqrt(rho_hat))
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
        Epsilon = np.sqrt(np.abs(fv.divide(pL-pR, pL+pR)))
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