import numpy as np

from functions import constructor, fv

##############################################################################
# Approximate linearised and non-linearised Riemann solvers
##############################################################################

# Intercell numerical fluxes between L and R interfaces based on Riemann solver
def calculate_Riemann_flux(data, sim_variables):
    # HLL-type solvers
    if sim_variables.solver_category == "hll":
        if sim_variables.solver.endswith("d"):
            Riemann_solver = calculate_HLLD_flux
        else:
            Riemann_solver = calculate_HLLC_flux
    # 'Complete Riemann' solvers
    elif sim_variables.solver_category == "complete":
        if sim_variables.solver.startswith("o"):
            Riemann_solver = calculate_DOTS_flux
        else:
            Riemann_solver = calculate_ES_flux
    # Roe-type/Lax-type solvers
    else:
        if sim_variables.solver.endswith("w"):
            Riemann_solver = calculate_LaxWendroff_flux
        else:
            Riemann_solver = calculate_LaxFriedrich_flux

    fluxes = {}
    for axis, arrays in data.items():
        eigmax = sim_variables.ds[axis]/fv.compute_eigmax(arrays['characteristics'], axis=axis)

        # Calculate the interface-averaged fluxes
        intf_fluxes_avgd = Riemann_solver(axis, sim_variables, **arrays)

        if sim_variables.dimension == 2 and sim_variables.higher_order:
            # Compute the orthogonal L/R Riemann states and fluxes at higher-order
            higher_order_intfs = {}
            for key, array in arrays.items():
                if key == "characteristics":
                    higher_order_intfs[key] = array
                elif len(array) == 2:
                    plus_intf, minus_intf = array
                    higher_order_intfs[key] = fv.high_order_convert('face', 'avg', plus_intf, sim_variables), fv.high_order_convert('face', 'avg', minus_intf, sim_variables)

            intf_fluxes_cntrd = Riemann_solver(axis, sim_variables, **higher_order_intfs)

            # Compute higher-order fluxes using approximation with fluxes from transverse interfaces
            final_fluxes = fv.high_order_compute_flux(intf_fluxes_cntrd, intf_fluxes_avgd, sim_variables)
        else:
            # Orthogonal Laplacian in 1D is zero
            final_fluxes = intf_fluxes_avgd

        # Add additional dissipation for strong shocks, if switched on (should not apply for mag. fields) [McCorquodale & Colella, 2011]
        if "artf_visc" in arrays.keys():
            final_fluxes = final_fluxes + arrays['artf_visc']

        fluxes[axis] = {'flux':final_fluxes, 'eigmax':eigmax}

    return fluxes


# (Local) Lax-Friedrich solver (1st-order; highly diffusive)
def calculate_LaxFriedrich_flux(axis, sim_variables, **kwargs):
    cons_plus, cons_minus = kwargs["cons_interfaces"]
    flux_plus, flux_minus = kwargs["flux_interfaces"]
    characteristics = kwargs["characteristics"]

    local_max_eigvals = np.max(np.abs(characteristics), axis=-1)
    max_eigvals = np.maximum(fv.slice_(local_max_eigvals, axis, end=-1), fv.slice_(local_max_eigvals, axis, start=1))
    return .5*(flux_minus+flux_plus) - .5*((cons_plus-cons_minus) * max_eigvals[...,None])


# Lax-Wendroff solver (2nd-order, Jacobian method; contains overshoots)
def calculate_LaxWendroff_flux(axis, sim_variables, **kwargs):
    cons_plus, cons_minus = kwargs["cons_interfaces"]
    flux_plus, flux_minus = kwargs["flux_interfaces"]
    characteristics = kwargs["characteristics"]

    # Sound speed for each cell (2-Riemann invariant; entropy wave or contact discontinuity); indexing 1 only works for hydrodynamics
    sound_speed = np.unique(characteristics, axis=-1)[...,1+axis]
    normalised_eigvals = fv.divide(sound_speed**2, np.max(np.abs(characteristics), axis=-1))
    max_normalised_eigvals = np.maximum(fv.slice_(normalised_eigvals, axis, end=-1), fv.slice_(normalised_eigvals, axis, start=1))

    return .5*(flux_minus+flux_plus) - .5*((cons_plus-cons_minus) * max_normalised_eigvals[...,None])


# HLLC Riemann solver [Fleischmann et al., 2020]
def calculate_HLLC_flux(axis, sim_variables, low_mach=False, **kwargs):
    Ma_limit = .1
    gamma = sim_variables.gamma
    rho, pressure = 0, 4
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

    # Generic HLLC solver [Toro et al., 1994]
    # Calculate sound speeds
    cL, cR = np.sqrt(fv.divide(gamma*pL, rhoL)), np.sqrt(fv.divide(gamma*pR, rhoR))
    u_hat = fv.divide(uL*np.sqrt(rhoL) + uR*np.sqrt(rhoR), np.sqrt(rhoL) + np.sqrt(rhoR))
    c_hat = np.sqrt(
        fv.divide(np.sqrt(rhoL)*cL**2 + np.sqrt(rhoR)*cR**2, np.sqrt(rhoL) + np.sqrt(rhoR))
        + .5 * fv.divide(np.sqrt(rhoL) * np.sqrt(rhoR), (np.sqrt(rhoL) + np.sqrt(rhoR))**2) * (uR-uL)**2
    )

    # Calculate the non-linear signal speeds
    sL, sR = np.minimum(uL-cL, u_hat-c_hat), np.maximum(uR+cR, u_hat+c_hat)
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
    rho, pressure, vels, Bfields = 0, 4, slice(1,4), slice(5,8)
    energy, momentums = pressure, vels
    axes = (axis + np.array(range(3)))%3
    abscissa, ordinate, applicate = axes
    Bx, By, Bz = 5 + axes
    momx, momy, momz = 1 + axes

    prim_plus, prim_minus = kwargs["prim_interfaces"]
    cons_plus, cons_minus = kwargs["cons_interfaces"]
    flux_plus, flux_minus = kwargs["flux_interfaces"]
    characteristics = kwargs['characteristics']

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
    local_min_eigvals, local_max_eigvals = np.min(characteristics, axis=-1), np.max(characteristics, axis=-1)
    sL = np.minimum(fv.slice_(local_min_eigvals, axis, end=-1), fv.slice_(local_min_eigvals, axis, start=1))
    sR = np.maximum(fv.slice_(local_max_eigvals, axis, end=-1), fv.slice_(local_max_eigvals, axis, start=1))
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
    gamma, roots, weights = sim_variables.gamma, sim_variables.roots, sim_variables.weights

    # Define the path integral for the Osher-Solomon dissipation term
    arr_plus, arr_minus = np.repeat(cons_plus[None,:], len(roots), axis=0), np.repeat(cons_minus[None,:], len(roots), axis=0)
    psi = arr_minus + (roots*(arr_plus - arr_minus).T).T

    # Define the right eigenvectors
    _right_eigenvectors = constructor.make_right_eigenvectors(axis, psi, gamma)

    # Generate the diagonal matrix of eigenvalues
    _lambda = np.zeros_like(_right_eigenvectors)
    rhos, vxs, pressures, B_fields = psi[...,0], psi[...,1], psi[...,4], psi[...,5:8]

    # Define speeds
    sound_speed = np.sqrt(gamma * fv.divide(pressures, rhos))
    alfven_speed = fv.divide(fv.norm(B_fields), np.sqrt(rhos))
    alfven_speed_x = fv.divide(B_fields[...,axis], np.sqrt(rhos))
    fast_magnetosonic_wave = np.sqrt(.5 * (sound_speed**2 + alfven_speed**2 + np.sqrt(((sound_speed**2 + alfven_speed**2)**2) - (4*(sound_speed**2)*(alfven_speed_x**2)))))
    slow_magnetosonic_wave = np.sqrt(.5 * (sound_speed**2 + alfven_speed**2 - np.sqrt(((sound_speed**2 + alfven_speed**2)**2) - (4*(sound_speed**2)*(alfven_speed_x**2)))))

    # Compute the diagonal matrix of eigenvalues
    _lambda[...,0,0] = vxs - fast_magnetosonic_wave
    _lambda[...,1,1] = vxs - alfven_speed_x
    _lambda[...,2,2] = vxs - slow_magnetosonic_wave
    _lambda[...,3,3] = vxs
    _lambda[...,4,4] = vxs
    _lambda[...,5,5] = vxs + slow_magnetosonic_wave
    _lambda[...,6,6] = vxs + alfven_speed_x
    _lambda[...,7,7] = vxs + fast_magnetosonic_wave
    _eigenvalues = np.abs(_lambda)

    # Compute the absolute value of the Jacobian
    abs_A = _right_eigenvectors @ _eigenvalues @ np.linalg.pinv(_right_eigenvectors)

    # Compute the Dumbser-Toro Jacobian with the Gauss-Legendre quadrature
    jacobian = np.sum((weights*abs_A.T).T, axis=0)

    # Compute the Osher-Solomon dissipation term
    _q_plus = jacobian @ cons_plus[...,None]
    _q_minus = jacobian @ cons_minus[...,None]
    _q_plus = _q_plus.reshape(_q_plus.shape[:-1])
    _q_minus = _q_minus.reshape(_q_minus.shape[:-1])

    return .5*(flux_plus+flux_minus) - .5*(_q_plus-_q_minus)


# Entropy-stable flux calculation based on left and right interpolated primitive variables [Winters & Gassner, 2015; Derigs et al., 2016]
def calculate_ES_flux(axis, sim_variables, **kwargs):
    prim_plus, prim_minus = kwargs["prim_interfaces"]
    rho, pressure, vels, Bfields = 0, 4, slice(1,4), slice(5,8)
    abscissa, ordinate, applicate = axis%3, (axis+1)%3, (axis+2)%3
    gamma = sim_variables.gamma

    # To construct the entropy-stable flux, 2 components are needed:
    # the entropy-conserving flux component, and the dissipation term to make the flux entropy-stable

    # Entropy-conserving flux section [Winters & Gassner, 2015]
    ec_flux = np.zeros_like(prim_plus)

    # Compute arithmetic mean
    def arith_mean(term):
        return .5 * (term[0] - term[1])

    # Stable numerical procedure for computing logarithmic mean [Ismail & Roe, 2009]
    def lon(term):
        return fv.divide(term[0] - term[1], fv.log(term[0]) - fv.log(term[1]))

    # Define frequently used terms; here we use L & R states for simplicity, i.e. L state = w-, R state = w+
    rhoL, vecL, PL, B_fieldL = prim_minus[...,rho], prim_minus[...,vels], prim_minus[...,pressure], prim_minus[...,Bfields]
    rhoR, vecR, PR, B_fieldR = prim_plus[...,rho], prim_plus[...,vels], prim_plus[...,pressure], prim_plus[...,Bfields]

    z1 = np.array([np.sqrt(fv.divide(rhoL, PL)), np.sqrt(fv.divide(rhoR, PR))])
    z5 = np.array([np.sqrt(rhoL*PL), np.sqrt(rhoR*PR)])
    vx, vy, vz = np.array([vecL[...,0], vecR[...,0]]), np.array([vecL[...,1], vecR[...,1]]), np.array([vecL[...,2], vecR[...,2]])
    Bx, By, Bz = np.array([B_fieldL[...,0], B_fieldR[...,0]]), np.array([B_fieldL[...,1], B_fieldR[...,1]]), np.array([B_fieldL[...,2], B_fieldR[...,2]])

    # Compute the averages
    rho_hat = arith_mean(z1) * lon(z5)
    P1_hat = fv.divide(arith_mean(z5), arith_mean(z1))
    P2_hat = ((gamma+1)/(2*gamma))*(fv.divide(lon(z5), lon(z1))) + ((gamma-1)/(2*gamma))*(fv.divide(arith_mean(z5), arith_mean(z1)))
    u1_hat = fv.divide(arith_mean(vx*z1), arith_mean(z1))
    v1_hat = fv.divide(arith_mean(vy*z1), arith_mean(z1))
    w1_hat = fv.divide(arith_mean(vz*z1), arith_mean(z1))
    u2_hat = fv.divide(arith_mean(vx*z1**2), arith_mean(z1**2))
    v2_hat = fv.divide(arith_mean(vy*z1**2), arith_mean(z1**2))
    w2_hat = fv.divide(arith_mean(vz*z1**2), arith_mean(z1**2))
    B1_hat = arith_mean(Bx)
    B1_dot = arith_mean(Bx**2)
    B2_hat = arith_mean(By)
    B2_dot = arith_mean(By**2)
    B3_hat = arith_mean(Bz)
    B3_dot = arith_mean(Bz**2)
    B1B2 = arith_mean(Bx*By)
    B1B3 = arith_mean(Bx*Bz)

    # Update the entropy-conserving flux vector; suitable for smooth solutions
    ec_flux[...,rho] = rho_hat * u1_hat
    ec_flux[...,abscissa+1] = P1_hat + rho_hat*u1_hat**2 + .5*(B1_dot+B2_dot+B3_dot) - B1_dot
    ec_flux[...,ordinate+1] = rho_hat*u1_hat*v1_hat - B1B2
    ec_flux[...,applicate+1] = rho_hat*u1_hat*w1_hat - B1B3
    ec_flux[...,pressure] = (gamma/(gamma-1))*u1_hat*P2_hat + .5*rho_hat*u1_hat*(u1_hat**2 + v1_hat**2 + w1_hat**2) + u2_hat*(B2_hat**2 + B3_hat**2) - B1_hat*(v2_hat*B2_hat + w2_hat*B3_hat)
    ec_flux[...,ordinate+5] = u2_hat*B2_hat - v2_hat*B1_hat
    ec_flux[...,applicate+5] = u2_hat*B3_hat - w2_hat*B1_hat


    # Entropy-stable flux with dissipation term section [Derigs et al., 2016]
    # Make the right eigenvectors for each cell in each grid using the averaged primitive variables
    right_eigenvectors = constructor.make_ES_right_eigenvectors(axis, np.array([rho_hat.T, u1_hat.T, v1_hat.T, w1_hat.T, P1_hat.T, B1_hat.T, B2_hat.T, B3_hat.T]).T, gamma)

    # Define speeds
    sound_speed = np.sqrt(gamma * fv.divide(P1_hat, rho_hat))
    alfven_speed = fv.divide(fv.norm(np.array([B1_hat.T, B2_hat.T, B3_hat.T]).T), np.sqrt(rho_hat))
    alfven_speed_x = fv.divide(B1_hat, np.sqrt(rho_hat))
    fast_magnetosonic_wave = np.sqrt(.5 * (sound_speed**2 + alfven_speed**2 + np.sqrt(((sound_speed**2 + alfven_speed**2)**2) - (4*(sound_speed**2)*(alfven_speed_x**2)))))
    slow_magnetosonic_wave = np.sqrt(.5 * (sound_speed**2 + alfven_speed**2 - np.sqrt(((sound_speed**2 + alfven_speed**2)**2) - (4*(sound_speed**2)*(alfven_speed_x**2)))))

    # Compute the diagonal matrix of eigenvalues for Roe
    roe_eigenvalues = np.zeros_like(right_eigenvectors)
    roe_eigenvalues[...,0,0] = u1_hat + fast_magnetosonic_wave
    roe_eigenvalues[...,1,1] = u1_hat + alfven_speed_x
    roe_eigenvalues[...,2,2] = u1_hat + slow_magnetosonic_wave
    roe_eigenvalues[...,3,3] = u1_hat
    roe_eigenvalues[...,4,4] = u1_hat
    roe_eigenvalues[...,5,5] = u1_hat - slow_magnetosonic_wave
    roe_eigenvalues[...,6,6] = u1_hat - alfven_speed_x
    roe_eigenvalues[...,7,7] = u1_hat - fast_magnetosonic_wave
    roe_eigenvalues = np.abs(roe_eigenvalues)

    # Compute the diagonal matrix of eigenvalues for Local Lax-Friedrich
    lff_eigenvalues = np.zeros_like(right_eigenvectors)
    i, j = np.diag_indices(lff_eigenvalues.shape[-1])
    max_values = np.maximum.reduce([np.abs(u1_hat+fast_magnetosonic_wave), np.abs(u1_hat+alfven_speed_x), np.abs(u1_hat+slow_magnetosonic_wave), np.abs(u1_hat), np.abs(u1_hat-slow_magnetosonic_wave), np.abs(u1_hat-alfven_speed_x), np.abs(u1_hat-fast_magnetosonic_wave)])
    lff_eigenvalues[..., i,j] = max_values[..., None]

    # Define the jump in the entropy vector
    entropy_vector = np.zeros_like(prim_plus)
    entropy_vector[...,rho] = ((gamma-np.log(PL*rhoL**-gamma))/(gamma-1) - fv.divide(.5*rhoL*fv.norm(vecL)**2, PL)) - ((gamma-np.log(PR*rhoR**-gamma))/(gamma-1) - fv.divide(.5*rhoR*fv.norm(vecR)**2, PR))
    entropy_vector[...,pressure] = fv.divide(rhoR, PR) - fv.divide(rhoL, PL)
    entropy_vector[...,vels] = fv.divide(vecL * rhoL[...,None], PL[...,None]) - fv.divide(vecR * rhoR[...,None], PR[...,None])
    entropy_vector[...,Bfields] = fv.divide(B_fieldL * rhoL[...,None], PL[...,None]) - fv.divide(B_fieldR * rhoR[...,None], PR[...,None])
    entropy_vector *= -1

    # Compute the hydrid entropy stabilisation
    Epsilon = np.sqrt(np.abs(fv.divide(PR-PL, PR+PL)))
    eigenvalues = (1-Epsilon)[...,None,None]*roe_eigenvalues + Epsilon[...,None,None]*lff_eigenvalues

    # Calculate the dissipation term
    abs_A = right_eigenvectors @ eigenvalues @ right_eigenvectors.transpose(0,2,1)
    _dissipation = abs_A @ entropy_vector[...,None]
    dissipation = _dissipation.reshape(len(entropy_vector), len(entropy_vector[0]))

    return ec_flux + .5*dissipation