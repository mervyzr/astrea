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
        eigmax = fv.compute_eigmax(arrays['characteristics'], axis=axis)

        # Calculate the interface-averaged fluxes
        intf_fluxes_avgd = Riemann_solver(axis, sim_variables, **arrays)

        if sim_variables.dimension == 2 and sim_variables.higher_order:
            # Compute the orthogonal L/R Riemann states and fluxes at higher-order
            higher_order_intfs = {}
            for key, array in arrays.items():
                if key == "primitive" or key == "characteristics":
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
    prim_plus, prim_minus = kwargs["prim_interfaces"]
    cons_plus, cons_minus = kwargs["cons_interfaces"]
    flux_plus, flux_minus = kwargs["flux_interfaces"]
    gamma = sim_variables.gamma
    rho, pressure = 0, 4

    """The convention here uses L & R states, i.e. L state = w-, R state = w+
        |                        w(i-1/2)                    w(i+1/2)                       |
        |-->         i-1         <--|-->          i          <--|-->         i+1         <--|
        |   w_R(i-1)     w_L(i-1)   |   w_R(i)         w_L(i)   |   w_R(i+1)     w_L(i+1)   |
    --> |   w+(i-3/2)   w-(i-1/2)   |   w+(i-1/2)   w-(i+1/2)   |  w+(i+1/2)    w-(i+3/2)   |
    """
    rhoL, uL, pL, QL = prim_minus[...,rho], prim_minus[...,1+axis], prim_minus[...,pressure], cons_minus
    rhoR, uR, pR, QR = prim_plus[...,rho], prim_plus[...,1+axis], prim_plus[...,pressure], cons_plus

    # Generic HLLC solver [Toro et al., 1994]
    # Calculate Roe averages at boundary
    cL, cR = np.sqrt(gamma*fv.divide(pL, rhoL)), np.sqrt(gamma*fv.divide(pR, rhoR))
    u_hat = fv.divide(uL*np.sqrt(rhoL) + uR*np.sqrt(rhoR), np.sqrt(rhoL) + np.sqrt(rhoR))
    c2_hat = fv.divide(np.sqrt(rhoL)*cL**2 + np.sqrt(rhoR)*cR**2, np.sqrt(rhoL) + np.sqrt(rhoR)) + .5*((uR-uL)**2)*fv.divide(np.sqrt(rhoL)*np.sqrt(rhoR), (np.sqrt(rhoL)+np.sqrt(rhoR))**2)

    # Calculate the non-linear signal speeds
    sL, sR = np.minimum(uL-cL, u_hat-np.sqrt(c2_hat)), np.maximum(uR+cR, u_hat+np.sqrt(c2_hat))
    sM = fv.divide(pR - pL + rhoL*uL*(sL-uL) - rhoR*uR*(sR-uR), rhoL*(sL-uL) - rhoR*(sR-uR))

    # Modification to HLLC solver for low Mach shocks [Fleischmann et al., 2020]
    if low_mach:
        Ma_local = np.maximum(np.abs(fv.divide(uL,cL)), np.abs(fv.divide(uR,cR)))
        phi = np.sin(.5 * np.pi * np.minimum(1, Ma_local/.1))
        sL = np.copy(phi * sL)
        sR = np.copy(phi * sR)

    # Calculate the intermediate states
    coeffL, coeffR = fv.divide(sL-uL, sL-sM), fv.divide(sR-uR, sR-sM)
    QL_star, QR_star = QL * coeffL[...,None], QR * coeffR[...,None]
    QL_star[...,1+axis] = rhoL * coeffL * sM
    QR_star[...,1+axis] = rhoR * coeffR * sM
    QL_star[...,pressure] = QL_star[...,pressure] + coeffL*(sM-uL)*(rhoL*sM + fv.divide(pL, sL-uL))
    QR_star[...,pressure] = QR_star[...,pressure] + coeffR*(sM-uR)*(rhoR*sM + fv.divide(pR, sR-uR))

    # Calculate the flux
    flux = np.copy(flux_plus)
    fLs_star, fRs_star = flux_minus + (QL_star-QL) * sL[...,None], flux_plus + (QR_star-QR) * sR[...,None]
    flux[(sL <= 0) & (0 < sM)] = fLs_star[(sL <= 0) & (0 < sM)]
    flux[(sM <= 0) & (0 <= sR)] = fRs_star[(sM <= 0) & (0 <= sR)]
    flux[sR < 0] = flux_plus[sR < 0]
    return flux


# HLLD Riemann solver [Miyoshi & Kusano, 2005]
def calculate_HLLD_flux(axis, sim_variables, **kwargs):
    rho, pressure, vels, Bfields = 0, 4, slice(1,4), slice(5,8)

    def make_speeds(_wF, _gamma, _axis):
        _rhos, _pressures, _Bfields = _wF[...,rho], _wF[...,pressure], _wF[...,Bfields]

        _sound_speed = np.sqrt(fv.divide(_gamma*_pressures, _rhos))
        _alfven_speed = fv.divide(fv.norm(_Bfields), np.sqrt(_rhos))
        _alfven_speed_x = fv.divide(_Bfields[...,_axis], np.sqrt(_rhos))
        _fast_magnetosonic = np.sqrt(.5 * (_sound_speed**2 + _alfven_speed**2 + np.sqrt(((_sound_speed**2 + _alfven_speed**2)**2) - (4*(_sound_speed**2)*(_alfven_speed_x**2)))))
        #_slow_magnetosonic = np.sqrt(.5 * (_sound_speed**2 + _alfven_speed**2 - np.sqrt(((_sound_speed**2 + _alfven_speed**2)**2) - (4*(_sound_speed**2)*(_alfven_speed_x**2)))))

        return _fast_magnetosonic

    primitive = kwargs["primitive"]
    prim_plus, prim_minus = kwargs["prim_interfaces"]
    cons_plus, cons_minus = kwargs["cons_interfaces"]
    flux_plus, flux_minus = kwargs["flux_interfaces"]
    gamma = sim_variables.gamma
    abscissa, ordinate, applicate = axis%3, (axis+1)%3, (axis+2)%3

    """The convention here uses L & R states, i.e. L state = w-, R state = w+
        |                        w(i-1/2)                    w(i+1/2)                       |
        |-->         i-1         <--|-->          i          <--|-->         i+1         <--|
        |   w_R(i-1)     w_L(i-1)   |   w_R(i)         w_L(i)   |   w_R(i+1)     w_L(i+1)   |
    --> |   w+(i-3/2)   w-(i-1/2)   |   w+(i-1/2)   w-(i+1/2)   |  w+(i+1/2)    w-(i+3/2)   |
    """
    primitive = fv.slice_(fv.add_boundary(primitive, sim_variables.boundary, axis=axis), axis, start=1)
    rhoL, vecL, pL, BL, QL = prim_minus[...,rho], prim_minus[...,vels], prim_minus[...,pressure], prim_minus[...,Bfields], cons_minus
    rhoR, vecR, pR, BR, QR = prim_plus[...,rho], prim_plus[...,vels], prim_plus[...,pressure], prim_plus[...,Bfields], cons_plus

    # Compute the wave speeds
    cafL, cafR = make_speeds(prim_minus, gamma, axis), make_speeds(prim_plus, gamma, axis)

    sL, sR = np.minimum(vecL[...,axis], vecR[...,axis]) - np.maximum(cafL, cafR), np.minimum(vecL[...,axis], vecR[...,axis]) + np.maximum(cafL, cafR)
    sM = fv.divide(pR - pL + rhoL*vecL[...,axis]*(sL-vecL[...,axis]) - rhoR*vecR[...,axis]*(sR-vecR[...,axis]) + .5*(fv.norm(BR)**2) - .5*(fv.norm(BL)**2), rhoL*(sL-vecL[...,axis]) - rhoR*(sR-vecR[...,axis]))

    # Calculate the star states
    rhoL_star, rhoR_star = rhoL * fv.divide(sL-vecL[...,axis], sL-sM), rhoR * fv.divide(sR-vecR[...,axis], sR-sM)
    sL_star, sR_star = sM - fv.divide(BL[...,axis], np.sqrt(rhoL_star)), sM - fv.divide(BR[...,axis], np.sqrt(rhoR_star))

    p_star = fv.divide(rhoL*(pR+.5*fv.norm(BR)**2)*(sL-vecL[...,axis]) - rhoR*(pL+.5*fv.norm(BL)**2)*(sR-vecR[...,axis]) + rhoR*rhoL*(sL-vecL[...,axis])*(sR-vecR[...,axis]), rhoL*(sL-vecL[...,axis]) - rhoR*(sR-vecR[...,axis]))
    vyL_star = vecL[...,ordinate] - primitive[...,abscissa+5]*BL[...,ordinate]*fv.divide(sM-vecL[...,axis], rhoL*(sL-vecL[...,axis])*(sL-sM) - primitive[...,abscissa+5]**2)
    vyR_star = vecR[...,ordinate] - primitive[...,abscissa+5]*BR[...,ordinate]*fv.divide(sM-vecR[...,axis], rhoR*(sR-vecR[...,axis])*(sR-sM) - primitive[...,abscissa+5]**2)
    vzL_star = vecL[...,applicate] - primitive[...,abscissa+5]*BL[...,applicate]*fv.divide(sM-vecL[...,axis], rhoL*(sL-vecL[...,axis])*(sL-sM) - primitive[...,abscissa+5]**2)
    vzR_star = vecR[...,applicate] - primitive[...,abscissa+5]*BR[...,applicate]*fv.divide(sM-vecR[...,axis], rhoR*(sR-vecR[...,axis])*(sR-sM) - primitive[...,abscissa+5]**2)
    ByL_star = BL[...,ordinate] * fv.divide(rhoL*(sL-vecL[...,axis])**2 - primitive[...,abscissa+5]**2, rhoL*(sL-vecL[...,axis])*(sL-sM) - primitive[...,abscissa+5]**2)
    ByR_star = BR[...,ordinate] * fv.divide(rhoR*(sR-vecR[...,axis])**2 - primitive[...,abscissa+5]**2, rhoR*(sR-vecR[...,axis])*(sR-sM) - primitive[...,abscissa+5]**2)
    BzL_star = BL[...,applicate] * fv.divide(rhoL*(sL-vecL[...,axis])**2 - primitive[...,abscissa+5]**2, rhoL*(sL-vecL[...,axis])*(sL-sM) - primitive[...,abscissa+5]**2)
    BzR_star = BR[...,applicate] * fv.divide(rhoR*(sR-vecR[...,axis])**2 - primitive[...,abscissa+5]**2, rhoR*(sR-vecR[...,axis])*(sR-sM) - primitive[...,abscissa+5]**2)

    QL_star, QR_star = np.zeros_like(QL), np.zeros_like(QR)
    QL_star[...,rho], QR_star[...,rho] = rhoL_star, rhoR_star
    QL_star[...,abscissa+1], QR_star[...,abscissa+1] = rhoL * sM, rhoR * sM
    QL_star[...,ordinate+1], QR_star[...,ordinate+1] = rhoL * vyL_star, rhoR * vyR_star
    QL_star[...,applicate+1], QR_star[...,applicate+1] = rhoL * vzL_star, rhoR * vzR_star
    QL_star[...,abscissa+5], QR_star[...,abscissa+5] = np.copy(QL[...,abscissa+5]), np.copy(QR[...,abscissa+5])
    QL_star[...,ordinate+5], QR_star[...,ordinate+5] = ByL_star, ByR_star
    QL_star[...,applicate+5], QR_star[...,applicate+5] = BzL_star, BzR_star
    QL_star[...,pressure] = fv.divide(QL[...,pressure]*(sL-vecL[...,axis]) - vecL[...,axis]*(pL+.5*fv.norm(BL)**2) + p_star*sM + primitive[...,abscissa+5]*(np.sum(vecL*BL, axis=-1) - np.sum(QL_star[...,vels]*QL_star[...,Bfields], axis=-1)), sL-sM)
    QR_star[...,pressure] = fv.divide(QR[...,pressure]*(sR-vecR[...,axis]) - vecR[...,axis]*(pR+.5*fv.norm(BR)**2) + p_star*sM + primitive[...,abscissa+5]*(np.sum(vecR*BR, axis=-1) - np.sum(QR_star[...,vels]*QR_star[...,Bfields], axis=-1)), sR-sM)

    fLs_star, fRs_star = np.copy(flux_minus), np.copy(flux_plus)
    fLs_star, fRs_star = fLs_star + (QL_star - QL) * sL[...,None], fRs_star + (QR_star - QR) * sR[...,None]

    # Calculate the double-star states
    vy_starstar = fv.divide(vyR_star*np.sqrt(rhoR_star) + vyL_star*np.sqrt(rhoL_star) + np.sign(primitive[...,abscissa+5])*(ByL_star-ByR_star), np.sqrt(rhoL_star) + np.sqrt(rhoR_star))
    vz_starstar = fv.divide(vzR_star*np.sqrt(rhoR_star) + vzL_star*np.sqrt(rhoL_star) + np.sign(primitive[...,abscissa+5])*(BzL_star-BzR_star), np.sqrt(rhoL_star) + np.sqrt(rhoR_star))
    By_starstar = fv.divide(ByL_star*np.sqrt(rhoR_star) + ByR_star*np.sqrt(rhoL_star) + np.sign(primitive[...,abscissa+5])*(vyL_star-vyR_star)*np.sqrt(rhoR_star*rhoL_star), np.sqrt(rhoL_star) + np.sqrt(rhoR_star))
    Bz_starstar = fv.divide(BzL_star*np.sqrt(rhoR_star) + BzR_star*np.sqrt(rhoL_star) + np.sign(primitive[...,abscissa+5])*(vzL_star-vzR_star)*np.sqrt(rhoR_star*rhoL_star), np.sqrt(rhoL_star) + np.sqrt(rhoR_star))

    QL_starstar, QR_starstar = np.zeros_like(QL), np.zeros_like(QR)
    QL_starstar[...,rho], QR_starstar[...,rho] = rhoL_star, rhoR_star
    QL_starstar[...,abscissa+1], QR_starstar[...,abscissa+1] = sM, sM
    QL_starstar[...,ordinate+1], QR_starstar[...,ordinate+1] = vy_starstar, vy_starstar
    QL_starstar[...,applicate+1], QR_starstar[...,applicate+1] = vz_starstar, vz_starstar
    QL_starstar[...,abscissa+5], QR_starstar[...,abscissa+5] = np.copy(QL_star[...,abscissa+5]), np.copy(QR_star[...,abscissa+5])
    QL_starstar[...,ordinate+5], QR_starstar[...,ordinate+5] = By_starstar, By_starstar
    QL_starstar[...,applicate+5], QR_starstar[...,applicate+5] = Bz_starstar, Bz_starstar
    QL_starstar[...,pressure] = np.copy(QL_star[...,pressure] - np.sqrt(rhoL_star)*np.sign(primitive[...,abscissa+5])*(np.sum(QL_star[...,vels]*QL_star[...,Bfields], axis=-1) - np.sum(QL_starstar[...,vels]*QL_starstar[...,Bfields], axis=-1)))
    QR_starstar[...,pressure] = np.copy(QR_star[...,pressure] - np.sqrt(rhoR_star)*np.sign(primitive[...,abscissa+5])*(np.sum(QR_star[...,vels]*QR_star[...,Bfields], axis=-1) - np.sum(QR_starstar[...,vels]*QR_starstar[...,Bfields], axis=-1)))

    fLs_starstar, fRs_starstar = np.copy(flux_minus), np.copy(flux_plus)
    fLs_starstar, fRs_starstar = fLs_starstar + (QL_starstar - QL_star) * sL_star[...,None], fRs_starstar + (QR_starstar - QR_star) * sR_star[...,None]

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
    #abs_A = _right_eigenvectors @ _eigenvalues @ _right_eigenvectors.transpose(0,1,3,2)

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