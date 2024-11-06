from collections import namedtuple, defaultdict

import numpy as np

from functions import fv, constructors

##############################################################################
# Approximate linearised and non-linearised Riemann solvers
##############################################################################

# Intercell numerical fluxes between L and R interfaces based on Riemann solver
def calculate_Riemann_flux(sim_variables: namedtuple, data: defaultdict):

    # Get the frequently used variables based on the subgrid model
    def retrieve_variables(_arrays, _subgrid, _scheme):
        if _subgrid in ["pcm", "constant", "c"]:
            _wLs, _wRs = _arrays["w"][1:], _arrays["w"][:-1]
            _qLs, _qRs = _arrays["q"][1:], _arrays["q"][:-1]
            _fLs, _fRs = _arrays["f"][1:], _arrays["f"][:-1]
        else:
            _wLs, _wRs = _arrays["wLs"], _arrays["wRs"]
            _qLs, _qRs = _arrays["qLs"], _arrays["qRs"]
            _fLs, _fRs = _arrays["fLs"], _arrays["fRs"]
        return {"wLs":_wLs, "wRs":_wRs, "qLs":_qLs, "qRs":_qRs, "fLs":_fLs, "fRs":_fRs, "wS":_arrays["wS"]}

    # Select Riemann solver based on scheme
    def run_Riemann_solver(_axis, _sim_variables, _characteristics, **kwargs):
        _wLs, _wRs, _qLs, _qRs, _fLs, _fRs, _wS = kwargs["wLs"], kwargs["wRs"], kwargs["qLs"], kwargs["qRs"], kwargs["fLs"], kwargs["fRs"], kwargs["wS"]

        # HLL-type schemes
        if _sim_variables.scheme_category == "hll":
            if _sim_variables.scheme.endswith("d"):
                return calculate_HLLD_flux(_axis, _wS, _wLs, _wRs, _qLs, _qRs, _fRs, _fLs, _sim_variables)
            else:
                return calculate_HLLC_flux(_axis, _wLs, _wRs, _qLs, _qRs, _fRs, _fLs, _sim_variables)
        # 'Complete Riemann' schemes
        elif _sim_variables.scheme_category == "complete":
            if _sim_variables.scheme.startswith("o"):
                return calculate_ES_flux(_wLs, _wRs, _sim_variables)
            else:
                return calculate_DOTS_flux(_qLs, _qRs, _fLs, _fRs, _sim_variables)
        # Roe-type/Lax-type schemes
        else:
            if _sim_variables.scheme.endswith("w"):
                return calculate_LaxWendroff_flux(_qLs, _qRs, _fLs, _fRs, _characteristics)
            else:
                return calculate_LaxFriedrich_flux(_qLs, _qRs, _fLs, _fRs, _characteristics)

    Riemann_flux = namedtuple('Riemann_flux', ['flux', 'eigmax'])
    fluxes = {}

    # Rotate grid and apply algorithm for each axis/dimension for interfaces
    axis = 0
    for axes, arrays in data.items():
        axis %= 3

        # Determine the eigenvalues for the computation of time stepping in each direction
        characteristics, eigmax = fv.compute_eigen(arrays['jacobian'])

        # Calculate the interface-averaged fluxes
        interface_variables = retrieve_variables(arrays, sim_variables.subgrid, sim_variables.scheme)
        intf_fluxes_avg = run_Riemann_solver(axis, sim_variables, characteristics, **interface_variables)

        if sim_variables.dimension == 2:
            # Compute the orthogonal L/R Riemann states and fluxes
            higher_order_interface_variables = {}
            for intf, arr in interface_variables.items():
                higher_order_interface_variables[intf] = fv.convert_mode(arr, sim_variables, "face")

            intf_fluxes_cntr = run_Riemann_solver(axis, sim_variables, characteristics, **higher_order_interface_variables)

            # Compute the higher-order fluxes
            _fluxes = fv.compute_high_approx_flux(intf_fluxes_cntr, intf_fluxes_avg, sim_variables.boundary)
        else:
            # Orthogonal Laplacian in 1D is zero
            _fluxes = intf_fluxes_avg

        fluxes[axes] = Riemann_flux(_fluxes, eigmax)
        axis += 1
    return fluxes


# (Local) Lax-Friedrich scheme (1st-order; highly diffusive)
def calculate_LaxFriedrich_flux(qLs, qRs, fLs, fRs, characteristics):
    local_max_eigvals = np.max(np.abs(characteristics), axis=-1)
    max_eigvals = np.max([local_max_eigvals[:-1], local_max_eigvals[1:]], axis=0)
    return .5*(fRs+fLs) - .5*(max_eigvals.T*(qLs-qRs).T).T


# Lax-Wendroff scheme (2nd-order, Jacobian method; contains overshoots)
def calculate_LaxWendroff_flux(qLs, qRs, fLs, fRs, characteristics):
    # Sound speed for each cell (2-Riemann invariant; entropy wave or contact discontinuity); indexing 1 only works for hydrodynamics
    sound_speed = np.unique(characteristics, axis=-1)[...,1]
    normalised_eigvals = fv.divide(sound_speed**2, np.max(np.abs(characteristics), axis=-1))
    max_normalised_eigvals = np.max([normalised_eigvals[:-1], normalised_eigvals[1:]], axis=0)

    return .5*(fRs+fLs) - .5*(max_normalised_eigvals.T*(qLs-qRs).T).T
    #return .5 * ((qLs+qRs) - fv.divide(fS[1:]-fS[:-1], max_eigvals[:, np.newaxis]))


# HLLC Riemann solver [Fleischmann et al., 2020]
def calculate_HLLC_flux(axis, wLs, wRs, qLs, qRs, fLs, fRs, sim_variables, low_mach=False):
    gamma = sim_variables.gamma

    """The interface convention here is swapped (L -> +/R, R -> -/L)
        |                        w(i-1/2)                    w(i+1/2)                       |
        |-->         i-1         <--|-->          i          <--|-->         i+1         <--|
        |   w_L(i-1)     w_R(i-1)   |   w_L(i)         w_R(i)   |   w_L(i+1)     w_R(i+1)   |
    --> |   w+(i-3/2)   w-(i-1/2)   |   w+(i-1/2)   w-(i+1/2)   |  w+(i+1/2)    w-(i+3/2)   |
    """
    rhoL, uL, pL, QL = wRs[...,0], wRs[...,axis+1], wRs[...,4], qRs
    rhoR, uR, pR, QR = wLs[...,0], wLs[...,axis+1], wLs[...,4], qLs

    # Generic HLLC solver [Toro et al., 1994]
    # Calculate Roe averages at boundary
    cL, cR = np.sqrt(gamma*fv.divide(pL, rhoL)), np.sqrt(gamma*fv.divide(pR, rhoR))
    u_hat = fv.divide(uL*np.sqrt(rhoL) + uR*np.sqrt(rhoR), np.sqrt(rhoL) + np.sqrt(rhoR))
    c2_hat = fv.divide(np.sqrt(rhoL)*cL**2 + np.sqrt(rhoR)*cR**2, np.sqrt(rhoL) + np.sqrt(rhoR)) + .5*((uR-uL)**2)*fv.divide(np.sqrt(rhoL)*np.sqrt(rhoR), (np.sqrt(rhoL)+np.sqrt(rhoR))**2)

    # Calculate the non-linear signal speeds
    sL, sR = np.minimum(uL-cL, u_hat-np.sqrt(c2_hat)), np.maximum(uR+cR, u_hat+np.sqrt(c2_hat))
    sM = fv.divide(pR - pL + (rhoL*uL*(sL-uL)) - (rhoR*uR*(sR-uR)), rhoL*(sL-uL) - rhoR*(sR-uR))

    # Modification to HLLC solver for low Mach shocks [Fleischmann et al., 2020]
    if low_mach:
        Ma_local = np.maximum(np.abs(fv.divide(uL,cL)), np.abs(fv.divide(uR,cR)))
        phi = np.sin(.5 * np.pi * np.minimum(1, Ma_local/.1))
        sL = np.copy((phi.T * sL.T).T)
        sR = np.copy((phi.T * sR.T).T)

    # Calculate the intermediate states
    coeffL, coeffR = fv.divide(sL-uL, sL-sM), fv.divide(sR-uR, sR-sM)
    QL_star, QR_star = (coeffL.T * QL.T).T, (coeffR.T * QR.T).T
    QL_star[...,1] = rhoL * coeffL * sM
    QR_star[...,1] = rhoR * coeffR * sM
    QL_star[...,4] = QL_star[...,4] + coeffL*(sM-uL)*(rhoL*sM + fv.divide(pL, sL-uL))
    QR_star[...,4] = QR_star[...,4] + coeffR*(sM-uR)*(rhoR*sM + fv.divide(pR, sR-uR))

    # Calculate the flux
    flux = np.copy(fLs)
    fLs_star, fRs_star = fLs + (sL.T * (QL_star-QL).T).T, fRs + (sR.T * (QR_star-QR).T).T
    flux[(sL <= 0) & (0 < sM)] = fLs_star[(sL <= 0) & (0 < sM)]
    flux[(sM <= 0) & (0 <= sR)] = fRs_star[(sM <= 0) & (0 <= sR)]
    flux[sR < 0] = fRs[sR < 0]
    return flux


# HLLD Riemann solver [Miyoshi & Kusano, 2005]
def calculate_HLLD_flux(axis, wS, wLs, wRs, qLs, qRs, fLs, fRs, sim_variables):

    def make_speeds(_wF, _gamma):
        _rhos, _pressures, _B_fields = _wF[...,0], _wF[...,4], _wF[...,5:8]

        _sound_speed = np.sqrt(fv.divide(_gamma*_pressures, _rhos))
        _alfven_speed = fv.divide(fv.norm(_B_fields), np.sqrt(_rhos))
        _alfven_speed_x = fv.divide(_B_fields[...,0], np.sqrt(_rhos))
        _fast_magnetosonic = np.sqrt(.5 * (_sound_speed**2 + _alfven_speed**2 + np.sqrt(((_sound_speed**2 + _alfven_speed**2)**2) - (4*(_sound_speed**2)*(_alfven_speed_x**2)))))
        #_slow_magnetosonic = np.sqrt(.5 * (_sound_speed**2 + _alfven_speed**2 - np.sqrt(((_sound_speed**2 + _alfven_speed**2)**2) - (4*(_sound_speed**2)*(_alfven_speed_x**2)))))

        return _fast_magnetosonic

    gamma = sim_variables.gamma

    """The interface convention here is swapped (L -> +/R, R -> -/L)
        |                        w(i-1/2)                    w(i+1/2)                       |
        |-->         i-1         <--|-->          i          <--|-->         i+1         <--|
        |   w_L(i-1)     w_R(i-1)   |   w_L(i)         w_R(i)   |   w_L(i+1)     w_R(i+1)   |
    --> |   w+(i-3/2)   w-(i-1/2)   |   w+(i-1/2)   w-(i+1/2)   |  w+(i+1/2)    w-(i+3/2)   |
    """
    wS = fv.add_boundary(wS, sim_variables.boundary)[1:]
    rhoL, vecL, pL, BL, QL = wRs[...,0], wRs[...,1:4], wRs[...,4], wRs[...,5:8], qRs
    rhoR, vecR, pR, BR, QR = wLs[...,0], wLs[...,1:4], wLs[...,4], wLs[...,5:8], qLs

    cafL, cafR = make_speeds(wLs, gamma), make_speeds(wRs, gamma)

    sL, sR = np.minimum(vecL[...,axis], vecR[...,axis]) - np.maximum(cafL, cafR), np.minimum(vecL[...,axis], vecR[...,axis]) + np.maximum(cafL, cafR)
    sM = fv.divide(pL - pR + rhoR*vecR[...,axis]*(sR-vecR[...,axis]) - rhoL*vecL[...,axis]*(sL-vecL[...,axis]) + .5*(fv.norm(BL)**2 - fv.norm(BR)**2), rhoR*(sR-vecR[...,axis]) - rhoL*(sL-vecL[...,axis]))

    # Calculate the star states
    rhoL_star, rhoR_star = rhoL * fv.divide(sL-vecL[...,axis], sL-sM), rhoR * fv.divide(sR-vecR[...,axis], sR-sM)
    sL_star, sR_star = sM - fv.divide(BL[...,axis], np.sqrt(rhoL_star)), sM - fv.divide(BR[...,axis], np.sqrt(rhoR_star))

    p_star = fv.divide(rhoR*(pL+.5*fv.norm(BL)**2)*(sR-vecR[...,axis]) - rhoL*(pR+.5*fv.norm(BR)**2)*(sL-vecL[...,axis]) + rhoL*rhoR*(sR-vecR[...,axis])*(sL-vecL[...,axis])*(vecR[...,axis]-vecL[...,axis]), rhoR*(sR-vecR[...,axis]) - rhoL*(sL-vecL[...,axis]))
    vyL_star = vecL[...,(axis+1)%3] - wS[...,(axis+0)%3+5]*BL[...,(axis+1)%3]*fv.divide(sM-vecL[...,axis], rhoL*(sL-vecL[...,axis])*(sL-sM) - wS[...,(axis+0)%3+5]**2)
    vyR_star = vecR[...,(axis+1)%3] - wS[...,(axis+0)%3+5]*BR[...,(axis+1)%3]*fv.divide(sM-vecR[...,axis], rhoR*(sR-vecR[...,axis])*(sR-sM) - wS[...,(axis+0)%3+5]**2)
    vzL_star = vecL[...,(axis+2)%3] - wS[...,(axis+0)%3+5]*BL[...,(axis+2)%3]*fv.divide(sM-vecL[...,axis], rhoL*(sL-vecL[...,axis])*(sL-sM) - wS[...,(axis+0)%3+5]**2)
    vzR_star = vecR[...,(axis+2)%3] - wS[...,(axis+0)%3+5]*BR[...,(axis+2)%3]*fv.divide(sM-vecR[...,axis], rhoR*(sR-vecR[...,axis])*(sR-sM) - wS[...,(axis+0)%3+5]**2)
    ByL_star = BL[...,(axis+1)%3] * fv.divide(rhoL*(sL-vecL[...,axis])**2 - wS[...,(axis+0)%3+5]**2, rhoL*(sL-vecL[...,axis])*(sL-sM) - wS[...,(axis+0)%3+5]**2)
    ByR_star = BR[...,(axis+1)%3] * fv.divide(rhoR*(sR-vecR[...,axis])**2 - wS[...,(axis+0)%3+5]**2, rhoR*(sR-vecR[...,axis])*(sR-sM) - wS[...,(axis+0)%3+5]**2)
    BzL_star = BL[...,(axis+2)%3] * fv.divide(rhoL*(sL-vecL[...,axis])**2 - wS[...,(axis+0)%3+5]**2, rhoL*(sL-vecL[...,axis])*(sL-sM) - wS[...,(axis+0)%3+5]**2)
    BzR_star = BR[...,(axis+2)%3] * fv.divide(rhoR*(sR-vecR[...,axis])**2 - wS[...,(axis+0)%3+5]**2, rhoR*(sR-vecR[...,axis])*(sR-sM) - wS[...,(axis+0)%3+5]**2)

    QL_star, QR_star = np.zeros_like(QL), np.zeros_like(QR)
    QL_star[...,0], QR_star[...,0] = rhoL_star, rhoR_star
    QL_star[...,(axis+0)%3+1], QR_star[...,(axis+0)%3+1] = rhoL * sM, rhoR * sM
    QL_star[...,(axis+1)%3+1], QR_star[...,(axis+1)%3+1] = rhoL * vyL_star, rhoR * vyR_star
    QL_star[...,(axis+2)%3+1], QR_star[...,(axis+2)%3+1] = rhoL * vzL_star, rhoR * vzR_star
    QL_star[...,(axis+1)%3+5], QR_star[...,(axis+1)%3+5] = ByL_star, ByR_star
    QL_star[...,(axis+2)%3+5], QR_star[...,(axis+2)%3+5] = BzL_star, BzR_star
    QL_star[...,4] = np.copy(fv.divide(QL[...,4]*(sL-vecL[...,axis]) - vecL[...,axis]*(pL+.5*fv.norm(BL)**2) + p_star*sM + wS[...,(axis+0)%3+5]*(np.sum(vecL*BL, axis=-1) - np.sum(QL_star[...,1:4]*QL_star[...,5:8], axis=-1)), sL-sM))
    QR_star[...,4] = np.copy(fv.divide(QR[...,4]*(sR-vecR[...,axis]) - vecR[...,axis]*(pR+.5*fv.norm(BR)**2) + p_star*sM + wS[...,(axis+0)%3+5]*(np.sum(vecR*BR, axis=-1) - np.sum(QR_star[...,1:4]*QR_star[...,5:8], axis=-1)), sR-sM))

    fLs_star, fRs_star = np.copy(fLs), np.copy(fRs)
    fLs_star, fRs_star = fLs_star + (sL.T * (QL_star - QL).T).T, fRs_star + (sR.T * (QR_star - QR).T).T

    # Calculate the double-star states
    vy_starstar = fv.divide(vyL_star*np.sqrt(rhoL_star) + vyR_star*np.sqrt(rhoR_star) + np.sign(wS[...,(axis+0)%3+5])*(ByR_star-ByL_star), np.sqrt(rhoL_star) + np.sqrt(rhoR_star))
    vz_starstar = fv.divide(vzL_star*np.sqrt(rhoL_star) + vzR_star*np.sqrt(rhoR_star) + np.sign(wS[...,(axis+0)%3+5])*(BzR_star-BzL_star), np.sqrt(rhoL_star) + np.sqrt(rhoR_star))
    By_starstar = fv.divide(ByR_star*np.sqrt(rhoL_star) + ByL_star*np.sqrt(rhoR_star) + np.sign(wS[...,(axis+0)%3+5])*(vyR_star-vyL_star)*np.sqrt(rhoL_star*rhoR_star), np.sqrt(rhoL_star) + np.sqrt(rhoR_star))
    Bz_starstar = fv.divide(BzR_star*np.sqrt(rhoL_star) + BzL_star*np.sqrt(rhoR_star) + np.sign(wS[...,(axis+0)%3+5])*(vzR_star-vzL_star)*np.sqrt(rhoL_star*rhoR_star), np.sqrt(rhoL_star) + np.sqrt(rhoR_star))

    QL_starstar, QR_starstar = np.zeros_like(QL), np.zeros_like(QR)
    QL_starstar[...,0], QR_starstar[...,0] = rhoL_star, rhoR_star
    QL_starstar[...,(axis+0)%3+1], QR_starstar[...,(axis+0)%3+1] = sM, sM
    QL_starstar[...,(axis+1)%3+1], QR_starstar[...,(axis+1)%3+1] = vy_starstar, vy_starstar
    QL_starstar[...,(axis+2)%3+1], QR_starstar[...,(axis+2)%3+1] = vz_starstar, vz_starstar
    QL_starstar[...,(axis+1)%3+5], QR_starstar[...,(axis+1)%3+5] = By_starstar, By_starstar
    QL_starstar[...,(axis+2)%3+5], QR_starstar[...,(axis+2)%3+5] = Bz_starstar, Bz_starstar
    QL_starstar[...,4] = np.copy(QL_star[...,4] - np.sqrt(rhoL_star)*np.sign(wS[...,(axis+0)%3+5])*(np.sum(QL_star[...,1:4]*QL_star[...,5:8], axis=-1) - np.sum(QL_starstar[...,1:4]*QL_starstar[...,5:8], axis=-1)))
    QR_starstar[...,4] = np.copy(QR_star[...,4] - np.sqrt(rhoR_star)*np.sign(wS[...,(axis+0)%3+5])*(np.sum(QR_star[...,1:4]*QR_star[...,5:8], axis=-1) - np.sum(QR_starstar[...,1:4]*QR_starstar[...,5:8], axis=-1)))

    fLs_starstar, fRs_starstar = np.copy(fLs), np.copy(fRs)
    fLs_starstar, fRs_starstar = fLs_starstar + (sL_star.T * (QL_starstar - QL_star).T).T, fRs_starstar + (sR_star.T * (QR_starstar - QR_star).T).T

    # Compute the flux
    flux = np.copy(fLs)
    flux[(sL <= 0) & (0 < sL_star)] = fLs_star[(sL <= 0) & (0 < sL_star)]
    flux[(sL_star <= 0) & (0 < sM)] = fLs_starstar[(sL_star <= 0) & (0 < sM)]
    flux[(sM <= 0) & (0 < sR_star)] = fRs_starstar[(sM <= 0) & (0 < sR_star)]
    flux[(sR_star <= 0) & (0 <= sR)] = fRs_star[(sR_star <= 0) & (0 <= sR)]
    flux[sR < 0] = fRs[sR < 0]
    return flux


# Osher-Solomon(-Dumbser-Toro) Riemann solver [Dumbser & Toro, 2011]
def calculate_DOTS_flux(qLs, qRs, fLs, fRs, sim_variables):
    gamma, roots, weights = sim_variables.gamma, sim_variables.roots, sim_variables.weights

    # Define the path integral for the Osher-Solomon dissipation term
    arr_L, arr_R = np.repeat(qLs[None,:], len(roots), axis=0), np.repeat(qRs[None,:], len(roots), axis=0)
    psi = arr_R + (roots*(arr_L-arr_R).T).T

    # Define the right eigenvectors
    _right_eigenvectors = constructors.make_OS_right_eigenvectors(psi, gamma)

    # Generate the diagonal matrix of eigenvalues
    _lambda = np.zeros_like(_right_eigenvectors)
    rhos, vxs, pressures, B_fields = psi[...,0], psi[...,1], psi[...,4], psi[...,5:8]

    # Define speeds
    sound_speed = np.sqrt(gamma * fv.divide(pressures, rhos))
    alfven_speed = fv.divide(fv.norm(B_fields), np.sqrt(rhos))
    alfven_speed_x = fv.divide(B_fields[...,0], np.sqrt(rhos))
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
    _qLs = jacobian @ qLs[..., np.newaxis]
    _qRs = jacobian @ qRs[..., np.newaxis]
    _qLs = _qLs.reshape(len(_qLs), len(_qLs[0]))
    _qRs = _qRs.reshape(len(_qRs), len(_qRs[0]))

    return .5*(fLs+fRs) - .5*(_qLs-_qRs)


# Entropy-stable flux calculation based on left and right interpolated primitive variables [Winters & Gassner, 2015; Derigs et al., 2016]
def calculate_ES_flux(wLs, wRs, sim_variables):
    gamma = sim_variables.gamma

    # To construct the entropy-stable flux, 2 components are needed:
    # the entropy-conserving flux component, and the dissipation term to make the flux entropy-stable

    # Entropy-conserving flux section [Winters & Gassner, 2015]
    ec_flux = np.zeros_like(wLs)

    # Compute arithmetic mean
    def arith_mean(term):
        return .5 * (term[0] + term[1])

    # Stable numerical procedure for computing logarithmic mean [Ismail & Roe, 2009]
    def lon(term):
        L, R = term
        zeta = np.divide(L, R, out=np.zeros_like(L), where=R!=0)
        f = np.divide(zeta-1, zeta+1, out=np.zeros_like(zeta), where=(zeta+1)!=0)
        u = f*f

        if (u < 1e-2).any():
            F = 1 + u/3 + u*u/5 + u*u*u/7
        else:
            F = np.log(zeta)/2/f
        return (L+R)/(2*F)

    # Define frequently used terms
    z1 = np.array([np.sqrt(fv.divide(wLs[...,0], wLs[...,4])), np.sqrt(fv.divide(wRs[...,0], wRs[...,4]))])
    z5 = np.array([np.sqrt(wLs[...,0]*wLs[...,4]), np.sqrt(wRs[...,0]*wRs[...,4])])
    vx, vy, vz = np.array([wLs[...,1], wRs[...,1]]), np.array([wLs[...,2], wRs[...,2]]), np.array([wLs[...,3], wRs[...,3]])

    # Compute the averages
    rho_hat = arith_mean(z1) * lon(z5)
    P1_hat = fv.divide(arith_mean(z5), arith_mean(z1))
    P2_hat = ((gamma+1)/(2*gamma))*(fv.divide(lon(z5), lon(z1))) + ((gamma-1)/(2*gamma))*(fv.divide(arith_mean(z5), arith_mean(z1)))
    u1_hat = fv.divide(arith_mean(vx*z1), arith_mean(z1))
    v1_hat = fv.divide(arith_mean(vy*z1), arith_mean(z1))
    w1_hat = fv.divide(arith_mean(vz*z1), arith_mean(z1))
    u2_hat = fv.divide(arith_mean(vx*(z1**2)), arith_mean(z1**2))
    v2_hat = fv.divide(arith_mean(vy*(z1**2)), arith_mean(z1**2))
    w2_hat = fv.divide(arith_mean(vz*(z1**2)), arith_mean(z1**2))
    B1_hat = arith_mean(np.array([wLs[...,5], wRs[...,5]]))
    B1_dot = arith_mean(np.array([wLs[...,5]**2, wRs[...,5]**2]))
    B2_hat = arith_mean(np.array([wLs[...,6], wRs[...,6]]))
    B2_dot = arith_mean(np.array([wLs[...,6]**2, wRs[...,6]**2]))
    B3_hat = arith_mean(np.array([wLs[...,7], wRs[...,7]]))
    B3_dot = arith_mean(np.array([wLs[...,7]**2, wRs[...,7]**2]))
    B1B2 = arith_mean(np.array([wLs[...,5]*wLs[...,6], wRs[...,5]*wRs[...,6]]))
    B1B3 = arith_mean(np.array([wLs[...,5]*wLs[...,7], wRs[...,5]*wRs[...,7]]))

    # Update the entropy-conserving flux vector; suitable for smooth solutions
    ec_flux[...,0] = rho_hat * u1_hat
    ec_flux[...,1] = P1_hat + rho_hat*u1_hat**2 + .5*(B1_dot+B2_dot+B3_dot) - B1_dot
    ec_flux[...,2] = rho_hat*u1_hat*v1_hat - B1B2
    ec_flux[...,3] = rho_hat*u1_hat*w1_hat - B1B3
    ec_flux[...,4] = (gamma/(gamma-1))*u1_hat*P2_hat + .5*rho_hat*u1_hat*(u1_hat**2 + v1_hat**2 + w1_hat**2) + u2_hat*(B2_hat**2 + B3_hat**2) - B1_hat*(v2_hat*B2_hat + w2_hat*B3_hat)
    ec_flux[...,6] = u2_hat*B2_hat - v2_hat*B1_hat
    ec_flux[...,7] = u2_hat*B3_hat - w2_hat*B1_hat


    # Entropy-stable flux with dissipation term section [Derigs et al., 2016]
    # Make the right eigenvectors for each cell in each grid using the averaged primitive variables
    right_eigenvectors = constructors.make_ES_right_eigenvectors(np.array([rho_hat.T, u1_hat.T, v1_hat.T, w1_hat.T, P1_hat.T, B1_hat.T, B2_hat.T, B3_hat.T]).T, gamma)

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
    entropy_vector = np.zeros_like(wLs)
    entropy_vector[...,0] = ((gamma-np.log(wRs[...,4]*wRs[...,0]**-gamma))/(gamma-1) - fv.divide(.5*wRs[...,0]*fv.norm(wRs[...,1:4])**2, wRs[...,4])) - ((gamma-np.log(wLs[...,4]*wLs[...,0]**-gamma))/(gamma-1) - fv.divide(.5*wLs[...,0]*fv.norm(wLs[...,1:4])**2, wLs[...,4]))
    entropy_vector[...,4] = fv.divide(wLs[...,0], wLs[...,4]) - fv.divide(wRs[...,0], wRs[...,4])
    entropy_vector[...,1:4] = fv.divide((wRs[...,0].T * wRs[...,1:4].T), wRs[...,4].T).T - fv.divide((wLs[...,0].T * wLs[...,1:4].T), wLs[...,4].T).T
    entropy_vector[...,5:8] = fv.divide((wRs[...,0].T * wRs[...,5:8].T), wRs[...,4].T).T - fv.divide((wLs[...,0].T * wLs[...,5:8].T), wLs[...,4].T).T
    entropy_vector = -entropy_vector

    # Compute the hydrid entropy stabilisation
    Epsilon = np.sqrt(np.abs(fv.divide(wLs[...,4]-wRs[...,4], wLs[...,4]+wRs[...,4])))
    eigenvalues = ((1-Epsilon).T * roe_eigenvalues.T).T + (Epsilon.T * lff_eigenvalues.T).T

    # Calculate the dissipation term
    abs_A = right_eigenvectors @ eigenvalues @ right_eigenvectors.transpose(0,2,1)
    _dissipation = abs_A @ entropy_vector[..., np.newaxis]
    dissipation = _dissipation.reshape(len(entropy_vector), len(entropy_vector[0]))

    return ec_flux + .5*dissipation


"""# HLLC Riemann solver [Toro, 2019]
def calculate_Toro_flux(wLs, wRs, qLs, qRs, fLs, fRs, sim_variables):
    gamma = sim_variables.gamma

    rhoL, uL, pL = wRs[...,0], wRs[...,1], wRs[...,4]
    rhoR, uR, pR = wLs[...,0], wLs[...,1], wLs[...,4]
    QL, QR = qRs, qLs

    zeta = (gamma-1)/(2*gamma)
    aL, aR = np.sqrt(gamma*fv.divide(pL, rhoL)), np.sqrt(gamma*fv.divide(pR, rhoR))
    two_rarefaction_approx = fv.divide(aL+aR-(((gamma-1)/2)*(uR-uL)), fv.divide(aL, pL**zeta)+fv.divide(aR, pR**zeta))**(1/zeta)

    qL, qR = np.ones_like(pL), np.ones_like(pR)
    _qL, _qR = np.sqrt(1 + (((gamma+1)/(2*gamma))*(fv.divide(two_rarefaction_approx, pL)-1))), np.sqrt(1 + (((gamma+1)/(2*gamma))*(fv.divide(two_rarefaction_approx, pR)-1)))
    qL[two_rarefaction_approx > pL] = _qL[two_rarefaction_approx > pL]
    qR[two_rarefaction_approx > pR] = _qR[two_rarefaction_approx > pR]

    sL, sR = uL - aL*qL, uR - aR*qR
    s_star = fv.divide(pR - pL + (rhoL*uL*(sL-uL)) - (rhoR*uR*(sR-uR)), rhoL*(sL-uL) - rhoR*(sR-uR))

    coeffL, coeffR = fv.divide(sL-uL, sL-s_star), fv.divide(sR-uR, sR-s_star)
    _QL, _QR = np.copy((coeffL.T * QL.T).T), np.copy((coeffR.T * QR.T).T)

    _QL[...,1] = rhoL * coeffL * s_star
    _QR[...,1] = rhoR * coeffR * s_star
    _pL, _pR = np.copy(_QL[...,4]), np.copy(_QR[...,4])
    _BL, _BR = np.copy(_QL[...,5:8]), np.copy(_QR[...,5:8])
    _QL[...,4] = rhoL * coeffL * (fv.divide(_pL, rhoL) + ((s_star-uL)*(s_star+fv.divide(pL, rhoL*(sL-uL)))))
    _QR[...,4] = rhoR * coeffR * (fv.divide(_pR, rhoR) + ((s_star-uR)*(s_star+fv.divide(pR, rhoR*(sR-uR)))))
    _QL[...,5:8] = ((rhoL * coeffL).T * _BL.T).T
    _QR[...,5:8] = ((rhoR * coeffR).T * _BR.T).T

    flux = np.copy(fLs)
    _fLs, _fRs = np.copy(fLs + (sL.T * (_QL-QL).T).T), np.copy(fRs + (sR.T * (_QR-QR).T).T)
    flux[(sL <= 0) & (0 <= s_star)] = _fLs[(sL <= 0) & (0 <= s_star)]
    flux[(s_star <= 0) & (0 <= sR)] = _fRs[(s_star <= 0) & (0 <= sR)]
    flux[0 >= sR] = fRs[0 >= sR]
    return flux"""