from collections import namedtuple, defaultdict

import numpy as np

from functions import fv, constructors

##############################################################################
# Approximate linearised and non-linearised Riemann solvers
##############################################################################

# Ideally, the 4th-order averaged fluxes should be computed from the face-averaged variables
# But because the simulation is only 1D, the "normal"-Laplacian (Taylor expansion) of the face-averaged states and fluxes are zero
# Thus, the face-averaged and face-centred values are the same (<w>_i+1/2 = w_i+1/2)
# Same for the averaged and centred fluxes (<F>_i+1/2 = F_i+1/2)


# Intercell numerical fluxes between L and R interfaces based on Riemann solver
def calculate_Riemann_flux(sim_variables: namedtuple, arrays: defaultdict, **kwargs):
    Data = namedtuple('Data', ['flux', 'eigmax'])

    # Rotate grid and apply algorithm for each axis
    for axis, axes in enumerate(sim_variables.permutations):

        # Determine the eigenvalues for the computation of time stepping
        characteristics = np.linalg.eigvals(arrays[axes]['jacobian'])
        local_max_eigvals = np.max(np.abs(characteristics), axis=-1)  # Local max eigenvalue for each cell (1- or 3-Riemann invariant; shock wave or rarefaction wave)
        max_eigvals = np.max([local_max_eigvals[:-1], local_max_eigvals[1:]], axis=0)  # Local max eigenvalue between consecutive pairs of cell

        eigmax = np.max([np.max(max_eigvals), np.finfo(sim_variables.precision).eps])  # Maximum wave speed (max eigenvalue) for time evolution

        # HLL-type schemes
        if sim_variables.scheme in ["hllc", "c"]:
            if sim_variables.subgrid in ["plm", "linear", "l", "ppm", "parabolic", "p", "weno", "w"]:
                wLs, wRs = kwargs["wLs"], kwargs["wRs"]
                fLs, fRs = kwargs["fLs"], kwargs["fRs"]
            else:
                wLs, wRs = kwargs["w"][1:], kwargs["w"][:-1]
                fLs, fRs = kwargs["f"][1:], kwargs["f"][:-1]
            return Data(calculate_HLLC_flux(wLs, wRs, fRs, fLs, sim_variables), eigmax)

        # Osher-Solomon schemes
        elif sim_variables.scheme in ["os", "osher-solomon", "osher", "solomon"]:
            if sim_variables.subgrid in ["plm", "linear", "l", "ppm", "parabolic", "p", "weno", "w"]:
                qLs, qRs = kwargs["qLs"], kwargs["qRs"]
                fluxes = kwargs["fLs"] + kwargs["fRs"]
            else:
                qLs, qRs = kwargs["qS"][:-1], kwargs["qS"][1:]
                fluxes = kwargs["f"][1:] + kwargs["f"][:-1]
            return Data(calculate_DOTS_flux(qLs, qRs, fluxes, sim_variables.gamma, sim_variables.roots, sim_variables.weights), eigmax)

        # Roe-type/Lax-type schemes
        else:
            if sim_variables.subgrid in ["plm", "linear", "l", "ppm", "parabolic", "p", "weno", "w"]:
                qDiff = (kwargs["qLs"] - kwargs["qRs"]).T
                fluxes = kwargs["fLs"] + kwargs["fRs"]
                wLs, wRs = kwargs["wLs"], kwargs["wRs"]
            else:
                qDiff = (kwargs["qS"][1:] - kwargs["qS"][:-1]).T
                fluxes = kwargs["f"][1:] + kwargs["f"][:-1]
                wLs, wRs = kwargs["w"][:-1], kwargs["w"][1:]

            if sim_variables.scheme in ["entropy", "stable", "entropy-stable", "es"]:
                return Data(calculate_ES_flux(wLs, wRs, sim_variables.gamma), eigmax)
            elif sim_variables.scheme in ["lw", "lax-wendroff", "wendroff"]:
                return Data(calculate_LaxWendroff_flux(fluxes, qDiff, local_max_eigvals, kwargs["characteristics"]), eigmax)
            else:
                return Data(calculate_LaxFriedrich_flux(fluxes, qDiff, max_eigvals), eigmax)


# (Local) Lax-Friedrich scheme (1st-order; highly diffusive)
def calculate_LaxFriedrich_flux(fluxes, qDiff, eigenvalues):
    return .5 * (fluxes - ((eigenvalues * qDiff).T))


# Lax-Wendroff scheme (2nd-order, Jacobian method; contains overshoots)
def calculate_LaxWendroff_flux(fluxes, qDiff, eigenvalues, characteristics):
    # Sound speed for each cell (2-Riemann invariant; entropy wave or contact discontinuity); indexing 1 only works for hydrodynamics
    sound_speed = np.unique(characteristics, axis=1)[...,1]
    normalised_eigvals = fv.divide(sound_speed**2, eigenvalues)
    max_normalised_eigvals = np.max([normalised_eigvals[:-1], normalised_eigvals[1:]], axis=0)

    return .5 * (fluxes - ((max_normalised_eigvals * qDiff).T))
    #return .5 * ((qLs+qRs) - fv.divide(fS[1:]-fS[:-1], max_eigvals[:, np.newaxis]))


# HLLC Riemann solver [Fleischmann et al., 2020]
def calculate_HLLC_flux(wLs, wRs, fLs, fRs, sim_variables):
    gamma = sim_variables.gamma

    # The convention here is using the opposite (LR -> RL)
    rhoL, uL, pL = wRs[...,0], wRs[...,1], wRs[...,4]
    rhoR, uR, pR = wLs[...,0], wLs[...,1], wLs[...,4]
    QL, QR = fv.convert_primitive(wRs, sim_variables), fv.convert_primitive(wLs, sim_variables)

    cL, cR = np.sqrt(gamma*fv.divide(pL, rhoL)), np.sqrt(gamma*fv.divide(pR, rhoR))
    u_hat = fv.divide(uL*np.sqrt(rhoL) + uR*np.sqrt(rhoR), np.sqrt(rhoL) + np.sqrt(rhoR))
    c2_hat = fv.divide(np.sqrt(rhoL)*cL**2 + np.sqrt(rhoR)*cR**2, np.sqrt(rhoL) + np.sqrt(rhoR)) + .5*((uR-uL)**2)*fv.divide(np.sqrt(rhoL)*np.sqrt(rhoR), (np.sqrt(rhoL)+np.sqrt(rhoR))**2)

    sL, sR = np.minimum(uL-cL, u_hat-np.sqrt(c2_hat)), np.maximum(uR+cR, u_hat+np.sqrt(c2_hat))
    s_star = fv.divide(pR - pL + (rhoL*uL*(sL-uL)) - (rhoR*uR*(sR-uR)), rhoL*(sL-uL) - rhoR*(sR-uR))

    coeffL, coeffR = fv.divide(sL-uL, sL-s_star), fv.divide(sR-uR, sR-s_star)
    _QL, _QR = (coeffL.T * QL.T).T, (coeffR.T * QR.T).T

    _QL[...,1] = rhoL * coeffL * s_star
    _QR[...,1] = rhoR * coeffR * s_star
    _QL[...,4] = _QL[...,4] + coeffL*(s_star-uL)*(rhoL*s_star + fv.divide(pL, sL-uL))
    _QR[...,4] = _QR[...,4] + coeffR*(s_star-uR)*(rhoR*s_star + fv.divide(pR, sR-uR))

    flux = np.copy(fLs)
    _fLs, _fRs = fLs + (sL.T * (_QL-QL).T).T, fRs + (sR.T * (_QR-QR).T).T
    flux[(sL < 0) & (s_star >= 0)] = _fLs[(sL < 0) & (s_star >= 0)]
    flux[(sR > 0) & (s_star <= 0)] = _fRs[(sR > 0) & (s_star <= 0)]
    flux[sR <= 0] = fRs[sR <= 0]
    return flux


# Osher-Solomon(-Dumbser-Toro) Riemann solver [Dumbser & Toro, 2011]
def calculate_DOTS_flux(qLs, qRs, fluxes, gamma, roots, weights):
    # Define the path integral for the Osher-Solomon dissipation term
    arr_L, arr_R = np.repeat(qLs[None,:], len(roots), axis=0), np.repeat(qRs[None,:], len(roots), axis=0)
    psi = arr_R + (roots*(arr_L-arr_R).T).T

    # Define the right eigenvectors
    _right_eigenvectors = constructors.make_OS_right_eigenvectors(psi, gamma)

    # Generate the diagonal matrix of eigenvalues
    _lambda = np.zeros_like(_right_eigenvectors)
    rhos, vxs, pressures, B_fields = psi[...,0], psi[...,1], psi[...,4], psi[...,5:8]/np.sqrt(4*np.pi)

    # Define speeds
    sound_speed = np.sqrt(gamma * fv.divide(pressures, rhos))
    alfven_speed = np.sqrt(fv.divide(fv.norm(B_fields)**2, rhos))
    alfven_speed_x = fv.divide(B_fields[...,0], np.sqrt(rhos))
    fast_magnetosonic_wave = .5 * (sound_speed**2 + alfven_speed**2 + np.sqrt(((sound_speed**2 + alfven_speed**2)**2) - (4*(sound_speed**2)*(alfven_speed_x**2))))
    slow_magnetosonic_wave = .5 * (sound_speed**2 + alfven_speed**2 - np.sqrt(((sound_speed**2 + alfven_speed**2)**2) - (4*(sound_speed**2)*(alfven_speed_x**2))))

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

    return .5*(fluxes - (_qLs-_qRs))


# Entropy-stable flux calculation based on left and right interpolated primitive variables [Winters & Gassner, 2015; Derigs et al., 2016]
def calculate_ES_flux(wLs, wRs, gamma):
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
    # Make the right eigenvectors for each cell in each tube using the averaged primitive variables
    right_eigenvectors = constructors.make_ES_right_eigenvectors(np.array([rho_hat.T, u1_hat.T, v1_hat.T, w1_hat.T, P1_hat.T, B1_hat.T, B2_hat.T, B3_hat.T]).T, gamma)

    # Define speeds
    sound_speed = np.sqrt(gamma * fv.divide(P1_hat, rho_hat))
    alfven_speed = np.sqrt(fv.divide(fv.norm(np.array([B1_hat.T, B2_hat.T, B3_hat.T]).T)**2, rho_hat))
    alfven_speed_x = fv.divide(B1_hat, np.sqrt(rho_hat))
    fast_magnetosonic_wave = .5 * (sound_speed**2 + alfven_speed**2 + np.sqrt(((sound_speed**2 + alfven_speed**2)**2) - (4*(sound_speed**2)*(alfven_speed_x**2))))
    slow_magnetosonic_wave = .5 * (sound_speed**2 + alfven_speed**2 - np.sqrt(((sound_speed**2 + alfven_speed**2)**2) - (4*(sound_speed**2)*(alfven_speed_x**2))))

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
def calculate_Toro_flux(wLs, wRs, fLs, fRs, sim_variables):
    gamma = sim_variables.gamma

    rhoL, uL, pL = wRs[...,0], wRs[...,1], wRs[...,4]
    rhoR, uR, pR = wLs[...,0], wLs[...,1], wLs[...,4]
    QL, QR = fv.convert_primitive(wRs, sim_variables), fv.convert_primitive(wLs, sim_variables)

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