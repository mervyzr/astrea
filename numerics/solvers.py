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
def calculateRiemannFlux(simVariables: namedtuple, arrays: defaultdict, **kwargs):
    Data = namedtuple('Data', ['flux', 'eigmax'])

    # Determine the eigenvalues for the computation of time stepping
    eigvals = np.max(np.abs(kwargs["characteristics"]), axis=1)  # Local max eigenvalue for each cell (1- or 3-Riemann invariant; shock wave or rarefaction wave)
    maxEigvals = np.max([eigvals[:-1], eigvals[1:]], axis=0)  # Local max eigenvalue between consecutive pairs of cell

    eigmax = np.max([np.max(maxEigvals), np.finfo(simVariables.precision).eps])  # Maximum wave speed (max eigenvalue) for time evolution

    # HLL-type schemes
    if simVariables.scheme in ["hllc", "c"]:
        if simVariables.subgrid in ["plm", "linear", "l", "ppm", "parabolic", "p", "weno", "w"]:
            wLs, wRs = kwargs["wLs"], kwargs["wRs"]
            fLs, fRs = kwargs["fLs"], kwargs["fRs"]
        else:
            wLs, wRs = kwargs["w"][1:], kwargs["w"][:-1]
            fLs, fRs = kwargs["f"][1:], kwargs["f"][:-1]
        return Data(calculateHLLCFlux(wLs, wRs, fRs, fLs, simVariables), eigmax)

    # Osher-Solomon schemes
    elif simVariables.scheme in ["os", "osher-solomon", "osher", "solomon"]:
        if simVariables.subgrid in ["plm", "linear", "l", "ppm", "parabolic", "p", "weno", "w"]:
            qS = [kwargs["qLs"], kwargs["qRs"]]
            fluxes = kwargs["fLs"] + kwargs["fRs"]
        else:
            qS = [kwargs["qS"][:-1], kwargs["qS"][1:]]
            fluxes = kwargs["f"][1:] + kwargs["f"][:-1]
        return Data(calculateDOTSFlux(qS, fluxes, simVariables.gamma, simVariables.roots, simVariables.weights), eigmax)

    # Roe-type/Lax-type schemes
    else:
        if simVariables.subgrid in ["plm", "linear", "l", "ppm", "parabolic", "p", "weno", "w"]:
            qDiff = (kwargs["qLs"] - kwargs["qRs"]).T
            fluxes = kwargs["fLs"] + kwargs["fRs"]
            wLs, wRs = kwargs["wLs"], kwargs["wRs"]
        else:
            qDiff = (kwargs["qS"][1:] - kwargs["qS"][:-1]).T
            fluxes = kwargs["f"][1:] + kwargs["f"][:-1]
            wLs, wRs = kwargs["w"][:-1], kwargs["w"][1:]

        if simVariables.scheme in ["entropy", "stable", "entropy-stable", "es"]:
            return Data(calculateESFlux([wLs, wRs], simVariables.gamma), eigmax)
        elif simVariables.scheme in ["lw", "lax-wendroff", "wendroff"]:
            return Data(calculateLaxWendroffFlux(fluxes, qDiff, eigvals, kwargs["characteristics"]), eigmax)
        else:
            return Data(calculateLaxFriedrichFlux(fluxes, qDiff, maxEigvals), eigmax)


# (Local) Lax-Friedrich scheme (1st-order; highly diffusive)
def calculateLaxFriedrichFlux(fluxes, qDiff, eigenvalues):
    return .5 * (fluxes - ((eigenvalues * qDiff).T))


# Lax-Wendroff scheme (2nd-order, Jacobian method; contains overshoots)
def calculateLaxWendroffFlux(fluxes, qDiff, eigenvalues, characteristics):
    # Sound speed for each cell (2-Riemann invariant; entropy wave or contact discontinuity); indexing 1 only works for hydrodynamics
    soundSpeed = np.unique(characteristics, axis=1)[...,1]
    normalisedEigvals = fv.divide(soundSpeed**2, eigenvalues)
    maxNormalisedEigvals = np.max([normalisedEigvals[:-1], normalisedEigvals[1:]], axis=0)

    return .5 * (fluxes - ((maxNormalisedEigvals * qDiff).T))
    #return .5 * ((qLs+qRs) - fv.divide(fS[1:]-fS[:-1], maxEigvals[:, np.newaxis]))


# HLLC Riemann solver [Fleischmann et al., 2020]
def calculateHLLCFlux(wLs, wRs, fLs, fRs, simVariables):
    gamma = simVariables.gamma

    # The convention here is using the opposite (LR -> RL)
    rhoL, uL, pL = wRs[...,0], wRs[...,1], wRs[...,4]
    rhoR, uR, pR = wLs[...,0], wLs[...,1], wLs[...,4]
    QL, QR = fv.convertPrimitive(wRs, simVariables), fv.convertPrimitive(wLs, simVariables)

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
def calculateDOTSFlux(qS, fluxes, gamma, roots, weights):
    qLs, qRs = qS

    # Define the path integral for the Osher-Solomon dissipation term
    arr_L, arr_R = np.repeat(qLs[None,:], len(roots), axis=0), np.repeat(qRs[None,:], len(roots), axis=0)
    psi = arr_R + (roots*(arr_L-arr_R).T).T

    # Define the right eigenvectors
    _rightEigenvectors = constructors.makeOSRightEigenvectors(psi, gamma)

    # Generate the diagonal matrix of eigenvalues
    _lambda = np.zeros_like(_rightEigenvectors)
    rhos, vxs, pressures, Bfields = psi[...,0], psi[...,1], psi[...,4], psi[...,5:8]/np.sqrt(4*np.pi)

    # Define speeds
    soundSpeed = np.sqrt(gamma * fv.divide(pressures, rhos))
    alfvenSpeed = np.sqrt(fv.divide(fv.norm(Bfields)**2, rhos))
    alfvenSpeedx = fv.divide(Bfields[...,0], np.sqrt(rhos))
    fastMagnetosonicWave = .5 * (soundSpeed**2 + alfvenSpeed**2 + np.sqrt(((soundSpeed**2 + alfvenSpeed**2)**2) - (4*(soundSpeed**2)*(alfvenSpeedx**2))))
    slowMagnetosonicWave = .5 * (soundSpeed**2 + alfvenSpeed**2 - np.sqrt(((soundSpeed**2 + alfvenSpeed**2)**2) - (4*(soundSpeed**2)*(alfvenSpeedx**2))))

    # Compute the diagonal matrix of eigenvalues
    _lambda[...,0,0] = vxs - fastMagnetosonicWave
    _lambda[...,1,1] = vxs - alfvenSpeedx
    _lambda[...,2,2] = vxs - slowMagnetosonicWave
    _lambda[...,3,3] = vxs
    _lambda[...,4,4] = vxs
    _lambda[...,5,5] = vxs + slowMagnetosonicWave
    _lambda[...,6,6] = vxs + alfvenSpeedx
    _lambda[...,7,7] = vxs + fastMagnetosonicWave
    _eigenvalues = np.abs(_lambda)

    # Compute the absolute value of the Jacobian
    absA = _rightEigenvectors @ _eigenvalues @ np.linalg.pinv(_rightEigenvectors)
    #absA = _rightEigenvectors @ _eigenvalues @ _rightEigenvectors.transpose(0,1,3,2)

    # Compute the Dumbser-Toro Jacobian with the Gauss-Legendre quadrature
    jacobian = np.sum((weights*absA.T).T, axis=0)

    # Compute the Osher-Solomon dissipation term
    _qLs = jacobian @ qLs[..., np.newaxis]
    _qRs = jacobian @ qRs[..., np.newaxis]
    _qLs = _qLs.reshape(len(_qLs), len(_qLs[0]))
    _qRs = _qRs.reshape(len(_qRs), len(_qRs[0]))

    return .5*(fluxes - (_qLs-_qRs))


# Entropy-stable flux calculation based on left and right interpolated primitive variables [Winters & Gassner, 2015; Derigs et al., 2016]
def calculateESFlux(wS, gamma):
    wLs, wRs = wS

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
    rightEigenvectors = constructors.makeESRightEigenvectors(np.array([rho_hat.T, u1_hat.T, v1_hat.T, w1_hat.T, P1_hat.T, B1_hat.T, B2_hat.T, B3_hat.T]).T, gamma)

    # Define speeds
    soundSpeed = np.sqrt(gamma * fv.divide(P1_hat, rho_hat))
    alfvenSpeed = np.sqrt(fv.divide(fv.norm(np.array([B1_hat.T, B2_hat.T, B3_hat.T]).T)**2, rho_hat))
    alfvenSpeedx = fv.divide(B1_hat, np.sqrt(rho_hat))
    fastMagnetosonicWave = .5 * (soundSpeed**2 + alfvenSpeed**2 + np.sqrt(((soundSpeed**2 + alfvenSpeed**2)**2) - (4*(soundSpeed**2)*(alfvenSpeedx**2))))
    slowMagnetosonicWave = .5 * (soundSpeed**2 + alfvenSpeed**2 - np.sqrt(((soundSpeed**2 + alfvenSpeed**2)**2) - (4*(soundSpeed**2)*(alfvenSpeedx**2))))

    # Compute the diagonal matrix of eigenvalues for Roe
    roeEigenvalues = np.zeros_like(rightEigenvectors)
    roeEigenvalues[...,0,0] = u1_hat + fastMagnetosonicWave
    roeEigenvalues[...,1,1] = u1_hat + alfvenSpeedx
    roeEigenvalues[...,2,2] = u1_hat + slowMagnetosonicWave
    roeEigenvalues[...,3,3] = u1_hat
    roeEigenvalues[...,4,4] = u1_hat
    roeEigenvalues[...,5,5] = u1_hat - slowMagnetosonicWave
    roeEigenvalues[...,6,6] = u1_hat - alfvenSpeedx
    roeEigenvalues[...,7,7] = u1_hat - fastMagnetosonicWave
    roeEigenvalues = np.abs(roeEigenvalues)

    # Compute the diagonal matrix of eigenvalues for Local Lax-Friedrich
    lffEigenvalues = np.zeros_like(rightEigenvectors)
    i, j = np.diag_indices(lffEigenvalues.shape[-1])
    max_values = np.maximum.reduce([np.abs(u1_hat+fastMagnetosonicWave), np.abs(u1_hat+alfvenSpeedx), np.abs(u1_hat+slowMagnetosonicWave), np.abs(u1_hat), np.abs(u1_hat-slowMagnetosonicWave), np.abs(u1_hat-alfvenSpeedx), np.abs(u1_hat-fastMagnetosonicWave)])
    lffEigenvalues[..., i,j] = max_values[..., None]

    # Define the jump in the entropy vector
    entropyVector = np.zeros_like(wLs)
    entropyVector[...,0] = ((gamma-np.log(wRs[...,4]*wRs[...,0]**-gamma))/(gamma-1) - fv.divide(.5*wRs[...,0]*fv.norm(wRs[...,1:4])**2, wRs[...,4])) - ((gamma-np.log(wLs[...,4]*wLs[...,0]**-gamma))/(gamma-1) - fv.divide(.5*wLs[...,0]*fv.norm(wLs[...,1:4])**2, wLs[...,4]))
    entropyVector[...,4] = fv.divide(wLs[...,0], wLs[...,4]) - fv.divide(wRs[...,0], wRs[...,4])
    entropyVector[...,1:4] = fv.divide((wRs[...,0].T * wRs[...,1:4].T), wRs[...,4].T).T - fv.divide((wLs[...,0].T * wLs[...,1:4].T), wLs[...,4].T).T
    entropyVector[...,5:8] = fv.divide((wRs[...,0].T * wRs[...,5:8].T), wRs[...,4].T).T - fv.divide((wLs[...,0].T * wLs[...,5:8].T), wLs[...,4].T).T
    entropyVector = -entropyVector

    # Compute the hydrid entropy stabilisation
    Epsilon = np.sqrt(np.abs(fv.divide(wLs[...,4]-wRs[...,4], wLs[...,4]+wRs[...,4])))
    eigenvalues = ((1-Epsilon).T * roeEigenvalues.T).T + (Epsilon.T * lffEigenvalues.T).T

    # Calculate the dissipation term
    absA = rightEigenvectors @ eigenvalues @ rightEigenvectors.transpose(0,2,1)
    _dissipation = absA @ entropyVector[..., np.newaxis]
    dissipation = _dissipation.reshape(len(entropyVector), len(entropyVector[0]))

    return ec_flux + .5*dissipation


"""# HLLC Riemann solver [Toro, 2019]
def calculateToroFlux(wLs, wRs, fLs, fRs, simVariables):
    gamma = simVariables.gamma

    rhoL, uL, pL = wRs[...,0], wRs[...,1], wRs[...,4]
    rhoR, uR, pR = wLs[...,0], wLs[...,1], wLs[...,4]
    QL, QR = fv.convertPrimitive(wRs, simVariables), fv.convertPrimitive(wLs, simVariables)

    zeta = (gamma-1)/(2*gamma)
    aL, aR = np.sqrt(gamma*fv.divide(pL, rhoL)), np.sqrt(gamma*fv.divide(pR, rhoR))
    twoRarefactionApprox = fv.divide(aL+aR-(((gamma-1)/2)*(uR-uL)), fv.divide(aL, pL**zeta)+fv.divide(aR, pR**zeta))**(1/zeta)

    qL, qR = np.ones_like(pL), np.ones_like(pR)
    _qL, _qR = np.sqrt(1 + (((gamma+1)/(2*gamma))*(fv.divide(twoRarefactionApprox, pL)-1))), np.sqrt(1 + (((gamma+1)/(2*gamma))*(fv.divide(twoRarefactionApprox, pR)-1)))
    qL[twoRarefactionApprox > pL] = _qL[twoRarefactionApprox > pL]
    qR[twoRarefactionApprox > pR] = _qR[twoRarefactionApprox > pR]

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