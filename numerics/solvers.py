from collections import namedtuple

import numpy as np

from functions import fv

##############################################################################

# Ideally, the 4th-order averaged fluxes should be computed from the face-averaged variables
# But because the simulation is only 1D, the "normal"-Laplacian (Taylor expansion) of the face-averaged states and fluxes are zero
# Thus, the face-averaged and face-centred values are the same (<w>_i+1/2 = w_i+1/2)
# Same for the averaged and centred fluxes (<F>_i+1/2 = F_i+1/2)


# Intercell numerical fluxes between L and R interfaces based on Riemann solver
def calculateRiemannFlux(simVariables, *args, **kwargs):
    Data = namedtuple('Data', ['flux', 'eigmax'])

    # Determine the eigenvalues for the computation of time stepping
    eigvals = np.max(np.abs(kwargs["characteristics"]), axis=1)  # Local max eigenvalue for each cell (1- or 3-Riemann invariant; shock wave or rarefaction wave)
    maxEigvals = np.max([eigvals[:-1], eigvals[1:]], axis=0)  # Local max eigenvalue between consecutive pairs of cell

    eigmax = np.max([np.max(maxEigvals), np.finfo(simVariables.precision).eps])  # Maximum wave speed (max eigenvalue) for time evolution

    # HLL-type schemes
    if simVariables.scheme in ["hllc", "c"]:
        if simVariables.subgrid in ["plm", "linear", "l", "ppm", "parabolic", "p", "weno", "w"]:
            wLs, wRs = kwargs["wLs"], kwargs["wRs"]
        else:
            wLs, wRs = kwargs["w"][:-1], kwargs["w"][1:]
        return Data(calculateHLLCFlux(wLs, wRs, simVariables.gamma, simVariables.boundary), eigmax)

    # Osher-Solomon schemes
    elif simVariables.scheme in ["os", "osher-solomon", "osher", "solomon"]:
        if simVariables.subgrid in ["plm", "linear", "l", "ppm", "parabolic", "p", "weno", "w"]:
            qS = [kwargs["qLs"], kwargs["qRs"]]
            fluxes = kwargs["fLs"] + kwargs["fRs"]
        else:
            qS = [kwargs["qS"][:-1], kwargs["qS"][1:]]
            fluxes = kwargs["f"][1:] + kwargs["f"][:-1]
        return Data(calculateDOTSFlux(kwargs["w"], qS, fluxes, simVariables.gamma, simVariables.roots, simVariables.weights), eigmax)

    # Roe-type/Lax-type schemes
    else:
        if simVariables.subgrid in ["plm", "linear", "l", "ppm", "parabolic", "p", "weno", "w"]:
            qDiff = (kwargs["qLs"] - kwargs["qRs"]).T
            fluxes = kwargs["fLs"] + kwargs["fRs"]
        else:
            qDiff = (kwargs["qS"][1:] - kwargs["qS"][:-1]).T
            fluxes = kwargs["f"][1:] + kwargs["f"][:-1]

        if simVariables.scheme in ["lw", "lax-wendroff", "wendroff"]:
            return Data(calculateLaxWendroffFlux(fluxes, qDiff, eigvals, kwargs["characteristics"]), eigmax)
        else:
            return Data(calculateLaxFriedrichFlux(fluxes, qDiff, maxEigvals), eigmax)


# (Local) Lax-Friedrich scheme (1st-order; highly diffusive)
def calculateLaxFriedrichFlux(fluxes, qDiff, eigenvalues):
    return .5 * (fluxes - ((eigenvalues * qDiff).T))


# Lax-Wendroff scheme (2nd-order, Jacobian method; contains overshoots)
def calculateLaxWendroffFlux(fluxes, qDiff, eigenvalues, characteristics):
    # Sound speed for each cell (2-Riemann invariant; entropy wave or contact discontinuity); indexing 1 only works for hydrodynamics
    soundSpeed = np.unique(characteristics, axis=1)[:,1]
    normalisedEigvals = fv.divide(soundSpeed**2, eigenvalues)
    maxNormalisedEigvals = np.max([normalisedEigvals[:-1], normalisedEigvals[1:]], axis=0)

    return .5 * (fluxes - ((maxNormalisedEigvals * qDiff).T))
    #return .5 * ((qLs+qRs) - fv.divide(fS[1:]-fS[:-1], maxEigvals[:, np.newaxis]))


# HLLC Riemann solver [Fleischmann et al., 2020]
def calculateHLLCFlux(wLs, wRs, gamma, boundary):
    # The convention here is using the opposite (LR -> RL)
    rhoL, uL, pL = wRs[:,0], wRs[:,1], wRs[:,4]
    rhoR, uR, pR = wLs[:,0], wLs[:,1], wLs[:,4]
    QL, QR = fv.convertPrimitive(wRs, gamma, boundary), fv.convertPrimitive(wLs, gamma, boundary)
    fL, fR = fv.makeFlux(wRs, gamma), fv.makeFlux(wLs, gamma)

    zeta = (gamma-1)/(2*gamma)
    cL, cR = np.sqrt(gamma*fv.divide(pL, rhoL)), np.sqrt(gamma*fv.divide(pR, rhoR))
    u_hat = fv.divide(uL*np.sqrt(rhoL) + uR*np.sqrt(rhoR), np.sqrt(rhoL) + np.sqrt(rhoR))
    c2_hat = fv.divide(np.sqrt(rhoL)*cL**2 + np.sqrt(rhoR)*cR**2, np.sqrt(rhoL) + np.sqrt(rhoR)) + .5*((uR-uL)**2)*fv.divide(np.sqrt(rhoL)*np.sqrt(rhoR), (np.sqrt(rhoL)+np.sqrt(rhoR))**2)

    sL, sR = np.minimum(uL-cL, u_hat-np.sqrt(c2_hat)), np.maximum(uR+cR, u_hat+np.sqrt(c2_hat))
    s_star = fv.divide(pR - pL + (rhoL*uL*(sL-uL)) - (rhoR*uR*(sR-uR)), rhoL*(sL-uL) - rhoR*(sR-uR))

    coeffL, coeffR = fv.divide(sL-uL, sL-s_star), fv.divide(sR-uR, sR-s_star)
    _QL, _QR = (coeffL*QL.T).T, (coeffR*QR.T).T

    _QL[:,1] = rhoL * coeffL * s_star
    _QR[:,1] = rhoR * coeffR * s_star
    _QL[:,4] = _QL[:,4] + coeffL*(s_star-uL)*(rhoL*s_star + fv.divide(pL, sL-uL))
    _QR[:,4] = _QR[:,4] + coeffR*(s_star-uR)*(rhoR*s_star + fv.divide(pR, sR-uR))

    flux = np.copy(fL)
    _fL, _fR = fL + (sL*(_QL-QL).T).T, fR + (sR*(_QR-QR).T).T
    flux[(sL < 0) & (s_star >= 0)] = _fL[(sL < 0) & (s_star >= 0)]
    flux[(sR > 0) & (s_star <= 0)] = _fR[(sR > 0) & (s_star <= 0)]
    flux[sR <= 0] = fR[sR <= 0]
    return flux


"""# Osher-Solomon Riemann solver [Castro et al., 2016]
def calculateOSFlux(wS, qS, gamma, boundary, roots, weights):
    wLs, wRs = wS
    qLs, qRs = qS

    avg_wS = getRoeAverage([wLs, wRs], [qLs, qRs], gamma)

    arr_L, arr_R = np.repeat(qLs[None,:], len(roots), axis=0), np.repeat(qRs[None,:], len(roots), axis=0)
    psi = arr_R + (roots*(arr_L-arr_R).T).T

    A = fv.makeJacobian(psi, gamma)
    characteristics = np.linalg.eigvals(A)

    _D_plus = .5 * (qLs-qRs) * (avg_wS + np.sum((weights * np.abs(characteristics).T).T, axis=0))
    D_minus = .5 * (qLs-qRs) * (avg_wS - np.sum((weights * np.abs(characteristics).T).T, axis=0))
    D_plus = fv.makeBoundary(_D_plus, boundary)[:-2]
    return D_minus+D_plus"""


# Osher-Solomon(-Dumbser-Toro) Riemann solver [Dumbser & Toro, 2011]
def calculateDOTSFlux(w, qS, fluxes, gamma, roots, weights):
    qLs, qRs = qS

    # Define the right eigenvectors
    rightEigenvectors = fv.makeRightEigenvector(w[1:], gamma)
    _rightEigenvectors = np.repeat(rightEigenvectors[None,:], len(roots), axis=0)

    # Define the path integral for the Osher-Solomon dissipation term
    arr_L, arr_R = np.repeat(qLs[None,:], len(roots), axis=0), np.repeat(qRs[None,:], len(roots), axis=0)
    psi = arr_R + (roots*(arr_L-arr_R).T).T

    """# Generate the diagonal matrix of eigenvalues
    _lambda = np.zeros_like(rightEigenvectors)
    rhos, vecs, pressures, Bfield = w[1:][:,0], w[1:][:,1:4], w[1:][:,4], w[1:][:,5:8]/np.sqrt(4*np.pi)

    # Define speeds
    soundSpeed = np.sqrt(gamma * fv.divide(pressures, rhos))
    alfvenSpeed = np.sqrt(fv.divide(np.linalg.norm(Bfield, axis=1)**2, rhos))
    alfvenSpeedx = fv.divide(Bfield[:,0], np.sqrt(rhos))
    fastMagnetosonicWave = .5 * (soundSpeed**2 + alfvenSpeed**2 + np.sqrt(((soundSpeed**2 + alfvenSpeed**2)**2) - (4*(soundSpeed**2)*(alfvenSpeedx**2))))
    slowMagnetosonicWave = .5 * (soundSpeed**2 + alfvenSpeed**2 - np.sqrt(((soundSpeed**2 + alfvenSpeed**2)**2) - (4*(soundSpeed**2)*(alfvenSpeedx**2))))

    # Compute the diagonal matrix of eigenvalues
    i,j = np.diag_indices(_lambda.shape[-1])"""


    # Compute the Jacobian of the path integral and get the eigenvalues
    A = fv.makeJacobian(psi, gamma)
    characteristics = np.linalg.eigvals(A)

    # Determine the absolute value of the eigenvalues
    _Gamma = np.abs(characteristics)
    _lambda = np.zeros_like(A)
    i,j = np.diag_indices(psi.shape[-1])
    _lambda[...,i,j] = _Gamma[...,i]

    # Compute the absolute value of the Jacobian
    absA = _rightEigenvectors @ _lambda @ np.linalg.inv(_rightEigenvectors)

    # Compute the Dumbser-Toro Jacobian with the Gauss-Legendre quadrature
    jacobian = np.sum((weights*absA.T).T, axis=0)

    # Compute the Osher-Solomon dissipation term
    _qLs = jacobian @ qLs[..., np.newaxis]
    _qRs = jacobian @ qRs[..., np.newaxis]
    _qLs = _qLs.reshape(len(_qLs), len(_qLs[0]))
    _qRs = _qRs.reshape(len(_qRs), len(_qRs[0]))

    return .5*(fluxes - (_qLs-_qRs))


# HLLC Riemann solver [Toro, 2019]
def calculateToroFlux(wLs, wRs, gamma, boundary):
    rhoL, uL, pL = wRs[:,0], wRs[:,1], wRs[:,4]
    rhoR, uR, pR = wLs[:,0], wLs[:,1], wLs[:,4]
    QL, QR = fv.convertPrimitive(wRs, gamma, boundary), fv.convertPrimitive(wLs, gamma, boundary)
    fL, fR = fv.makeFlux(wRs, gamma), fv.makeFlux(wLs, gamma)

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
    _QL, _QR = np.copy((coeffL*QL.T).T), np.copy((coeffR*QR.T).T)

    _QL[:,1] = rhoL * coeffL * s_star
    _QR[:,1] = rhoR * coeffR * s_star
    _pL, _pR = np.copy(_QL[:,4]), np.copy(_QR[:,4])
    _BL, _BR = np.copy(_QL[:,5:8]), np.copy(_QR[:,5:8])
    _QL[:,4] = rhoL * coeffL * (fv.divide(_pL, rhoL) + ((s_star-uL)*(s_star+fv.divide(pL, rhoL*(sL-uL)))))
    _QR[:,4] = rhoR * coeffR * (fv.divide(_pR, rhoR) + ((s_star-uR)*(s_star+fv.divide(pR, rhoR*(sR-uR)))))
    _QL[:,5:8] = (rhoL * coeffL * _BL.T).T
    _QR[:,5:8] = (rhoR * coeffR * _BR.T).T

    flux = np.copy(fL)
    _fL, _fR = np.copy(fL + (sL*(_QL-QL).T).T), np.copy(fR + (sR*(_QR-QR).T).T)
    flux[(sL <= 0) & (0 <= s_star)] = _fL[(sL <= 0) & (0 <= s_star)]
    flux[(s_star <= 0) & (0 <= sR)] = _fR[(s_star <= 0) & (0 <= sR)]
    flux[0 >= sR] = fR[0 >= sR]
    return flux


# Calculate the Roe-averaged primitive variables from the left- & right-interface states for use in Roe solver in order to better capture shocks [Brio & Wu, 1988; LeVeque, 2002; Stone et al., 2008]
def getRoeAverage(wS, qS, gamma):
    wL, wR = wS
    qL, qR = qS

    avg = np.zeros_like(wL)
    rho_L, rho_R = np.sqrt(wL[:,0]), np.sqrt(wR[:,0])

    avg[:,0] = rho_L * rho_R
    avg[:,1:4] = fv.divide((rho_L*wL[:,1:4].T).T + (rho_R*wR[:,1:4].T).T, (rho_L + rho_R)[:,np.newaxis])
    avg[:,6:8] = fv.divide((rho_R*wL[:,6:8].T).T + (rho_L*wR[:,6:8].T).T, (rho_L + rho_R)[:,np.newaxis])

    H_L, H_R = fv.divide(qL[:,4] + wL[:,4] + .5*np.linalg.norm(wL[:,5:8], axis=1)**2, wL[:,0]), fv.divide(qR[:,4] + wR[:,4] + .5*np.linalg.norm(wR[:,5:8], axis=1)**2, wR[:,0])
    H = fv.divide(rho_L*H_L + rho_R*H_R, rho_L + rho_R)
    avg[:,4] = ((gamma-1)/gamma) * (avg[:,0]*H - .5*(avg[:,0]*np.linalg.norm(avg[:,1:4], axis=1)**2 + np.linalg.norm(avg[:,5:8], axis=1)**2))

    return avg