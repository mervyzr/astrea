import numpy as np

from functions import fv

##############################################################################

# Ideally, the 4th-order averaged fluxes should be computed from the face-averaged variables
# But because the simulation is only 1D, the "normal"-Laplacian (Taylor expansion) of the face-averaged states and fluxes are zero
# Thus, the face-averaged and face-centred values are the same (<w>_i+1/2 = w_i+1/2)
# Same for the averaged and centred fluxes (<F>_i+1/2 = F_i+1/2)


# (Local) Lax-Friedrich scheme (1st-order; highly diffusive)
def calculateLaxFriedrichFlux(flux, qDiff, eigenvalues):
    return .5 * ((flux[1:]+flux[:-1]) - ((eigenvalues * qDiff).T))


# Lax-Wendroff scheme (2nd-order, Jacobian method; contains overshoots)
def calculateLaxWendroffFlux(flux, qDiff, eigenvalues, characteristics):
    # Sound speed for each cell (2-Riemann invariant; entropy wave or contact discontinuity); indexing 1 only works for hydrodynamics
    soundSpeed = np.unique(characteristics, axis=1)[:,1]
    normalisedEigvals = fv.divide(soundSpeed**2, eigenvalues)
    maxNormalisedEigvals = np.max([normalisedEigvals[:-1], normalisedEigvals[1:]], axis=0)

    return .5 * ((flux[1:]+flux[:-1]) - ((maxNormalisedEigvals * qDiff).T))
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


# Osher-Solomon(-Dumbser) Riemann solver [Dumbser & Toro, 2011]
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
    return D_minus+D_plus


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