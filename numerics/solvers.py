import numpy as np
import scipy as sp

from functions import fv

##############################################################################

from numerics.reconstruct import modified, dissipate



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



# Solve the Riemann (flux) problem (Local Lax-Friedrichs; approximate Riemann solver)
def calculateRiemannFlux(tube, solutions, simVariables):
    gamma, subgrid, scheme, precision, boundary = simVariables.gamma, simVariables.subgrid, simVariables.scheme, simVariables.precision, simVariables.boundary

    # Get the average of the solutions
    if subgrid in ["ppm", "parabolic", "p"]:
        if dissipate and modified:
            leftSolution, rightSolution = solutions[0]
            _mu = solutions[1]
        else:
            leftSolution, rightSolution = solutions
            _mu = np.zeros_like(tube)
        leftInterface, rightInterface = fv.makeBoundary(leftSolution, boundary)[1:], fv.makeBoundary(rightSolution, boundary)[:-1]
        avg_wS = .5 * (leftSolution + rightSolution)
    elif subgrid in ["plm", "linear", "l"]:
        leftSolution, rightSolution = solutions
        _mu = np.zeros_like(tube)
        leftInterface, rightInterface = fv.makeBoundary(leftSolution, boundary)[1:], fv.makeBoundary(rightSolution, boundary)[:-1]
        avg_wS = .5 * (leftSolution + rightSolution)
    else:
        _mu = np.zeros_like(tube)
        avg_wS = solutions

    # Ideally, the 4th-order averaged fluxes should be computed from the face-averaged variables
    # But because the simulation is only 1D, the "normal"-Laplacian (Taylor expansion) of the face-averaged states and fluxes are zero
    # Thus, the face-averaged and face-centred values are the same (<w>_i+1/2 = w_i+1/2)
    # Same for the averaged and centred fluxes (<F>_i+1/2 = F_i+1/2)
    
    # HLLC Riemann solver [Fleischmann et al., 2020]
    if scheme in ["hllc", "c"]:
        rhoL, uL, pL = rightInterface[:,0], rightInterface[:,1], rightInterface[:,4]
        rhoR, uR, pR = leftInterface[:,0], leftInterface[:,1], leftInterface[:,4]
        QL, QR = fv.pointConvertPrimitive(rightInterface, gamma), fv.pointConvertPrimitive(leftInterface, gamma)
        fL, fR = fv.makeFlux(rightInterface, gamma), fv.makeFlux(leftInterface, gamma)

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

        wS = fv.makeBoundary(avg_wS, boundary)
        fS = fv.makeFlux(wS, gamma)
        mu = fv.makeBoundary(_mu, boundary)
        fS += mu
        A = fv.makeJacobian(wS, gamma)
        characteristics = np.linalg.eigvals(A)
        eigvals = np.max(np.abs(characteristics), axis=1)  # Local max eigenvalue for each cell (1- or 3-Riemann invariant; shock wave or rarefaction wave)
        maxEigvals = np.max([eigvals[:-1], eigvals[1:]], axis=0)  # Local max eigenvalue between consecutive pairs of cell
        eigmax = np.max([np.max(maxEigvals), np.finfo(precision).eps])  # Maximum wave speed (max eigenvalue) for time evolution




        """rhoL, uL, pL = rightInterface[:,0], rightInterface[:,1], rightInterface[:,4]
        rhoR, uR, pR = leftInterface[:,0], leftInterface[:,1], leftInterface[:,4]
        QL, QR = fv.pointConvertPrimitive(rightInterface, gamma), fv.pointConvertPrimitive(leftInterface, gamma)
        fL, fR = fv.makeFlux(rightInterface, gamma), fv.makeFlux(leftInterface, gamma)

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

        wS = fv.makeBoundary(avg_wS, boundary)
        fS = fv.makeFlux(wS, gamma)
        mu = fv.makeBoundary(_mu, boundary)
        fS += mu
        A = fv.makeJacobian(wS, gamma)
        characteristics = np.linalg.eigvals(A)
        eigvals = np.max(np.abs(characteristics), axis=1)  # Local max eigenvalue for each cell (1- or 3-Riemann invariant; shock wave or rarefaction wave)
        maxEigvals = np.max([eigvals[:-1], eigvals[1:]], axis=0)  # Local max eigenvalue between consecutive pairs of cell
        eigmax = np.max([np.max(maxEigvals), np.finfo(precision).eps])  # Maximum wave speed (max eigenvalue) for time evolution"""

        return flux, eigmax

    # Osher-Solomon(-Dumbser) Riemann solver [Dumbser & Toro, 2011]
    elif scheme in ["osher-solomon", "osher", "solomon", "os"]:
        roots, weights = simVariables.roots, simVariables.weights

        qLs, qRs = fv.pointConvertPrimitive(leftInterface, gamma), fv.pointConvertPrimitive(rightInterface, gamma)
        avg_wS = getRoeAverage([leftInterface, rightInterface], [qLs, qRs], gamma)

        arr_L, arr_R = np.repeat(qLs[None,:], len(roots), axis=0), np.repeat(qRs[None,:], len(roots), axis=0)
        psi = arr_R + (roots*(arr_L-arr_R).T).T

        A = fv.makeJacobian(psi, gamma)
        characteristics = np.linalg.eigvals(A)

        _D_plus = .5 * (qLs-qRs) * (avg_wS + np.sum((weights * np.abs(characteristics).T).T, axis=0))
        D_minus = .5 * (qLs-qRs) * (avg_wS - np.sum((weights * np.abs(characteristics).T).T, axis=0))
        D_plus = fv.makeBoundary(_D_plus, boundary)[:-2]

        wS = fv.makeBoundary(avg_wS, boundary)
        _A = fv.makeJacobian(wS, gamma)
        _characteristics = np.linalg.eigvals(_A)
        eigvals = np.max(np.abs(_characteristics), axis=1)
        maxEigvals = np.max([eigvals[:-1], eigvals[1:]], axis=0)
        eigmax = np.max([np.max(maxEigvals), np.finfo(precision).eps])

        return D_minus+D_plus, eigmax

    #  Approximate (linearised) Riemann solver
    else:
        wS = fv.makeBoundary(avg_wS, boundary)
        fS = fv.makeFlux(wS, gamma)
        mu = fv.makeBoundary(_mu, boundary)
        fS += mu
        A = fv.makeJacobian(wS, gamma)
        characteristics = np.linalg.eigvals(A)

        """# Entropy-stable flux component
        wS = fv.makeBoundary(avg_wS, boundary)
        wLs, wRs = fv.makeBoundary(leftSolution, boundary), fv.makeBoundary(rightSolution, boundary)
        fS = fv.makeFlux([wLs, wRs], gamma)

        A = fv.makeJacobian(wS, gamma)
        characteristics = np.linalg.eigvals(A)
        D = np.zeros((characteristics.shape[0], characteristics.shape[1], characteristics.shape[1]))
        _diag = np.arange(characteristics.shape[1])
        D[:, _diag, _diag] = characteristics

        sL, sR = getEntropyVector(wLs, gamma), getEntropyVector(wRs, gamma)
        dfS = .5 * np.einsum('ijk,ij->ik', A*(D*A.transpose([0,2,1])), sR-sL)
        fS -= dfS"""

        if subgrid in ["ppm", "parabolic", "p", "plm", "linear", "l"]:
            # The conversion can be pointwise conversion for the face-averaged values
            qLs, qRs = fv.pointConvertPrimitive(leftInterface, gamma), fv.pointConvertPrimitive(rightInterface, gamma)
            qDiff = (qLs - qRs).T
        else:
            qLs, qRs = fv.pointConvertPrimitive(wS[:-1], gamma), fv.pointConvertPrimitive(wS[1:], gamma)
            qDiff = (qRs - qLs).T

        # Determine the eigenvalues for the computation of the flux and time stepping
        eigvals = np.max(np.abs(characteristics), axis=1)  # Local max eigenvalue for each cell (1- or 3-Riemann invariant; shock wave or rarefaction wave)
        maxEigvals = np.max([eigvals[:-1], eigvals[1:]], axis=0)  # Local max eigenvalue between consecutive pairs of cell
        eigmax = np.max([np.max(maxEigvals), np.finfo(precision).eps])  # Maximum wave speed (max eigenvalue) for time evolution

        # Local Lax-Friedrich scheme (1st-order; highly diffusive)
        if scheme in ["lf", "llf", "lax-friedrich", "friedrich"]:
            return .5 * ((fS[1:]+fS[:-1]) - ((maxEigvals * qDiff).T)), eigmax
        else:
            soundSpeed = np.unique(characteristics, axis=1)[:,1]  # Sound speed for each cell (2-Riemann invariant; entropy wave or contact discontinuity); indexing 1 only works for hydrodynamics
            normalisedEigvals = fv.divide(soundSpeed**2, eigvals)
            maxNormalisedEigvals = np.max([normalisedEigvals[:-1], normalisedEigvals[1:]], axis=0)

            # Lax-Wendroff scheme (2nd-order, Jacobian method; overshoots)
            if scheme in ["lw", "lax-wendroff", "wendroff"]:
                return .5 * ((fS[1:]+fS[:-1]) - ((maxNormalisedEigvals * qDiff).T)), eigmax
                #return .5 * ((qLs+qRs) - fv.divide(fS[1:]-fS[:-1], maxEigvals[:, np.newaxis])), eigmax
            # Fromm scheme (2nd-order)
            elif scheme in ["fr", "fromm"]:
                laxWendroffFlux = .5 * ((fS[1:]+fS[:-1]) - ((maxNormalisedEigvals * qDiff).T))
                beamWarmingFlux = .5 * ((3*fS[1:]-fS[:-1]) - ((maxNormalisedEigvals * qDiff).T))
                return .5 * (laxWendroffFlux + beamWarmingFlux), eigmax
            # Revert to Local Lax-Friedrich scheme
            else:
                return .5 * ((fS[1:]+fS[:-1]) - ((maxEigvals * qDiff).T)), eigmax


# Operator L as a function of the reconstruction values; calculate the flux through the surface [F(i+1/2) - F(i-1/2)]/dx
def calculateFlux(fluxes, dx):
    return -np.diff(fluxes, axis=0)/dx


# Calculate the entropy vector (jump between the left and right states)
def getEntropyVector(w, g):
    arr = np.copy(w)
    factor = w[:,0]/w[:,4]

    arr[:,0] = ((g-np.log(w[:,4]*w[:,0]**-g))/(g-1)) - (.5*w[:,0]*np.linalg.norm(w[:,1:4], axis=1)**2)/w[:,4]
    arr[:,1:4] = (w[:,1:4].T * factor).T
    arr[:,4] = -factor
    arr[:,5:8] = (w[:,5:8].T * factor).T
    return arr


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