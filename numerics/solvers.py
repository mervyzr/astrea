import numpy as np

from functions import fv

##############################################################################

from numerics.reconstruct import modified, dissipate

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
        #avg_wS = .5 * (leftSolution + rightSolution)

        avg_wS = np.copy(tube)
        rho_L, rho_R = np.sqrt(leftSolution[:,0]), np.sqrt(rightSolution[:,0])

        avg_wS[:,0] = rho_L * rho_R
        avg_wS[:,1:4] = fv.divide((rho_L*leftSolution[:,1:4].T).T + (rho_R*rightSolution[:,1:4].T).T, (rho_L + rho_R)[:,np.newaxis])

        H_L = fv.divide(leftSolution[:,4], leftSolution[:,0])*gamma/(gamma-1) + .5*np.linalg.norm(leftSolution[:,1:4], axis=1)**2 + fv.divide(np.linalg.norm(leftSolution[:,5:8], axis=1)**2, leftSolution[:,0])
        H_R = fv.divide(rightSolution[:,4], rightSolution[:,0])*gamma/(gamma-1) + .5*np.linalg.norm(rightSolution[:,1:4], axis=1)**2 + fv.divide(np.linalg.norm(rightSolution[:,5:8], axis=1)**2, rightSolution[:,0])
        H = fv.divide(rho_L*H_L + rho_R*H_R, rho_L + rho_R)
        avg_wS[:,4] = (avg_wS[:,0]*H - .5*avg_wS[:,0]*np.linalg.norm(avg_wS[:,1:4], axis=1)**2 - np.linalg.norm(avg_wS[:,5:8], axis=1)**2) * (gamma-1)/gamma

        avg_wS[:,6:8] = fv.divide((rho_R*leftSolution[:,6:8].T).T + (rho_L*rightSolution[:,6:8].T).T, (rho_L + rho_R)[:,np.newaxis])

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
    
    # HLLC Riemann solver [Toro, 2019]
    if scheme in ["hllc", "hll", "c"]:
        pass

    # Roe solver
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
            soundSpeed = np.unique(characteristics, axis=1)[:,1]  # Sound speed for each cell (2-Riemann invariant; entropy wave or contact discontinuity)
            normalisedEigvals = fv.divide(soundSpeed**2, eigvals)
            maxNormalisedEigvals = np.max([normalisedEigvals[:-1], normalisedEigvals[1:]], axis=0)

            # Lax-Wendroff scheme (2nd-order, Jacobian method; overshoots)
            if scheme in ["lw", "lax-wendroff", "wendroff"]:
                return .5 * ((fS[1:]+fS[:-1]) - ((maxNormalisedEigvals * qDiff).T)), eigmax
            # Beam-Warming scheme (2nd-order; overshoots)
            elif scheme in ["bw", "beam-warming", "beam", "warming"]:
                return .5 * (), eigmax
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