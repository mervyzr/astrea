import numpy as np

from functions import fv, constructors
from numerics import limiters, solvers

##############################################################################
# Piecewise linear reconstruction method (PLM) [van Leer, 1979]
##############################################################################

# Current convention: |               w(i-1/2)                    w(i+1/2)              |
#                     | i-1          <-- | -->         i         <-- | -->          i+1 |
#                     |        w_R(i-1)  |   w_L(i)          w_R(i)  |  w_L(i+1)        |
def run(tube, simVariables):
    gamma, boundary, permutations = simVariables.gamma, simVariables.boundary, simVariables.permutations

    # Rotate grid and apply algorithm for each axis
    for axis, axes in enumerate(permutations):

        # Convert to primitive variables; able to use pointwise conversion as it is still 2nd-order
        wS = fv.pointConvertConservative(tube.transpose(axes), gamma)

        # Pad array with boundary & apply (TVD) slope limiters
        w = fv.addBoundary(wS, boundary)
        limitedValues = limiters.minmodLimiter(w)

        # Linear reconstruction [Derigs et al., 2017]
        gradients = .5 * limitedValues
        wL, wR = np.copy(wS-gradients), np.copy(wS+gradients)  # (eq. 4.13)

        # Pad the reconstructed interfaces
        wLs, wRs = fv.addBoundary(wL, boundary)[1:], fv.addBoundary(wR, boundary)[:-1]

        # Convert the primitive variables
        # The conversion can be pointwise conversion for face-average values as it is still 2nd-order
        qLs, qRs = fv.pointConvertPrimitive(wLs, gamma), fv.pointConvertPrimitive(wRs, gamma)

        # Compute the fluxes and the Jacobian
        fLs, fRs = constructors.makeFluxTerm(wLs, gamma, axis), constructors.makeFluxTerm(wRs, gamma, axis)
        A = constructors.makeJacobian(w, gamma, axis)
        characteristics = np.linalg.eigvals(A)

    return solvers.calculateRiemannFlux(simVariables, fLs=fLs, fRs=fRs, wLs=wLs, wRs=wRs, qLs=qLs, qRs=qRs, characteristics=characteristics)