from collections import namedtuple

import numpy as np

from functions import fv
from numerics import solvers

##############################################################################

# WENO reconstruction method [Shu, 2009]
# Current convention: |               w(i-1/2)                    w(i+1/2)              |
#                     | i-1          <-- | -->         i         <-- | -->          i+1 |
#                     |        w_R(i-1)  |   w_L(i)          w_R(i)  |  w_L(i+1)        |
def run(tube, simVariables):
    gamma, precision, scheme, boundary = simVariables.gamma, simVariables.precision, simVariables.scheme, simVariables.boundary
    Data = namedtuple('Data', ['flux', 'eigmax'])

    # Convert to primitive variables
    wS = fv.convertConservative(tube, gamma, boundary)
    w = fv.makeBoundary(wS, boundary)

    # Function to generate the WENO interface values
    def makeFaceValue(_wS, _boundary):
        # Pad array with boundary
        w2 = fv.makeBoundary(_wS, _boundary, 2)

        # Define frequently used terms
        minusOne, minusTwo = w2[1:-3], w2[:-4]
        plusOne, plusTwo = w2[3:-1], w2[4:]

        # Define the stencils
        u1 = (minusTwo/3) - (minusOne*7/6) + (_wS*11/6)
        u2 = -(minusOne/6) + (_wS*5/6) + (plusOne/3)
        u3 = (_wS/3) + (plusOne*5/6) - (plusTwo/6)

        # Define the linear weights
        x1, x2, x3 = 1/10, 3/5, 3/10

        # Determine the smoothness indicators
        b1 = (13/12 * (minusTwo - 2*minusOne + _wS)**2) + (.25 * (minusTwo - 4*minusOne + 3*_wS)**2)
        b2 = (13/12 * (minusOne - 2*_wS + plusOne)**2) + (.25 * (minusOne - plusOne)**2)
        b3 = (13/12 * (_wS - 2*plusOne + plusTwo)**2) + (.25 * (3*_wS - 4*plusOne + plusTwo)**2)

        # Determine the non-linear weights
        alpha1 = x1/((1e-6 + b1)**2)
        alpha2 = x2/((1e-6 + b2)**2)
        alpha3 = x3/((1e-6 + b3)**2)

        weight1 = fv.divide(alpha1, alpha1+alpha2+alpha3)
        weight2 = fv.divide(alpha2, alpha1+alpha2+alpha3)
        weight3 = fv.divide(alpha3, alpha1+alpha2+alpha3)

        return weight1*u1 + weight2*u2 + weight3*u3

    # WENO reconstruction [Shu, 2009]
    wL, wR = makeFaceValue(w[2:], boundary), makeFaceValue(wS, boundary)

    # Pad the reconstructed interfaces
    wLs, wRs = fv.makeBoundary(wL, boundary)[1:], fv.makeBoundary(wR, boundary)[:-1]

    # Convert the primitive variables, and compute the state differences
    qLs, qRs = fv.convertPrimitive(wLs, gamma, boundary), fv.convertPrimitive(wRs, gamma, boundary)

    # Compute the fluxes and the Jacobian
    fLs, fRs = fv.makeFluxTerm(wLs, gamma), fv.makeFluxTerm(wRs, gamma)
    A = fv.makeJacobian(w, gamma)
    characteristics = np.linalg.eigvals(A)

    return solvers.calculateRiemannFlux(simVariables, fLs=fLs, fRs=fRs, wLs=wLs, wRs=wRs, qLs=qLs, qRs=qRs, characteristics=characteristics)