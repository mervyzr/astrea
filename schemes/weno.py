from collections import namedtuple

import numpy as np

from functions import fv

##############################################################################

# Extrapolate the cell averages to face averages
# Current convention: |  i-1     ---> |  i       ---> |  i+1     ---> |
#                     |       w(i-1/2)|       w(i+1/2)|       w(i+3/2)|
def extrapolate(tube, simVariables):
    gamma, precision, scheme, boundary = simVariables.gamma, simVariables.precision, simVariables.scheme, simVariables.boundary
    Data = namedtuple('Data', ['flux', 'eigmax'])

    # Convert to primitive variables
    wS = fv.convertConservative(tube, gamma, boundary)

    # Extrapolate the cell averages to face averages
    # Current convention: |  i-1     ---> |  i       ---> |  i+1     ---> |
    #                     |       w(i-1/2)|       w(i+1/2)|       w(i+3/2)|

    # Pad array with boundary
    w2 = fv.makeBoundary(wS, boundary, 2)
    w = np.copy(w2[1:-1])

    # Define the stencils
    u1 = (w2[:-4]/3) - (w[:-2]*7/6) + (wS*11/6)
    u2 = -(w[:-2]/6) + (wS*5/6) + (w[2:]/3)
    u3 = (wS/3) + (w[2:]*5/6) - (w2[4:]/6)

    # Define the linear weights
    x1, x2, x3 = 1/10, 3/5, 3/10

    # Determine the smoothness indicators
    b1 = (13/12 * (w2[:-4] - 2*w[:-2] + wS)**2) + (.25 * (w2[:-4] - 4*w[:-2] + 3*wS)**2)
    b2 = (13/12 * (w[:-2] - 2*wS + w[2:])**2) + (.25 * (w[:-2] - w[2:])**2)
    b3 = (13/12 * (wS - 2*w[2:] + w2[4:])**2) + (.25 * (3*wS - 4*w[2:] + w2[4:])**2)

    # Determine the non-linear weights
    alpha1 = x1/((1e-6 + b1)**2)
    alpha2 = x2/((1e-6 + b2)**2)
    alpha3 = x3/((1e-6 + b3)**2)

    weight1 = fv.divide(alpha1, alpha1+alpha2+alpha3)
    weight2 = fv.divide(alpha2, alpha1+alpha2+alpha3)
    weight3 = fv.divide(alpha3, alpha1+alpha2+alpha3)

    wF = weight1*u1 + weight2*u2 + weight3*u3

    pass