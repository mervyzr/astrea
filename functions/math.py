from math import gcd
from decimal import Decimal

import numpy as np

##############################################################################
# Math functions
##############################################################################

# Generic Gaussian function
def gauss_func(r, test_specifics):
    return test_specifics['y_offset'] + test_specifics['ampl']*np.exp(-(r**2)/test_specifics['fwhm'])


# Generic sin function
def sine_func(r, test_specifics):
    return test_specifics['y_offset'] + test_specifics['ampl']*np.sin(test_specifics['freq']*np.pi*r)


# Smoothing kernel
def smoothing_kernel(quantity, r, d=1, mu=0, sigma=1):
    return quantity * (2 * np.pi * sigma**2)**(-d/2) * np.exp(-.5 * ((r - mu)/sigma)**2)


# Magic function to make errors disappear (!! physics would most likely be messed up so be very careful using this function !!)
def nan_to_num(arr, eps=1e-16):
    return np.nan_to_num(arr, copy=True, nan=0., posinf=eps, neginf=-eps)


# For handling division-by-zero warnings during array divisions
# !! MONITOR THE PHYSICS WHEN USING THIS; ZEROS IN DIVISOR MIGHT MEAN YOUR CODE IS INCORRECT !!
# The result buffer is left uninitialised and only the masked-out (zero-divisor) entries are
# written afterwards. Pre-filling it with 1/eps, as this used to, wrote a full-size buffer that
# was then almost entirely overwritten: measured 3.4x slower, and this is called ~100 times per
# timestep on full grids. Pass out= to write into an existing buffer.
def divide(dividend, divisor, eps=1e-16, out=None):
    dividend, divisor = np.real(dividend), np.real(divisor)

    if out is None:
        out = np.empty(
            np.broadcast_shapes(np.shape(dividend), np.shape(divisor)),
            dtype=np.result_type(dividend, divisor, np.float64),
        )

    nonzero = divisor != 0
    np.divide(dividend, divisor, out=out, where=nonzero)
    if not nonzero.all():
        np.copyto(out, 1/eps, where=~nonzero)
    return out


# For handling log zero and log negative values
# !! MONITOR THE PHYSICS WHEN USING THIS; NEGATIVE OR ZERO VALUES MIGHT MEAN YOUR CODE IS INCORRECT INSTEAD !!
def log(arr, eps=1e-16):
    positive = np.log(np.full(arr.shape, eps))
    return np.log(arr, out=positive, where=arr>0)


# There are situations where oscillations may produce negative densities/pressures
# This function is for clipping those values; ideally there should be no negative values
# !! MONITOR THE PHYSICS WHEN USING THIS; IMAGINARY PARTS DISCARDED, MONITOR FOR RANDOM OSCILLATIONS !!
def sqrt(arr):
    return np.sqrt(np.maximum(0, arr))


# For handling norms; typically would always be using the last axis
# einsum rather than np.linalg.norm: measured 5.1x faster on a full grid. np.linalg.norm
# rescales to avoid intermediate overflow, which only matters for |arr| > ~1e154 and so
# cannot arise for physical states here.
def norm(arr):
    return np.sqrt(norm2(arr))


# Same as norm, but returns the squared value
# Note this is the primitive: squaring np.linalg.norm took a square root and then undid it,
# which is both slower and less accurate than summing the squares directly.
def norm2(arr):
    return np.einsum('...i,...i->...', arr, arr)


# Customised rounding function
def round_off(value):
    if value%int(value) >= .5:
        return int(value) + 1
    else:
        return int(value)


# Catalan's function (used in mass distribution for spiral galaxies)
# G = sum_{n=0}^inf{(-1)^n / (2n + 1)^2}
def catalan(n=1000):
    _range = np.array(range(1, n+1, 2))**2
    coeff = np.array(([1,-1] * int(n//4 + n%4))[:len(_range)])
    return np.sum(coeff/_range)


# Get greatest common denominator (GCD) of decimal (commonly used for the coeff. in the SSP-RK methods)
def get_fraction(number):
    if not isinstance(number, str):
        try:
            number = str(number)
        except:
            raise TypeError

    try:
        places = abs(Decimal(str(number)).as_tuple().exponent)
    except:
        raise ValueError
    else:
        denom = 10**places
        numer = number * denom
        common = gcd(numer, denom)

        return numer//common, denom//common