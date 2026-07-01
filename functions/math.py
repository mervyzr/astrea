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
def divide(dividend, divisor, eps=1e-16):
    return np.divide(np.real(dividend), np.real(divisor), out=np.full_like(dividend, 1/eps), where=np.real(divisor)!=0)


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
def norm(arr):
    return np.linalg.norm(arr, axis=-1)


# Same as norm, but returns the squared value
def norm2(arr):
    return np.linalg.norm(arr, axis=-1) ** 2


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