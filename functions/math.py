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
# This function is for handling those scenarios; ideally there should be no negative values
# !! MONITOR THE PHYSICS WHEN USING THIS; IMAGINARY PARTS DISCARDED, MONITOR FOR RANDOM OSCILLATIONS !!
def sqrt(arr):
    return np.sqrt(np.real(arr), out=np.zeros_like(arr), where=arr>=0)


# For handling norms; typically would always be using the last axis
def norm(arr):
    return np.linalg.norm(arr, axis=-1)


# Customised rounding function
def round_off(value):
    if value%int(value) >= .5:
        return int(value) + 1
    else:
        return int(value)