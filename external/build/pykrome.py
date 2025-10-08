import ctypes
import numpy as np
import os

# Load the shared library
lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "libkrome.so"))

# Declare prototypes
lib.krome_init_c.restype = None
lib.krome_init_c.argtypes = []

lib.krome_c.restype = None
lib.krome_c.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # x
    ctypes.POINTER(ctypes.c_double),  # Tgas
    ctypes.POINTER(ctypes.c_double),  # dt
]

# Example constants (replace with your network values)
nsp = 3        # number of species
idx_H = 0        # adjust if krome_idx_H != 1 in your network

def run_example():
    # allocate abundances
    x = np.full(nsp, 1e-20, dtype=np.float64)
    x[idx_H] = 1.0e4

    Tgas = ctypes.c_double(1e3)
    spy = 3.65e2 * 2.4e1 * 3.6e3
    dt = ctypes.c_double(1e6 * spy)

    # Init
    lib.krome_init_c()

    # Run
    x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    lib.krome_c(x_ptr, ctypes.byref(Tgas), ctypes.byref(dt))

    return x

if __name__ == "__main__":
    out = run_example()
    print("Updated abundances:", out[:10])  # print first 10 species

