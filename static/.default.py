import numpy as np

##############################################################################

# Test configurations
# : Sod, Sedov, Shu-Osher, Ryu-Jones, Gaussian, sin-wave, sinc-wave, square wave, Toro1/2/3/4/5
config = "sod"

# Shock tube parameters
cells = 128
cfl = .5
gamma = 1.4

# Numerical methods
dimension = 1
precision = np.float64
# : PCM, PLM, PPM, WENO
subgrid = "ppm"
# : Euler, RK4, SSPRK(2,2), SSPRK(3,3), SSPRK(4,3), SSPRK(5,3), SSPRK(5,4)
timestep = "ssprk(5,4)"
# : Lax-Friedrich (LF), Lax-Wendroff (LW), HLLC (C), Entropy-stable (ES), Osher-Solomon (OS)
scheme = "lf"

# Runtime parameters
# : Single, Multiple
run_type = "single"
live_plot = False

# Media options
# : Save plots, with number of snapshots
save_plots = False
snapshots = 1
# : Save a video of the simulation
save_video = False
# : Save the HDF5 file of the simulation
save_file = False