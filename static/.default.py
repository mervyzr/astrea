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
precision = np.float64
# : PCM, PLM, PPM
subgrid = "ppm"
# : Euler, RK4, SSPRK(2,2), SSPRK(3,3), SSPRK(4,3), SSPRK(5,3), SSPRK(5,4)
timestep = "ssprk(5,4)"
# : Lax-Friedrich (LF), Lax-Wendroff (LW)
scheme = "lf"

# Runtime parameters
# : Single, Multiple
runType = "single"
livePlot = False

# Media options
# : Save plots, with number of snapshots
savePlots = False
snapshots = 1
# : Save a video of the simulation
saveVideo = False
# : Save the HDF5 file of the simulation
saveFile = False