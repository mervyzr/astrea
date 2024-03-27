import sys

##############################################################################

# Test configurations
# : Sod, Sedov, Shu-Osher, Gaussian, sin wave, square wave, Toro1/2/3/4/5
config = "gauss"

# Shock tube parameters
cells = 200
cfl = .5
gamma = 1.4

# Numerical methods
# : PCM, PLM, PPM
solver = "ppm"
# : Euler, RK4, SSPRK(2,2), SSPRK(3,3), SSPRK(4,3), SSPRK(5,3), SSPRK(5,4)
timestep = "ssprk(5,4)"

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
# : Save the HDF5 file of the simulation (!! Might go up to 94GB !!)
saveFile = False




variables = {
    'config': config.lower(),
    'cells': cells,
    'cfl': cfl,
    'gamma': gamma,
    'solver': solver.lower(),
    'timestep': timestep.lower(),
    'runType': runType.lower(),
    'livePlot': livePlot,
    'savePlots': savePlots,
    'snapshots': snapshots,
    'saveVideo': saveVideo,
    'saveFile': saveFile
}