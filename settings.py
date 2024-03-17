
##############################################################################

config = "sod"  # Sod, Sedov, Shu-Osher, sin wave, square wave, Toro1/2/3/4/5
cells = 200
cfl = .5
gamma = 1.4
solver = "ppm"  # PCM, PLM, PPM
timestep = "ssprk(5,4)"  # Euler, RK4, SSPRK(2,2), SSPRK(3,3), SSPRK(4,3), SSPRK(5,3), SSPRK(5,4)

runType = "single"  # Single/Multiple
livePlot = False

saveFile = False
snapshots = 1
saveVideo = False

variables = [config, cells, cfl, gamma, solver, timestep, livePlot]