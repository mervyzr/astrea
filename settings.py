
##############################################################################

config = "sod"  # Sod, Sedov, Shu-Osher, Toro1/2/3/4/5, SIN
cells = 100
cfl = .5
gamma = 1.4
solver = "ppm"  # PCM, PLM, PPM
timestep = "euler"  # Euler, RK4, SSPRK(3,3), SSPRK(4,3), SSPRK(5,3), SSPRK(5,4)

runType = "single"  # Single/Multiple
livePlot = True

saveFile = False
snapshots = 1
saveVideo = False

variables = [config, cells, cfl, gamma, solver, timestep, livePlot]