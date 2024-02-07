
##############################################################################

config = "sod"  # Sod, Sedov, Shu-Osher, Toro1/2/3/4/5, SIN
cells = 100
cfl = .5
gamma = 1.4
solver = "ppm"  # PCM, PLM, PPM
timestep = "ssprk(3,3)"  # RK4, SSPRK(3,3), SSPRK(5,4), Euler

runType = "single"  # Single/Multiple
livePlot = True

saveFile = False
snapshots = 1
saveVideo = False

variables = [config, cells, cfl, gamma, solver, timestep, livePlot]