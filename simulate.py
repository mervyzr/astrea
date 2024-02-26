import time
from datetime import datetime, timedelta

import numpy as np

import limiters
import tests as tst
import settings as cfg
import solvers as solv
import timestepper as tmstp
import plotting_functions as plotter
from functions import generic, fv

##############################################################################

# Run code
def simulateShock(_configVariables, _testVariables):
    simulation = {}
    _config, _N, _cfl, _gamma, _solver, _timestep, _livePlot = generic.lowerList(_configVariables)
    _startPos, _endPos, _shockPos, _tEnd, _boundary, _wL, _wR = generic.lowerList(_testVariables)

    _N += (_N%2)  # Make N into an even number
    domain = tst.initialise(_config, _N, _gamma, _startPos, _endPos, _shockPos, _wL, _wR)
    
    # Compute dx and set t = 0
    dx = abs(_endPos-_startPos)/_N
    t = 0

    if _livePlot:
        fig, ax, plots = plotter.initiateLivePlot(_startPos, _endPos, _N)

    while t <= _tEnd:
        # Saves each instance of the system at time t
        tubeSnapshot = fv.pointConvertConservative(domain, _gamma)
        simulation[t] = np.copy(tubeSnapshot)

        if _livePlot:
            plotter.updatePlot(tubeSnapshot, t, fig, ax, plots)

        # Initiate the shock tube
        shockTube = solv.RiemannSolver(domain, _solver, _gamma, dx, _boundary, limiters.applyLimiter)

        # Compute the numerical fluxes at each interface
        fluxes = fv.evolveSpace(shockTube, domain)

        # Compute the full time step dt
        dt = _cfl * shockTube.dx/shockTube.eigmax

        # Update the solution with the numerical fluxes using iterative methods
        domain = tmstp.evolveTime(shockTube, dt, fluxes, _timestep)
        t += dt
    return simulation

##############################################################################

if __name__ == "__main__":
    runs = []

    # Error condition(s)
    if cfg.solver.lower() not in ["ppm", "parabolic", "p", "plm", "linear", "l", "pcm", "constant", "c"]:
        print(f"{generic.bcolours.WARNING}Reconstruct unknown; reverting to piecewise constant reconstruction method..{generic.bcolours.ENDC}")
    if cfg.timestep.lower() not in ["euler", "rk4", "ssprk(2,2)","ssprk(3,3)", "ssprk(4,3)", "ssprk(5,3)", "ssprk(5,4)"]:
        print(f"{generic.bcolours.WARNING}Timestepper unknown; reverting to Forward Euler timestepping..{generic.bcolours.ENDC}")

    if cfg.runType[0].lower() == "m":
        cfg.variables[-1] = False
        for n in range(2,16):
            cells = 10*n**2
            cfg.variables[1] = cells
            lap, now = time.time(), datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            run = simulateShock(cfg.variables, tst.variables)
            generic.printOutput(now, cfg.config, cells, cfg.solver, cfg.timestep, cfg.cfl, str(timedelta(seconds=time.time()-lap)), len(run))
            runs.append(run)
        if cfg.saveFile:
            plotter.plotQuantities(runs, cfg.snapshots, [cfg.config.lower(), cfg.gamma, cfg.solver, cfg.timestep, tst.startPos, tst.endPos, tst.shockPos])
            if cfg.config.lower() == "sin":
                plotter.plotSolutionErrors(runs, [cfg.config.lower(), cfg.solver, cfg.timestep, tst.startPos, tst.endPos])
    else:
        if cfg.runType[0].lower() != "s":
            print(f"{generic.bcolours.WARNING}RunType unknown; running single test..{generic.bcolours.ENDC}")
        if cfg.saveFile:
            cfg.variables[-1] = False
        lap, now = time.time(), datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        run = simulateShock(cfg.variables, tst.variables)
        generic.printOutput(now, cfg.config, cfg.cells, cfg.solver, cfg.timestep, cfg.cfl, str(timedelta(seconds=time.time()-lap)), len(run))
        runs.append(run)
        if cfg.saveFile:
            plotter.plotQuantities(runs, cfg.snapshots, [cfg.config.lower(), cfg.gamma, cfg.solver, cfg.timestep, tst.startPos, tst.endPos, tst.shockPos])
        if cfg.saveVideo:
            plotter.makeVideo(runs, [cfg.config.lower(), cfg.solver, cfg.timestep, tst.startPos, tst.endPos])