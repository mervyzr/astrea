import time
from datetime import datetime, timedelta

import numpy as np

import tests as tst
import settings as cfg
import functions as fn
import solvers as solv
import timestepper as tmstp
import plotting_functions as plotter


##############################################################################

# Run code
def simulateShock(_configVariables, _testVariables):
    simulation = {}
    _config, _N, _cfl, _gamma, _solver, _timestep, _livePlot = fn.lowerList(_configVariables)
    _startPos, _endPos, _shockPos, _tEnd, _boundary, _wL, _wR = fn.lowerList(_testVariables)

    _N += (_N%2)  # Make N into an even number
    domain = tst.initialise(_config, _N, _gamma, _startPos, _endPos, _shockPos, _wL, _wR)
    
    # Compute dx and set t = 0
    dx = abs(_endPos-_startPos)/_N
    t = 0

    if _livePlot:
        fig, ax, plots = plotter.initiateLivePlot(_startPos, _endPos, _N)

    while t <= _tEnd:
        # Saves each instance of the system at time t
        tube = fn.pointConvertConservative(domain, _gamma)
        simulation[t] = np.copy(tube)

        if _livePlot:
            plotter.updatePlot(tube, t, fig, ax, plots)

        # Compute the numerical fluxes at each interface
        hydroTube = solv.RiemannSolver(domain, _boundary, _gamma)
        fluxes = hydroTube.calculateRiemannFlux(_solver)

        # Compute the full time step dt
        dt = _cfl * dx/hydroTube.eigmax

        # Update the solution with the numerical fluxes using iterative methods
        #stepper = tmstp.TimeStepper(domain, fluxes, dt, dx, _boundary, _gamma, _solver)
        #domain = stepper.evolveSystem(_timestep)
        domain -= ((dt/dx) * np.diff(fluxes, axis=0))
        t += dt
    return simulation

##############################################################################

if __name__ == "__main__":
    runs = []
    if cfg.runType[0].lower() == "m":
        cfg.variables[-1] = False
        for n in range(5,11):
            cells = 5*2**n
            cfg.variables[1] = cells
            lap = time.time()
            run = simulateShock(cfg.variables, tst.variables)
            fn.printOutput(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), cfg.config, cells, cfg.solver, cfg.timestep, str(timedelta(seconds=time.time()-lap)), len(run))
            runs.append(run)
        if cfg.saveFile:
            plotter.plotQuantities(runs, cfg.snapshots, cfg.config.lower(), cfg.gamma, tst.startPos, tst.endPos, tst.shockPos)
            if cfg.config.lower() == "sin":
                plotter.plotSolutionErrors(runs, cfg.config.lower(), tst.startPos, tst.endPos)
    else:
        if cfg.runType[0].lower() != "s":
            print(f"RunType unknown; running single test\n")
        if cfg.saveFile:
            cfg.variables[-1] = False
        lap = time.time()
        run = simulateShock(cfg.variables, tst.variables)
        fn.printOutput(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), cfg.config, cfg.cells, cfg.solver, cfg.timestep, str(timedelta(seconds=time.time()-lap)), len(run))
        runs.append(run)
        if cfg.saveFile:
            plotter.plotQuantities(runs, cfg.snapshots, cfg.config.lower(), cfg.gamma, tst.startPos, tst.endPos, tst.shockPos)
        if cfg.saveVideo:
            plotter.makeVideo(runs, cfg.config.lower(), tst.startPos, tst.endPos)