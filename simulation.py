import time
from datetime import timedelta

import numpy as np

import tests as tst
import settings as cfg
import functions as fn
import solvers as solv
import plotting_functions as plotter


##############################################################################

# Run code
def runSimulation(_config, _N, _cfl, _gamma, _solver, _variables):
    simulation = {}
    _N += (_N%2)  # Make N into an even number
    _startPos, _endPos, _shockPos, _tEnd, _boundary, _wL, _wR = _variables
    domain = tst.initialise(_config, _N, _gamma, _startPos, _endPos, _shockPos, _wL, _wR)
    
    # Compute dx and set t = 0
    dx = abs(_endPos-_startPos)/_N
    t = 0

    if cfg.livePlot:
        fig, ax, plots = plotter.initiateLivePlot(_startPos, _endPos, _N)

    while t <= _tEnd:
        # Saves each instance of the system at time t
        tube = fn.pointConvertConservative(domain, _gamma)
        simulation[t] = np.copy(tube)

        if cfg.livePlot:
            plotter.updatePlot(tube, t, fig, ax, plots)

        # Compute the numerical fluxes at each interface
        hydroTube = solv.RiemannSolver(domain, _boundary, _gamma)
        fluxes = hydroTube.calculateRiemannFlux(_solver)

        # Compute new time step
        dt = _cfl * dx/hydroTube.eigmax

        # Update the new solution with the computed time step and the numerical fluxes
        domain -= ((dt/dx) * np.diff(fluxes, axis=0))
        t += dt
    return simulation

##############################################################################

runs = []
if cfg.runType[0].lower() == "m":
    cfg.livePlot = False
    for n in [20, 100, 300, 1000, 5000]:
        lap = time.time()
        run = runSimulation(cfg.config, n, cfg.cfl, cfg.gamma, cfg.solver, tst.variables)
        print(f"[Test={cfg.config}, N={n}, solver={cfg.solver}; {len(run)} timesteps]  Elapsed: {str(timedelta(seconds=time.time()-lap))} s")
        runs.append(run)
    if cfg.saveFile:
        plotter.plotQuantities(runs, cfg.snapshots, cfg.config, cfg.gamma, tst.startPos, tst.endPos, tst.shockPos)
        plotter.plotSolutionErrors(runs, cfg.config, tst.startPos, tst.endPos)
else:
    if cfg.saveFile:
        cfg.livePlot = False
    lap = time.time()
    run = runSimulation(cfg.config, cfg.cells, cfg.cfl, cfg.gamma, cfg.solver, tst.variables)
    print(f"[Test={cfg.config}, N={cfg.cells}, solver={cfg.solver}; {len(run)} timesteps]  Elapsed: {str(timedelta(seconds=time.time()-lap))} s")
    runs.append(run)
    if cfg.saveFile:
        plotter.plotQuantities(runs, cfg.snapshots, cfg.config, cfg.gamma, tst.startPos, tst.endPos, tst.shockPos)
    if cfg.saveVideo:
        plotter.makeVideo(runs, cfg.config, tst.startPos, tst.endPos)