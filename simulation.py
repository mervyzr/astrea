import time
from datetime import timedelta

import numpy as np

import configs as cfg
import functions as fn
import solvers as solver
import plotting_functions as plotter


##############################################################################

# Run code
def runSimulation(_N, _config, _cfl, _gamma, _startPos, _endPos, _shockPos, _tEnd):
    simulation = {}
    _N += (_N%2)  # Make N into an even number
    domain = fn.initialise(_N, _config, _gamma, _startPos, _endPos, _shockPos)
    
    # Compute dx and set t = 0
    dx = abs(_endPos-_startPos)/_N
    t = 0

    if cfg.livePlot:
        fig, ax, plots = plotter.initiateLivePlot(_startPos, _endPos, _N)

    while t <= _tEnd:
        # Saves each instance of the system at time t
        tube = fn.convertConservative(domain, _gamma)
        simulation[t] = np.copy(tube)

        if cfg.livePlot:
            plotter.updatePlot(tube, t, fig, ax, plots)

        # Compute the numerical fluxes at each interface
        hydroTube = solver.plmSolver(domain, _config, _gamma)
        fluxes = hydroTube.calculateRiemannFlux()

        # Compute new time step
        dt = _cfl * dx/hydroTube.eigmax

        # Update the new solution with the computed time step and the (numerical fluxes?)
        domain -= ((dt/dx) * np.diff(fluxes, axis=0))
        t += dt
    return simulation

##############################################################################

lap = time.time()
run = runSimulation(cfg.cells, cfg.config, cfg.cfl, cfg.gamma, cfg.startPos, cfg.endPos, cfg.shockPos, cfg.tEnd)
print(f"[Test={cfg.config}, N={cfg.cells}; {len(run)} files]  Elapsed: {str(timedelta(seconds=time.time()-lap))} s")

"""runs = []
for n in [20, 100, 300, 1000, 4096]:
    config = "sod"
    lap = time.time()
    run = runSimulation(n, config)
    print(f"[Test={config}, N={n}; {len(run)} files]  Elapsed: {str(timedelta(seconds=time.time()-lap))} s")
    runs.append(run)

#plotter.plotQuantities(runs, index=-1, start=startPos, end=endPos)
#plotter.plotSolutionErrors(runs, start=startPos, end=endPos)
#plotter.makeVideo(run, start=startPos, end=endPos)
"""