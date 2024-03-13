import os
import time
import shutil
import random
from datetime import datetime, timedelta

import h5py

import limiters
import tests as tst
import settings as cfg
import solvers as solv
import timestepper as tmstp
import plotting_functions as plotter
from functions import generic, fv

##############################################################################

currentdir = os.getcwd()
seed = random.randint(0, 10000000)

# Run code
def simulateShock(_configVariables, _testVariables, grp):
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
        dataset = grp.create_dataset(str(t), data=tubeSnapshot)
        dataset.attrs['t'] = t

        if _livePlot:
            plotter.updatePlot(tubeSnapshot, t, fig, ax, plots)

        # Initiate the shock tube
        shockTube = solv.RiemannSolver(domain, _solver, _gamma, dx, _boundary, limiters)

        # Compute the numerical fluxes at each interface
        fluxes = fv.evolveSpace(shockTube, domain)

        # Compute the full time step dt
        dt = _cfl * shockTube.dx/shockTube.eigmax

        # Update the solution with the numerical fluxes using iterative methods
        domain = tmstp.evolveTime(shockTube, dt, fluxes, _timestep)
        t += dt
    return None

##############################################################################

if __name__ == "__main__":
    filename = f"{currentdir}/.shockTemp_{seed}.hdf5"
    f = h5py.File(filename, "w")

    if not os.path.exists(f"{currentdir}/datasets"):
        os.makedirs(f"{currentdir}/datasets")

    if not os.path.exists(f"{currentdir}/plots"):
        os.makedirs(f"{currentdir}/plots")

    # Error condition(s)
    if cfg.solver.lower() not in ["ppm", "parabolic", "p", "plm", "linear", "l", "pcm", "constant", "c"]:
        print(f"{generic.bcolours.WARNING}Reconstruct unknown; reverting to piecewise constant reconstruction method..{generic.bcolours.ENDC}")
    if cfg.timestep.lower() not in ["euler", "rk4", "ssprk(2,2)","ssprk(3,3)", "ssprk(4,3)", "ssprk(5,3)", "ssprk(5,4)"]:
        print(f"{generic.bcolours.WARNING}Timestepper unknown; reverting to Forward Euler timestepping..{generic.bcolours.ENDC}")


    if cfg.runType[0].lower() == "m":
        cfg.variables[-1] = False  # Turn off the live plot
        
        for n in range(3,5):
            cells = 5*2**n
            cfg.variables[1] = cells  # Change cell values

            grp = f.create_group(str(cells))
            grp.attrs['config'] = cfg.solver
            grp.attrs['cells'] = cells
            grp.attrs['gamma'] = cfg.gamma
            grp.attrs['cfl'] = cfg.cfl
            grp.attrs['solver'] = cfg.solver
            grp.attrs['timestepper'] = cfg.timestep

            lap, now = time.time(), datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            simulateShock(cfg.variables, tst.variables, grp)
            generic.printOutput(now, cfg.config, cells, cfg.cfl, cfg.solver, cfg.timestep, str(timedelta(seconds=time.time()-lap)), len(list(grp.keys())))

        if cfg.saveFile:
            plotter.plotQuantities(f, cfg.snapshots, [cfg.config.lower(), cfg.gamma, cfg.solver, cfg.timestep, tst.startPos, tst.endPos, tst.shockPos])
            if cfg.config.lower() == "sin":
                plotter.plotSolutionErrors(f, [cfg.config.lower(), cfg.solver, cfg.timestep, tst.startPos, tst.endPos])

    else:
        if cfg.runType.lower() != "single":
            print(f"{generic.bcolours.WARNING}RunType unknown; running single test..{generic.bcolours.ENDC}")

        if cfg.saveFile or cfg.saveVideo:
            cfg.variables[-1] = False  # Turn off the live plot

        grp = f.create_group(str(cfg.cells))
        grp.attrs['config'] = cfg.solver
        grp.attrs['cells'] = cfg.cells
        grp.attrs['gamma'] = cfg.gamma
        grp.attrs['cfl'] = cfg.cfl
        grp.attrs['solver'] = cfg.solver
        grp.attrs['timestepper'] = cfg.timestep

        lap, now = time.time(), datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        simulateShock(cfg.variables, tst.variables, grp)
        generic.printOutput(now, cfg.config, cfg.cells, cfg.cfl, cfg.solver, cfg.timestep, str(timedelta(seconds=time.time()-lap)), len(list(grp.keys())))
        
        if cfg.saveFile:
            plotter.plotQuantities(f, cfg.snapshots, [cfg.config.lower(), cfg.gamma, cfg.solver, cfg.timestep, tst.startPos, tst.endPos, tst.shockPos])

        if cfg.saveVideo:
            plotter.makeVideo(f, [cfg.config.lower(), cfg.solver, cfg.timestep, tst.startPos, tst.endPos])
    
    f.close()
    if cfg.saveFile:
        shutil.move(filename, f"{currentdir}/datasets/shockTube_{cfg.config.lower()}_{cfg.solver}_{cfg.timestep}_{datetime.now().strftime('%Y%m%d%H%M')}.hdf5")