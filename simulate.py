import os
import shutil
import random
import traceback
from datetime import datetime
from time import time, process_time

import h5py
import yaml
import numpy as np

import tests as tst
import settings as cfg
import evolvers as evo
from functions import generic, fv, plotting

##############################################################################

currentdir = os.getcwd()
seed = random.randint(0, 10000000)


# Run code
def simulateShock(_configVariables, _testVariables, grp):
    _N = _configVariables['cells']
    _N += (_N%2)  # Make N into an even number
    domain = fv.initialise(_configVariables, _testVariables)
    
    # Compute dx and set t = 0
    dx = abs(_testVariables['endPos']-_testVariables['startPos'])/_N
    t = 0

    if _configVariables['livePlot']:
        fig, ax, plots = plotting.initiateLivePlot(_testVariables['startPos'], _testVariables['endPos'], _N)

    while t <= _testVariables['tEnd']:
        # Saves each instance of the system at time t
        tubeSnapshot = fv.pointConvertConservative(domain, _configVariables['gamma'])
        dataset = grp.create_dataset(str(t), data=tubeSnapshot)
        dataset.attrs['t'] = t

        if _configVariables['livePlot']:
            plotting.updatePlot(tubeSnapshot, t, fig, ax, plots)
        
        # Compute the numerical fluxes at each interface
        fluxes, eigmax = evo.evolveSpace(domain, _configVariables['gamma'], _configVariables['solver'], _testVariables['boundary'])
        
        # Compute the full time step dt
        dt = _configVariables['cfl'] * dx/eigmax

        # Update the solution with the numerical fluxes using iterative methods
        domain = evo.evolveTime(domain, fluxes, dx, dt, _configVariables['timestep'], _configVariables['gamma'], _configVariables['solver'], _testVariables['boundary'])
        t += dt
    return None

##############################################################################

if __name__ == "__main__":
    filename = f"{currentdir}/.shockTemp_{seed}.hdf5"

    configVariables = cfg.variables
    testVariables = tst.variables

    # Error condition(s)
    if configVariables['solver'] not in ["ppm", "parabolic", "p", "plm", "linear", "l", "pcm", "constant", "c"]:
        print(f"{generic.bcolours.WARNING}Reconstruct unknown; reverting to piecewise constant reconstruction method..{generic.bcolours.ENDC}")
    if configVariables['timestep'] not in ["euler", "rk4", "ssprk(2,2)","ssprk(3,3)", "ssprk(4,3)", "ssprk(5,3)", "ssprk(5,4)"]:
        print(f"{generic.bcolours.WARNING}Timestepper unknown; reverting to Forward Euler timestepping..{generic.bcolours.ENDC}")

    
    if configVariables['runType'].startswith('m'):
        configVariables['livePlot'] = False  # Turn off the live plot
        nList = 5 * 2**np.arange(3,12)
    else:
        if not configVariables['runType'].startswith('s'):
            print(f"{generic.bcolours.WARNING}RunType unknown; running single test..{generic.bcolours.ENDC}")
        if configVariables['savePlots'] or configVariables['saveVideo']:
            configVariables['livePlot'] = False  # Turn off the live plot
        nList = [configVariables['cells']]

    try:
        with h5py.File(filename, "w") as f:
            for cells in nList:
                configVariables['cells'] = cells  # Set cell values

                grp = f.create_group(str(cells))
                grp.attrs['config'] = configVariables['config']
                grp.attrs['cells'] = configVariables['cells']
                grp.attrs['gamma'] = configVariables['gamma']
                grp.attrs['cfl'] = configVariables['cfl']
                grp.attrs['solver'] = configVariables['solver']
                grp.attrs['timestep'] = configVariables['timestep']

                lap, now = process_time(), datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                simulateShock(configVariables, testVariables, grp)
                elapsed = process_time() - lap
                grp.attrs['elapsed'] = elapsed
                generic.printOutput(now, configVariables, elapsed, len(list(grp.keys())))

            if configVariables['savePlots']:
                savepath = f"{currentdir}/plots"
                if not os.path.exists(savepath):
                    os.makedirs(savepath)

                plotting.plotQuantities(f, configVariables, testVariables, savepath)
                if configVariables['runType'].startswith('m') and (configVariables['config'].startswith('sin') or configVariables['config'].startswith('gauss')):
                    plotting.plotSolutionErrors(f, configVariables, testVariables, savepath)

            if configVariables['saveVideo']:
                if configVariables['runType'].startswith('s'):
                    savepath = f"{currentdir}/videos"
                    if not os.path.exists(savepath):
                        os.makedirs(savepath)

                    plotting.makeVideo(f, configVariables, testVariables, savepath)
                else:
                    print(f"{generic.bcolours.WARNING}Error; can only save video with runType='single'{generic.bcolours.ENDC}")
    except Exception as e:
        print(f"{generic.bcolours.WARNING}-- Error: {e} --{generic.bcolours.ENDC}\n")
        print(traceback.format_exc())
        os.remove(filename)
    else:
        if configVariables['saveFile']:
            if not os.path.exists(f"{currentdir}/datasets"):
                os.makedirs(f"{currentdir}/datasets")
            shutil.move(filename, f"{currentdir}/datasets/shockTube_{configVariables['config']}_{configVariables['solver']}_{configVariables['timestep']}_{seed}.hdf5")
        else:
            os.remove(filename)