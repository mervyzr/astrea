import os
import sys
import shutil
import getopt
import random
import traceback
from datetime import datetime
from time import time, process_time

import h5py
import numpy as np

import tests as tst
import settings as cfg
import evolvers as evo
from functions import generic, fv, plotting

##############################################################################

currentdir = os.getcwd()
seed = random.randint(0, 1e8)
np.random.seed(seed)


# Run finite volume code
def simulateShock(_configVariables, _testVariables, grp):
    # Initialise the discrete solution array with conserved variables <q>
    # Even though the solution array is discrete, the variables are averages (FV) instead of points (FD)
    domain = fv.initialise(_configVariables, _testVariables)

    # Compute dx and set t = 0
    dx = abs(_testVariables['endPos']-_testVariables['startPos'])/_configVariables['cells']
    t = 0

    if _configVariables['livePlot']:
        fig, ax, plots = plotting.initiateLivePlot(_testVariables['startPos'], _testVariables['endPos'], _configVariables['cells'])

    while t <= _testVariables['tEnd']:
        # Saves each instance of the system at time t
        if _configVariables['solver'] in ["ppm", "parabolic", "p"]:
            tubeSnapshot = fv.convertConservative(domain, _configVariables['gamma'], _testVariables['boundary'])
        else:
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

    if len(sys.argv) > 1:
        try:
            opts, args = getopt.getopt(sys.argv[1:], "", ["config=", "cells=", "cfl=", "gamma=", "solver=", "timestep=", "runType=", "livePlot=", "savePlots=", "snapshots=", "saveVideo=", "saveFile="])
        except getopt.GetoptError as e:
            print(f'{generic.bcolours.WARNING}Error: {e}{generic.bcolours.ENDC}')
            sys.exit(2)
        else:
            for opt, arg in opts:
                opt = opt.replace("--","")
                if opt in ["cells", "snapshots"]:
                    configVariables[opt] = int(arg)
                elif opt in ["cfl", "gamma"]:
                    configVariables[opt] = float(arg)
                elif opt in ["livePlot", "savePlot", "saveVideo", "saveFile"]:
                    configVariables[opt] = arg == "True"
                else:
                    configVariables[opt] = arg

    # Error condition(s)
    if configVariables['solver'] not in ["ppm", "parabolic", "p", "plm", "linear", "l", "pcm", "constant", "c"]:
        print(f"{generic.bcolours.WARNING}Reconstruct unknown; reverting to piecewise constant reconstruction method..{generic.bcolours.ENDC}")
    if configVariables['timestep'] not in ["euler", "rk4", "ssprk(2,2)","ssprk(3,3)", "ssprk(4,3)", "ssprk(5,3)", "ssprk(5,4)"]:
        print(f"{generic.bcolours.WARNING}Timestepper unknown; reverting to Forward Euler timestepping..{generic.bcolours.ENDC}")


    if configVariables['runType'].startswith('m'):
        configVariables['livePlot'] = False  # Turn off the live plot
        nList = 10 * 2**np.arange(3,11)
    else:
        if not configVariables['runType'].startswith('s'):
            print(f"{generic.bcolours.WARNING}RunType unknown; running single test..{generic.bcolours.ENDC}")
        if configVariables['savePlots'] or configVariables['saveVideo']:
            configVariables['livePlot'] = False  # Turn off the live plot
        nList = [configVariables['cells']]

    try:
        scriptStart = datetime.now().strftime('%Y%m%d%H%M')
        savepath = f"{currentdir}/savedData/{scriptStart}_{seed}"
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
                generic.printOutput(now, seed, configVariables)
                simulateShock(configVariables, testVariables, grp)
                elapsed = process_time() - lap
                generic.printOutput(now, seed, configVariables, elapsed=elapsed, runLength=len(list(grp.keys())))
                grp.attrs['elapsed'] = elapsed

            if (configVariables['savePlots'] or configVariables['saveVideo'] or configVariables['saveFile']) and not os.path.exists(savepath):
                os.makedirs(savepath)

            if configVariables['savePlots']:
                plotting.plotQuantities(f, configVariables, testVariables, savepath)
                #plotting.plotTotalVariation(f, configVariables, savepath)
                if configVariables['runType'].startswith('m') and (configVariables['config'].startswith('sin') or configVariables['config'].startswith('gauss')):
                    plotting.plotSolutionErrors(f, configVariables, testVariables, savepath)

            if configVariables['saveVideo']:
                if configVariables['runType'].startswith('s'):
                    vidpath = f"{currentdir}/.vidplots"
                    if not os.path.exists(vidpath):
                        os.makedirs(vidpath)
                    plotting.makeVideo(f, configVariables, testVariables, savepath, vidpath)
                else:
                    print(f"{generic.bcolours.FAIL}Videos can only be saved with runType='single'{generic.bcolours.ENDC}")
    except Exception as e:
        print(f"{generic.bcolours.WARNING}-- Error: {e} --{generic.bcolours.ENDC}\n")
        print(traceback.format_exc())
        os.remove(filename)
    else:
        if configVariables['saveFile']:
            shutil.move(filename, f"{savepath}/shockTube_{configVariables['config']}_{configVariables['solver']}_{configVariables['timestep']}_{seed}.hdf5")
        else:
            os.remove(filename)