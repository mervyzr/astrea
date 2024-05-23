import os
import sys
import shutil
import getopt
import traceback
from datetime import datetime
from time import time, process_time
from collections import namedtuple

import h5py
import numpy as np

import tests
import settings as cfg
import evolvers as evo
from functions import fv, generic, plotting

##############################################################################

# Globals
currentdir = os.getcwd()
seed = np.random.randint(0, 1e8)
np.random.seed(seed)


# Finite volume shock function
def runSimulation(grp, _simVariables):
    # Initialise the discrete solution array with conserved variables <q>
    # Even though the solution array is discrete, the variables are averages (FV) instead of points (FD)
    domain = fv.initialise(_simVariables)

    # Initiate live plotting, if enabled
    if _simVariables.livePlot:
        fig, ax, graphs = plotting.initiateLivePlot(_simVariables)

    # Start simulation run
    t = 0.0
    while t <= _simVariables.tEnd:
        # Saves each instance of the system at time t
        tubeSnapshot = fv.pointConvertConservative(domain, _simVariables.gamma)
        dataset = grp.create_dataset(str(t), data=tubeSnapshot)
        dataset.attrs['t'] = t

        # Update the live plot, if enabled
        if _simVariables.livePlot:
            plotting.updatePlot(tubeSnapshot, t, fig, ax, graphs)

        # Compute the numerical fluxes at each interface
        fluxes, eigmax = evo.evolveSpace(domain, _simVariables)

        # Compute the full time step dt
        dt = _simVariables.cfl * _simVariables.dx/eigmax

        # Update the solution with the numerical fluxes using iterative methods
        domain = evo.evolveTime(domain, fluxes, dt, _simVariables)
        t += dt
    return None

##############################################################################

# Below script is run when simulate.py is run
if __name__ == "__main__":
    # Save the HDF5 file (with seed) to store the temporary data
    filename = f"{currentdir}/.shockTemp_{seed}.hdf5"

    # Generate the simulation variables (dict)
    configVariables = cfg.variables
    testVariables = tests.generateTestConditions(configVariables['config'])
    simVariables = configVariables | testVariables
    simVariables['dx'] = abs(simVariables['endPos']-simVariables['startPos'])/simVariables['cells']
    noprint = False

    # CLI arguments handler; updates the simulation variables (dict)
    if len(sys.argv) > 1:
        try:
            opts, args = getopt.getopt(sys.argv[1:], "", ["test=", "config=", "N=", "cells=", "cfl=", "gamma=", "subgrid=", "timestep=", "scheme=", "runType=", "livePlot=", "savePlots=", "snapshots=", "saveVideo=", "saveFile=", "noprint", "cheer"])
        except getopt.GetoptError as e:
            print(f'{generic.bcolours.WARNING}Error: {e}{generic.bcolours.ENDC}')
            sys.exit(2)
        else:
            for opt, arg in opts:
                opt = opt.replace("--","")
                if opt in ["cells", "N"]:
                    simVariables[opt] = int(arg) - int(arg)%2
                elif opt in ["snapshots"]:
                    simVariables[opt] = int(arg)
                elif opt in ["cfl", "gamma"]:
                    simVariables[opt] = float(arg)
                elif opt in ["livePlot", "savePlot", "saveVideo", "saveFile"]:
                    simVariables[opt] = arg.lower() == "true"
                elif opt == "test":
                    simVariables["config"] = arg.lower()
                elif opt == "noprint":
                    noprint = True
                elif opt == "cheer":
                    print(f"{generic.bcolours.OKGREEN}{generic.quotes[np.random.randint(len(generic.quotes))]}{generic.bcolours.ENDC}")
                    sys.exit(2)
                else:
                    simVariables[opt] = arg.lower()

    # Error condition(s) handler; revert to default values for the simulation variables (dict) if unknown
    if simVariables['config'] not in ["sod", "sin", "sin-wave", "sinc", "sinc-wave", "sedov", "shu-osher", "shu", "osher", "gaussian", "gauss", "sq", "square", "square-wave", "toro1", "toro2", "toro3", "toro4", "toro5", "ryu-jones", "ryu", "jones", "rj"]:
        print(f"{generic.bcolours.WARNING}Test unknown; reverting to Sod shock tube test..{generic.bcolours.ENDC}")
        simVariables['config'] = "sod"
    if simVariables['subgrid'] not in ["ppm", "parabolic", "p", "plm", "linear", "l", "pcm", "constant", "c"]:
        print(f"{generic.bcolours.WARNING}Subgrid option unknown; reverting to piecewise constant method..{generic.bcolours.ENDC}")
        simVariables['subgrid'] = "pcm"
    if simVariables['timestep'] not in ["euler", "rk4", "ssprk(2,2)","ssprk(3,3)", "ssprk(4,3)", "ssprk(5,3)", "ssprk(5,4)"]:
        print(f"{generic.bcolours.WARNING}Timestepper unknown; reverting to Forward Euler timestepping..{generic.bcolours.ENDC}")
        simVariables['timestep'] = "euler"
    if simVariables['scheme'] not in ["llf", "lf", "lax","friedrich", "lax-friedrich", "lw", "lax-wendroff", "wendroff"]:
        print(f"{generic.bcolours.WARNING}Scheme unknown; reverting to Lax-Friedrich scheme..{generic.bcolours.ENDC}")
        simVariables['scheme'] = 'lf'
    if not (simVariables['runType'].startswith('s') or simVariables['runType'].startswith('m')):
        print(f"{generic.bcolours.WARNING}Run-type unknown; reverting to runType='single' simulation..{generic.bcolours.ENDC}")
        simVariables['runType'] = "single"
    if simVariables['saveVideo'] and not simVariables['runType'].startswith('s'):
        print(f"{generic.bcolours.WARNING}Videos can only be saved with runType='single'..{generic.bcolours.ENDC}")
        simVariables['saveVideo'] = False

    # Simulation condition handler
    if simVariables['runType'].startswith('m'):
        # Auto-generate the resolutions/grid-sizes for multiple simulations
        nList = 10 * 2**np.arange(3,11)

        # Turn off live plot feature when multiple simulations are run
        if simVariables['livePlot']:
            print(f"{generic.bcolours.WARNING}Live plots can only be switched on for single simulation runs..{generic.bcolours.ENDC}")
        simVariables['livePlot'] = False
    else:
        nList = [simVariables['cells']]

        # Turn off the live plot feature when saving plots or videos; live plot interferes with matplotlib savefig
        if simVariables['savePlots'] or simVariables['saveVideo']:
            simVariables['livePlot'] = False


    ###################################### SCRIPT INITIATE ######################################
    # Start the script; run in a try-except-else to handle crashes and prevent exiting code entirely
    try:
        scriptStart = datetime.now().strftime('%Y%m%d%H%M')
        savepath = f"{currentdir}/savedData/{scriptStart}_{seed}"

        # Save simulation variables into namedtuple
        variableConstructor = namedtuple('simulationVariables', simVariables)
        _simVariables = variableConstructor(**simVariables)

        # Initiate the HDF5 database to store data temporarily
        with h5py.File(filename, "w") as f:
            for N in nList:
                ############################# INDIVIDUAL SIMULATION #############################
                # Update cells (and grid width) in simulation variables (namedtuple)
                _simVariables = _simVariables._replace(cells=N)
                _simVariables = _simVariables._replace(dx=abs(_simVariables.endPos-_simVariables.startPos)/_simVariables.cells)

                # Save simulation variables into HDF5 file
                grp = f.create_group(str(_simVariables.cells))
                grp.attrs['config'] = _simVariables.config
                grp.attrs['cells'] = _simVariables.cells
                grp.attrs['gamma'] = _simVariables.gamma
                grp.attrs['cfl'] = _simVariables.cfl
                grp.attrs['subgrid'] = _simVariables.subgrid
                grp.attrs['timestep'] = _simVariables.timestep
                grp.attrs['scheme'] = _simVariables.scheme

                ################### CORE ###################
                lap, now = process_time(), datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                if not noprint:
                    generic.printOutput(now, seed, _simVariables)
                runSimulation(grp, _simVariables)
                elapsed = process_time() - lap
                grp.attrs['elapsed'] = elapsed
                if not noprint:
                    generic.printOutput(now, seed, _simVariables, elapsed=elapsed, runLength=len(list(grp.keys())))
                ################### CORE ###################
                ############################# END SIMULATION #############################

            # Make directory if it does not exist
            if (_simVariables.savePlots or _simVariables.saveVideo or _simVariables.saveFile) and not os.path.exists(savepath):
                os.makedirs(savepath)

            # Save plots; primitive quantities, total variation, conservation equation quantities, solution errors (only for runType=multiple)
            if _simVariables.savePlots:
                plotting.plotQuantities(f, _simVariables, savepath)
                if not _simVariables.runType.startswith('m'):
                    plotting.plotTotalVariation(f, _simVariables, savepath)
                    plotting.plotConservationEquations(f, _simVariables, savepath)
                if _simVariables.runType.startswith('m') and (_simVariables.config.startswith('sin') or _simVariables.config.startswith('gauss')):
                    plotting.plotSolutionErrors(f, _simVariables, savepath)

            # Save video (only for runType=single)
            if _simVariables.saveVideo and _simVariables.runType.startswith('s'):
                vidpath = f"{currentdir}/.vidplots"
                if not os.path.exists(vidpath):
                    os.makedirs(vidpath)
                plotting.makeVideo(f, _simVariables, savepath, vidpath)

    # Exception handling; deletes the temporary HDF5 database to prevent clutter
    except Exception as e:
        print(f"{generic.bcolours.WARNING}-- Error: {e} --{generic.bcolours.ENDC}\n")
        print(traceback.format_exc())
        os.remove(filename)

    # If no errors;
    else:
        # Save the temporary HDF5 database (!! Possibly large file sizes > 100GB !!)
        if _simVariables.saveFile:
            shutil.move(filename, f"{savepath}/shockTube_{_simVariables.config}_{_simVariables.subgrid}_{_simVariables.timestep}_{seed}.hdf5")
        else:
            os.remove(filename)
    ###################################### SCRIPT END ######################################