import os
import sys
import shutil
import getopt
import traceback
from datetime import datetime
from time import perf_counter
from collections import namedtuple

import h5py
import numpy as np
import scipy as sp

import settings
import evolvers
from static import tests
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
    domain = fv.initialise(_simVariables, convert=True)

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
        fluxes, eigmax = evolvers.evolveSpace(domain, _simVariables)

        # Compute the full time step dt
        dt = _simVariables.cfl * _simVariables.dx/eigmax

        # Update the solution with the numerical fluxes using iterative methods
        domain = evolvers.evolveTime(domain, fluxes, dt, _simVariables)
        t += dt
    return None

##############################################################################

# Main script; includes handlers and core execution of simulation code
def main():
    # Save the HDF5 file (with seed) to store the temporary data
    filename = f"{currentdir}/.shockTemp_{seed}.hdf5"
    noprint = False

    # Generate the simulation variables (dict)
    configList = [var for var in dir(settings) if '__' not in var and var != 'np']
    configVariables = generic.tidyDict({k:v for k,v in vars(settings).items() if k in configList})

    # CLI arguments handler; updates the simulation variables (dict)
    if len(sys.argv) > 1:
        try:
            opts, args = getopt.getopt(sys.argv[1:], "", ["test=", "config=", "N=", "n=", "cells=", "cfl=", "gamma=", "dim=", "subgrid=", "timestep=", "scheme=", "runType=", "livePlot=", "savePlots=", "snapshots=", "saveVideo=", "saveFile=", "noprint", "echo"])
        except getopt.GetoptError as e:
            print(f'{generic.bcolours.WARNING}Error: {e}{generic.bcolours.ENDC}')
            sys.exit(2)
        else:
            for opt, arg in opts:
                opt = opt.replace("--","")
                if opt in ["cells", "N", "n"]:
                    configVariables[opt] = int(arg) - int(arg)%2
                elif opt in ["snapshots", "dim"]:
                    configVariables[opt] = int(arg)
                elif opt in ["cfl", "gamma"]:
                    configVariables[opt] = float(arg)
                elif opt in ["livePlot", "savePlots", "saveVideo", "saveFile"]:
                    configVariables[opt] = arg.lower() == "true"
                elif opt in ["test", "config"]:
                    configVariables["config"] = arg.lower()
                elif opt == "noprint":
                    noprint = True
                elif opt == "echo":
                    print(f"{generic.bcolours.OKGREEN}{generic.quotes[np.random.randint(len(generic.quotes))]}{generic.bcolours.ENDC}")
                    sys.exit(2)
                else:
                    configVariables[opt] = arg.lower()

    # Generate test configuration
    testVariables = tests.generateTestConditions(configVariables['config'])
    simVariables = configVariables | testVariables
    simVariables['dx'] = abs(simVariables['endPos']-simVariables['startPos'])/simVariables['cells']
    if simVariables['scheme'] in ['osher-solomon', 'osher', 'solomon', 'os']:
        _roots, _weights = sp.special.roots_legendre(3)  # 3rd-order Gauss-Legendre quadrature with interval [-1,1]
        simVariables['roots'] = .5*_roots + .5  # Gauss-Legendre quadrature with interval [0,1]
        simVariables['weights'] = _weights/2  # Gauss-Legendre quadrature with interval [0,1]

    # Error condition(s) handler; filter erroneous entries
    simVariables = generic.handleErrors(simVariables)

    # Simulation condition handler
    if simVariables['runType'].startswith('m'):
        # Auto-generate the resolutions/grid-sizes for multiple simulations
        coeff = 5
        nList = coeff*2**np.arange(2,12)

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
                lap, now = perf_counter(), datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                if not noprint:
                    generic.printOutput(now, seed, _simVariables)
                runSimulation(grp, _simVariables)
                elapsed = perf_counter() - lap
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
                    plotting.plotSolutionErrors(f, _simVariables, savepath, coeff)

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


if __name__ == "__main__":
    main()