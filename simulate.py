import os
import sys
import math
import shutil
import getopt
import traceback
import itertools
from datetime import datetime
from time import perf_counter
from collections import namedtuple

import h5py
import numpy as np
import scipy as sp

import settings
import evolvers
from static import tests
from functions import fv, generic, plotting, constructors

##############################################################################
# Main script
##############################################################################

# Globals
np.set_printoptions(linewidth=400, suppress=True)
CURRENT_DIR = os.getcwd()
SEED = np.random.randint(0, 1e8)
np.random.seed(SEED)


# Finite volume shock function
def run_simulation(grp: h5py, _sim_variables: namedtuple):
    # Initialise the discrete solution array with conserved variables <q>
    # Even though the solution array is discrete, the variables are averages (FV) instead of points (FD)
    domain = constructors.initialise(_sim_variables)

    # Initiate live plotting, if enabled
    if _sim_variables.live_plot:
        plotting_params = plotting.initiate_live_plot(_sim_variables)

    # Start simulation run
    t = 0.0
    while t <= _sim_variables.t_end:
        # Saves each instance of the system at time t
        tube_snapshot = fv.point_convert_conservative(domain, _sim_variables.gamma)
        dataset = grp.create_dataset(str(t), data=tube_snapshot)
        dataset.attrs['t'] = t

        # Update the live plot, if enabled
        if _sim_variables.live_plot:
            plotting.update_plot(tube_snapshot, t, _sim_variables.dimension, *plotting_params)

        # Compute the numerical fluxes at each interface
        interface_fluxes = evolvers.evolve_space(domain, _sim_variables)

        # Compute the full time step dt
        eigmaxes = [_sim_variables.dx/Riemann_flux.eigmax for Riemann_flux in list(interface_fluxes.values())]
        dt = _sim_variables.cfl * min(eigmaxes)

        # Update the solution with the numerical fluxes using iterative methods
        domain = evolvers.evolve_time(domain, interface_fluxes, dt, _sim_variables)

        # Handle the time update for machine precision
        if t+dt > _sim_variables.t_end:
            if t == _sim_variables.t_end:
                break
            else:
                t = _sim_variables.t_end
                continue
        else:
            t += dt
    return None

##############################################################################

# Main script; includes handlers and core execution of simulation code
def main() -> None:
    # Save the HDF5 file (with seed) to store the temporary data
    file_name = f"{CURRENT_DIR}/.tempShockData_{SEED}.hdf5"
    noprint, debug = False, False

    # Generate the simulation variables (dict)
    config_list = [var for var in dir(settings) if '__' not in var and var != 'np']
    config_variables = generic.tidy_dict({k:v for k,v in vars(settings).items() if k in config_list})

    # CLI arguments handler; updates the simulation variables (dict)
    if len(sys.argv) > 1:
        try:
            opts, args = getopt.getopt(sys.argv[1:], "", ["config=", "N=", "n=", "cells=", "cfl=", "gamma=", "dim", "dimension=", "subgrid=", "timestep=", "scheme=", "run_type=", "live_plot=", "save_plots=", "snapshots=", "save_video=", "save_file=", "test", "TEST", "debug", "DEBUG", "noprint", "echo", "quote"])
        except getopt.GetoptError as e:
            print(f'{generic.BColours.WARNING}-- Error: {e}{generic.BColours.ENDC}')
            sys.exit(2)
        else:
            for opt, arg in opts:
                opt = opt.replace("--","")
                if opt in ["cells", "N", "n"]:
                    config_variables[opt] = int(arg) - int(arg)%2
                elif opt in ["snapshots"]:
                    config_variables[opt] = int(arg)
                elif opt in ["cfl", "gamma"]:
                    config_variables[opt] = float(arg)
                elif opt in ["live_plot", "save_plots", "save_video", "save_file"]:
                    config_variables[opt] = arg.lower() == "true"
                elif opt in ["dim", "dimension"]:
                    config_variables[opt] = arg
                elif opt in ["DEBUG", "debug"]:
                    debug = True
                elif opt in ["TEST", "test"]:
                    continue
                elif opt == "noprint":
                    noprint = True
                elif opt in ["echo", "quote"]:
                    print(f"{generic.BColours.OKGREEN}{generic.quotes[np.random.randint(len(generic.quotes))]}{generic.BColours.ENDC}")
                    sys.exit(2)
                else:
                    config_variables[opt] = arg.lower()

    # Generate test configuration
    test_variables = tests.generate_test_conditions(config_variables['config'])
    sim_variables = config_variables | test_variables

    # Error condition(s) handler; filter erroneous entries
    sim_variables = generic.handle_errors(sim_variables)

    # Generate frequently used variables
    sim_variables['dx'] = abs(sim_variables['end_pos']-sim_variables['start_pos'])/sim_variables['cells']
    sim_variables['permutations'] = [axes for axes in list(itertools.permutations(list(range(math.ceil(sim_variables['dimension']+1))))) if axes[-1] == math.ceil(sim_variables['dimension'])]
    if sim_variables['scheme'] in ['osher-solomon', 'osher', 'solomon', 'os']:
        _roots, _weights = sp.special.roots_legendre(3)  # 3rd-order Gauss-Legendre quadrature with interval [-1,1]
        sim_variables['roots'] = .5*_roots + .5  # Gauss-Legendre quadrature with interval [0,1]
        sim_variables['weights'] = _weights/2  # Gauss-Legendre quadrature with interval [0,1]

    # Simulation condition handler
    if sim_variables['run_type'].startswith('m'):
        # Auto-generate the resolutions/grid-sizes for multiple simulations
        coeff = 5
        n_list = coeff*2**np.arange(2,12)
    else:
        n_list = [sim_variables['cells']]

    ###################################### SCRIPT INITIATE ######################################
    # Save simulation variables into namedtuple
    variable_constructor = namedtuple('simulation_variables', sim_variables)
    _sim_variables = variable_constructor(**sim_variables)

    # Start the script; run in a try-except-else to handle crashes and prevent exiting code entirely
    script_start = datetime.now().strftime('%Y%m%d%H%M')
    save_path = f"{CURRENT_DIR}/savedData/{script_start}_{SEED}"

    try:
        # Initiate the HDF5 database to store data temporarily
        with h5py.File(file_name, "w") as f:
            for N in n_list:
                ############################# INDIVIDUAL SIMULATION #############################
                # Update cells (and grid width) in simulation variables (namedtuple)
                _sim_variables = _sim_variables._replace(cells=N)
                _sim_variables = _sim_variables._replace(dx=abs(_sim_variables.end_pos-_sim_variables.start_pos)/_sim_variables.cells)

                # Save simulation variables into HDF5 file
                grp = f.create_group(str(_sim_variables.cells))
                grp.attrs['config'] = _sim_variables.config
                grp.attrs['cells'] = _sim_variables.cells
                grp.attrs['gamma'] = _sim_variables.gamma
                grp.attrs['cfl'] = _sim_variables.cfl
                grp.attrs['subgrid'] = _sim_variables.subgrid
                grp.attrs['timestep'] = _sim_variables.timestep
                grp.attrs['scheme'] = _sim_variables.scheme

                ################### CORE ###################
                lap, now = perf_counter(), datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                if not noprint:
                    generic.print_output(now, SEED, _sim_variables)
                run_simulation(grp, _sim_variables)
                elapsed = perf_counter() - lap
                grp.attrs['elapsed'] = elapsed
                if not noprint:
                    generic.print_output(now, SEED, _sim_variables, elapsed=elapsed, run_length=len(list(grp.keys())))
                ################### CORE ###################
                ############################# END SIMULATION #############################

            # Make directory if it does not exist
            if (_sim_variables.save_plots or _sim_variables.save_video or _sim_variables.save_file) and not os.path.exists(save_path):
                os.makedirs(save_path)

            # Save plots; primitive quantities, total variation, conservation equation quantities, solution errors (only for run_type=multiple)
            if _sim_variables.save_plots:
                plotting.plot_quantities(f, _sim_variables, save_path)
                if not _sim_variables.run_type.startswith('m'):
                    plotting.plot_total_variation(f, _sim_variables, save_path)
                    plotting.plot_conservation_equations(f, _sim_variables, save_path)
                if _sim_variables.run_type.startswith('m') and (_sim_variables.config.startswith('sin') or _sim_variables.config.startswith('gauss')):
                    plotting.plot_solution_errors(f, _sim_variables, save_path, coeff)

            # Save video (only for run_type=single)
            if _sim_variables.save_video and _sim_variables.run_type.startswith('s'):
                vidpath = f"{CURRENT_DIR}/.vidplots"
                if not os.path.exists(vidpath):
                    os.makedirs(vidpath)
                plotting.make_video(f, _sim_variables, save_path, vidpath)

    # Exception handling; deletes the temporary HDF5 database to prevent clutter
    except Exception as e:
        print(end='\x1b[2K')
        if debug:
            print(f"\n{generic.BColours.FAIL}-------    Error    -------{generic.BColours.ENDC}")
            print(traceback.format_exc())
        else:
            print(f"{generic.BColours.FAIL}-- Error: {e}{generic.BColours.ENDC} (use --DEBUG option for more details)")
        os.remove(file_name)

    # If no errors;
    else:
        # Save the temporary HDF5 database (!! Possibly large file sizes > 100GB !!)
        if _sim_variables.save_file:
            shutil.move(file_name, f"{save_path}/mHydyS_{_sim_variables.config}_{_sim_variables.subgrid}_{_sim_variables.timestep}_{SEED}.hdf5")
        else:
            os.remove(file_name)
    ###################################### SCRIPT END ######################################


if __name__ == "__main__":
    main()