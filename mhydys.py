#!/usr/bin/env python3

import os
import sys
import shutil
import signal
import traceback
from datetime import datetime
from time import perf_counter
from collections import namedtuple

import h5py
import yaml
import dotenv
import yaml
import dotenv
import numpy as np

import evolvers
from static import tests
from functions import fv, generic, plotting, constructors

##############################################################################
# Main script
##############################################################################

# Globals
CURRENT_DIR = os.getcwd()
SEED = np.random.randint(0, 1e8)
LOAD_ENV = False


# Finite volume shock function
def core_run(grp: h5py, _sim_variables: namedtuple, *args, **kwargs):
    # Initialise the discrete solution array with primitive variables <w> and convert them to conservative variables
    grid = constructors.initialise(_sim_variables, convert=True)
    plot_axes = _sim_variables.permutations[-1]

    # Initiate live or snapshot plotting, if enabled
    if _sim_variables.live_plot:
        plotting_params = plotting.initiate_live_plot(_sim_variables)
    elif _sim_variables.save_snaps:
        tol = _sim_variables.t_end/(_sim_variables.snapshots*_sim_variables.cells)
        timings = np.linspace(0, _sim_variables.t_end, _sim_variables.snapshots+1)

    # Define the conversion based on subgrid model
    if _sim_variables.subgrid.startswith("w") or _sim_variables.subgrid in ["ppm", "parabolic", "p"]:
        convert = fv.convert_conservative
    else:
        convert = fv.point_convert_conservative

    # Start simulation run
    t = 0.0
    while t <= _sim_variables.t_end:
        # Saves each instance of the system at time t
        grid_snapshot = convert(grid, _sim_variables).transpose(plot_axes)
        dataset = grp.create_dataset(str(float(t)), data=grid_snapshot)
        dataset.attrs['t'] = float(t)

        # Update the live plot, if enabled, or save snapshot
        if _sim_variables.live_plot:
            plotting.update_plot(grid_snapshot, t, _sim_variables.dimension, *plotting_params)
        elif _sim_variables.save_snaps:
            if (np.abs(t-timings) <= tol).any():
                plotting.plot_snapshot(grid_snapshot, t, _sim_variables, save_path=f"./savedData/snap{_sim_variables.seed}")

        # Handle the simulation end
        if t == _sim_variables.t_end:
            break
        else:
            # Compute the numerical fluxes at each interface
            interface_fluxes = evolvers.evolve_space(grid, _sim_variables)

            # Compute the maximum eigenvalues for determining the full time step
            eigmaxes = [_sim_variables.dx/Riemann_flux.eigmax for Riemann_flux in list(interface_fluxes.values())]
            dt = _sim_variables.cfl * min(eigmaxes)

            # Handle dt close to simulation end
            if t+dt > _sim_variables.t_end:
                dt = _sim_variables.t_end - t

            # Update the solution with the numerical fluxes using iterative methods
            grid = evolvers.evolve_time(grid, interface_fluxes, dt, _sim_variables)

            # Update time step
            t += dt

    return grp

##############################################################################

# Main script; includes handlers and core execution of simulation code
def run() -> None:
    np.random.seed(SEED)

    # Save the HDF5 file (with seed) to store the temporary data
    file_name = f"{CURRENT_DIR}/.tempSimData_{SEED}.hdf5"

    # Signal handler for Ctrl+C
    def graceful_exit(sig, frame):
        sys.stdout.write('\033[2K\033[1G')
        print(f"Received SIGINT; exiting gracefully...")
        os.remove(file_name)
        sys.exit(0)

    # Load env variables
    if LOAD_ENV and (sys.version_info.major == 3 and sys.version_info.minor >= 13):
        dotenv.load_dotenv(f"{CURRENT_DIR}/static/.env")

    # Generate the simulation variables from settings (dict)
    with open(f"{CURRENT_DIR}/settings.yml", "r") as settings_file:
        config_variables = yaml.safe_load(settings_file)

    # Check CLI arguments
    if len(sys.argv) > 1:
        cli_variables, debug, noprint = generic.handle_CLI()
    else:
        cli_variables, debug, noprint = {}, False, False

    if not debug:
        np.seterr(all='ignore')

    # Variables handler; filter erroneous entries and default values
    config_variables = generic.handle_variables(SEED, config_variables, cli_variables)

    # Generate test configuration and final variables
    test_variables = tests.generate_test_conditions(config_variables['config'], config_variables['cells'])
    sim_variables = config_variables | test_variables

    # Auto-generate the resolutions/grid-sizes for run type
    if sim_variables['run_type'].startswith('m'):
        coeff = 1
        if sim_variables['dimension'] == 2:
            n_list = coeff*2**np.arange(2,8)
        else:
            n_list = coeff*2**np.arange(3,11)
    else:
        n_list = [sim_variables['cells']]

    # Save simulation variables into namedtuple
    variable_constructor = namedtuple('simulation_variables', sim_variables)
    _sim_variables = variable_constructor(**sim_variables)

    ###################################### SCRIPT INITIATE ######################################
    script_start = datetime.now().strftime('%Y%m%d%H%M')
    save_path = f"{CURRENT_DIR}/savedData/sim{script_start}_{SEED}"

    # Make directories if they do not exist
    if (_sim_variables.save_plots or _sim_variables.save_video or _sim_variables.save_file) and not os.path.exists(save_path):
        os.makedirs(save_path)
    if _sim_variables.save_snaps and not os.path.exists(f"{CURRENT_DIR}/savedData/snap{SEED}"):
        os.makedirs(f"{CURRENT_DIR}/savedData/snap{SEED}")

    # Run in a try-except-else to handle crashes and prevent exiting code entirely, with signal handler
    original_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, graceful_exit)

    try:
        # Initiate the HDF5 database to store data temporarily
        with h5py.File(file_name, "w") as f:
            f.attrs['datetime'] = script_start
            f.attrs['seed'] = _sim_variables.seed
            for N in n_list:
                ############################# INDIVIDUAL SIMULATION #############################
                # Update cells (and grid width) in simulation variables (namedtuple)
                _sim_variables = _sim_variables._replace(cells=N)
                _sim_variables = _sim_variables._replace(dx=abs(_sim_variables.end_pos-_sim_variables.start_pos)/_sim_variables.cells)

                # Save simulation variables into HDF5 file
                grp = f.create_group(str(_sim_variables.cells))
                grp.attrs['config'] = _sim_variables.config
                grp.attrs['cells'] = _sim_variables.cells
                grp.attrs['cfl'] = _sim_variables.cfl
                grp.attrs['gamma'] = _sim_variables.gamma
                grp.attrs['dimension'] = _sim_variables.dimension
                grp.attrs['subgrid'] = _sim_variables.subgrid
                grp.attrs['timestep'] = _sim_variables.timestep
                grp.attrs['scheme'] = _sim_variables.scheme

                ################### CORE ###################
                lap, now = perf_counter(), datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                if not noprint:
                    generic.print_output(now, SEED, _sim_variables)
                core_run(grp, _sim_variables)
                elapsed = perf_counter() - lap
                grp.attrs['elapsed'] = elapsed
                if not noprint:
                    generic.print_output(now, SEED, _sim_variables, elapsed=elapsed, run_length=len(list(grp.keys())))
                ################### CORE ###################
                ############################# END SIMULATION #############################

            # Save plots; primitive quantities, total variation, conservation equation quantities, solution errors (errors only for run_type=multiple)
            if _sim_variables.save_plots:
                plotting.plot_quantities(f, _sim_variables, save_path)
                if _sim_variables.run_type.startswith("m"):
                    if _sim_variables.config_category == "smooth":
                        plotting.plot_solution_errors(f, _sim_variables, save_path, coeff)
                else:
                    plotting.plot_total_variation(f, _sim_variables, save_path)
                    plotting.plot_conservation_equations(f, _sim_variables, save_path)

            # Save video (only for run_type=single)
            if _sim_variables.save_video:
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
    
    finally:
        signal.signal(signal.SIGINT, original_sigint_handler)
    ###################################### SCRIPT END ######################################

if __name__ == "__main__":
    run()