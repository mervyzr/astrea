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

from static import tests
from num_methods import evolvers
from functions import constructor, fv, generic, plotting

##############################################################################
# Main script
##############################################################################

# Globals
CURRENT_DIR = os.getcwd()
SEED = np.random.randint(0, 1e8)
LOAD_ENV = False


# Finite volume shock function
def core_run(grp: h5py, sim_variables: namedtuple, *args, **kwargs):
    # Initialise the discrete solution array with primitive variables <w> and convert them to conservative variables
    grid = constructor.initialise(sim_variables, convert=True)
    plot_axes = sim_variables.permutations[-1]

    # Initiate live or snapshot plotting, if enabled
    if sim_variables.live_plot:
        plotting_params = plotting.initiate_live_plot(sim_variables)
    elif sim_variables.take_snaps:
        tol = sim_variables.t_end/(sim_variables.snapshots*sim_variables.cells)
        timings = np.linspace(0, sim_variables.t_end, sim_variables.snapshots+1)

    # Define the conversion based on subgrid model
    if sim_variables.subgrid.startswith("w") or sim_variables.subgrid in ["ppm", "parabolic", "p"]:
        convert = fv.convert_conservative
    else:
        convert = fv.point_convert_conservative

    # Start simulation run
    t = 0.0
    while t <= sim_variables.t_end:
        # Saves each instance of the system at time t
        grid_snapshot = convert(grid, sim_variables).transpose(plot_axes)
        dataset = grp.create_dataset(str(float(t)), data=grid_snapshot)
        dataset.attrs['t'] = float(t)
        if not sim_variables.quiet:
            generic.print_progress(t, sim_variables)

        # Update the live plot, if enabled, or save snapshot
        if sim_variables.live_plot:
            plotting.update_plot(grid_snapshot, t, sim_variables.dimension, *plotting_params)
        elif sim_variables.take_snaps:
            if (np.abs(t-timings) <= tol).any():
                plotting.plot_snapshot(grid_snapshot, t, sim_variables, save_path=f"./savedData/snap{sim_variables.seed}")

        # Handle the simulation end
        if t == sim_variables.t_end:
            break
        else:
            # Compute the numerical fluxes at each interface
            interface_fluxes = evolvers.evolve_space(grid, sim_variables)

            # Compute the maximum eigenvalues for determining the full time step
            eigmaxes = [sim_variables.dx/Riemann_flux.eigmax for Riemann_flux in list(interface_fluxes.values())]
            dt = sim_variables.cfl * min(eigmaxes)

            # Handle dt close to simulation end
            if t+dt > sim_variables.t_end:
                dt = sim_variables.t_end - t

            # Update the solution with the numerical fluxes using iterative methods
            grid = evolvers.evolve_time(grid, interface_fluxes, dt, sim_variables)

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
    with open(f"{CURRENT_DIR}/parameters.yml", "r") as settings_file:
        config_variables = yaml.safe_load(settings_file)

    # Check CLI arguments
    if len(sys.argv) > 1:
        cli_variables, debug = generic.handle_CLI()
    else:
        cli_variables, debug = {}, False

    if not debug:
        np.seterr(all='ignore')

    # Variables handler; filter erroneous entries and default values
    config_variables = generic.handle_variables(SEED, config_variables, cli_variables)

    # Generate test configuration and final variables
    test_variables = tests.generate_test_conditions(config_variables['config'], config_variables['cells'])
    _sim_variables = config_variables | test_variables

    # Auto-generate the resolutions/grid-sizes for run type
    if _sim_variables['run_type'].startswith('m'):
        coeff = 1
        if _sim_variables['dimension'] == 2:
            n_list = coeff*2**np.arange(2,8)
        else:
            n_list = coeff*2**np.arange(3,11)
    else:
        n_list = [_sim_variables['cells']]

    # Save simulation variables into namedtuple
    variable_constructor = namedtuple('simulation_variables', _sim_variables)
    sim_variables = variable_constructor(**_sim_variables)

    ###################################### SCRIPT INITIATE ######################################
    script_start = datetime.now().strftime('%Y%m%d%H%M')
    save_path = f"{CURRENT_DIR}/savedData/sim{script_start}_{SEED}"

    # Make directories if they do not exist
    if (sim_variables.save_plots or sim_variables.save_video or sim_variables.save_file) and not os.path.exists(save_path):
        os.makedirs(save_path)
    if sim_variables.take_snaps and not os.path.exists(f"{CURRENT_DIR}/savedData/snap{SEED}"):
        os.makedirs(f"{CURRENT_DIR}/savedData/snap{SEED}")

    # Run in a try-except-else to handle crashes and prevent exiting code entirely, with signal handler
    original_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, graceful_exit)

    try:
        # Initiate the HDF5 database to store data temporarily
        with h5py.File(file_name, "w") as f:
            f.attrs['datetime'] = script_start
            f.attrs['seed'] = sim_variables.seed
            for N in n_list:
                ############################# INDIVIDUAL SIMULATION #############################
                now = datetime.now()

                # Update cells (and grid width) in simulation variables (namedtuple)
                sim_variables = sim_variables._replace(now=now)
                sim_variables = sim_variables._replace(cells=N)
                sim_variables = sim_variables._replace(dx=abs(sim_variables.end_pos-sim_variables.start_pos)/sim_variables.cells)

                # Save simulation variables into HDF5 file
                grp = f.create_group(str(sim_variables.cells))
                grp.attrs['config'] = sim_variables.config
                grp.attrs['cells'] = sim_variables.cells
                grp.attrs['cfl'] = sim_variables.cfl
                grp.attrs['gamma'] = sim_variables.gamma
                grp.attrs['dimension'] = sim_variables.dimension
                grp.attrs['subgrid'] = sim_variables.subgrid
                grp.attrs['timestep'] = sim_variables.timestep
                grp.attrs['scheme'] = sim_variables.scheme

                ################### CORE ###################
                lap = perf_counter()
                core_run(grp, sim_variables)
                elapsed = perf_counter() - lap
                ################### CORE ###################

                sim_variables = sim_variables._replace(elapsed=elapsed)
                grp.attrs['elapsed'] = elapsed
                if not sim_variables.quiet:
                    generic.print_final(grp, sim_variables)
                ############################# END INDIVIDUAL SIMULATION #############################

            # Save plots; primitive quantities, total variation, conservation equation quantities, solution errors (errors only for run_type=multiple)
            if sim_variables.save_plots:
                plotting.plot_quantities(f, sim_variables, save_path)
                if sim_variables.run_type.startswith("m"):
                    if sim_variables.config_category == "smooth":
                        plotting.plot_solution_errors(f, sim_variables, save_path, coeff)
                else:
                    plotting.plot_total_variation(f, sim_variables, save_path)
                    plotting.plot_conservation_equations(f, sim_variables, save_path)

            # Save video (only for run_type=single)
            if sim_variables.save_video:
                vidpath = f"{CURRENT_DIR}/.vidplots"
                if not os.path.exists(vidpath):
                    os.makedirs(vidpath)
                plotting.make_video(f, sim_variables, save_path, vidpath)

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
        if sim_variables.save_file:
            shutil.move(file_name, f"{save_path}/mhydys_{sim_variables.config}_{sim_variables.subgrid}_{sim_variables.timestep}_{SEED}.hdf5")
        else:
            os.remove(file_name)

    finally:
        signal.signal(signal.SIGINT, original_sigint_handler)
    ###################################### SCRIPT END ######################################

if __name__ == "__main__":
    run()