#!/usr/bin/env python3

import os
import sys
import shutil
import signal
import traceback
from datetime import datetime
from time import perf_counter

import h5py
import yaml
import dotenv
import numpy as np

from static import tests
from num_methods import evolvers
from functions import constructor, fv, generic, io, plotting
from functions.io import SimulationVariables
from functions.generic import BColours

##############################################################################
# Main script
##############################################################################

# Globals
CURRENT_DIR = os.getcwd()
SAVE_DIR = "saved_data"
SEED = np.random.randint(0, 1e8)

DB_PATH = "static/.db.json"
PLOT_STYLE = "default"
BEAUTIFY_1D_PLOTS = False


# Finite volume shock function
def core_run(hdf5, sim_variables):
    # Initialise the discrete solution array with primitive variables <w> and convert them to conservative variables <q>
    grid = constructor.initialise(sim_variables)
    grid = fv.point_convert("primitive", grid, sim_variables, sim_variables.magnetic)

    # Initiate live or snapshot plotting, if enabled
    plot_snapshot = True if sim_variables.save_plots else False
    if sim_variables.live_plot:
        plotting_params = plotting.initiate_live_plot(sim_variables)

    # Activates only when checkpoints > 0; checkpoints still required to be > 0 for simulation to run
    chkpt = sim_variables.t_end/sim_variables.checkpoints if sim_variables.checkpoints > 0 else sim_variables.t_end
    write_chkpt = True if sim_variables.checkpoints > 0 else False
 
    # Start simulation run
    t, idx = 0., 1
    while t <= sim_variables.t_end:
        # Transform grid for visualisation; always use centred grid for visualisation, not staggered grid
        if sim_variables.magnetic:
            centred_grid = fv.inverse_reconstruct(grid, sim_variables)
        else:
            centred_grid = grid
        grid_snapshot = sim_variables.convert("conservative", centred_grid, sim_variables)

        # Save each instance of the system (primitive variables) at time t, if full_set_required
        if sim_variables.full_set_required:
            with h5py.File(hdf5, "a") as f:
                dataset = f[sim_variables.access_key].create_dataset(str(float(t)), data=grid_snapshot, compression="gzip", compression_opts=9)
                dataset.attrs['t'] = float(t)

        # Miscellaneous media/print options
        if not sim_variables.quiet:
            generic.print_status(sim_variables, t=t)
        if sim_variables.live_plot:
            plotting.update_plot(grid_snapshot, t, sim_variables, *plotting_params)
        if plot_snapshot:
            plotting.plot_snapshot(grid_snapshot, t, sim_variables)
            plot_snapshot = False
        if write_chkpt:
            io.write_chkpt(grid_snapshot, t, sim_variables)
            write_chkpt = False

        # Actual computation starts here
        if t == sim_variables.t_end:
            # Hard exact stop for the simulation; prevents adding an additional computation step
            break
        else:
            # Compute the numerical fluxes at each interface
            fluxes, eigmax = evolvers.evolve_space(grid, sim_variables, first_stage=True)

            # Compute the maximum eigenvalues for determining the full time step
            dt = sim_variables.cfl * eigmax

            # Handle dt
            if t+dt >= chkpt*idx:
                dt = chkpt*idx - t
                if sim_variables.save_plots:
                    plot_snapshot = True
                if sim_variables.checkpoints > 0:
                    write_chkpt = True
                idx += 1

            # Update the solution with the numerical fluxes using iterative methods
            grid = evolvers.evolve_time(grid, fluxes, dt, sim_variables)

            # Update time step
            t += dt
            sim_variables.timesteps += 1

            # Change the order of the axis sweep
            sim_variables.axes = sim_variables.axes[::-1]

##############################################################################

# Main script; includes handlers and core execution of simulation code
def run(seed, current_dir, save_dir, db_path, plot_style, beautify) -> None:
    np.random.seed(seed)

    # Save the HDF5 file (with seed) to store the temporary data, if full_set_required
    file_name = f"{current_dir}/.astrea_hdf5_temp_{seed}"

    # Signal handler for Ctrl+C
    def graceful_exit(sig, frame):
        sys.stdout.write('\033[2K\033[1G')
        print(f"{BColours.WARNING}Simulation end by SIGINT; exiting gracefully..{BColours.ENDC}")
        sys.exit(0)

    # Generate the simulation variables from settings (dict)
    with open(f"{current_dir}/parameters.yml", "r") as settings_file:
        config_variables = yaml.safe_load(settings_file)

    # Check CLI arguments
    if len(sys.argv) > 1:
        cli_variables, debug = io.handle_CLI(f"{current_dir}/{db_path}")
    else:
        cli_variables, debug = {}, False

    if not debug:
        np.seterr(all='ignore')

    # Tidy up configuration variables
    config_variables = io.parse_cli_variables(config_variables, cli_variables, f"{current_dir}/{db_path}")

    # Generate test configuration based on configuration
    test_variables = tests.generate_test_conditions(config_variables['config'], config_variables['cells'])

    # Initialise simulation variables
    sim_variables = SimulationVariables(seed, config_variables, test_variables, f"{current_dir}/{db_path}")

    # Auto-generate the resolutions/grid-sizes for run type
    if sim_variables.run_type.startswith('m'):
        if sim_variables.dimension == 2:
            _range = 2**np.arange(2,8)
        else:
            _range = 2**np.arange(3,11)
        grid_sizes = np.array([_range,] * sim_variables.dimension).T
    else:
        grid_sizes = [sim_variables.cells]
    grid_axes = [sim_variables.x_axis, sim_variables.y_axis]

    ###################################### SCRIPT INITIATE ######################################
    script_start = datetime.now().strftime('%Y%m%d%H%M')
    save_path = f"{current_dir}/{save_dir}/sim{script_start}_{seed}"
    sim_variables.plot_style = plot_style
    sim_variables.beautify = beautify

    # Make directories if they do not exist
    if sim_variables.save_plots or sim_variables.full_plots or sim_variables.save_video or sim_variables.save_file or sim_variables.checkpoints > 0:
        sim_variables.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    if sim_variables.save_plots and not os.path.exists(f"{save_path}/snapshots"):
        os.makedirs(f"{save_path}/snapshots")

    # Run in a try-except-else to handle crashes and prevent exiting code entirely, with signal handler
    original_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, graceful_exit)

    try:
        # Initiate the HDF5 database to store data, if full_set_required
        if sim_variables.full_set_required:
            with h5py.File(file_name, "w") as f:
                f.attrs['datetime'] = script_start
                f.attrs['seed'] = seed
                f.attrs['code'] = 'astrea'

        for grid_size in grid_sizes:
            ############################# INDIVIDUAL SIMULATION #############################
            now = datetime.now()

            # Update cells (and cell widths) in simulation variables
            sim_variables.access_key = now.strftime('%Y%m%d%H%M%S')+str(now.microsecond)
            sim_variables.now = now
            sim_variables.cells = grid_size
            for ax in sim_variables.ds.keys():
                sim_variables.ds[ax] = np.abs(np.diff(grid_axes[ax]))/grid_size[ax]

            # Save simulation variables into HDF5 file
            if sim_variables.full_set_required:
                with h5py.File(file_name, "a") as f:
                    grp = f.create_group(sim_variables.access_key)
                    grp.attrs['config'] = sim_variables.config
                    grp.attrs['cells'] = sim_variables.cells
                    grp.attrs['cfl'] = sim_variables.cfl
                    grp.attrs['gamma'] = sim_variables.gamma
                    grp.attrs['permeability'] = sim_variables.permeability
                    grp.attrs['dimension'] = sim_variables.dimension
                    grp.attrs['subgrid'] = sim_variables.subgrid
                    grp.attrs['time_evo'] = sim_variables.time_evo
                    grp.attrs['solver'] = sim_variables.solver

            ################### CORE ###################
            lap = perf_counter()
            core_run(file_name, sim_variables)
            elapsed = perf_counter() - lap
            ################### CORE ###################

            # Save attributes after individual run is completed
            sim_variables.elapsed = elapsed
            if sim_variables.full_set_required:
                with h5py.File(file_name, "a") as f:
                    f[sim_variables.access_key].attrs['elapsed'] = elapsed
            if not sim_variables.quiet:
                generic.print_status(sim_variables, final=True)
            ############################# END INDIVIDUAL SIMULATION #############################

        # Save plots; primitive quantities, total variation, conservation equation quantities, solution errors (errors only for run_type=multiple)
        if sim_variables.full_plots:
            with h5py.File(file_name, "r") as f:
                plotting.plot_quantities(f, sim_variables)
                if sim_variables.run_type.startswith("m"):
                    if sim_variables.config_category == "smooth":
                        plotting.plot_solution_errors(f, sim_variables, error_norm=1)
                else:
                    plotting.plot_total_variation(f, sim_variables)
                    plotting.plot_conservation_equations(f, sim_variables)

        # Save video (only for run_type=single)
        if sim_variables.save_video:
            with h5py.File(file_name, "r") as f:
                vidpath = f"{save_path}/.vidplots"
                if not os.path.exists(vidpath):
                    os.makedirs(vidpath)
                plotting.make_video(f, sim_variables, vidpath)

    # Exception handling; deletes the temporary HDF5 database to prevent clutter
    except Exception as e:
        print(end='\x1b[2K')
        if debug:
            print(f"\n{BColours.FAIL}-------    Error    -------{BColours.ENDC}")
            print(traceback.format_exc())
        else:
            print(f"{BColours.FAIL}-- Error: {e}{BColours.ENDC} (use --debug option for more details)")

    finally:
        # Save the temporary HDF5 database (!! Possibly large file sizes > 100 GB !!)
        if sim_variables.full_set_required:
            if sim_variables.save_file:
                shutil.move(file_name, f"{save_path}/astrea_{sim_variables.config}_{sim_variables.subgrid}_{sim_variables.time_evo}_{sim_variables.seed}.hdf5")
            else:
                os.remove(file_name)

        signal.signal(signal.SIGINT, original_sigint_handler)

    ###################################### SCRIPT END ######################################

if __name__ == "__main__":
    # Load env variables
    for dirpath, dirnames, filenames in os.walk(CURRENT_DIR):
        _ = [_filename for _filename in filenames if _filename.endswith('.env')]
        if len(_) == 1:
            dotenv.load_dotenv(os.path.join(dirpath, _[0]))
    _globals = [SEED, CURRENT_DIR, SAVE_DIR, DB_PATH, PLOT_STYLE, BEAUTIFY_1D_PLOTS]

    run(*_globals)