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

from external import krome_funcs
from functions import constructor, fv, io, plotting
from functions.io import SimulationVariables
from functions.generic import BColours
from num_methods import evolvers
from static import tests

##############################################################################
# Main script
##############################################################################

# Globals
SAVE_DIR = "saved_data"
SEED = np.random.randint(0, 1e8)


# Finite volume simulation
def core_run(sim_variables, **kwargs):
    # Initialise or load the discrete solution array with primitive variables <w>
    if sim_variables.chkpt_file:
        primitive_grid, t, idx = kwargs['grid'], kwargs['time'], kwargs['idx']
    else:
        primitive_grid, t, idx = constructor.initialise(sim_variables), 0., 1

    # For magnetic simulations, inverse reconstruction needed for the conversion, as well as returning converted centred grid to staggered values
    centred_grid = fv.inverse_reconstruct(primitive_grid, sim_variables) if sim_variables.magnetic else primitive_grid

    # Convert primitive grid to conservative variables <q>
    grid = fv.point_convert("primitive", centred_grid, sim_variables)
    grid[...,5+sim_variables.axes] = primitive_grid[...,5+sim_variables.axes]

    ########################

    # Initialise the chemical grid if activated;.
    # Abundances can be overriden; accepts a dictionary of atom/molecule/ion name as key and the number densities [1/cm3] or mass fraction [X] as value
    if sim_variables.chemistry:
        chem_grid = krome_funcs.initialise(sim_variables, perturb=True)

    ########################

    # Initiate live or snapshot plotting, if enabled
    plot_snapshot = True if sim_variables.save_snaps else False
    if sim_variables.live_plot:
        plotting_params = plotting.initiate_live_plot(sim_variables)

    # Activates only when checkpoints > 0; checkpoints still required to be > 0 for simulation to run
    chkpt = sim_variables.t_end/sim_variables.checkpoints if sim_variables.checkpoints > 0 else sim_variables.t_end
    create_chkpt_file = True if sim_variables.write_chkpt else False

    ########################

    while t <= sim_variables.t_end:
        # Transform grid for visualisation; always use centred grid for visualisation, not staggered grid
        centred_grid = fv.inverse_reconstruct(grid, sim_variables) if sim_variables.magnetic else grid
        grid_snapshot = sim_variables.convert("conservative", centred_grid, sim_variables)

        ########################

        # Save each instance of the system (primitive variables) at time t, if full_set_required
        if sim_variables.full_set_required:
            with h5py.File(kwargs['hdf5'], "a") as f:
                dataset = f[sim_variables.access_key].create_dataset(str(float(t)), data=grid_snapshot, compression="gzip", compression_opts=9)
                dataset.attrs['t'] = float(t)

        # Miscellaneous media/print options
        if not sim_variables.quiet:
            sim_variables.print_status(sim_variables, t=t)
        if sim_variables.live_plot:
            plotting.update_plot(grid_snapshot, t, sim_variables, *plotting_params)
        if plot_snapshot:
            plotting.plot_snapshot(grid_snapshot, t, sim_variables)
            plot_snapshot = False
        if create_chkpt_file:
            io.write_chkpt_file(grid_snapshot, t, idx, sim_variables)
            create_chkpt_file = False

        ########################

        if t == sim_variables.t_end:
            # Hard exact stop for the simulation; prevents adding an additional computation step
            break
        else:
            # Compute the numerical fluxes at each interface
            fluxes, eigmax = evolvers.evolve_space(grid, sim_variables, first_stage=True)

            # Compute the maximum eigenvalues for determining the full time step
            dt = sim_variables.cfl * eigmax

            # Limit dt to get next checkpoint timing; plot the snapshot or write the checkpoint file at next timestep
            if t+dt >= chkpt*idx:
                dt = chkpt*idx - t
                if sim_variables.save_snaps:
                    plot_snapshot = True
                if sim_variables.write_chkpt:
                    create_chkpt_file = True
                idx += 1

            # Update the solution with the numerical fluxes using iterative methods
            grid = evolvers.evolve_time(grid, fluxes, dt, sim_variables)

            if sim_variables.chemistry:
                chem_grid = krome_funcs.krome_run(chem_grid, grid, dt, sim_variables)

            # Update time step
            t += dt
            sim_variables.timesteps += 1

            # Roll the order of the axis sweep
            sim_variables.axes = np.roll(sim_variables.axes, shift=-1)

    ########################

##############################################################################

# __main__ script; includes handlers and core execution of simulation
def run(seed, save_dir) -> None:
    np.random.seed(seed)

    current_dir = os.getcwd()
    arguments = {}

    # Save the HDF5 file (with seed) to store the temporary data, if full_set_required
    file_name = f"{current_dir}/.astrea_hdf5_temp_{seed}"
    arguments['hdf5'] = file_name

    # Signal handler for Ctrl+C
    def graceful_exit(sig, frame):
        sys.stdout.write('\033[2K\033[1G')
        print(f"{BColours.WARNING}Simulation end by SIGINT; exiting gracefully..{BColours.ENDC}")
        sys.exit(0)

    # Generate the simulation variables from settings (dict); priority: checkpoint file > CLI > parameters.yml
    config_variables = {}
    with open(f"{current_dir}/parameters.yml", "r") as settings_file:
        _config_variables = yaml.safe_load(settings_file)

        # Remove nested dictionary from config_variables
        for parameters in _config_variables.values():
            for k,v in parameters.items():
                config_variables[k] = v

    config_variables['home'] = current_dir
    db_path = f"{current_dir}/static/.db.json"
    config_variables['db_path'] = db_path

    # Check CLI arguments
    arguments.update(io.handle_CLI(db_path))

    # Check for checkpoint file loading
    checkpoint_file = arguments['chkpt_file']
    if checkpoint_file:
        try:
            seed, config_variables, dct = io.load_chkpt_file(config_variables, checkpoint_file)
            arguments.update(dct)
        except Exception as e:
            print(f"{BColours.FAIL}Unable to load checkpoint file: {e}..{BColours.ENDC}")
            sys.exit(0)
    config_variables = io.parse_cli_variables(config_variables, arguments)

    # Generate test configuration based on configuration
    test_variables = tests.generate_test_conditions(config_variables['config'], config_variables['cells'], config_variables['gamma'])

    # Initialise simulation variables
    sim_variables = SimulationVariables(seed, config_variables, test_variables)

    # Auto-generate the resolutions/grid-sizes for run type
    if sim_variables.run_type.startswith('m'):
        if sim_variables.multidimensional:
            _range = 2**np.arange(2,8)
        else:
            _range = 2**np.arange(3,11)
        grid_sizes = np.array([_range,] * sim_variables.dimension).T
    else:
        grid_sizes = [sim_variables.cells]

    ###################################### SCRIPT INITIATE ######################################
    script_start = datetime.now().strftime('%Y%m%d%H%M')
    save_path = f"{current_dir}/{save_dir}/sim{script_start}_{seed}"
    sim_variables.save_path = save_path

    # Make directories if they do not exist
    if sim_variables.save_snaps or sim_variables.save_plots or sim_variables.save_video or sim_variables.save_file or sim_variables.write_chkpt:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    if sim_variables.save_snaps and not os.path.exists(f"{save_path}/snapshots"):
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
                sim_variables.ds[ax] = np.abs(np.diff(sim_variables.axis_coord))/grid_size[ax]

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
                    grp.attrs['precision'] = sim_variables.precision
                    grp.attrs['subgrid'] = sim_variables.subgrid
                    grp.attrs['time_evo'] = sim_variables.time_evo
                    grp.attrs['solver'] = sim_variables.solver

            sim_variables.print_status(sim_variables, status='init')

            ################### CORE ###################
            lap = perf_counter()
            core_run(sim_variables, **arguments)
            elapsed = perf_counter() - lap
            ################### CORE ###################

            # Save attributes after individual run is completed
            sim_variables.elapsed = elapsed
            if sim_variables.full_set_required:
                with h5py.File(file_name, "a") as f:
                    f[sim_variables.access_key].attrs['elapsed'] = elapsed
            if not sim_variables.quiet:
                sim_variables.print_status(sim_variables, status='final')
            ############################# END INDIVIDUAL SIMULATION #############################

        # Save plots; primitive quantities, total variation, conservation equation quantities, solution errors (errors only for run_type=multiple)
        if sim_variables.save_plots:
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
        print(f"\n{BColours.FAIL}-------    Error    -------{BColours.ENDC}")
        print(traceback.format_exc())

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
    env_files = [os.path.join(root, file) for root, _, files in os.walk(os.getcwd()) for file in files if file.endswith('.env')]
    if env_files:
        dotenv.load_dotenv(env_files[0])

    run(*[SEED, SAVE_DIR])