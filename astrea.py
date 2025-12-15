#!/usr/bin/env python3

import sys
import shutil
import signal
import warnings
import traceback
from pathlib import Path
from datetime import datetime
from time import perf_counter, process_time

import h5py
import yaml
import numpy as np

from external import krome_funcs
from functions import constructor, fv, generic, io, plotting
from num_methods import ct, evolvers, tracers
from static import tests

##############################################################################
# Main script
##############################################################################

# Globals
SAVE_DIR = "saved_data"
SEED = np.random.randint(0, 1e8)

# Global settings
warnings.filterwarnings('ignore')
np.set_printoptions(linewidth=1000, suppress=True)

# Finite volume simulation
def core_run(sim_variables, **kwargs):
    # Initialise or load the discrete solution array with primitive variables <w>
    if sim_variables.chkpt_file:
        primitive_grid, t, idx = kwargs['grid'], kwargs['time'], kwargs['idx']
    else:
        primitive_grid, t, idx = constructor.initialise(sim_variables), 0., 1

    # For magnetic simulations, inverse reconstruction needed for the conversion, as well as returning converted centred grid to staggered values
    centred_grid = ct.inverse_reconstruct(primitive_grid, sim_variables) if sim_variables.magnetic else primitive_grid

    # Convert primitive grid to conservative variables <q>
    grid = fv.point_convert("primitive", centred_grid, sim_variables)
    grid[...,5+sim_variables.axes] = primitive_grid[...,5+sim_variables.axes]

    ########################

    # Initialise the chemical grid if activated
    # Abundances can be overriden; accepts a dictionary of atom/molecule/ion name as key and the number densities [1/cm3] or mass fraction [X] as value
    if sim_variables.chemistry:
        chem_grid = krome_funcs.initialise(sim_variables, perturb=True)

    ########################

    # Initialise the tracer particles if activated
    if sim_variables.tracers:
        tracer_positions = tracers.initialise(sim_variables)

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
        centred_grid = ct.inverse_reconstruct(grid, sim_variables) if sim_variables.magnetic else grid
        grid_snapshot = sim_variables.convert("conservative", centred_grid, sim_variables)

        ########################

        # Save each instance of the system (primitive variables) at time t, if full_set_required
        if sim_variables.full_set_required:
            with h5py.File(sim_variables.hdf5, "a") as f:
                dataset = f[sim_variables.access_key].create_dataset(str(float(t)), data=grid_snapshot, compression="gzip", compression_opts=9)
                dataset.attrs['t'] = float(t)

        # Miscellaneous media/print options
        if not sim_variables.quiet:
            sim_variables.print_status(sim_variables, t=t)
        if sim_variables.live_plot:
            plotting.update_plot(grid_snapshot, t, sim_variables, *plotting_params)
        if plot_snapshot:
            plotting.plot_snapshot(grid_snapshot, t, sim_variables)
            plotting.plot_tracer_particles(tracer_positions, t, sim_variables)
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

            # Update chemical grid
            if sim_variables.chemistry:
                chem_grid = krome_funcs.krome_run(chem_grid, grid, dt, sim_variables)

            # Update tracer particles
            if sim_variables.tracers:
                tracer_positions = tracers.update(tracer_positions, grid, dt, sim_variables)

            # Update time step
            t += dt
            sim_variables.timesteps += 1

            # Roll the order of the axis sweep
            sim_variables.axes = np.roll(sim_variables.axes, shift=-1)

            # Post update steps (if any)

    ########################

##############################################################################

# Signal handler for Ctrl+C
def graceful_exit(sig, frame):
    sys.stdout.write('\033[2K\033[1G')
    print(f"{generic.BColours.WARNING}Simulation end by SIGINT; exiting gracefully..{generic.BColours.ENDC}")
    sys.exit(0)


# __main__ script; includes handlers and core execution of simulation
def run(seed=SEED, save_dir=SAVE_DIR) -> None:
    np.random.seed(seed)

    current_dir = Path(__file__).resolve().parent
    db_path = current_dir/"static"/".db.json"
    file_name = current_dir/f".astrea_hdf5_temp_{seed}"

    config_variables = {
        'home': current_dir,
        'db_path': db_path,
        'hdf5': file_name,
    }

    cli_arguments = io.handle_CLI(db_path)

    init = cli_arguments.get('init', False)
    if init:
        generic.check_init(current_dir)
        sys.exit(0)
    else:
        # Generate the simulation variables from settings (dict)
        try:
            with open(current_dir/"parameters.yml", "r") as settings_file:
                _config_variables = yaml.safe_load(settings_file)
        except Exception:
            print(f"{generic.BColours.FAIL}parameters.yml file not found; please run astrea again with the --init option..{generic.BColours.ENDC}")
        else:
            # Remove nested dictionary from config_variables
            for parameters in _config_variables.values():
                for k,v in parameters.items():
                    config_variables[k] = v

    # Replace the relevant configuration variables; priority: checkpoint file > CLI > parameters.yml
    config_variables.update(cli_arguments)

    checkpoint_file = cli_arguments['chkpt_file']
    if checkpoint_file:
        try:
            seed, config_variables, chkpt_kwargs = io.load_chkpt_file(config_variables, checkpoint_file)
        except Exception as e:
            print(f"{generic.BColours.FAIL}Unable to load checkpoint file: {e}..{generic.BColours.ENDC}")
        else:
            print(f"{generic.BColours.OKGREEN}Checkpoint file loaded! Running simulation from checkpoint..{generic.BColours.ENDC}")

    config_variables = io.filter_variables(config_variables)

    # Generate test configuration based on configuration
    test_variables = tests.generate_test_conditions(config_variables)

    # Initialise simulation variables
    sim_variables = io.SimulationVariables(seed, config_variables, test_variables)

    # Auto-generate the resolutions/grid-sizes for run type
    if sim_variables.test:
        if sim_variables.multidimensional:
            _range = 2**np.arange(2,8)
        else:
            _range = 2**np.arange(3,11)
        grid_sizes = np.array([_range,] * sim_variables.dimensions).T
    else:
        grid_sizes = [sim_variables.cells]

    ###################################### SCRIPT INITIATE ######################################
    script_start = datetime.now().strftime('%Y%m%d%H%M')
    save_path = current_dir/f"{save_dir}/sim{script_start}_{seed}"
    sim_variables.save_path = save_path

    # Make directories if they do not exist
    if sim_variables.save_snaps or sim_variables.save_plots or sim_variables.save_video or sim_variables.save_file or sim_variables.write_chkpt:
        Path(save_path).mkdir(parents=True, exist_ok=True)
    if sim_variables.save_snaps:
        Path(save_path/"snapshots").mkdir(parents=True, exist_ok=True)

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
            sim_variables.ds = {ax: np.abs(np.diff(sim_variables.axis_coord[ax]))/grid_size[ax] for ax in range(len(grid_size))}

            # Save simulation variables into HDF5 file
            if sim_variables.full_set_required:
                with h5py.File(file_name, "a") as f:
                    grp = f.create_group(sim_variables.access_key)
                    grp.attrs['config'] = sim_variables.config
                    grp.attrs['cells'] = sim_variables.cells
                    grp.attrs['cfl'] = sim_variables.cfl
                    grp.attrs['gamma'] = sim_variables.gamma
                    grp.attrs['dimensions'] = sim_variables.dimensions
                    grp.attrs['precision'] = sim_variables.precision
                    grp.attrs['eps'] = sim_variables.eps
                    grp.attrs['subgrid'] = sim_variables.subgrid
                    grp.attrs['time_evo'] = sim_variables.time_evo
                    grp.attrs['solver'] = sim_variables.solver

            sim_variables.print_status(sim_variables, status='init')

            ################### CORE ###################
            lap, cpu_start = perf_counter(), process_time()
            core_run(sim_variables, **chkpt_kwargs) if sim_variables.chkpt_file else core_run(sim_variables)
            elapsed, cpu_elapsed = perf_counter() - lap, process_time() - cpu_start
            ################### CORE ###################

            # Save attributes after individual run is completed
            sim_variables.elapsed, sim_variables.cpu_elapsed = elapsed, cpu_elapsed
            if sim_variables.full_set_required:
                with h5py.File(file_name, "a") as f:
                    f[sim_variables.access_key].attrs['elapsed'] = elapsed
            if not sim_variables.quiet:
                sim_variables.print_status(sim_variables, status='final')
            ############################# END INDIVIDUAL SIMULATION #############################

        if sim_variables.save_plots:
            with h5py.File(file_name, "r") as f:
                plotting.plot_quantities(f, sim_variables)
                if sim_variables.test:
                    if sim_variables.config_category == "smooth":
                        plotting.plot_solution_errors(f, sim_variables, error_norm=1)
                else:
                    plotting.plot_total_variation(f, sim_variables)
                    plotting.plot_conservation_equations(f, sim_variables)
                    if "kelvin" in sim_variables.config or "helmholtz" in sim_variables.config or "khi" in sim_variables.config:
                        plotting.plot_turbulence_spectrum(f, sim_variables, bins=8, normalise=False)

        if sim_variables.save_video:
            with h5py.File(file_name, "r") as f:
                vidpath = save_path/".vidplots"
                Path(vidpath).mkdir(parents=True, exist_ok=True)
                plotting.make_video(f, sim_variables, vidpath)

    # Exception handling; deletes the temporary HDF5 database to prevent clutter
    except Exception as e:
        print(end='\x1b[2K')
        print(f"\n{generic.BColours.FAIL}{'='*15} Error : {seed} {'='*15}{generic.BColours.ENDC}")
        print(traceback.format_exc())

    finally:
        # Save the temporary HDF5 database (!! Possibly large file sizes > 100 GB !!)
        if sim_variables.full_set_required:
            if sim_variables.save_file:
                shutil.move(file_name, f"{save_path}/astrea_{sim_variables.config}_{sim_variables.subgrid}_{sim_variables.time_evo}_{sim_variables.seed}.hdf5")
            else:
                Path.unlink(file_name, missing_ok=True)

        signal.signal(signal.SIGINT, original_sigint_handler)

    ###################################### SCRIPT END ######################################

if __name__ == "__main__":
    run()