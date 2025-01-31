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
import numpy as np

from static import tests
from num_methods import evolvers
from functions import constructor, generic, plotting

##############################################################################
# Main script
##############################################################################

# Globals
CURRENT_DIR = os.getcwd()
SEED = np.random.randint(0, 1e8)


# Finite volume shock function
def core_run(hdf5: str, sim_variables: namedtuple, *args, **kwargs):
    try:
        chkpts = kwargs['checkpoints']
    except KeyError:
        chkpts = 10
    chkpt = sim_variables.t_end/chkpts

    # Initialise the discrete solution array with primitive variables <w> and convert them to conservative variables
    grid = constructor.initialise(sim_variables, convert=True)
    plot_axes = sim_variables.permutations[-1]

    # Initiate live or snapshot plotting, if enabled
    if sim_variables.live_plot:
        plotting_params = plotting.initiate_live_plot(sim_variables)
    elif sim_variables.take_snaps:
        chkpt = sim_variables.t_end/sim_variables.snapshots
        take_snapshot = True

    # Start simulation run
    t, idx = 0., 1
    while t <= sim_variables.t_end:
        # Saves each instance of the system (primitive variables) at time t
        grid_snapshot = sim_variables.convert_conservative(grid, sim_variables).transpose(plot_axes)
        with h5py.File(hdf5, "a") as f:
            dataset = f[sim_variables.access_key].create_dataset(str(float(t)), data=grid_snapshot)
            dataset.attrs['t'] = float(t)

        # Miscellaneous media/print options
        if not sim_variables.quiet:
            generic.print_progress(t, sim_variables)

        if sim_variables.live_plot:
            plotting.update_plot(grid_snapshot, t, sim_variables, *plotting_params)
        elif sim_variables.take_snaps and take_snapshot:
            plotting.plot_snapshot(grid_snapshot, t, sim_variables, save_path=f"./savedData/snap{sim_variables.seed}")
            take_snapshot = False

        if t == sim_variables.t_end:
            # Exact stop for the simulation; prevents adding an additional computation step
            break
        else:
            # Compute the numerical fluxes at each interface
            fluxes = evolvers.evolve_space(grid, sim_variables)

            # Compute the maximum eigenvalues for determining the full time step
            eigmaxes = [sim_variables.dx/Riemann_values['eigmax'] for Riemann_values in list(fluxes.values())]
            dt = sim_variables.cfl * min(eigmaxes)

            # Handle dt
            if t+dt >= chkpt*idx:
                dt = chkpt*idx - t
                if sim_variables.take_snaps:
                    take_snapshot = True
                idx += 1

            # Update the solution with the numerical fluxes using iterative methods
            grid = evolvers.evolve_time(grid, fluxes, dt, sim_variables)

            # Update time step
            t += dt

##############################################################################

# Main script; includes handlers and core execution of simulation code
def run() -> None:
    np.random.seed(SEED)

    # Save the HDF5 file (with seed) to store the temporary data
    file_name = f"{CURRENT_DIR}/.tempSimData_{SEED}.hdf5"

    # Signal handler for Ctrl+C
    def graceful_exit(sig, frame):
        sys.stdout.write('\033[2K\033[1G')
        print(f"{generic.BColours.WARNING}Simulation end by SIGINT; exiting gracefully..{generic.BColours.ENDC}")
        sys.exit(0)

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
        if _sim_variables['dimension'] == 2:
            itr_list = 2**np.arange(2,8)
        else:
            itr_list = 2**np.arange(3,11)
    else:
        itr_list = [_sim_variables['cells']]

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
        # Initiate the HDF5 database to store data
        with h5py.File(file_name, "w") as f:
            f.attrs['datetime'] = script_start
            f.attrs['seed'] = sim_variables.seed

        for _var in itr_list:
            ############################# INDIVIDUAL SIMULATION #############################
            now = datetime.now()

            # Update cells (and grid width) in simulation variables (namedtuple)
            sim_variables = sim_variables._replace(access_key=now.strftime('%Y%m%d%H%M%S')+str(now.microsecond))
            sim_variables = sim_variables._replace(now=now)
            sim_variables = sim_variables._replace(cells=_var)
            sim_variables = sim_variables._replace(dx=abs(sim_variables.end_pos-sim_variables.start_pos)/sim_variables.cells)

            # Save simulation variables into HDF5 file
            with h5py.File(file_name, "a") as f:
                grp = f.create_group(sim_variables.access_key)
                grp.attrs['config'] = sim_variables.config
                grp.attrs['cells'] = sim_variables.cells
                grp.attrs['cfl'] = sim_variables.cfl
                grp.attrs['gamma'] = sim_variables.gamma
                grp.attrs['dimension'] = sim_variables.dimension
                grp.attrs['subgrid'] = sim_variables.subgrid
                grp.attrs['timestep'] = sim_variables.timestep
                grp.attrs['solver'] = sim_variables.solver

            ################### CORE ###################
            lap = perf_counter()
            core_run(file_name, sim_variables)
            elapsed = perf_counter() - lap
            ################### CORE ###################

            # Save attributes after individual run is completed
            sim_variables = sim_variables._replace(elapsed=elapsed)
            with h5py.File(file_name, "a") as f:
                f[sim_variables.access_key].attrs['elapsed'] = elapsed
                timestep_count = len(f[sim_variables.access_key])
            if not sim_variables.quiet:
                generic.print_final(sim_variables, timestep_count)
            ############################# END INDIVIDUAL SIMULATION #############################

        # Save plots; primitive quantities, total variation, conservation equation quantities, solution errors (errors only for run_type=multiple)
        with h5py.File(file_name, "r") as f:
            if sim_variables.save_plots:
                plotting.plot_quantities(f, sim_variables, save_path)
                if sim_variables.run_type.startswith("m"):
                    if sim_variables.config_category == "smooth":
                        plotting.plot_solution_errors(f, sim_variables, save_path, error_norm=1)
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
            print(f"{generic.BColours.FAIL}-- Error: {e}{generic.BColours.ENDC} (use --debug option for more details)")

    finally:
        # Save the temporary HDF5 database (!! Possibly large file sizes > 100GB !!)
        if sim_variables.save_file:
            shutil.move(file_name, f"{save_path}/mhydys_{sim_variables.config}_{sim_variables.subgrid}_{sim_variables.timestep}_{SEED}.hdf5")
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

    run()