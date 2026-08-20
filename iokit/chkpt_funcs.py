import json

import h5py

from functions.generic import BColours

##############################################################################
# I/O functions for checkpoint files
##############################################################################

# Write grid to HDF5 checkpoint files
def write_chkpt_file(grid, t, idx, sim_variables):
    if sim_variables.test:
        file_name = f"astrea_hdf5_{sim_variables.cells}_chkpt_{sim_variables.timesteps:05}_{t:.6f}".replace('.','')
    else:
        file_name = f"astrea_hdf5_chkpt_{sim_variables.timesteps:05}_{t:.6f}".replace('.','')

    with h5py.File(f"{sim_variables.save_path}/{file_name}", "w") as f:
        f.attrs['datetime'] = sim_variables.access_key
        f.attrs['seed'] = sim_variables.seed
        f.attrs['code'] = 'astrea'
        f.attrs['time'] = float(t)
        f.attrs['idx'] = int(idx)

        f.attrs['config'] = sim_variables.config
        f.attrs['cells'] = sim_variables.cells
        f.attrs['cfl'] = sim_variables.cfl
        f.attrs['gamma'] = sim_variables.gamma
        f.attrs['dimensions'] = sim_variables.dimensions
        f.attrs['eps'] = sim_variables.eps
        f.attrs['subgrid'] = sim_variables.subgrid
        f.attrs['guards'] = sim_variables.guards
        f.attrs['time_evo'] = sim_variables.time_evo
        f.attrs['solver'] = sim_variables.solver
        f.attrs['magnetic'] = sim_variables.magnetic
        f.attrs['units'] = sim_variables.units
        f.attrs['self_gravity'] = sim_variables.self_gravity
        f.attrs['ext_gravity'] = sim_variables.ext_gravity
        f.attrs['boundary'] = sim_variables.boundary
        f.attrs['test_specifics'] = json.dumps(sim_variables.test_specifics)
        f.attrs['coordinates'] = tuple(sim_variables.coordinates.values())
        f.attrs['box_lengths'] = tuple(sim_variables.box_lengths.values())

        f.create_dataset('grid', data=grid, compression="gzip", compression_opts=9)


# Load HDF5 checkpoint files to grid
def load_chkpt_file(file):
    with h5py.File(file, "r") as f:
        try:
            grid = f['grid'][:]
            time = float(f.attrs['time'])
            idx = int(f.attrs['idx'])
        except Exception as e:
            print(f"{BColours.FAIL}Unable to load checkpoint state: {e}..{BColours.ENDC}")
        else:
            return grid, time, idx


# Load HDF5 checkpoint variables
def load_chkpt_variables(file):
    config_variables = {}
    error_message = f"{BColours.WARNING}Checkpoint file not created by astrea..{BColours.ENDC}"

    with h5py.File(file, "r") as f:
        try:
            code = f.attrs['code']
        except Exception:
            print(error_message)
            return None
        else:
            if code != 'astrea':
                print(error_message)
                return None
            else:
                config_variables['seed'] = int(f.attrs['seed'])
                config_variables['config'] = f.attrs['config']
                config_variables['cells'] = f.attrs['cells']
                config_variables['cfl'] = float(f.attrs['cfl'])
                config_variables['gamma'] = float(f.attrs['gamma'])
                config_variables['dimensions'] = int(f.attrs['dimensions'])
                config_variables['eps'] = f.attrs['eps']
                config_variables['subgrid'] = f.attrs['subgrid']
                config_variables['guards'] = f.attrs['guards']
                config_variables['time_evo'] = f.attrs['time_evo']
                config_variables['solver'] = f.attrs['solver']
                config_variables['magnetic'] = f.attrs['magnetic']
                config_variables['units'] = f.attrs['units']
                config_variables['self_gravity'] = f.attrs['self_gravity']
                config_variables['ext_gravity'] = f.attrs['ext_gravity']
                config_variables['boundary'] = f.attrs['boundary']
                config_variables['test_specifics'] = json.loads(f.attrs['test_specifics'])
                config_variables['coordinates'] = {ax:axis_coord for ax, axis_coord in enumerate(f.attrs['coordinates'])}
                config_variables['box_lengths'] = {ax:start_end for ax, start_end in enumerate(f.attrs['box_lengths'])}

                return config_variables