import os
from concurrent.futures import ProcessPoolExecutor

import yaml
import numpy as np

from functions import grid as gutils
from functions.generic import BColours
from numkit import c_transport as ct

##############################################################################
# Functions for pychem routines
##############################################################################

# Fractional abundances (n_i/n_H) evolved by pychem; C+ is only evolved by networks 15 & 17 (pychem is built with network 5)
SPECIES = {'H2': .49, 'H+': 1e-7, 'CO': 1e-9, 'C+': 0.}

# Arguments of flchem_init and the radiation field arguments of flchem_evolve, in order; defaults from physics/pychem/test.py
INIT_PARAMS = {'abundC': 1.4e-4, 'abundO': 3.2e-4, 'dust_to_gas_ratio': 1., 'cosmic_ray_ion_rate': 3e-17, 'NH_ext': 1e20, 'Z_atom': 1.}
FIELD_PARAMS = {'G0': 1.7, 'G0RP': 0., 'fshield_H2': 1., 'fshield_CO': 1., 'AV_mean': 0., 'chi_mean': 1.}


# Worker initialiser; flchem is only imported in the workers because importing it re-enables the GIL in free-threaded Python
def _init_worker(init_params):
    global flchem
    import flchem
    flchem.flchem_init(*init_params)


# Evolve a single cell; returns (istatus, abh2, abhp, abco, abcp, temperature, dust_temp, eint)
def _evolve_cell(*args):
    return flchem.flchem_evolve(*args)[1:9]


# Initialise the pool of pychem workers (processes, as the Fortran COMMON blocks are not thread-safe) and the chemical grid [*cells, species]
# Defaults can be overridden with a flat .yml file of species and/or parameters, e.g. {H2: 1e-6, G0: 1.7}
def initialise(sim_variables):
    abundances, init_params, field_params = dict(SPECIES), dict(INIT_PARAMS), dict(FIELD_PARAMS)

    if sim_variables.abundances:
        try:
            with open(sim_variables.abundances, "r") as abd:
                overrides = {key: float(value) for key, value in yaml.safe_load(abd).items()}
        except Exception as e:
            print(f"{BColours.WARNING}Unable to load initial abundance file:\n{e}\nUsing default abundances..{BColours.ENDC}")
        else:
            for params in (abundances, init_params, field_params):
                params.update({key: value for key, value in overrides.items() if key in params})

    if sim_variables.units == "code":
        print(f"{BColours.WARNING}Chemistry is running with code units; the grid values are taken as CGS units..{BColours.ENDC}")

    sim_variables.species, sim_variables.chem_params = list(abundances), field_params
    sim_variables.chem_pool = ProcessPoolExecutor(initializer=_init_worker, initargs=(tuple(init_params.values()),))

    return np.full(list(sim_variables.cells)+[len(abundances),], list(abundances.values()), dtype=float)


# Evolve the abundances from t to t+dt with the updated conservative grid, in CGS units
# If CHEM_COOLING is set, the heating/cooling from pychem is added to the energy of the conservative grid (in-place)
def pychem_run(chem_grid, conserv_grid, dt, sim_variables):
    scales = sim_variables.constants.cgs_scales

    convert = ct.convert if sim_variables.magnetic else gutils.convert
    primitive_grid = convert("conservative", conserv_grid, sim_variables)

    # pychem takes T = 2/3 eint/(n k_B), so eint = 3/2 P keeps T consistent with the hydro pressure for any gamma
    dens = primitive_grid[...,sim_variables.rho] * scales['density']
    eint = 1.5 * primitive_grid[...,sim_variables.pressure] * scales['pressure']
    divv = sum(np.gradient(primitive_grid[...,sim_variables.vx+ax], sim_variables.ds[ax].item(), axis=ax) for ax in range(sim_variables.dimensions)) / scales['time']
    dl = np.min(list(sim_variables.ds.values())) * scales['length']
    dt_cgs = dt * scales['time']

    # Cells with non-positive density or energy (e.g. negative pressures from the hydro) cannot be evolved and keep their abundances
    valid = (dens > 0) & (eint > 0)

    # Arguments of flchem_evolve for each cell; total time = timestep = dt for a single step
    args = (dt_cgs, dt_cgs, *sim_variables.chem_params.values(), dl, divv, dens, eint, *np.moveaxis(chem_grid, -1, 0))
    columns = [np.broadcast_to(arg, dens.shape)[valid].tolist() for arg in args]

    chunksize = max(1, np.count_nonzero(valid) // (4*(os.cpu_count() or 1)))
    results = np.array(list(sim_variables.chem_pool.map(_evolve_cell, *columns, chunksize=chunksize))).reshape(-1, 8)

    # pychem returns the input values for cells that failed (istatus != 999)
    failed = dens.size - np.count_nonzero(results[:,0] == 999)
    if failed:
        print(f"\033[2K{BColours.WARNING}pychem could not evolve {failed} cell(s); keeping their previous abundances..{BColours.ENDC}")

    if sim_variables.chem_cooling:
        conserv_grid[...,sim_variables.energy][valid] += 2/3 * (results[:,-1] - eint[valid]) / scales['pressure'] / (sim_variables.gamma-1)

    new_chem_grid = np.copy(chem_grid)
    new_chem_grid[valid] = results[:,1:5]
    return new_chem_grid
