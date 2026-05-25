import sys
import shutil

import numpy as np
from tinydb import TinyDB, Query

from functions.generic import BColours
from simio import chkpt_funcs, cli_funcs, param_funcs

##############################################################################
# Handler for I/O functions; consolidates and allocates user input
##############################################################################

def allocate(seed, basedir, db_path, filename):
    config_variables = {
        'seed': seed,
        'home': basedir,
        'db_path': db_path,
        'hdf5': filename,
    }

    cli_arguments = cli_funcs.parse_CLI(db_path)

    init = cli_arguments.get('init', False)
    if init:
        check_init(basedir)
        sys.exit(0)
    else:
        # Generate the simulation variables from parameters.yml (dict)
        yaml_variables = param_funcs.load_parameters(basedir/"parameters.yml")

    # Priority: checkpoint file > CLI > parameters.yml
    config_variables.update(yaml_variables)
    config_variables.update(cli_arguments)

    checkpoint_file = cli_arguments.get('chkpt_file', False)
    if checkpoint_file:
        try:
            checkpoint_variables = chkpt_funcs.load_chkpt_variables(checkpoint_file)
        except Exception as e:
            print(f"{BColours.FAIL}Unable to load checkpoint file: {e}..{BColours.ENDC}")
        else:
            print(f"{BColours.OKGREEN}Checkpoint file loaded! Running simulation from checkpoint..{BColours.ENDC}")
            config_variables.update(checkpoint_variables)

    return filter_variables(config_variables)


# Check if required dependencies are working
# Copy static/.default.yml -> parameters.yml
def check_init(basedir):
    default = basedir/"static/.default.yml"
    dest = basedir/"parameters.yml"
    if not dest.exists() and default.exists():
        shutil.copy2(default, dest)
        print(f"{BColours.OKGREEN}Created parameters.yml file!{BColours.ENDC}")
    else:
        print(f"{BColours.WARNING}parameters.yml file already exists!{BColours.ENDC}")

    try:
        import git
        import h5py
        import yaml
        import numpy
        import scipy
        import tinydb
        import dotenv
        import pynvml
        import skimage
        import tabulate
        import matplotlib
        import threadpoolctl
    except Exception:
        print(f"{BColours.FAIL}Unable to import some modules. Check if installation is installed properly!{BColours.ENDC}")
    else:
        print(f"{BColours.OKGREEN}Import modules working!{BColours.ENDC}")
    pass


# Check variables if they are valid
def filter_variables(config_variables):
    db, params = TinyDB(config_variables['db_path']), Query()
    eps = np.finfo(float).eps
    config_variables['eps'] = eps

    # Check validity of variables; revert to default values if not valid
    for k,v in config_variables.items():
        if k in ['live_plot', 'save_snaps', 'save_plots', 'save_video', 'save_file']:
            if not isinstance(v, bool):
                v = False
        elif k in ['checkpoints', 'dimensions']:
            if not isinstance(v, int):
                v = 1
            if k == 'dimensions' and not (1 <= v <= 3):
                v = 1
        elif k == "cells":
            if isinstance(v, (int, float)):
                v = [int(v)-int(v)%2,] * config_variables['dimensions']
            elif isinstance(v, str):
                try:
                    v = [int(n)-int(n)%2 for n in v.strip('()').replace(' ','').replace('x',',').split(',')]
                    if len(v) <= 1:
                        v *= config_variables['dimensions']
                except Exception:
                    v = [128,] * config_variables['dimensions']
                else:
                    if len(v) > config_variables['dimensions']:
                        v = v[:config_variables['dimensions']]
            elif isinstance(v, list) or isinstance(v, np.ndarray):
                try:
                    v = [int(_)-int(_)%2 for _ in v]
                except Exception:
                    v = [128,] * config_variables['dimensions']
                else:
                    v = v[:config_variables['dimensions']]
            else:
                v = [128,] * config_variables['dimensions']
        elif k in ['gamma', 'cfl']:
            if not isinstance(v, (int, float)):
                if "/" in v:
                    num, dem = v.split('/')
                    v = float(num)/float(dem)
                else:
                    if k == "gamma":
                        v = 1.4
                    elif k == "cfl":
                        v = .5
                    else:
                        v = 1.
            if k == "gamma" and v == 1:
                v += eps
            if k == "cfl":
                if v <= 0:
                    v = eps
        elif k == "gravity":
            if isinstance(v, str):
                if v.lower() not in ['true', '1', 'self', 'ext', 'external']:
                    v = False
            else:
                if not isinstance(v, bool):
                    if isinstance(v, int):
                        if v not in (0,1):
                            v = False
                        else:
                            v = bool(v)
                    else:
                        v = False
        elif k == "plot_options":
            accepted_plot_options, valid, invalid = db.get(params.type == k)['accepted'], [], []
            try:
                if isinstance(v, str):
                    v = v.replace(' ','').replace('-',',').replace('/',',').replace('|',',').split(',')
                for option in v:
                    _option = option.replace(' ','').replace('-','').replace('_','')
                    if _option.lower() not in accepted_plot_options:
                        invalid.append(option)
                    else:
                        valid.append(option)
                v = [i.lower() for i in valid]
                _ = v[0]  # Check for empty list
            except (IndexError, TypeError):
                v = db.get(params.type == 'default')[k]
                print(f"{BColours.WARNING}No valid plot options; reverting to default values..{BColours.ENDC}")
            finally:
                if invalid != []:
                    print(f"{BColours.WARNING}Invalid plot options: {invalid}{BColours.ENDC}")
        else:
            if k in ['config', 'subgrid', 'time_evo', 'solver']:
                if isinstance(v, str):
                    v = v.lower()

                found = False
                for dct in db.search(params.type == k):
                    if v in dct['accepted']:
                        found = True
                        break

                if not found:
                    v = db.get(params.type == 'default')[k]
                    print(f"{BColours.WARNING}{k.upper()} value not valid; reverting back to default value: {v}..{BColours.ENDC}")

        config_variables[k] = v

    return config_variables


# Make simulation variables when testing functions in Python REPL; 
# most functions require sim_variables, so it might be useful to have a function auto-generate one as needed
def make_sim_variables(file):
    config_variables = {
        'seed': -1,
        'home': '.',
        'db_path': "static/.db.json",
        'hdf5': ".astrea_hdf5_temp_-1",
        'chemistry': False,
        'tracers': False,
        'gravity': False,
        'init': False,
        'verbose': False,
        'write_chkpt': False,
        'test': False,
        'quiet': False,
    }

    yaml_variables = param_funcs.load_parameters(file)
    config_variables.update(yaml_variables)

    config_variables = filter_variables(config_variables)

    from static import tests
    test_variables = tests.generate_test_conditions(config_variables)

    from io import simulation
    return simulation.Variables(config_variables, test_variables)