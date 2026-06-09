import os
import argparse

import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

##############################################################################
# Make simulation variables when testing functions in Python REPL;
# most functions require sim_variables, so it might be useful to have a 
# function auto-generate one as needed
#
# Usage:
# ```python3
# >>> from utilities import make_simvars
# >>> sim_variables = make_simvars.run()
# ```
##############################################################################

def run():
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

    from iokit import handler, param_funcs, simulation
    from static import tests

    try:
        yaml_variables = param_funcs.load_parameters("parameters.yml")
    except Exception as e:
        print(f"Error: {e}")
    else:
        config_variables.update(yaml_variables)
        config_variables = handler.filter_variables(config_variables)

        test_variables = tests.generate_test_conditions(config_variables)

        return simulation.Variables(config_variables, test_variables)