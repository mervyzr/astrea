from pathlib import Path

from iokit import handler, simulation
from static import tests

##############################################################################
# Make simulation variables when testing functions in Python REPL;
# most functions require sim_variables, so it might be useful to have a 
# function auto-generate one as needed
#
# Usage:
# ```python3
# >>> from utilities import make_simvars
# >>> sim_variables = simvars.create()
# ```
##############################################################################

def create(**kwargs):
    if kwargs:
        try:
            seed = kwargs['seed']
        except:
            seed = -1

        try:
            basedir = kwargs['basedir']
        except:
            basedir = Path(__file__).resolve().parent

        try:
            db_path = kwargs['db']
        except:
            try:
                db_path = kwargs['db_path']
            except:
                db_path = basedir/"static"/".db.json"

        try:
            filename = kwargs['file']
        except:
            try:
                filename = kwargs['filename']
            except:
                filename = basedir/f".astrea_hdf5_temp_{seed}"

    else:
        seed = -1
        basedir = Path().cwd()
        db_path = basedir/"static"/".db.json"
        filename = basedir/f".astrea_hdf5_temp_{seed}"

    config_variables = handler.allocate(seed, basedir, db_path, filename)
    test_variables = tests.generate_test_conditions(config_variables)
    return simulation.Variables(config_variables, test_variables)