import os
import ctypes
from pathlib import Path

import numpy as np

from functions import grid as gutils
from functions import math as mfuncs
from functions.generic import BColours
from functions.generic import verbose_timer
from numkit import c_transport as ct
from physics import constants
from physics.krome import krome_funcs

##############################################################################
# Chemistry module, for krome, CHIMES or pychem
##############################################################################

# Initialise the chemistry grid
def initialise(sim_variables):

    if sim_variables.chemistry == "pychem":
        pass

    elif sim_variables.chemistry == "chimes":
        pass

    else:
        # Compile krome, with optional network file
        if not (Path.is_file(sim_variables.network) and os.access(sim_variables.network, os.R_OK)):
            sim_variables.network = ''

        options = [
            '-iRHS',
            '-noRecCheck',
            '-coolFile=data/coolZ.dat',
            '-cooling=ATOMIC,H2,DUST,Z,CI,OI,CII',
            '-heating=COMPRESS,PHOTO,CHEM,PHOTODUST'
        ]
        sim_variables.pykrome, sim_variables.species, sim_variables.useX = krome_funcs.build_krome(sim_variables.home, sim_variables.chem_path, sim_variables.network, options)

        if sim_variables.pykrome == None or sim_variables.species == None:
            print(f"{BColours.WARNING}krome built but cannot be accessed. Switching off chemistry..{BColours.ENDC}")
            sim_variables.chemistry = False

        if sim_variables.chemistry:
            chem_grid = krome_funcs.initialise(sim_variables)

    return chem_grid


# Update chemical grid within the timestep dt, requires conservative grid info
def update(chemical_grid, grid, dt, sim_variables):

    if sim_variables.chemistry == "pychem":
        pass

    elif sim_variables.chemistry == "chimes":
        pass

    else:
        return krome_funcs.krome_run(chemical_grid, grid, dt, sim_variables)