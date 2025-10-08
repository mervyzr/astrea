import os
import ctypes
import subprocess
import concurrent.futures
from textwrap import dedent
from itertools import repeat

import numpy as np

from functions import fv
from functions.generic import BColours
from static import constants

##############################################################################
# Functions for krome routines
##############################################################################

# Set the initial abundances of chemical species of interest, 
# accepts a dictionary of atom/molecule/ion name as key and the number densities [1/cm3] or mass fraction [X] as value
ABUNDANCES = {
    'H': 1e4,
    'H2': 1e4,
}


# Optional .f90 wrapper for explicit C symbols; increases type safety and robustness between Python/ctypes and Fortran
def write_krome_ctypes(filename, useX):
    args = "Tgas, dt"
    if useX:
        args = "rho, " + args

    text = dedent(f"""\
        module krome_ctypes_mod
          use iso_c_binding
          use krome_main
          use krome_user
          implicit none
        contains

        subroutine krome_init_() bind(C, name="krome_init_")
          call krome_init()
        end subroutine krome_init_

        subroutine krome_(x, {args}) bind(C, name="krome_")
          integer, parameter :: nsp = krome_nmols
          real(c_double), intent(inout) :: x(nsp)
          real(c_double), intent(in) :: {args}

          call krome(x, {args})
        end subroutine krome_

        end module krome_ctypes_mod""").strip("\n")
    with open(filename, 'w') as writer:
        writer.write(text)
    pass


# Build krome with network file and write .f90 file for loading into astrea
def build_krome(paths, robust=True, options=['-noRecCheck', '-iRHS']):
    astrea_path, krome_path, network_path = paths
    pykrome, species = None, None
    useX = '-useX' in options

    if krome_path:
        os.chdir(krome_path)

        # Pre-process krome to create build folder if chemical network provided
        if network_path:
            if os.path.isfile(network_path) and os.access(network_path, os.R_OK):
                print(f"Chemistry switched on. Follow the prompts in krome for setup :\n")
                subprocess.run(["./krome", f"-n={network_path}"] + options)
            else:
                print(f"{BColours.WARNING}Error reading network file..{BColours.ENDC}")

        try:
            os.chdir(os.path.join(krome_path, 'build'))

            # Save the species used for chemical networks
            species = []
            with open('species.gps', 'r') as reader:
                lines = reader.readlines()
                for line in lines:
                    if line.startswith('krome_idx'):
                        krome_idx = line.split(' ')[0]
                        # CR: cosmic ray, g: photons, Tgas: dust, dummy: dummy
                        if not krome_idx.endswith(('CR', 'g', 'Tgas', 'dummy')):
                            species.append(krome_idx.split('_')[-1])

            # Compile the .f90 & .f files with gfortran
            subprocess.run(["make", "gfortran"])
            subprocess.run(["./test"])

            # Build and expose the Fortran routines to Python with ctypes wrapper
            subprocess.run(["gfortran", "-ffree-line-length-none", "-w", "-fallow-argument-mismatch", "-fPIC", "-O3", "-c", "*.f", "*.f90"])
            if robust:
                filename = 'krome_ctypes.f90'
                if not os.path.isfile(os.path.join(krome_path, 'build', filename)):
                    write_krome_ctypes(filename, useX)
                subprocess.run(["gfortran", "-ffree-line-length-none", "-w", "-fallow-argument-mismatch", "-fPIC", "-O3", "-c", filename])
            subprocess.run(["gfortran", "-shared", "-o", "libkrome.so", "*.o"])

            # Load the shared library
            pykrome = ctypes.CDLL(os.path.join(krome_path, 'build', 'libkrome.so'))

            # ---------------------------------------------
            # Declare subroutine prototypes for ctypes
            # ---------------------------------------------

            # krome init
            pykrome.krome_init_.restype = None
            pykrome.krome_init_.argtypes = []

            # krome solver: krome(x(:), [rho], Tgas, dt)
            # void krome(double* x, double* Tgas, double* dt)
            pykrome.krome_.restype = None
            if useX:
                pykrome.krome_.argtypes = [ctypes.POINTER(ctypes.c_double)] * 4  # x(:), rho [g/cm3], Tgas [K], dt [s]
            else:
                pykrome.krome_.argtypes = [ctypes.POINTER(ctypes.c_double)] * 3  # x(:), Tgas [K], dt [s]

            # ---------------------------------------------
            # Initialize krome
            # ---------------------------------------------
            pykrome.krome_init_()

        except Exception as e:
            print(f"{BColours.WARNING}Failed to build krome:\n{e}{BColours.ENDC}")

        finally:
            os.chdir(astrea_path)

    return pykrome, species, useX


# Initialise the grid
def initialise(cells, species, abundances=ABUNDANCES):
    size = cells + [len(species),]
    network = np.full(shape=size, fill_value=1e-20, dtype=np.float64)

    for mol, abundance in abundances:
        try:
            mol_idx = np.where(np.array(species) == mol)[0][0]
        except:
            pass
        else:
            network[...,mol_idx] = abundance
    return network


# Solve the chemical ODEs for each grid cell;
# WARNING: depending on the grid size and the number of chemical species, the computation time can explode
def run(chem_grid, conserv_grid, dt, sim_variables):
    conversion_factor = (constants.pc/constants.Myr)**2 * (constants.mu * constants.m_p)/constants.k_B

    def krome_per_cell(_abundance, _cell):
        abundance = _abundance.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        _dt = ctypes.c_double(constants.Myr * dt)

        Tgas = ctypes.c_double(conversion_factor * fv.divide(_cell[...,sim_variables.pressure], _cell[...,sim_variables.rho]))

        if sim_variables.useX:
            sim_variables.pykrome.krome_(abundance, _cell[...,sim_variables.rho]*(constants.m_sun/(constants.pc**3)), ctypes.byref(Tgas), ctypes.byref(_dt))
        else:
            sim_variables.pykrome.krome_(abundance, ctypes.byref(Tgas), ctypes.byref(_dt))

        return abundance

    primitive_grid = fv.convert_variable('energy', conserv_grid, sim_variables)
    _chem_grid = chem_grid.reshape(-1, chem_grid.shape[-1])
    _prim_grid = primitive_grid.reshape(-1, primitive_grid.shape[-1])

    with concurrent.futures.ThreadPoolExecutor() as executor:
        jobs = executor.map(krome_per_cell, list(_chem_grid), list(_prim_grid))
        new_chem_grid = np.array([job for job in jobs]).reshape(chem_grid.shape)

    return new_chem_grid