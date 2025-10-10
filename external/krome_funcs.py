import os
import glob
import ctypes
import subprocess
from textwrap import dedent

import yaml
import numpy as np

from functions import fv
from functions.generic import BColours
from functions.generic import verbose_timer
from static import constants

##############################################################################
# Functions for krome routines
##############################################################################

# Optional .f90 wrapper for explicit C symbols; increases type safety and robustness between Python/ctypes and Fortran
def write_krome_ctypes(filename, useX):
    args = "Tgas, dt"
    batch_args = "Tgas(i), dt"
    batch_args2 = "Tgas(cells)"
    if useX:
        args = "rho, " + args
        batch_args = "rho(i), " + batch_args
        batch_args2 = "rho(cells), " + batch_args2

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
          integer, parameter            :: nsp = krome_nmols
          real(c_double), intent(inout) :: x(nsp)
          real(c_double), value         :: {args}

          call krome(x, {args})
        end subroutine krome_

        subroutine krome_batch_(xall, {args}, cells) bind(C, name="krome_batch_")
          integer, parameter            :: nsp = krome_nmols
          integer, value                :: cells
          real(c_double), intent(inout) :: xall(nsp,cells)
          real(c_double), intent(in)    :: {batch_args2}
          real(c_double), value         :: dt
          integer :: i

          do i = i, cells
            call krome(xall(:,i), {batch_args})
          end do
        end subroutine krome_batch_

        end module krome_ctypes_mod""").strip("\n")
    with open(filename, 'w') as writer:
        writer.write(text)
    pass


# Build krome with network file and write .f90 file for loading into astrea
def build_krome(paths, options):
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
            subprocess.run(["gfortran", "-ffree-line-length-none", "-w", "-fallow-argument-mismatch", "-fPIC", "-O3", "-c",] + glob.glob("*.f90") + glob.glob("*.f"))
            filename = 'krome_ctypes.f90'
            if not os.path.isfile(os.path.join(krome_path, 'build', filename)):
                write_krome_ctypes(filename, useX)
            subprocess.run(["gfortran", "-ffree-line-length-none", "-w", "-fallow-argument-mismatch", "-fPIC", "-O3", "-c", filename])
            subprocess.run(["gfortran", "-shared", "-o", "libkrome.so"] + glob.glob("*.o"))

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

            argtypes = [np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")]  # x(:)
            if useX:
                argtypes += [ctypes.POINTER(ctypes.c_double)]*3  # rho [g/cm3], Tgas [K], dt [s]
            else:
                argtypes += [ctypes.POINTER(ctypes.c_double)]*2  # Tgas [K], dt [s]
            pykrome.krome_.argtypes = argtypes

            # krome batch solver: krome(xall(:,i), [rho], Tgas(i), dt)
            pykrome.krome_batch_.restype = None
            batch_argtypes = [np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="F_CONTIGUOUS")]  # x(:,i)
            if useX:
                batch_argtypes += [np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")]  # rho(i) [g/cm3]
            batch_argtypes += [
                np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # Tgas(i) [K]
                ctypes.POINTER(ctypes.c_double),  # dt [s]
                ctypes.POINTER(ctypes.c_int)  # cells
            ]
            pykrome.krome_batch_.argtypes = batch_argtypes

            # ---------------------------------------------
            # Initialize krome
            # ---------------------------------------------
            pykrome.krome_init_()

        except Exception as e:
            print(f"{BColours.WARNING}Failed to build krome:\n{e}{BColours.ENDC}")

        finally:
            os.chdir(astrea_path)

    return pykrome, species, useX


# Initialise the (column-major) grid (because of Fortran)
def initialise(sim_variables, perturb=False):
    abundance_file = f"{sim_variables.home}/external/abundances.yml"
    file_valid = False

    if sim_variables.abundances:
        try:
            with open(sim_variables.abundances, "r") as abd:
                _ = yaml.safe_load(abd)
        except Exception as e:
            print(f"{BColours.WARNING}Unable to load initial abundance file:\n{e}\nUsing default abundances..{BColours.ENDC}")
        else:
            abundance_file = sim_variables.abundances
            file_valid = True

    else:
        print(f"{BColours.WARNING}Using default abundances..{BColours.ENDC}")

    with open(abundance_file, "r") as abd:
        _abundances = yaml.safe_load(abd)

        if file_valid:
            abundances = _abundances
        else:
            if sim_variables.useX:
                abundances = _abundances['mass_frac_abundances']
            else:
                abundances = _abundances['num_dens_abundances']

    size = [len(sim_variables.species),] + sim_variables.cells
    network = np.full(shape=size, fill_value=1e-20, dtype=np.float64, order="F")

    for mol, abundance in abundances.items():
        try:
            mol_idx = np.where(mol == np.array(sim_variables.species))[0][0]
        except:
            pass
        else:
            # Uniform population of species abundance across the grid
            network[mol_idx,...] = abundance

            # Add a small perturbation to the initial abundances by 2 orders of magnitude (1%)
            if perturb:
                network[mol_idx,...] += .1 * np.random.uniform(-abundance, abundance, size=sim_variables.cells)

    return network


# Solve the chemical ODEs for each grid cell;
# WARNING: depending on the grid size and the number of chemical species, the computation time can explode
#@verbose_timer
def krome_run(chem_grid, conserv_grid, dt, sim_variables):
    conversion_factor = (constants.mu * constants.m_p * constants.pc**3)/constants.m_sun

    centred_grid = fv.inverse_reconstruct(conserv_grid, sim_variables) if sim_variables.magnetic else conserv_grid
    primitive_grid = sim_variables.convert('conservative', centred_grid, sim_variables)

    Tgas = fv.divide(primitive_grid[...,sim_variables.pressure], primitive_grid[...,sim_variables.rho]).reshape(-1, order="F")

    xall = chem_grid.reshape(chem_grid.shape[0], -1) * conversion_factor
    ncells = xall.shape[-1]

    if sim_variables.useX:
        density = primitive_grid[...,sim_variables.rho].reshape(-1, order="F")
        sim_variables.pykrome.krome_batch_(np.asfortranarray(xall), density, Tgas, ctypes.c_double(dt), ctypes.c_int(ncells))
    else:
        sim_variables.pykrome.krome_batch_(np.asfortranarray(xall), Tgas, ctypes.c_double(dt), ctypes.c_int(ncells))

    return xall.reshape(chem_grid.shape)/conversion_factor