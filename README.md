# mHydyS
mHydyS (pronounced *"Hades"*; the *"m"* is silent 😀) is a one-dimensional (magneto-)hydrodynamics shock code for the purpose of simulating shocks with ions and neutrals, with possible radiative heating and cooling.

# Description
The code is written entirely in Python3 and employs a finite volume (Eulerian) subgrid model with a fixed evenly-spaced grid. The simulation allows for periodic or outlet boundary conditions. It also allows for magnetic fields in one dimension with the magnetic permeability set to one.

### Spatial discretisation
The simulation employs a method-of-lines approach with primitive variables, and can currently perform various reconstruction methods: the piecewise constant method (Godunov, 1959), the piecewise linear method (Derigs et al., 2018) with a *minmod* limiter (Roe, 1986), and the piecewise parabolic method (Felker & Stone, 2017) with face value limiters (Colella et al., 2011) and parabolic interpolations (Colella et al., 2011; Peterson & Hammett, 2013). The reconstruction methods also have the possibility of implementing a slope flattener (Saltzman, 1994), although this feature is not fully tested.

### Time discretisation
The time in the simulation is discretised and evolved with iterative methods. In the following, the strong-stability preserving (SSP) variants of the (explicit) Runge-Kutta (RK) methods are denoted as SSPRK (*i*,*j*), where *i* and *j* refers to *i*-stage and the *j*-th order iterative method respectively. Several SSPRK variants are included for this simulation, with the SSPRK (2,2), SSPRK (3,3) (Shu & Osher, 1988), SSPRK (5,3) (Spiteri & Ruuth, 2002), and SSPRK (5,4) (Gottlieb et al., 2009) methods. The ''classic'' RK4 or the Forward Euler method can also be used.

### Riemann solver
The Local Lax-Friedrichs method (LeVeque, 1992) is used for solving the Riemann problem at each interface for all reconstruction methods, although this method is highly dissipative and only first-order accurate.

### Hydrodynamical tests
Hydrodynamical tests in place are the Sod shock tube test (Sod, 1978), the Sedov blast test (Sedov, 1946), a ''sin-wave'' test, a square-wave advection test, the Shu-Osher shock tube problem (Shu & Osher, 1989), and five shock tube tests from Toro's book (Toro, 1999, p.225). An analytical solution for the Sod shock test (Pfrommer et al., 2006) is also plotted when the Sod shock tube test is run. The solution error (L1 error norm) can also be determined when the ''sin-wave'' test is run.

This code is created as part of the Master's thesis research project at the University of Cologne.

# Installation
Clone this repository onto your local machine, and navigate to the cloned repository. Install the Python dependencies for this code, which can be found in *`requirements.txt`*. It is best practice to create a virtual environment for each project, and installing the project dependencies in that virtual environment.

# Usage
To run the simulation, set your configurations in *`settings.py`* and run *`simulate.py`*.

## Organisation
There are several files in this code for different purposes:

- `simulate.py`: Runs the simulation, and contains the update loop
    | - `evolvers.py`: Collates the functions for space and time evolution
        | - `solvers.py`: Contains the Riemann solver
        | - `reconstruct.py`: Functions for the reconstruction methods
        | - `limiters.py`: Implements flux/slope limiters to prevent spurious oscillations in the reconstructed states
- functions
    | - `generic.py`: Generic functions used throughout the code
    | - `analytic.py`: Analytical solutions to hydrodynamics tests
    | - `fv.py`: Re-usable/generic code specific to the finite volume method
    | - `plotting.py`: Functions for (live-)plotting
- `tests.py`: Hydrodynamics test configurations
- `settings.py`: Parameters for the simulation
