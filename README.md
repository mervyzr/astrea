# shock1D
shock1D is a 1D-hydrodynamics code for the purpose of simulating shocks and turbulence with ions/neutrals interaction.

# Description
The code is written entirely in Python3 and employs a finite volume method (Eulerian) with a fixed evenly-spaced grid. The simulation allows for periodic boundary conditions or outflows.

The code currently employs a piecewise constant method (Godunov, 1959), a piecewise linear reconstruction method (Derigs et al., 2018) with a *min-mod* limiter (Roe, 1986), and a piecewise parabolic reconstruction method (Felker & Stone, 2017) with a parabolic limiter (Colella et al., 2011) or the XPPM limiter (Peterson & Hammett, 2013).

The simulation is evolved in time with iterative methods. In the following, the strong-stability preserving (SSP) variants of the (explicit) Runge-Kutta (RK) methods are denoted as SSPRK (*i*,*j*), where *i* and *j* refers to *i*-stage and the *j*-th order iterative method respectively. Several SSPRK variants are included for this simulation, with the SSPRK (3,3) (Shu & Osher, 1988), SSPRK (5,3) (Spiteri & Ruuth, 2002), and SSPRK (5,4) (Gottlieb et al., 2009) methods. The ''classic'' RK4 or the Forward Euler method can also be selected.

The Lax-Friedrichs method (LeVeque, 1992) is used for solving the Riemann problem at each interface for all reconstruction methods, although this method is highly dissipative and only first-order accurate.

Hydrodynamical tests in place are the Sod shock tube test (Sod, 1978), the Sedov blast test (Sedov, 1946), a ''sin-wave'' test, the Shu-Osher shock tube problem (Shu & Osher, 1989), and five shock tube tests from Toro's book (Toro, 1999, p.225). An analytical solution for the Sod shock test (Pfrommer et al., 2006) is also plotted when the Sod shock tube test is run. The solution error (L1 error norm) can also be determined when the ''sin-wave'' test is run.

This code is created as part of the Master's thesis project at the University of Cologne.

# Installation
Clone this repository and create a new Python environment. The dependencies for the environment can be found in *`requirements.txt`*. Initialise the environment and navigate to this folder on your local machine.

# Usage
To run the simulation, set your configurations in *`settings.py`* and run *`simulate.py`*.

## Organisation
There are several files in this code for different purposes:

- simulate.py: Runs the simulation, and contains the update loop
- solvers.py: Functions for the reconstruction methods and Riemann solvers
- limiters.py: Implements flux/slope limiters to prevent spurious oscillations in the reconstructed states
- timestepper.py: Functions for the (higher-order) time evolution
- functions.py: Generic functions that can be used throughout the code
- tests.py: Hydrodynamics test configurations
- plotting_functions.py: Functions for (live-)plotting
- settings.py: Parameter settings for the simulation
