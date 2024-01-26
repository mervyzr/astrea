# hydro_shock
hydro_shock is a 1D-hydrodynamics code for the purpose of simulating shocks and turbulence with ions/neutrals interaction.

# Description
The code is written entirely in Python 3 and employs a finite volume method (Eulerian) with a fixed evenly-spaced grid. The simulation allows for periodic boundary conditions or outflows.

The code can currently employ a piecewise constant method, a piecewise linear reconstruction method (Derigs et al., 2018) with a *min-mod* limiter (Roe, 1986), or a piecewise parabolic reconstruction method (Felker & Stone, 2017) with an *interface* and *parabolic interpolant* limiter (Colella et al., 2011). 

The Lax-Friedrichs method (LeVeque, 1992) is used for solving the Riemann problem at each interface for all reconstruction methods, although this method is highly dissipative and only first-order accurate. The simulation is also only first-order accurate in time as the states are evolved in full time-steps.

Hydrodynamical tests in place are the Sod shock tube test, the Sedov blast test, a ``sin-wave'' test, the Shu-Osher shock tube problem (Shu & Osher, 1989), and five shock tube tests from Toro's book (Toro, 1999, p.225).

This code is created as part of the Master's thesis project at the University of Cologne.

# Installation
Clone this repository and create a new Python environment. The dependencies for the environment can be found in *`requirements.txt`*. Initialise the environment and navigate to this folder on your local machine.

# Usage
To run the simulation, set your configurations in *`settings.py`* and run *`simulation.py`*.

## Organisation
There are several files in this code for different purposes:

- simulation.py: Employs the Riemann solver and runs the simulation
- solvers.py: Functions for the reconstruction methods
- limiters.py: Implements flux/slope limiters to prevent spurious oscillations in the reconstructed states
- functions.py: Generic functions that can be used throughout the code
- tests.py: Hydrodynamics test configurations
- plotting_functions.py: Functions for (live-)plotting
- settings.py: Settings for the simulation
