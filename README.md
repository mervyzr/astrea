# hydro_shock

hydro_shock is a 1D-hydrodynamics code for the purpose of simulating shocks and turbulence with ions/neutrals interaction.

# Description

The code is written entirely in Python 3 and employs a finite volume method (Eulerian) with a fixed evenly-spaced grid. The simulation allows for periodic boundary conditions or outflows.

The code can currently employ a piece-wise constant method, a piece-wise linear reconstruction method (Derigs et al., 2018) with a *min-mod* limiter (Roe, 1986), or a piece-wise parabolic reconstruction method (Felker & Stones, 2017) with an *interface* and *parabolic interpolant* limiter (Colella et al., 2011). 

The Lax-Friedrichs method (LeVeque, 1992) is used for solving the Riemann problem at each interface for all reconstruction methods, although this method is highly dissipative and only first-order accurate.

The simulation is also only first-order accurate in time as the states are evolved in full time-steps.

Hydrodynamical tests in place are the Sod shock tube test, the Sedov blast test, and a ``sin-wave'' test.

This code is created as part of the Master's thesis project at the University of Cologne.

# Installation
Ideally the code can be in its own folder and should be run inside a Python environment. The dependencies can be found in the requirements.txt file.

To run the simulation, edit the config variables in `configs.py` and run `python3 simulation.py`.

# Usage
There are several files in this code for different purposes:

- simulation.py: Main file to run the simulation
- solvers.py: Contains the classes for different solvers and reconstruction methods
- flux_limiters.py: Contains the various flux/slope limiters to prevent spurious oscillations in the reconstruction step
- functions.py: Contains generic functions that are used throughout the code
- plotting_functions.py: Contains the functions for plots
- configs.py: Contains the configurations for the simulation
