# hydro_shock

hydro_shock is a 1D-hydrodynamical code for the purpose of simulating shocks and turbulence with ions/neutrals interaction.

# Description

The code is written entirely in Python 3. It begins by simulating an evenly-spaced grid, which then updates the variables with an Eulerian approach. There are multiple solvers and flux limiters in place, but some are still works-in-progress. This code is created as part of the Master's thesis project at the University of Cologne.

# Installation
Ideally the code can be in its own folder and should be run inside a Python environment. The dependencies can be found in the requirements.txt file.

To run the simulation, run `python3 simulation.py`.

# Usage
There are several files in this code for different purposes:

- simulation.py: Main file to run the simulation
- solvers.py: Contains the classes for different solvers and reconstruction methods
- flux_limiters.py: Contains the various flux/slope limiters to prevent spurrous oscillations in the reconstruction step
- functions.py: Contains generic functions that are used throughout the code
- plotting_functions.py: Contains the functions for plots
- configs.py: Contains the configurations for the simulation
