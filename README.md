# mHydyS
`mHydyS` (pronounced *"Hades"*; the *"m"* is silent 😀) is a one-dimensional (magneto-)hydrodynamics shock code for the purpose of simulating shocks with ions and neutrals, with possible radiative heating and cooling.

This code is created as part of the Master's thesis research project at the University of Cologne.

# Description
The code is written entirely in Python3 and employs a finite volume (Eulerian) subgrid model with a structured uniform grid. The simulation allows for periodic or outlet boundary conditions. It also allows for magnetic fields in one dimension with the magnetic permeability set to one.

### Spatial discretisation
The simulation employs a finite volume method and can currently perform various reconstructions with primitive variables: the piecewise constant method (Godunov, 1959), the piecewise linear method with a *minmod* slope limiter (Derigs et al., 2018), and the piecewise parabolic method (Felker & Stone, 2017) with cell faces limiters (Colella et al., 2011) and parabolic interpolations limiters (Colella et al., 2011; McCorquodale & Colella, 2011). The parabolic reconstruction method also includes a slope flattener (Colella, 1990) in order to suppress oscillations at strong shocks.

### Time discretisation
Time in the simulation is discretised and evolved with a method-of-lines approach. In the following, the strong-stability preserving (SSP) variants of the (explicit) Runge-Kutta (RK) methods are denoted as SSPRK (*i*,*j*), where *i* and *j* refers to *i*-stage and the *j*-th order iterative method respectively. Several SSPRK variants are included for this simulation, with the SSPRK (2,2), SSPRK (3,3) (Shu & Osher, 1988), SSPRK (5,3) (Spiteri & Ruuth, 2002), and SSPRK (5,4) (Gottlieb et al., 2009) methods. The ''classic'' RK4 or the Forward Euler method can also be used.

### Riemann solver
The Local Lax-Friedrichs method (LeVeque, 1992) is used for solving the Riemann problem at each interface for all reconstruction methods, although this method is highly dissipative and only first-order accurate.

### Hydrodynamical tests
Hydrodynamical tests in place are the Sod shock tube test (Sod, 1978), the Sedov blast test (Sedov, 1946), simple advection wave tests (Gaussian, *sin*-curve, square), the Shu-Osher shock tube problem (Shu & Osher, 1989), and five shock tube tests from Toro (Toro, 1999, p.225). An additional magnetohydrodynamics test is included (Ryu & Jones, 1995). Analytical solutions for the Sod shock test (Pfrommer et al., 2006), Gaussian wave test and the *sin*-wave test are overplotted in the saved plots. The solution error (L1 error norm) is also determined when the *sin-wave* or Gaussian test is run.

# Installation
Clone this repository onto your local machine, and navigate to the cloned repository. Run *`python3 setup.py`* to install a venv (*.shock_venv*) in the home directory. This will also install the requirements for the project in the venv.

# Usage
Edit your parameters in *`settings.py`* and run *`simulate.py`* (preferably in the venv).

## Organisation

### Import structure
- `simulate.py`: Runs the simulation, and contains the update loop
    - `evolvers.py`: Collates the functions for space and time evolution
        - `solvers.py`: Contains the Riemann solver
        - `reconstruct.py`: Functions for the reconstruction methods
        - `limiters.py`: Implements flux/slope limiters to prevent spurious oscillations in the reconstructed states
    - `tests.py`: Hydrodynamics test configurations
    - `settings.py`: Parameters for the simulation
    - `functions`
        - `generic.py`: Generic functions used throughout the code
        - `analytic.py`: Analytical solutions to hydrodynamics tests
        - `fv.py`: Re-usable/generic code specific to the finite volume method
        - `plotting.py`: Functions for (live-)plotting

### Folder structure
```
├── LICENSE
├── README.md
├── __init__.py
├── evolvers.py
├── functions
│   ├── __init__.py
│   ├── analytic.py
│   ├── fv.py
│   ├── generic.py
│   └── plotting.py
├── limiters.py
├── reconstruct.py
├── requirements.txt
├── settings.py
├── simulate.py
├── solvers.py
└── tests.py
```
