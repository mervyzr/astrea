[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-brightgreen?logo=python&logoColor=white)](https://www.python.org)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
<!-- ![GitHub Tag](https://img.shields.io/github/v/tag/mervyzr/mHydyS) -->


# mHydyS
`mHydyS` (pronounced *"Hades"*; the *"m"* is silent 😀) is a one-dimensional (magneto-)hydrodynamics shock code for the purpose of simulating shocks with ions and neutrals, with possible radiative heating and cooling.

This code is created as part of the Master's thesis research project at the University of Cologne.

# Description
The code is written entirely in Python3 and employs a (Eulerian) finite volume subgrid model with a structured uniform grid and a method-of-lines approach. The simulation allows for periodic or outlet boundary conditions. It also allows for magnetic fields in one dimension with the magnetic permeability set to one. The simulation is updated in parallel.

### Spatial discretisation
The space in the simulation is discretised via a uniform Cartesian grid, and employs various reconstruction methods with primitive variables as part of the subgrid modelling: the piecewise constant method (Godunov, 1959), the piecewise linear method with a *minmod* slope limiter (Derigs et al., 2018), and the piecewise parabolic method (Felker & Stone, 2017) with cell faces limiters (Colella et al., 2011) and parabolic interpolations limiters (Colella et al., 2011; McCorquodale & Colella, 2011). The parabolic reconstruction method by McCorquodale & Colella (2011) also includes a slope flattener (Colella, 1990) and artificial viscosity as additional dissipation mechanisms to suppress oscillations at strong shocks.

### Time discretisation
The time in the simulation can be discretised and treated separately from the spatial component (because of the method-of-lines approach). Higher-order temporal discretisation methods can be employed to match the higher-order spatial components used. In the following, the strong-stability preserving (SSP) variants of the (explicit) Runge-Kutta (RK) methods are denoted as SSPRK (*i*,*j*), where *i* and *j* refers to *i*-stage and the *j*-th order iterative method respectively. Several SSPRK variants are included for this simulation, with the SSPRK (2,2), SSPRK (3,3) (Shu & Osher, 1988), SSPRK (5,3) (Spiteri & Ruuth, 2002), and SSPRK (5,4) (Gottlieb et al., 2009) methods. The ''classic'' RK4 or the Forward Euler method can also be used.

### Scheme and Riemann solver
With the discretisation of the space in the simulation, a Riemann problem is created at each interface between consecutive cells, with each cell containing the subgrid profile. In this code, approximate Riemann solvers are used (linear and non-linear) in order to perform the conservative update. The Local Lax-Friedrichs (LLF) scheme (LeVeque, 1992) is an approximate linearised Riemann solver (i.e. the method aims to find an *exact* solution to the *linearised* or *approximate* version of the (magneto-)hydrodynamic equations) that is very stable and robust, however it is highly dissipative and only first-order. The fluxes and the Jacobian matrices are calculated using the interpolated interfaces, with the Jacobian matrix using an average of these interfaces (Cargo & Gallice, 1997). The code also allows for the Lax-Wendroff scheme (Lax & Wendroff, 1960), which is another approximate linearised Riemann solver but it is second-order. The main issue with linear schemes is that the schemes will cause spurrious oscillations, according to Godunov's Theorem (Godunov, 1954). Therefore, the HLLC Riemann solver (Fleischmann et. al., 2020) can also be used for this code, that allows for higher-order reconstruction methods and also reduces the spurrious oscillations, albeit with some dissipation.

### Hydrodynamical tests
Hydrodynamical tests in place are the Sod shock tube test (Sod, 1978), the Sedov blast test (Sedov, 1946), simple advection wave tests (Gaussian, *sin*-curve, square), the Shu-Osher shock tube problem (Shu & Osher, 1989), and five shock tube tests from Toro (Toro, 1999, p.225). An additional magnetohydrodynamics test is included (Ryu & Jones, 1995). Analytical solutions for the Sod shock test (Pfrommer et al., 2006), Gaussian wave test and the *sin*-wave test are overplotted in the saved plots. The solution error (L1 error norm) is also determined when the *sin-wave* or Gaussian test is run.

# Installation
Clone this repository onto your local machine, and navigate to the cloned repository. Run *`python3 static/setup.py`* to install a venv (*.shock_venv*) in the home directory. This will also install the requirements for the project in the venv. *It is highly recommended to always run Python projects in separate virtual environments.*

# Usage
Edit your parameters in *`settings.py`* and run *`simulate.py`* (preferably in the venv).

## Organisation

### Import structure
- `simulate.py`: Runs the simulation, and contains the update loop
    - `settings.py`: Parameters for the simulation
    - `tests.py`: Hydrodynamics test configurations
    - `evolvers.py`: Collates the schemes for space and time evolution
    - `schemes`
        - `pcm.py`: Piecewise constant method [Godunov, 1959]
        - `plm.py`: Piecewise linear method [Derigs et al., 2018]
        - `ppm.py`: Piecewise parabolic method [Felker & Stone, 2015]
        - `weno.py`: WENO method [Shu, 2009]
    - `numerics`
        - `solvers.py`: Contains the Riemann solvers
        - `limiters.py`: Implements flux/slope limiters to prevent spurious oscillations in the reconstructed states
    - `functions`
        - `generic.py`: Generic functions used throughout the code
        - `fv.py`: Re-usable/generic code specific to the finite volume method
        - `plotting.py`: Functions for (live-)plotting
            - `analytic.py`: Analytical solutions to hydrodynamics tests

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
├── numerics
│   ├── __init__.py
│   ├── limiters.py
│   ├── solvers.py
├── schemes
│   ├── __init__.py
│   ├── pcm.py
│   ├── plm.py
│   ├── ppm.py
│   ├── weno.py
├── settings.py
├── simulate.py
├── static
│   ├── __init__.py
│   ├── .default.py
│   ├── requirements.txt
│   ├── setup.py
│   ├── tests.py
```
