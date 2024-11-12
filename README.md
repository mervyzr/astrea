[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Python](https://img.shields.io/badge/Python-3.1x-yellow?logo=python&logoColor=white)](https://www.python.org)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

<!-- ![GitHub Tag](https://img.shields.io/github/v/tag/mervyzr/mHydyS) -->

# m-hydys

m-hydys (pronounced _"Hades"_; the _"m"_ is silent 😀) is a (one-)/two-dimensional (**M**agneto-)**HY**dro**DY**namic**S** code for the purpose of simulating shocks with a chemical network of ions and neutrals, and possible implementation of radiative heating and cooling.

**_This code is created as part of the Master's thesis research project at the University of Cologne, under supervision by Prof. Dr. Stefanie Walch-Gassner._**

<p align='center'>
  <img src='./static/khi_rho.gif' width=400 alt='Kelvin-Helmholtz instability'>
  <img src='./static/lax_liu_6.gif' width=400 alt='Lax-Liu 6'>
</p>

# Description

### Code

The simulation employs a finite volume subgrid model (Eulerian) with a fixed and uniform Cartesian grid with periodic or outlet boundary conditions. The solution in the grid is updated in parallel. The simulation also allows for magnetic fields with the magnetic permeability set to one for simplicity.

The code is written entirely in Python 3, and uses the `numpy` and `h5py` modules extensively for calculations and data handling respectively. The last _^stable_ Python version supported is _**Python 3.12**_.

Some experimentation was done to parallelise the code with `Open MPI` and `MPICH`, or to enable multithreading. However, this is generally not recommended because of the global-interpreter-lock (GIL) in Python and the sequential nature of the simulation. Futhermore, `numpy` should already use multi-threading _wherever possible_, and 'parallelised Python' with `numpy` does not show a substantial increase in speed anyway over 'fully-parallel' code in Fortran or C (Ross, 2016).

^There are some issues with building the wheels for `h5py` and `scipy` in Python 3.13 with the GIL disabled and experimental JIT compiler enabled (see [here](https://github.com/h5py/h5py/issues/2475) and [here](https://docs.scipy.org/doc/scipy/dev/toolchain.html)). Therefore, the code can only run with Python 3.13 built without those two options.

### Spatial discretisation

The space in the simulation is discretised into a uniform Cartesian grid, and thus the computational domain is assumed to be identically mapped to the physical domain.

The code employs various reconstruction methods with _primitive variables_ as part of the subgrid modelling: the piecewise constant method (PCM) (Godunov, 1959), the piecewise linear method (PLM) (Derigs et al., 2018), the piecewise parabolic method (PPM) (Felker & Stone, 2017), and the WENO method (Shu, 2009; San & Kara, 2015).

In order to fulfil the Total Variation Diminishing (TVD) condition (Harten, 1983), which ensures that the reconstruction scheme is monotonicity-preserving, limiters have to be after the spatial reconstructions. The PCM does not require any limiters. The PLM employs the "minmod" slope limiter (Derigs et al., 2018). The PPM employs several limiters: when extrapolating from the cell centres to the interfaces (Colella et al., 2011) and when interpolating to the left and right of each cell interface (Colella et al., 2011; McCorquodale & Colella, 2011). The WENO method currently does not employ any limiters. There are other TVD slope limiters available in the code (e.g., superbee).

The parabolic reconstruction method by McCorquodale & Colella also includes a slope flattener (Colella, 1990) and artificial viscosity as additional dissipation mechanisms to suppress oscillations at strong shocks.

### Time discretisation

A method-of-lines approach is used for the temporal evolution of the simulation, thus the temporal component of the advection equation can be discretised and treated separately from the spatial component.

Higher-order temporal discretisation methods can be employed to match the higher-order spatial components used. These higher-order methods also need to fulfil the TVD condition, which leads to the use of strong-stability preserving (SSP) variants of the Runge-Kutta (RK) methods, denoted here as SSPRK. Some of the SSPRK variants use the "Shu-Osher representation" (Shu & Osher, 1988) of Butcher's tableau of RK coefficients(Butcher, 1975).

In the following, the (explicit) SSPRK methods are denoted as SSPRK (_i_,_j_), where _i_ and _j_ refers to _i_-stage and the _j_-th order iterative method respectively. Several SSPRK variants are included for this simulation, with the SSPRK (2,2) (Gottlieb et al., 2008), SSPRK (3,3) (Shu & Osher, 1988; Gottlieb et al., 2008), SSPRK(4,3), SSPRK (5,3) (Spiteri & Ruuth, 2002; Gottlieb et al., 2008), SSPRK (5,4) (Kraaijevanger, 1991; Ruuth & Spiter, 2002), and low-storage (Williamson, 1980) SSPRK(10,4) (Ketcheson, 2008) methods. The ''classic'' RK4 or the Forward Euler method can also be used.

For a _j_-order reconstruction scheme, _j_ > 4, the Dormand-Prince 8(7) (Dormand & Prince, 1981) method can be considered. However, this method is not a SSP variant as no methods with order _j_ > 4 with positive SSP coefficients can exist (Kraaijevanger, 1991; Ruuth & Spiteri, 2001), and therefore might not be suitable for solutions with discontinuities.

### Riemann solver and flux update

Due to the nature of the finite volume method and the discretisation of space in the grid, a Riemann problem is created at each interface between consecutive cells, with each cell containing the subgrid profile. In this code, approximate Riemann solvers are used (linear and non-linear) in order to compute the flux across interfaces.

The Local Lax-Friedrichs (LLF) scheme (LeVeque, 1992) is an approximate linearised Riemann solver (i.e. the method aims to find an _exact_ solution to the _linearised_ or _approximate_ version of the (magneto-)hydrodynamic equations). This scheme is very stable and robust, however it is highly dissipative and only first-order accurate. The code also allows for the Lax-Wendroff scheme (Lax & Wendroff, 1960), which is another approximate linearised Riemann solver and is second-order accurate.

The fluxes are calculated from the interpolated interfaces, and the Jacobian matrices are calculated from the Roe average (Roe & Pike, 1984) of these interfaces (Cargo & Gallice, 1997).

An issue that arises when linear schemes are made to be monotonicity-preserving (i.e. do not produce spurrious oscillations), then the scheme can be at most first-order accurate. This is known as Godunov's Theorem (Godunov, 1954). Since the main focus of this project is simulating shocks, where large discontinuities and possible spurrious oscillations are present (similar to Gibbs phenomenon), non-linear Riemann solvers, that attempt to restore some form of the eigenstructure of the characteristic waves, are therefore implemented into the code.

The Harten-Lax-van Leer-Contact (HLLC) Riemann solver (Toro et al., 1994; Fleischmann et. al., 2020) attempts to restore the contact discontinuity wave while tracing the rarefaction and shock wave (Riemann invariants), thus it provides a better resolution albeit with some dissipation. The HLLC Riemann solver crashes when magnetic fields are present. For that, the Harten-Lax-van Leer-discontinuities (HLLD) solver (Miyoshi & Kusano, 2005) should be used. The HLLD Riemann solver restores the magnetosonic and Alfvén waves, although this is not a complete Riemann solver; this implementation of the Riemann solver ignores the slow magnetosonic wave.

Riemann solvers that attempt to derive the flux from the full (_but not exact_) eigenstructure are also included in the code, such as the entropy-stable flux (Derigs et al., 2016) and the modified Osher-Solomon flux (Dumbser & Toro, 2011). However, these solvers are not as robust and stable, and run into errors frequently.

### Hydrodynamical tests

Several (magneto-)hydrodynamical tests are in place:

- Hydrodynamics
  - Sod shock tube test (Sod, 1978)
  - Sedov blast test (Sedov, 1946)
  - Shu-Osher shock test (Shu & Osher, 1989)
  - "Toro tests" (Toro, 1999, p.225)
  - "Lax-Liu tests" (Lax & Liu, 1998)
  - Slow-moving shock (Zingale, 2023, p.148)
  - Kelvin-Helmholtz instability
  - Simple advection wave tests
    - Gaussian
    - _sin_
    - square
    - isentropic vortex (Yee et al., 1999)
- Magnetohydrodynamics
  - Ryu-Jones 2a shock test (Ryu & Jones, 1995)
  - Brio-Wu shock test (Brio & Wu, 1998)

Analytical solutions for the Sod shock test (Pfrommer et al., 2006), Gaussian wave test and the _sin_ wave test are overplotted in the saved plots. The solution error norms are also calculated when the smooth advection wave tests are run (Gaussian, _sin_).

# Installation

_It is recommended to run Python projects in separate virtual environments._

Clone this repository onto your local machine, and navigate to the cloned repository. In the command line, run _`/path/to/venv/bin/python3 -m pip install .`_; this will install the minimum packages to run the simulation and create a `parameters.yml` file for simulation configurations.

# Usage

The main method to run the simulation would be to edit the simulation parameters in `parameters.yml` and running the main Python file:

```bash
python3 mhydys.py
```

OR

```bash
./mhydys.py
```

Alternatively, the code can be run with CLI options:

```bash
python3 mhydys.py --config==sedov --cells=256
```

See _`--help`_ for a list of available options.

Running the code in a Python interactive shell is also possible, although this is generally not recommended:

```python
import mhydys
mhydys.run()
```

## Organisation

```
├── LICENSE
├── README.md
├── __init__.py
├── functions
│   ├── __init__.py
│   ├── analytic.py      : Analytical solutions to hydrodynamics tests
│   ├── constructors.py  : Constructors for math objects, such as eigenvectors and Jacobian matrices
│   ├── fv.py            : Frequently used functions specific to finite volume
│   ├── generic.py       : Generic functions not specific to finite volume
│   └── plotting.py      : Functions for (live-)plotting
├── num_methods
│   ├── __init__.py
│   ├── evolvers.py      : Collates the schemes for space and time evolution
│   ├── limiters.py      : Implements flux/slope limiters in the reconstructed states
│   ├── solvers.py       : Contains the Riemann solvers
├── parameters.yml       : Parameters for the simulation
├── schemes
│   ├── __init__.py
│   ├── pcm.py           : Piecewise constant method [Godunov, 1959]
│   ├── plm.py           : Piecewise linear method [Derigs et al., 2018]
│   ├── ppm.py           : Piecewise parabolic method [Felker & Stone, 2015]
│   ├── weno.py          : WENO method [Shu, 2009; San & Kara, 2015]
├── setup.py             : Installation script
├── mhydys.py            : Runs the simulation, and contains the update loop
├── static
│   ├── __init__.py
│   ├── .db.json         : Database for parameters
│   ├── .default.yml     : Default parameters file
│   ├── requirements.txt
│   ├── tests.py         : Hydrodynamics test configurations
```
