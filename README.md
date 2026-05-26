[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Python](https://img.shields.io/badge/Python-3.10+-green?logo=python&logoColor=white)](https://www.python.org)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

<!-- ![GitHub Tag](https://img.shields.io/github/v/tag/mervyzr/astrea) -->

# astrea

**_astrea_** (**A**strophysical **S**hockwave and **T**urbulence **RE**search for interstellar **A**pplications) is a multi-dimensional ideal magnetohydrodynamics simulation toy-model code with an experimental chemical network solver and self-gravity for the purpose of modelling shockwaves in the interstellar medium.

**_This code was originally created as part of my Master's thesis research project at the University of Cologne, under supervision by Prof. Dr. Stefanie Walch-Gassner. The thesis has since been completed._**

<p align='center'>
  <img src='./static/khi-energy.gif' width=30% alt='Kelvin-Helmholtz instability'>
  <img src='./static/ll3-density.gif' width=30% alt='Lax-Liu config. 3'>
  <img src='./static/ot-magpressure.gif' width=30% alt='Orszag-Tang vortex'>
</p>


### Table of Contents  
- [Description](#description)
  - [Code](#code)
  - [Spatial discretisation](#spatial-discretisation)
  - [Riemann solver](#riemann-solver-and-flux-update)
  - [Time discretisation](#time-discretisation)
  - [Constrained transport](#constrained-transport)
  - [Self-gravity](#self-gravity)
  - [Chemical network](#chemical-network)
  - [Simulation benchmarks](#simulation-benchmarks)
- [Installation](#installation)
- [Usage](#usage)
- [Organisation](#organisation)


<a name="description"></a>

# Description

<a name="code"></a>

### Code

The simulation employs a higher-order finite volume subgrid model (Eulerian) with a fixed and uniform Cartesian grid with periodic or outlet boundary conditions. The solution in the grid is updated in parallel.

The code is mostly written with Python language, and uses the `numpy` and `h5py` modules extensively for calculations and data handling respectively. The last _stable^_ Python version supported is _**Python 3.13**_.

### Is this simulation slow? _Yes (relatively)_.
By nature, interpreted languages are slower than compiled languages.

### Would I consider this a production-ready code? _No_.
This code is not meant to replace or compete with other production-ready MHD simulations codes, such as FLASH, AREPO, or ATHENA++.

### Should this code still be used? _Absolutely_.
This code is meant to be a toy/test model for various numerical schemes, therefore it has the most bare-bones scripts in order to run the numerical simulations (relative to the aforementioned codes). The underlying principles/physics are the same as those codes, therefore I hope this code can allow others to experiment and learn on their own.

Some experimentation was done to parallelise the code with `Open MPI` and `MPICH`. However, this is generally not recommended because of the global-interpreter-lock (GIL); the GIL in Python makes it more difficult to achieve parallelisation. Attempts have been made with multi-processing (`multiprocessing`) and multi-threading (`concurrent.futures`), with limited success. Some of the main slow-downs come from the sequential nature of the (explicit) time evolution method and the I/O of the `hdf5` data file. Most of the functions have also been vectorised to make use of `numpy`'s multi-threading _wherever possible_. But ultimately the benefits of having a 'semi-parallelised' Python code with `numpy` might not outweigh having a fully compiled code such as Fortran or C (Ross, 2016).

^_`h5py` is not fully optimised for the experimental free-threading build or the just-in-time compiler introduced in Python 3.13t. The authors of `h5py` have indicated that users can run `h5py` with the free-threading build, but at the users' own risk!_


<a name="spatial-discretisation"></a>

### Spatial discretisation

The space in the simulation is discretised into a uniform Cartesian grid, and thus the computational domain is assumed to be identically mapped to the physical domain.

The code employs various reconstruction methods with _primitive variables_ as part of the subgrid modelling: the piecewise constant method (PCM) (Godunov, 1959), the piecewise linear method (PLM) (Derigs et al., 2018), the piecewise parabolic method (PPM) (Felker & Stone, 2018), the WENO method (Jiang & Shu, 1996; Balsara & Shu, 2000), the CWENO method (Levy et al., 1999, 2000), the WENO-Z method (Borges et al., 2008), and the TENO(-AA) method (Fu et al., 2016; Fu, 2021).

Godunov's theorem states that for a linear scheme that is monotonicity-preserving (i.e. do not produce spurrious oscillations), the scheme can be at most first-order accurate (Godunov, 1954). This has led to the development of several subgrid models that reduce these spurious oscillations while still maintaining a high-order accuracy. These models are known as Total Variation Diminishing (TVD) schemes (Harten, 1983). In order to fulfil the TVD condition, limiters have to be used after the spatial reconstructions. The PCM does not require any limiters. The PLM employs the "minmod" slope limiter (Derigs et al., 2018). The PPM employs several limiters: when _interpolating_ from the cell centres to the interfaces (Colella et al., 2011) and when _extrapolating_ to the left and right of each cell interface (Colella et al., 2011; McCorquodale & Colella, 2011). The WENO method currently does not employ any limiters. There are other TVD slope limiters available in the code too (e.g., superbee).

The parabolic reconstruction method by McCorquodale & Colella (2011) also allows for a slope flattener (Colella, 1990) and artificial viscosity as additional dissipation mechanisms to suppress oscillations at sharp discontinuities.


<a name="riemann-solver-and-flux-update"></a>

### Riemann solver and flux update

Due to the nature of the finite volume method and the discretisation of space in the grid, a Riemann problem is created at each interface between consecutive cells, with each cell containing the subgrid profile. In this code, approximate Riemann solvers are used (linear and non-linear) in order to compute the flux across interfaces. The Riemann solvers are solving for the compressible Euler equations, with possible source terms such as gravity (Grosheintz-Laval & Käppeli, 2019), in which the discontinuous jump conditions need to satisfy the Rankine-Hugoniot relations.

The Local Lax-Friedrichs (LLF) solver (LeVeque, 1992) is an approximate linearised Riemann solver (i.e. the method aims to find an exact solution to the _linearised_ or _approximate_ version of the ideal magnetohydrodynamic equations). This scheme is very stable and robust, however it is highly dissipative and only first-order accurate. The code also allows for the Lax-Wendroff scheme (Lax & Wendroff, 1960), which is another approximate linearised Riemann solver and is second-order accurate, and the GFORCE solver (Toro & Titarev, 2006), which is a linearly weighted combination of Lax-Friedrichs and Lax-Wendroff solvers. The Beam-Warming scheme (Beam & Warming, 1976) and the Fromm scheme (Fromm, 1968) are not included in this code as modifications to the update steps are required to adapt to those schemes.

The fluxes are calculated from the interpolated interfaces, and the Jacobian matrices are calculated from the Roe average of these interfaces (Roe & Pike, 1984; Cargo & Gallice, 1997).

Non-linear approximate Riemann solvers may also be used instead of linear solvers. Non-linear solvers tackle the non-linear form of the compressible Euler equations directly, instead of linearising the equations first. These solvers attempt to restore some form of the eigenstructure of the characteristic waves, and they are useful as they contain all the information. Since the main focus of this project is simulating shockwaves, where large discontinuities and possible spurrious oscillations are present (similar to Gibbs phenomenon), non-linear approximate Riemann solvers are therefore implemented into the code too.

The Harten-Lax-van Leer (HLL) Riemann solver (Harten et al., 1983) forms the basis of the so-called 'HLL-family' of approximate non-linear solvers. Most variations of the HLL-family of solvers build upon this initial solver. The Harten-Lax-van Leer-Contact (HLLC) Riemann solver (Toro et al., 1994; Fleischmann et al., 2020) attempts to restore the contact discontinuity wave while tracing the rarefaction and shockwave (Riemann invariants), thus it provides a better resolution albeit with some dissipation. The HLLC Riemann solver crashes when magnetic fields are present. For that, the Harten-Lax-van Leer-discontinuities (HLLD) solver (Miyoshi & Kusano, 2005) should be used. The HLLD Riemann solver restores the magnetosonic and Alfvén waves, although this is not a complete Riemann solver; this implementation of the Riemann solver ignores the slow magnetosonic wave.

Riemann solvers that attempt to derive the flux from the full (_but not exact_) eigenstructure are also included in the code, such as the entropy-stable flux (Derigs et al., 2018) and the modified Osher-Solomon flux (Dumbser & Toro, 2011). However, these solvers are not as robust and stable, and run into errors frequently.


<a name="time-discretisation"></a>

### Time discretisation

A method-of-lines approach is used for the temporal evolution of the simulation, thus the temporal component of the advection equation can be discretised and treated separately from the spatial component.

Higher-order temporal discretisation methods can be employed to match the higher-order spatial components used. These higher-order methods also need to fulfil the TVD condition, which leads to the use of strong-stability preserving (SSP) variants of the Runge-Kutta (RK) methods, denoted here as SSPRK. Some of the SSPRK variants use the "Shu-Osher representation" (Shu & Osher, 1988) of Butcher's tableau of RK coefficients (Butcher, 1975).

In the following, the (explicit) SSPRK methods are denoted as SSPRK (_i_,_j_), where _i_ and _j_ refers to _i_-stage and the _j_-th order iterative method respectively. Several SSPRK variants are included for this simulation, with the SSPRK (2,2) (Gottlieb et al., 2009), SSPRK (3,3) (Shu & Osher, 1988; Gottlieb et al., 2009), SSPRK(4,3), SSPRK (5,3) (Spiteri & Ruuth, 2002; Gottlieb et al., 2009), SSPRK (5,4) (Kraaijevanger, 1991; Ruuth & Spiteri, 2002), SSPRK (6,5) (Gottlieb et al., 2009), and low-storage (Williamson, 1980) SSPRK(10,4) (Ketcheson, 2008) methods. The ''classic'' RK4 or the Forward Euler method can also be used.

For a _j_-order reconstruction scheme, _j_ > 4, the Dormand-Prince 8(7) (Dormand & Prince, 1981) method can be considered. However, this method is not a SSP variant as no methods with order _j_ > 4 with positive SSP coefficients can exist (Kraaijevanger, 1991; Ruuth & Spiteri, 2002), and therefore might not be suitable for solutions with discontinuities.


<a name="constrained-transport"></a>

### Constrained transport

With the presence of magnetic fields, it is crucial for the divergence-free condition to be maintained; no monopoles should be created in the simulation. The induction equation, which governs the fluxes of the magnetic fields, must be solved along with the other conservation equations for ideal magnetohydrodynamics. For ideal MHD, the electromotive forces (emfs) are equivalent to the cross product between the velocities and magnetic fields. The magnetic permeability is also set to one for simplicity.

In order to compute the magnetic fluxes and maintain the divergence-free condition, the constrained transport (CT) approach is commonly used (Evans & Hawley, 1988). In the constrained transport method, the emfs are computed at the corners or edges of each cell for 2D and 3D grid configurations respectively. This ensures that the magnetic field lines are 'connected' essentially and thus the numerical errors can be kept close to machine precision. However, the complexity of this implementation is much higher; one has to consider the use of staggered grids and the 'location' of the quantities within the cells. Wrong allocation of cell-interface values to cell-centred values might lead to wrong computation of the physical quantities. This is especially complex when adaptive meshes or unstructured grids are used. It might be possible to avoid staggered grids altogether too (Helzel et al., 2011), but this is not included in this code.

Other methods include divergence cleaning (e.g., Dedner et al., 2002) and Powell's eight-wave formulation (Powell, 1994).

The higher-order CT implementation in this code mainly follows the works of Felker & Stone (2018). The implementation also follows closely to the works of Verma et al. (2018) and Mignone & Del Zanna (2021).


<a name="self-gravity"></a>

### Self-gravity

Self-gravity is an important physical field that affects astrophysical simulations. Including self-gravity into the code would thus be beneficial; a simple Poisson solver is included into the code. While solving the Poisson equation is relatively trivial, one has to take note of several factors and methods of solving the equation. Due to the use of a uniform grid (for now), one can thus make use of a fast Fourier transform (FFT) methodology to solve the Poisson equation on the whole grid at once, which would provide the gravitational potential field.

The gravitational acceleration can then be computed as a simple cell-centred difference of the potential field, and this force would enter into the conservation of momentum equation where it would drive the gas and particles in the cells. A 'higher-order' centred difference can be used to compute the gravitational acceleration more accurately, similar to the higher-order spatial reconstruction methods above.


<a name="chemical-network"></a>

### Chemical network (_experimental_)

The inclusion of the chemical network is _experimental_ and achieved with the <a href='https://www.kromepackage.org' target='_blank'>`krome`</a> package. In order to include a chemical network in the simulation, the `krome` folder has to be placed in the base folder:
```bash
git clone https://bitbucket.org/tgrassi/krome.git
```

Additionally, the `--chemistry` option has to be indicated at runtime:
```bash
python3 astrea.py --chemistry --network=/path/to/network_file
```
If no network files are indicated in the runtime options, a custom network file will be used with the following species:
```
HI, HII, H2, CII, CO, O, OH, e-
```


<a name="simulation-benchmarks"></a>

### Simulation benchmarks

Several (magneto)hydrodynamics tests are in place:

<ul>
  <li>Hydrodynamics</li>
  <details>
  <summary>One-dimensional</summary>
  <ul>
    <li>Random noise field</li>
    <li>Turbulent (OU) driving motions (Federrath et al., 2010; Brucy et al., 2024)</li>
    <li>Sod shock tube (Sod, 1978)</li>
    <li>Sedov blast wave (Sedov, 1946)</li>
    <li>Slow-moving shockwave (Zingale, 2023, p.148)</li>
    <li>Shu-Osher shockwave (Shu & Osher, 1988)</li>
    <li>Toro tests (Toro, 1999, p.225)</li>
    <li>Tycho supernova (Markert et al., 2022)</li>
    <li>Smooth advection wave tests</li>
    <ul>
      <li>Gaussian wave</li>
      <li>sine-wave</li>
      <li>Manufactured Euler solution</li>
    </ul>
  </ul>
  </details>
  <details>
  <summary>Two-dimensional</summary>
  <ul>
    <li>Random noise field</li>
    <li>Turbulent (OU) driving motions</li>
    <li>Sedov blast wave (Sedov, 1946)</li>
    <li>Kelvin-Helmholtz instability</li>
    <li>Rayleigh-Taylor instability</li>
    <li>Noh problem (Noh, 1987)</li>
    <li>Gresho vortex (Gresho & Chan, 1990)</li>
    <li>"Lax-Liu tests" (Lax & Liu, 1998)</li>
    <li>Isentropic vortex (Pang & Wu, 2025)</li>
    <li>Smooth advection wave tests</li>
    <li>Tycho supernova</li>
    <ul>
      <li>Gaussian wave</li>
      <li>sine-wave</li>
      <li>Manufactured Euler solution</li>
    </ul>
  </ul>
  </details>
  <details>
  <summary>Three-dimensional</summary>
  <ul>
    <li>Random noise field</li>
    <li>Turbulent (OU) driving motions</li>
    <li>Sedov blast wave</li>
    <li>Tycho supernova</li>
    <li>Smooth advection wave tests</li>
    <ul>
      <li>Gaussian wave</li>
      <li>sine-wave</li>
      <li>Manufactured Euler solution</li>
    </ul>
  </ul>
  </details>
</ul>

<ul>
  <li>Magnetohydrodynamics</li>
  <details>
  <summary>One-dimensional</summary>
  <ul>
    <li>Random noise field</li>
    <li>Turbulent (OU) driving motions</li>
    <li>Ryu-Jones 2a shockwave (Ryu & Jones, 1995)</li>
    <li>Brio-Wu shockwave (Brio & Wu, 1988)</li>
  </ul>
  </details>
  <details>
  <summary>Two-dimensional</summary>
  <ul>
    <li>Random noise field</li>
    <li>Turbulent (OU) driving motions</li>
    <li>Magnetised Kelvin-Helmholtz instability</li>
    <li>Magnetised Rayleigh-Taylor instability</li>
    <li>Orszag-Tang vortex (Orszag & Tang, 1998)</li>
    <li>MHD vortex (Balsara, 2004)</li>
    <li>MHD rotor (Balsara & Spicer, 1999)</li>
    <li>MHD torus (Machida et al., 1999)</li>
    <li>MHD blast wave (Londrillo & Del Zanna, 2000)</li>
    <li>MHD current sheet (Gardiner & Stone, 2005)</li>
    <li>Yee-Sjögreen Riemann problem (Yee & Sjögreen, 2005)</li>
    <li>Shock cloud (Dai & Woodward, 1998)</li>
    <li>Astrophysical jet (Wu & Shu, 2018)</li>
    <li>Smooth advection wave tests</li>
    <ul>
      <li>Circular polarised Alfvén wave (Tóth, 2000)</li>
    </ul>
  </details>
  <details>
  <summary>Three-dimensional</summary>
  <ul>
    <li>Random noise field</li>
    <li>Turbulent (OU) driving motions</li>
    <li>Orszag-Tang vortex (Orszag & Tang, 1998)</li>
    <li>MHD vortex (Mignone et al., 2010)</li>
    <li>MHD torus (Machida et al., 1999)</li>
    <li>MHD blast wave (Londrillo & Del Zanna, 2000)</li>
  </ul>
  </details>
</ul>

Analytical solutions for the Sod shock-tube test (Pfrommer et al., 2006), Gaussian wave test and the sine wave test are overplotted in the saved plots. The solution error norms are also calculated when the smooth advection wave tests are run (Gaussian & sine waves).

<details>
  <summary>References</summary>
  <ol>
    <li>Balsara, D. S., & Shu, C.-W. (2000). Monotonicity Preserving weighted essentially non-oscillatory schemes with increasingly high order of accuracy. Journal of Computational Physics, 160, 405-452.</li>
    <li>Balsara, D. S., & Spicer, D. S. (1999). A Staggered Mesh Algorithm Using High Order Godunov Fluxes to Ensure Solenoidal Magnetic Fields in Magnetohydrodynamic Simulations. Journal of Computational Physics, 149, 270–292.</li>
    <li>Beam, R. M., & Warming, R. F. (1976). An implicit finite-difference algorithm for hyperbolic systems in conservation-law form. Journal of Computational Physics, 22(1), 87-110.</li>
    <li>Borges, R., Carmona, M., Costa, B., & Don, W. S. (2008). An improved weighted essentially non-oscillatory scheme for hyperbolic conservation laws. Journal of Computational Physics, 227, 3191-3211.</li>
    <li>Brio, M., & Wu, C. C. (1988). An upwind diﬀerencing scheme for the equations of ideal magnetohydrodynamics. Journal of Computational Physics, 75(2), 400–422.</li>
    <li>Brucy, N., Hennebelle, P., Colman, T., Klessen, R. S., & Le Yhuelic, C. (2024). Inefficient star formation in high Mach number environments: II. Numerical simulations and comparison with analytical models. Astronomy & Astrophysics, 690, A44.</li>
    <li>Butcher, J. C. (1975). A stability property of implicit Runge-Kutta methods. BIT, 15(4), 358–361.</li>
    <li>Cargo, P., & Gallice, G. (1997). Roe Matrices for Ideal MHD and Systematic Construction of Roe Matrices for Systems of Conservation Laws. Journal of Computational Physics, 136(2), 446–466.</li>
    <li>Colella, P., Dorr, M. R., Hittinger, J. A. F., & Martin, D. F. (2011). High-order, finite-volume methods in mapped coordinates. Journal of Computational Physics, 230(8), 2952–2976.</li>
    <li>Dedner, A., Kemm, F., Kröner, F., Munz, C.-D., Schnitzer, T., & Wesenberg, M. (2002). Hyperbolic Divergence Cleaning for the MHD Equations. Journal of Computational Physics, 175(2), 645-673.</li>
    <li>Derigs, D., Gassner, G. J., Walch, S., & Winters, A. R. (2017). Entropy Stable Finite Volume Approximations for Ideal Magnetohydrodynamics (arXiv:1708.03537). arXiv.</li>
    <li>Dumbser, M., & Toro, E. F. (2011). A Simple Extension of the Osher Riemann Solver to Non-conservative Hyperbolic Systems. Journal of Scientific Computing, 48(1–3), 70–88.</li>
    <li>Evans, C. R., & Hawley, J. F. (1988). Simulation of Magnetohydrodynamic Flows: A Constrained Transport Model. The Astrophysical Journal, 332, 659.</li>
    <li>Federrath, C., Roman-Duval, J., Klessen, R. S., Schmidt, W., & Mac-Low, M.-M. (2010). Comparing the statistics of interstellar turbulence in simulations and observations: Solenoidal versus compressive turbulence forcing. Astronomy & Astrophysics ,512, A81.</li>
    <li>Felker, K. G., & Stone, J. (2018). A fourth-order accurate finite volume method for ideal MHD via upwind constrained transport. Journal of Computational Physics, 375, 1365–1400.</li>
    <li>Fleischmann, N., Adami, S., & Adams, N. A. (2020). A shock-stable modification of the HLLC Riemann solver with reduced numerical dissipation. Journal of Computational Physics, 423, 109762.</li>
    <li>Fromm, J. E. (1968). A method for reducing dispersion in convective difference schemes. Journal of Computational Physics, 3, 176.</li>
    <li>Fu, L., Hu, X. Y., & Adams, N. A. (2016). A family of high-order targeted ENO schemes for compressible-fluid simulations. Journal of Computational Physics, 305, 333-359.</li>
    <li>Fu, L. (2021). Very-high-order TENO schemes with adaptive accuracy order and adaptive dissipation control. Computer Methods in Applied Mechanics and Engineering, 387, 114193.</li>
    <li>Gardiner, T. A. & Stone, J. M. (2005). An unsplit Godunov method for ideal MHD via constrained transport. Journal of Computational Physics, 205(2), 509–539.</li>
    <li>Godunov, S. K. (1959). A Difference Scheme for Numerical Solution of Discontinuous Solution of Hydrodynamic Equations. Mat. Sbornik, 47, 271-306. Translated US Joint Publ. Res. Service, JPRS 7226, 1969</li>
    <li>Gottlieb, S., Ketcheson, D. I., & Shu, C.-W. (2009). High Order Strong Stability Preserving Time Discretizations. Journal of Scientific Computing, 38(3), 251–289.</li>
    <li>Grosheintz-Laval, L., & Käppeli, R. (2019). High-order well-balanced finite volume schemes for the Euler equations with gravitation. Journal of Computational Physics, 378, 324-343.</li>
    <li>Harten, A. (1983). High Resolution Schemes for Hyperbolic Conservation Laws. Journal of Computational Physics, 49(3), 357–393.</li>
    <li>Harten, A., Lax, P., & van Leer, B. (1983). On upstream differencing and godunov-type schemes for hyperbolic conservation laws. SIAM Review, 25(1), 35–61.</li>
    <li>Helzel, C., Rossmanith, J. A., & Taetz, B. (2011). An unstaggered constrained transport method for the 3D ideal magnetohydrodynamic equations. Journal of Computational Physics, 230(10), 3803-3829.</li>
    <li>Jiang, G. S., & Shu, C.-W. (1996). Efficient Implementation of Weighted ENO Schemes. Journal of Computational Physics, 126(1), 202-228.</li>
    <li>Ketcheson, D. I. (2008). Highly Efficient Strong Stability-Preserving Runge–Kutta Methods with Low-Storage Implementations. SIAM Journal on Scientific Computing, 30(4), 2113–2136.</li>
    <li>Kraaijevanger, J. F. B. M. (1991). Contractivity of Runge-Kutta methods. BIT, 31(3), 482–528.</li>
    <li>Lax, P. D., & Wendroff, B. (1960). Systems of conservation laws. Commun. Pure Appl. Math. 13 (2), 217–237.</li>
    <li>Lax, P. D., & Liu, X.-D. (1998). Solution of Two-Dimensional Riemann Problems of Gas Dynamics by Positive Schemes. SIAM Journal on Scientific Computing, 19(2), 319–340.</li>
    <li>LeVeque, R. J. (1992). Numerical Methods for Conservation Laws (2nd ed.). Birkhäuser Basel.</li>
    <li>Levy, D., Puppo, G., & Russo, G. (1999). Central WENO Schemes for Hyperbolic Systems of Conservation Laws. Mathematical Modelling and Numerical Analysis, 33(3), 547-571.</li>
    <li>Levy, D., Puppo, G., & Russo, G. (2000). Compact Central WENO Schemes for Multidimensional Conservation Laws. SIAM Journal on Scientific Computing, 22(2), 656-672.</li>
    <li>Machida, M., Hayashi, M. R., & Matsumoto, R. (1999). Global Simulations of Differentially Rotating Magnetized Disks: Formation of Low-β Filaments and Structured Coronae. The Astrophysical Journal, 532, L67-L70.</li>
    <li>Markert, J., Walch, S., & Gassner, G. (2022). A discontinuous Galerkin solver in the <span style="font-variant:small-caps;">flash</span> multiphysics framework. Monthly Notices of the Royal Astronomical Society, 511(3), 4179-4200.</li>
    <li>McCorquodale, P., & Colella, P. (2011). A high-order finite-volume method for conservation laws on locally refined grids. Communications in Applied Mathematics and Computational Science, 6(1), 1–25.</li>
    <li>Mignone, A. & Del Zanna, L. (2021). Systematic construction of upwind constrained transport schemes for MHD. Journal of Computational Physics, 424, 109748.</li>
    <li>Miyoshi, T., & Kusano, K. (2005). A multi-state HLL approximate Riemann solver for ideal magnetohydrodynamics. Journal of Computational Physics, 208(1), 315–344.</li>
    <li>Noh, W. F. (1987). Errors for calculations of strong shocks using an artificial viscosity and an artificial heat flux. Journal of Computational Physics, 72(1), 78-120.</li>
    <li>Orszag, S. A., & Tang, C.-M. (1979). Small-scale structure of two-dimensional magnetohydrodynamic turbulence. Journal of Fluid Mechanics, 90, 129-143.</li>
    <li>Pfrommer, C., Springel, V., Ensslin, T. A., & Jubelgas, M. (2006). Detecting shock waves in cosmological smoothed particle hydrodynamics simulations. Monthly Notices of the Royal Astronomical Society, 367(1), 113–131.</li>
    <li>Powell, K. G. (1994). An approximate Riemann solver for magnetohydrodynamics (that works in more than one dimension). NASA Technical Reports, NAS 1.26:194902.</li>
    <li>Prince, P. J., & Dormand, J. R. (1981). High order embedded Runge-Kutta formulae. Journal of Computational and Applied Mathematics, 7(1), 67–75.</li>
    <li>Roe, P., & Pike, J. (1984). Efficient Conservation and Utilisation of Approximate Riemann Solution. Computing Methods in Applied Science and Engineering, 6, pp. 499-558.</li>
    <li>Roy, C. J., Nelson, C. C., Smith, T. M., & Ober, C. C. (2004). Verification of euler/navier–stokes codes using the method of manufactured solutions. International Journal for Numerical Methods in Fluids, 44(6), 599–620.</li>
    <li>Ryu, D., & Jones, T. W. (1995). Numerical magetohydrodynamics in astrophysics: Algorithm and tests for one-dimensional flow. The Astrophysical Journal, 442, 228.</li>
    <li>San, O., & Kara, K. (2015). Evaluation of Riemann flux solvers for WENO reconstruction schemes: Kelvin–Helmholtz instability. Computers & Fluids, 117, 24–41.</li>
    <li>Sedov, L. I. (1946). Propagation of strong shock waves. Journal of Applied Mathematics and Mechanics, 10, 241-250.</li>
    <li>Sod, G. A. (1978). A survey of several finite difference methods for systems of nonlinear hyperbolic conservation laws. Journal of Computational Physics, 27(1), 1-31.</li>
    <li>Shu, C.-W., & Osher, S. (1988). Efficient implementation of essentially non-oscillatory shock-capturing schemes. Journal of Computational Physics, 77(2), 439–471.</li>
    <li>Shu, C.-W. (2009). High Order Weighted Essentially Nonoscillatory Schemes for Convection Dominated Problems. SIAM Review, 51(1), 82–126.</li>
    <li>Shu, F. (1991). Physics of Astrophysics, Vol. II: Gas Dynamics. New York: University Science Books.</li>
    <li>Spiteri, R. J., & Ruuth, S. J. (2002). A New Class of Optimal High-Order Strong-Stability-Preserving Time Discretization Methods. SIAM Journal on Numerical Analysis, 40(2), 469–491.</li>
    <li>Toro, E. F., Spruce, M., & Speares, W. (1994). Restoration of the Contact Surface in the HLL Riemann Solver. Shock Waves, 4, 25-34.</li>
    <li>Toro, E. F., & Titarev, V. A. (2006). MUSTA fluxes for systems of conservation laws. Journal of Computational Physics, 216(2), 403–429.</li>
    <li>Verma, P. S., Jean-Mathieu, T., & Müller, W.-C. (2018). Fourth-order accurate finite-volume CWENO scheme for astrophysical MHD problems. Monthly Notices of the Royal Astronomical Society, 482(1), 416-437.</li>
    <li>Williamson, J. H. (1980). Low-storage Runge-Kutta schemes. Journal of Computational Physics, 35(1), 48–56.</li>
    <li>Yee, H-C., Sandham, N., & Djomehri, M., (1999). Low dissipative high order shock-capturing methods using characteristic-based filters. Journal of Computational Physics, 150(1), 199-238.</li>
    <li>Yee, H-C., & Sjögreen, B. (2005). Divergence Free High Order Filter Methods for the Compressible MHD Equations. Proc. International Conderence on High Performance Scientific Computing, 559-575.</li>
  </ol>
</details>


<a name="installation"></a>

# Installation

Clone this repository onto your local machine, and navigate to the cloned repository. 

In order to install the minimum packages to run the simulation, in the command line, run:
```bash
python3 -m pip install .
```

To test whether the installation has installed properly, run:
```bash
python3 astrea.py --init
```
This would also create a `parameters.yml` file for changing the simulation configurations.


<a name="usage"></a>

# Usage

The main method to run the simulation would be to edit the simulation parameters in `parameters.yml` and running the main Python file:

```bash
python3 astrea.py
```

OR

```bash
./astrea.py
```

Alternatively, the code can be run with CLI options:

```bash
python3 astrea.py --config=sedov --cells=256 --file=/path/to/checkpoint_file
```

See _`--help`_ for a list of available options.

_Running the code in a Python interactive shell is also possible, although this is generally not recommended:_

```python
>>> import astrea
>>> astrea.run()
```


<a name="organisation"></a>

# Organisation

```
astrea/
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
├── __init__.py
├── astrea.py               : Core script for running the simulation
├── functions
│   ├── __init__.py
│   ├── analytic.py   : Analytical solutions to smooth advection wave tests
│   ├── generic.py    : Generic functions not specific to FVM
│   ├── grid.py       : Grid functions used for padding, slicing, higher-order interpolations, etc.
│   ├── math.py       : Math functions, including specialised math functions
│   ├── numeric.py    : Numerical functions for computing eigenvectors, Jacobian matrices, wavespeeds, etc.
│   └── plotting.py   : Functions for media, e.g., (live-)plotting, saving videos
├── iokit
│   ├── __init__.py
│   ├── chkpt_funcs.py  : Checkpoint file I/O functions
│   ├── cli_funcs.py    : User input I/O functions
│   ├── handler.py      : Consolidator for all I/O functions; implements priority list for user input
│   ├── simulation.py   : Functions for generating simulation variables
│   ├── param_funcs.py  : Parameter file I/O functions
├── numkit
│   ├── __init__.py
│   ├── c_transport.py  : Functions for the constrained transport implementation
│   ├── limiters.py     : Implements slope limiters for the reconstructed states
│   ├── solvers.py      : Contains the various Riemann solvers
├── parameters.yml      : Parameters for the simulation (not tracked by git)
├── physics
│   ├── __init__.py
│   ├── krome
│   │   ├── build           : Build for default chemical network
│   │   ├── __init__.py
│   │   ├── abundances.yml  : Initial abundances for chemical species in default network
│   │   ├── krome_funcs.py  : Functions for building and parsing krome routines
│   ├── constants.py        : Conversion between code units & CGS units
│   ├── gravity.py          : Functions for self-gravity (FFT Poisson solver)
│   ├── tracers.py          : Functions for tracer particles
├── spatial
│   ├── __init__.py
│   ├── cweno.py    : Central weighted essentially non-oscillatory method (CWENO) [Levy et al., 1999]
│   ├── pcm.py      : Piecewise constant method (PCM) [Godunov, 1959]
│   ├── plm.py      : Piecewise linear method (PLM) [Derigs et al., 2018]
│   ├── ppm.py      : Piecewise parabolic method (PPM) [McCorquodale & Colella, 2011; Felker & Stone, 2018]
│   ├── spatial.py  : Handler for spatial reconstruction schemes
│   ├── teno.py     : Targeted ENO method (TENO) [Fu et al., 2016]
│   ├── weno.py     : Weighted essentially non-oscillatory method (WENO) [Jiang & Shu, 1996]
│   ├── wenoz.py    : WENO method with higher-order smoothness indicators (WENO-Z) [Borges et al., 2008]
├── static
│   ├── __init__.py
│   ├── .db.json        : Database for parameters
│   ├── .default.yml    : Default parameters file
|   ├── *.gif           : .gif files for graphics in README.md
│   ├── tests.py        : Initial conditions for (magneto)hydrodynamics tests
├── temporal
│   ├── __init__.py
│   ├── euler.py      : Forward Euler (explicit) scheme
│   ├── rk4.py        : (Standard) Runge-Kutta 3 scheme
│   ├── ssprk2.py     : Second-order strong stability-preserving scheme [Gottlieb et al., 2009]
│   ├── ssprk3.py     : Third-order strong stability-preserving schemes [Shu & Osher, 1988; Spiteri & Ruuth, 2002; Gottlieb et al., 2009]
│   ├── ssprk4.py     : Fourth-order strong stability-preserving schemes [Kraaijevanger, 1991; Ruuth & Spiteri, 2002; Ketcheson, 2008]
│   ├── ssprk5.py     : Fifth-order strong stability-preserving scheme [Gottlieb et al., 2009]
│   ├── temporal.py   : Handler for time integration schemes
├── utilities
│   ├── plot_chkpt.py    : Standalone plotting function for checkpoint files
```
