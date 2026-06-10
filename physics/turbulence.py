import numpy as np

##############################################################################
# Turbulence and perturbations module
##############################################################################

# Create a grid of perturbation values
def pertubations(grid, max_ampl):
    return np.random.uniform(-max_ampl/2, max_ampl, size=grid.shape)


# Initialise the turbulence driving grid
def initialise(sim_variables):
    return np.zeros(list(sim_variables.cells)+[sim_variables.dimensions,], dtype=float, order='C')


# Compute and set up the turbulent driving field based on the mixing ratio
def drive(forcing_field, dt, eigmax, sim_variables, proj='wiener'):
    cells, dimensions = sim_variables.cells, sim_variables.dimensions
    ds = list(sim_variables.ds.values())
    axes = tuple(range(dimensions))

    # Load turbulent driving parameters (values can be altered at any time during simulation too; just re-assign class attribute)
    zeta = sim_variables.test_specifics['zeta']
    mach = sim_variables.test_specifics['mach']
    f_rms = sim_variables.test_specifics['f_rms']
    k_min, k_max = sim_variables.test_specifics['k_range']

    def get_projections(field, _kvectors, _k2):
        # Compute dot product between k-vector and field
        dot_product = sum(_kvectors[i] * field[...,i] for i in range(dimensions))

        # Obtain compressive components (divergence)
        k_divs = [(dot_product/_k2) * _kvectors[i] for i in range(dimensions)]

        # Obtain solenoidal components (curl)
        k_curls = [field[...,i] - k_divs[i] for i in range(dimensions)]

        # Compute the projection operator from Helmholtz decomposition
        # zeta is the compressive (div) to solenoidal (curl) forcing mixing ratio
        projections = np.stack([(1 - zeta) * k_curls[i] + zeta * k_divs[i] for i in range(dimensions)], axis=-1)

        return projections

    # Perform Helmholtz decomposition of grid; get curl and divergence components of vector field
    # Convert grid to Fourier space (density)
    fourier_field = np.fft.fftn(forcing_field, axes=axes)

    # Construct k-vectors for each dimension with FFT
    kvectors = np.meshgrid(*[2 * np.pi * np.fft.fftfreq(n, d=dh) for n, dh in zip(cells, ds)], indexing='ij')

    # Compute k^2 and protect against division by zero
    k2 = sum(k**2 for k in kvectors)
    k2[k2 == 0] = 1.

    # Compute forcing mask and power spectrum
    k_norm = np.sqrt(k2, where=(k2>0), out=np.zeros_like(k2)) / (2*np.pi)
    mask = np.where((k_min < k_norm) & (k_norm < k_max))
    power_spectrum = np.zeros_like(k_norm)
    power_spectrum[mask] = 1 - (k_norm[mask] - 2)**2
    power_spectrum[k_norm == 0] = 0.

    # Ornstein-Uhlenbeck process
    # Compute the autocorrelation time scale
    L = np.average([np.diff(_) for _ in sim_variables.coordinates.values()])
    T = L/(2 * eigmax * mach)

    # Wiener process in Fourier space dW(t)
    wiener_dist = np.random.normal(loc=0, scale=np.sqrt(dt), size=fourier_field.shape)

    # Apply projection operator to the Wiener process
    if proj == 'wiener':
        projections = get_projections(wiener_dist, kvectors, k2)
        df_k = -fourier_field * dt/T + power_spectrum[...,None] * projections

    # Alternatively apply projection operator to the Fourier space
    elif proj == 'force':
        projections = get_projections(fourier_field, kvectors, k2)
        df_k = -projections * dt/T + power_spectrum[...,None] * wiener_dist

    # Update the Fourier field
    fourier_field += df_k

    # Project to real space
    new_forcing_field = np.real(np.fft.ifftn(fourier_field, axes=axes))

    # Apply normalising factor g_chi
    rms = np.sqrt(np.mean(new_forcing_field**2))
    if rms > 0:
        new_forcing_field *= f_rms/rms

    return new_forcing_field


# Update step for turbulence with conservative grid, given the timestep dt
def update(grid, forcing_field, dt, sim_variables):
    rho, momentums, energy = sim_variables.rho, 1+np.array(range(sim_variables.dimensions)), sim_variables.energy

    original_momentum = np.copy(grid[...,momentums])

    grid[...,momentums] += dt * grid[...,rho][...,None] * forcing_field
    grid[...,energy] += dt * np.sum(original_momentum * forcing_field, axis=-1)

    return grid