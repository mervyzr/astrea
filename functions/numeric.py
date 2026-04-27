import numpy as np

from functions import grid as gutils
from functions import math as mfuncs

##############################################################################
# Functions for constructing objects such as eigenvectors, Jacobian and flux terms
##############################################################################

# Make flux as a function of cell-averaged (primitive) variables
def compute_flux(grid, sim_variables, axis):
    abscissa, ordinate, applicate = (axis + np.array(range(3)))%3
    gamma, permeability = sim_variables.gamma, sim_variables.constants.mu_0

    # In code units
    rhos, vels, pressures, Bfields = grid[...,sim_variables.rho], grid[...,sim_variables.vels], grid[...,sim_variables.pressure], grid[...,sim_variables.Bfields]
    arr = np.zeros_like(grid)

    arr[...,0] = rhos * vels[...,abscissa]
    arr[...,1+abscissa] = rhos*vels[...,abscissa]**2 + pressures + .5*mfuncs.norm(Bfields)**2 - (Bfields[...,abscissa]**2)/permeability
    arr[...,1+ordinate] = rhos*vels[...,abscissa]*vels[...,ordinate] - (Bfields[...,abscissa]*Bfields[...,ordinate])/permeability
    arr[...,1+applicate] = rhos*vels[...,abscissa]*vels[...,applicate] - (Bfields[...,abscissa]*Bfields[...,applicate])/permeability
    arr[...,4] = vels[...,abscissa]*(.5*rhos*mfuncs.norm(vels)**2 + (gamma*pressures)/(gamma-1) + mfuncs.norm(Bfields)**2) - (Bfields[...,abscissa]*np.einsum('...i,...i->...', vels, Bfields))/permeability
    arr[...,5+ordinate] = Bfields[...,ordinate]*vels[...,abscissa] - Bfields[...,abscissa]*vels[...,ordinate]
    arr[...,5+applicate] = Bfields[...,applicate]*vels[...,abscissa] - Bfields[...,abscissa]*vels[...,applicate]

    return arr


# Jacobian matrix based on primitive variables [Winters & Gassner, 2016]
def compute_jacobian(grid, sim_variables, axis):
    gamma, permeability = sim_variables.gamma, sim_variables.constants.mu_0

    # In code units
    rhos, vels, pressures, Bfields = grid[...,sim_variables.rho], grid[...,sim_variables.vels], grid[...,sim_variables.pressure], grid[...,sim_variables.Bfields]

    # Create empty square arrays for each cell
    _arr = np.zeros_like(grid)
    arr = np.repeat(_arr[...,None], _arr.shape[-1], axis=-1)
    i, j = np.diag_indices(_arr.shape[-1])

    abscissa, ordinate, applicate = (axis + np.array(range(3)))%3
    Bx, By, Bz = Bfields[...,abscissa], Bfields[...,ordinate], Bfields[...,applicate]

    # Input matrix with values at position [row i, col j]. Positions refer to x-axis alignment; the coordinate rotation is already done by permuting the input variables
    # Hydrodynamic components
    arr[...,i,j] = vels[...,abscissa][...,None]  # diagonal elements
    arr[...,0,1] = rhos
    arr[...,1,4] = 1/rhos
    arr[...,4,1] = gamma * pressures

    # Magneto- components
    arr[...,2,6] = -mfuncs.divide(Bx, permeability*rhos)
    arr[...,3,7] = -mfuncs.divide(Bx, permeability*rhos)
    arr[...,1,6] = mfuncs.divide(By, permeability*rhos)
    arr[...,1,7] = mfuncs.divide(Bz, permeability*rhos)

    arr[...,6,2] = -Bx
    arr[...,7,3] = -Bx
    arr[...,6,1] = By
    arr[...,7,1] = Bz

    return arr


# Calculate the Roe-averaged primitive variables at the interface from the minus- & plus-interface states for use in Roe solver in order to better capture shocks [Roe & Pike, 1984; Brio & Wu, 1988; LeVeque, 2002; Stone et al., 2008]
def compute_Roe_average(interfaces, sim_variables):
    rho, pressure, vels, Bfields = sim_variables.rho, sim_variables.pressure, sim_variables.vels, sim_variables.Bfields

    plus_interface, minus_interface = interfaces
    avg = np.zeros_like(plus_interface)
    rho_plus, rho_minus = np.sqrt(plus_interface[...,rho]), np.sqrt(minus_interface[...,rho])

    avg[...,rho] = rho_minus * rho_plus
    avg[...,vels] = mfuncs.divide((plus_interface[...,vels] * rho_plus[...,None]) + (minus_interface[...,vels] * rho_minus[...,None]), (rho_minus + rho_plus)[...,None])
    avg[...,pressure] = mfuncs.divide((rho_plus * plus_interface[...,pressure]) + (rho_minus * minus_interface[...,pressure]), rho_minus + rho_plus)
    avg[...,Bfields] = mfuncs.divide((plus_interface[...,Bfields] * rho_minus[...,None]) + (minus_interface[...,Bfields] * rho_plus[...,None]), (rho_minus + rho_plus)[...,None])
    return avg


# Compute the max eigenvalues for calculating the time evolution
def compute_eigmax(characteristics, axis):
    # Local max eigenvalue for each cell (1- or 3-Riemann invariant; shock wave or rarefaction wave)
    local_max_eigvals = np.max(np.abs(characteristics), axis=-1)

    # Local max eigenvalue between consecutive pairs of cell
    max_eigvals = np.maximum(gutils.slice_(local_max_eigvals, axis, end=-1), gutils.slice_(local_max_eigvals, axis, start=1))

    # Maximum wave speed (max eigenvalue) for time evolution
    return np.max(max_eigvals)


# Compute wavespeeds for a grid
def compute_wavespeeds(grid, sim_variables, axis, waves='all'):
    gamma, permeability = sim_variables.gamma, sim_variables.constants.mu_0
    rho, pressure, Bfields = sim_variables.rho, sim_variables.pressure, sim_variables.Bfields

    waves = waves.lower()
    match = lambda substrings: any(wave in waves for wave in substrings)

    if match(['sound', 'fast', 'cff', 'slow', 'css', 'all']) or waves in ['cs', 'a']:
        sound_speed = np.sqrt(mfuncs.divide(gamma * grid[...,pressure], grid[...,rho]))
        if 'sound' in waves or waves in ['cs', 'a']:
            return sound_speed
    if match(['alfven', 'ca', 'fast', 'cff', 'slow', 'css', 'all']):
        if match(['fast', 'slow', 'all']):
            alfven_speed_x = mfuncs.divide(grid[...,5+axis], np.sqrt(grid[...,rho] * permeability))
            alfven_speed = mfuncs.divide(mfuncs.norm(grid[...,Bfields]), np.sqrt(grid[...,rho] * permeability))
        else:
            if waves.endswith(('x', 'y', 'z')):
                return mfuncs.divide(grid[...,5+axis], np.sqrt(grid[...,rho] * permeability))
            else:
                return mfuncs.divide(mfuncs.norm(grid[...,Bfields]), np.sqrt(grid[...,rho] * permeability))
    if match(['fast', 'cff', 'all']):
        fast_magnetosonic_wave = np.sqrt(.5 * (sound_speed**2 + alfven_speed**2 + np.sqrt((sound_speed**2 + alfven_speed**2)**2 - (2 * sound_speed * alfven_speed_x)**2)))
        if waves != 'all':
            return fast_magnetosonic_wave
    if match(['slow', 'css', 'all']):
        slow_magnetosonic_wave = np.sqrt(.5 * (sound_speed**2 + alfven_speed**2 - np.sqrt((sound_speed**2 + alfven_speed**2)**2 - (2 * sound_speed * alfven_speed_x)**2)))
        if waves != 'all':
            return slow_magnetosonic_wave

    if waves == 'all':
        return sound_speed, alfven_speed, alfven_speed_x, fast_magnetosonic_wave, slow_magnetosonic_wave


# Characteristics (diagonalised eigenmatrix) [Stone et al., 2008]
def compute_characteristics(grid, sim_variables, axis):
    uN = grid[...,1+axis]
    if sim_variables.magnetic:
        _, cA, _, cFF, cSS = compute_wavespeeds(grid, sim_variables, axis=axis)
        characteristics = np.array([uN - cFF, uN - cA, uN - cSS, uN, uN + cSS, uN + cA, uN + cFF]).transpose(np.roll(np.arange(sim_variables.dimensions+1), -1))
    else:
        cs = compute_wavespeeds(grid, sim_variables, axis=axis, waves='sound')
        characteristics = np.array([uN - cs, uN, uN, uN, uN + cs]).transpose(np.roll(np.arange(sim_variables.dimensions+1), -1))
    return characteristics


# Make the left & right eigenvectors for adiabatic magnetohydrodynamics [Roe & Balsara, 1996; Stone et al., 2008; Derigs et al., 2016]
# Here, Stone and Roe & Balsara only uses the 7-wave formulation due to constrained transport; the divergence wave is not needed
# Powell adds an 8th "divergence" wave to correct for the longitudinal magnetic field for divergence cleaning. Derigs modifies it further for entropy-stable formulation
def compute_eigenvectors(grids, sim_variables, axis, vectors="both"):
    abscissa, ordinate, applicate = (axis + np.array(range(3)))%3

    # In code units
    rhos, Bfields = grids[...,sim_variables.rho], grids[...,sim_variables.Bfields]
    Bx, By, Bz = Bfields[...,abscissa], Bfields[...,ordinate], Bfields[...,applicate]

    if sim_variables.magnetic:
        # Compute wavespeeds
        cs, cA, cAx, cFF, cSS = compute_wavespeeds(grids, sim_variables, axis=axis)

        # Define abbreviated terms
        alpha_f = np.sqrt(mfuncs.divide(cs**2 - cSS**2, cFF**2 - cSS**2))
        alpha_s = np.sqrt(mfuncs.divide(cFF**2 - cs**2, cFF**2 - cSS**2))

        # Magnetic field degenerate cases
        S = np.sign(Bx)
        transverse_field = np.sqrt(By**2 + Bz**2)
        non_degenerate_transverse_field = np.where(transverse_field != 0)
        beta_y = np.full_like(By, 1/np.sqrt(2))
        beta_z = np.full_like(Bz, 1/np.sqrt(2))
        beta_y[non_degenerate_transverse_field] = mfuncs.divide(By, transverse_field)[non_degenerate_transverse_field]
        beta_z[non_degenerate_transverse_field] = mfuncs.divide(Bz, transverse_field)[non_degenerate_transverse_field]

        # Handle degeneracy cases
        degenerate = np.where((cAx == cs) & (cA == cs))
        alpha_f[degenerate], alpha_s[degenerate] = 1, 0

        # Define frequently used terms
        Cff, Css = cFF * alpha_f, cSS * alpha_s
        Qff, Qss = Cff * S, Css * S
        Af, As = cs * alpha_f * np.sqrt(rhos), cs * alpha_s * np.sqrt(rhos)

        # Pop the longitudinal magnetic field
        _ = len(sim_variables.ambient) - 1

    else:
        # Compute sound speed
        cs = compute_wavespeeds(grids, sim_variables, axis=axis, waves='sound')

        # Pop the magnetic field components
        _ = len(sim_variables.ambient) - 3

    # Compute characteristics and generate the RIGHT eigenvectors; the coordinate rotation is already done by permuting the input variables
    if vectors.casefold().startswith(("b", "r")):
        right_eigenvectors = np.repeat(np.zeros_like(grids)[...,None], grids.shape[-1], axis=-1)

        if sim_variables.magnetic:
            ralphaf, ralphas = rhos * alpha_f, rhos * alpha_s
            r2alphaf, r2alphas = ralphaf * cs**2, ralphas * cs**2
            QssBy, QssBz = Qss * beta_y, Qss * beta_z
            QffBy, QffBz = Qff * beta_y, Qff * beta_z
            AsBy, AsBz = As * beta_y, As * beta_z
            AfBy, AfBz = Af * beta_y, Af * beta_z
            BySrho, BzSrho = beta_y * S * np.sqrt(rhos), beta_z * S * np.sqrt(rhos)

            # First column (Fast- magnetoacoustic wave)
            right_eigenvectors[...,0,0] = ralphaf
            right_eigenvectors[...,1,0] = -Cff
            right_eigenvectors[...,2,0] = QssBy
            right_eigenvectors[...,3,0] = QssBz
            right_eigenvectors[...,4,0] = r2alphaf
            right_eigenvectors[...,5,0] = AsBy
            right_eigenvectors[...,6,0] = AsBz
            # Second column (Alfven- wave)
            right_eigenvectors[...,2,1] = -beta_z
            right_eigenvectors[...,3,1] = beta_y
            right_eigenvectors[...,5,1] = -BzSrho
            right_eigenvectors[...,6,1] = BySrho
            # Third column (Slow- magnetoacoustic wave)
            right_eigenvectors[...,0,2] = ralphas
            right_eigenvectors[...,1,2] = -Css
            right_eigenvectors[...,2,2] = -QffBy
            right_eigenvectors[...,3,2] = -QffBz
            right_eigenvectors[...,4,2] = r2alphas
            right_eigenvectors[...,5,2] = -AfBy
            right_eigenvectors[...,6,2] = -AfBz
            # Fourth column (Entropy/contact wave)
            right_eigenvectors[...,0,3] = 1
            # Fifth column (Slow+ magnetoacoustic wave)
            right_eigenvectors[...,0,4] = ralphas
            right_eigenvectors[...,1,4] = Css
            right_eigenvectors[...,2,4] = QffBy
            right_eigenvectors[...,3,4] = QffBz
            right_eigenvectors[...,4,4] = r2alphas
            right_eigenvectors[...,5,4] = -AfBy
            right_eigenvectors[...,6,4] = -AfBz
            # Sixth column (Alfven+ wave)
            right_eigenvectors[...,2,5] = beta_z
            right_eigenvectors[...,3,5] = -beta_y
            right_eigenvectors[...,5,5] = -BzSrho
            right_eigenvectors[...,6,5] = BySrho
            # Seventh column (Fast+ magnetoacoustic wave)
            right_eigenvectors[...,0,6] = ralphaf
            right_eigenvectors[...,1,6] = Cff
            right_eigenvectors[...,2,6] = -QssBy
            right_eigenvectors[...,3,6] = -QssBz
            right_eigenvectors[...,4,6] = r2alphaf
            right_eigenvectors[...,5,6] = AsBy
            right_eigenvectors[...,6,6] = AsBz

        else:
            csrho = mfuncs.divide(cs, rhos)
            cs2 = cs**2

            # First column
            right_eigenvectors[...,0,0] = 1
            right_eigenvectors[...,1,0] = -csrho
            right_eigenvectors[...,4,0] = cs2
            # Second column
            right_eigenvectors[...,0,1] = 1
            # Third column
            right_eigenvectors[...,2,2] = 1
            # Fourth column
            right_eigenvectors[...,3,3] = 1
            # Fifth column
            right_eigenvectors[...,0,4] = 1
            right_eigenvectors[...,1,4] = csrho
            right_eigenvectors[...,4,4] = cs2

    # Compute characteristics and generate the LEFT eigenvectors; the coordinate rotation is already done by permuting the input variables
    if vectors.casefold().startswith(("b", "l")):
        left_eigenvectors = np.repeat(np.zeros_like(grids)[...,None], grids.shape[-1], axis=-1)

        if sim_variables.magnetic:
            Nf = Ns = 1/(2*cs**2)
            NfCff, NsCss = Nf * Cff, Ns * Css
            Nfalphaf, Nsalphas = mfuncs.divide(Nf * alpha_f, rhos), mfuncs.divide(Ns * alpha_s, rhos)
            NfQssBy, NfQssBz = Nf * Qss * beta_y, Nf * Qss * beta_z
            NsQffBy, NsQffBz = Ns * Qff * beta_y, Ns * Qff * beta_z
            NfAsBy, NfAsBz = mfuncs.divide(Nf * As * beta_y, rhos), mfuncs.divide(Nf * As * beta_z, rhos)
            NsAfBy, NsAfBz = mfuncs.divide(Ns * Af * beta_y, rhos), mfuncs.divide(Ns * Af * beta_z, rhos)
            ByS2rho, BzS2rho = mfuncs.divide(beta_y * S, 2 * np.sqrt(rhos)), mfuncs.divide(beta_z * S, 2 * np.sqrt(rhos))

            # First row (Fast- magnetoacoustic wave)
            left_eigenvectors[...,0,1] = -NfCff
            left_eigenvectors[...,0,2] = NfQssBy
            left_eigenvectors[...,0,3] = NfQssBz
            left_eigenvectors[...,0,4] = Nfalphaf
            left_eigenvectors[...,0,5] = NfAsBy
            left_eigenvectors[...,0,6] = NfAsBz
            # Second row (Alfven- wave)
            left_eigenvectors[...,1,2] = -beta_z/2
            left_eigenvectors[...,1,3] = beta_y/2
            left_eigenvectors[...,1,5] = -BzS2rho
            left_eigenvectors[...,1,6] = ByS2rho
            # Third row (Slow- magnetoacoustic wave)
            left_eigenvectors[...,2,1] = -NsCss
            left_eigenvectors[...,2,2] = -NsQffBy
            left_eigenvectors[...,2,3] = -NsQffBz
            left_eigenvectors[...,2,4] = Nsalphas
            left_eigenvectors[...,2,5] = -NsAfBy
            left_eigenvectors[...,2,6] = -NsAfBz
            # Fourth row (Entropy/contact wave)
            left_eigenvectors[...,3,0] = 1
            left_eigenvectors[...,3,4] = -2 * Nf
            # Fifth row (Slow+ magnetoacoustic wave)
            left_eigenvectors[...,4,1] = NsCss
            left_eigenvectors[...,4,2] = NsQffBy
            left_eigenvectors[...,4,3] = NsQffBz
            left_eigenvectors[...,4,4] = Nsalphas
            left_eigenvectors[...,4,5] = -NsAfBy
            left_eigenvectors[...,4,6] = -NsAfBz
            # Sixth row (Alfven+ wave)
            left_eigenvectors[...,5,2] = beta_z/2
            left_eigenvectors[...,5,3] = -beta_y/2
            left_eigenvectors[...,5,5] = -BzS2rho
            left_eigenvectors[...,5,6] = ByS2rho
            # Seventh row (Fast+ magnetoacoustic wave)
            left_eigenvectors[...,6,1] = NfCff
            left_eigenvectors[...,6,2] = -NfQssBy
            left_eigenvectors[...,6,3] = -NfQssBz
            left_eigenvectors[...,6,4] = Nfalphaf
            left_eigenvectors[...,6,5] = NfAsBy
            left_eigenvectors[...,6,6] = NfAsBz

        else:
            rho2cs = mfuncs.divide(rhos, 2*cs)
            a2 = 1/(2 * cs**2)

            # First row
            left_eigenvectors[...,0,1] = -rho2cs
            left_eigenvectors[...,0,4] = a2
            # Second row
            left_eigenvectors[...,1,0] = 1
            left_eigenvectors[...,1,4] = -2 * a2
            # Third row
            left_eigenvectors[...,2,2] = 1
            # Fourth row
            left_eigenvectors[...,3,3] = 1
            # Fifth row
            left_eigenvectors[...,4,1] = rho2cs
            left_eigenvectors[...,4,4] = a2

    if vectors.casefold().startswith("l"):
        return left_eigenvectors
    elif vectors.casefold().startswith("r"):
        return right_eigenvectors
    else:
        return left_eigenvectors, right_eigenvectors


# Make the right eigenvectors for adiabatic magnetohydrodynamics [Derigs et al., 2016]
def compute_right_eigenvectors(grids, sim_variables, axis):
    abscissa, ordinate, applicate = (axis + np.array(range(3)))%3
    gamma = sim_variables.gamma

    rhos, vels, pressures, Bfields = grids[...,sim_variables.rho], grids[...,sim_variables.vels], grids[...,sim_variables.pressure], grids[...,sim_variables.Bfields]
    vx, vy, vz = vels[...,abscissa], vels[...,ordinate], vels[...,applicate]
    Bx, By, Bz = Bfields[...,abscissa], Bfields[...,ordinate], Bfields[...,applicate]

    # Define the right eigenvectors for each cell in each grid
    right_eigenvectors = np.repeat(np.zeros_like(grids)[...,None], grids.shape[-1], axis=-1)

    # Compute wavespeeds
    cs, _, _, cFF, cSS = compute_wavespeeds(grids, sim_variables, axis)

    # Define frequently used components
    S = np.sign(Bx)
    alpha_f = np.sqrt(mfuncs.divide(cs**2 - cSS**2, cFF**2 - cSS**2))
    alpha_s = np.sqrt(mfuncs.divide(cFF**2 - cs**2, cFF**2 - cSS**2))
    b_perpend = np.sqrt(mfuncs.divide(By**2 + Bz**2, rhos))
    beta2 = mfuncs.divide(By, np.sqrt(By**2 + Bz**2))
    beta3 = mfuncs.divide(Bz, np.sqrt(By**2 + Bz**2))

    psi_plus_slow = (
        .5 * alpha_s * rhos * mfuncs.norm(vels)**2
        - cs * alpha_f * rhos * b_perpend
        + (alpha_s * rhos * cs**2)/(gamma - 1)
        + alpha_s * cSS * rhos * vx
        + alpha_f * cFF * rhos * S * (vy*beta2 + vz*beta3)
        )
    psi_minus_slow = (
        .5 * alpha_s * rhos * mfuncs.norm(vels)**2
        - cs * alpha_f * rhos * b_perpend
        + (alpha_s * rhos * cs**2)/(gamma - 1)
        - alpha_s * cSS * rhos * vx
        - alpha_f * cFF * rhos * S * (vy*beta2 + vz*beta3)
        )
    psi_plus_fast = (
        .5 * alpha_f * rhos * mfuncs.norm(vels)**2
        + cs * alpha_s * rhos * b_perpend
        + (alpha_f * rhos * cs**2)/(gamma - 1)
        + alpha_f * cFF * rhos * vx
        - alpha_s * cSS * rhos * S * (vy*beta2 + vz*beta3)
        )
    psi_minus_fast = (
        .5 * alpha_f * rhos * mfuncs.norm(vels)**2
        + cs * alpha_s * rhos * b_perpend
        + (alpha_f * rhos * cs**2)/(gamma - 1)
        - alpha_f * cFF * rhos * vx
        + alpha_s * cSS * rhos * S * (vy*beta2 + vz*beta3)
        )

    # Generate the right eigenvectors
    # First column (Fast- magnetoacoustic wave)
    right_eigenvectors[...,0,0] = rhos * alpha_f
    right_eigenvectors[...,1,0] = rhos * alpha_f * (vx - cFF)
    right_eigenvectors[...,2,0] = rhos * (alpha_f*vy + alpha_s*cSS*beta2*S)
    right_eigenvectors[...,3,0] = rhos * (alpha_f*vz + alpha_s*cSS*beta3*S)
    right_eigenvectors[...,4,0] = psi_minus_fast
    right_eigenvectors[...,6,0] = alpha_s * cs * beta2 * np.sqrt(rhos)
    right_eigenvectors[...,7,0] = alpha_s * cs * beta3 * np.sqrt(rhos)
    # Second column (Alfven- wave)
    right_eigenvectors[...,2,1] = -beta3 * rhos**1.5
    right_eigenvectors[...,3,1] = beta2 * rhos**1.5
    right_eigenvectors[...,4,1] = (beta2*vz - beta3*vy) * rhos**1.5
    right_eigenvectors[...,6,1] = -rhos * beta3
    right_eigenvectors[...,7,1] = rhos * beta2
    # Third column (Slow- magnetoacoustic wave)
    right_eigenvectors[...,0,2] = rhos * alpha_s
    right_eigenvectors[...,1,2] = rhos * alpha_s * (vx - cSS)
    right_eigenvectors[...,2,2] = rhos * (alpha_s*vy - alpha_f*cFF*beta2*S)
    right_eigenvectors[...,3,2] = rhos * (alpha_s*vz - alpha_f*cFF*beta3*S)
    right_eigenvectors[...,4,2] = psi_minus_slow
    right_eigenvectors[...,6,2] = -alpha_f * cs * beta2 * np.sqrt(rhos)
    right_eigenvectors[...,7,2] = -alpha_f * cs * beta3 * np.sqrt(rhos)
    # Fourth column (Entropy wave)
    right_eigenvectors[...,0,3] = 1
    right_eigenvectors[...,1,3] = vx
    right_eigenvectors[...,2,3] = vy
    right_eigenvectors[...,3,3] = vz
    right_eigenvectors[...,4,3] = .5 * mfuncs.norm(vels)**2
    # Fifth column (Divergence wave)
    right_eigenvectors[...,4,4] = Bx
    right_eigenvectors[...,6,4] = 1
    # Sixth column (Slow+ magnetoacoustic wave)
    right_eigenvectors[...,0,5] = rhos * alpha_s
    right_eigenvectors[...,1,5] = rhos * alpha_s * (vx + cSS)
    right_eigenvectors[...,2,5] = rhos * (alpha_s*vy + alpha_f*cFF*beta2*S)
    right_eigenvectors[...,3,5] = rhos * (alpha_s*vz + alpha_f*cFF*beta3*S)
    right_eigenvectors[...,4,5] = psi_plus_slow
    right_eigenvectors[...,6,5] = -alpha_f * cs * beta2 * np.sqrt(rhos)
    right_eigenvectors[...,7,5] = -alpha_f * cs * beta3 * np.sqrt(rhos)
    # Seventh column (Alfven+ wave)
    right_eigenvectors[...,2,6] = beta3 * rhos**1.5
    right_eigenvectors[...,3,6] = -beta2 * rhos**1.5
    right_eigenvectors[...,4,6] = (beta3*vy - beta2*vz) * rhos**1.5
    right_eigenvectors[...,6,6] = -rhos * beta3
    right_eigenvectors[...,7,6] = rhos * beta2
    # Eighth column (Fast+ magnetoacoustic wave)
    right_eigenvectors[...,0,7] = rhos * alpha_f
    right_eigenvectors[...,1,7] = rhos * alpha_f * (vx + cFF)
    right_eigenvectors[...,2,7] = rhos * (alpha_f*vy - alpha_s*cSS*beta2*S)
    right_eigenvectors[...,3,7] = rhos * (alpha_f*vz - alpha_s*cSS*beta3*S)
    right_eigenvectors[...,4,7] = psi_plus_fast
    right_eigenvectors[...,6,7] = alpha_s * cs * beta2 * np.sqrt(rhos)
    right_eigenvectors[...,7,7] = alpha_s * cs * beta3 * np.sqrt(rhos)

    # Scale the right eigenvectors with a diagonal scaling matrix, so as to prevent degeneracies [Barth, 1999]
    # For adiabatic magnetohydrodynamics in entropy-stable flux (primitive variables)
    if sim_variables.solver.startswith('e'):
        diag_scaler = np.zeros_like(right_eigenvectors)
        diag_scaler[...,0,0] = 1/(2*gamma*rhos)
        diag_scaler[...,1,1] = mfuncs.divide(pressures, 2*rhos**3)
        diag_scaler[...,2,2] = 1/(2*gamma*rhos)
        diag_scaler[...,3,3] = (rhos*(gamma-1))/gamma
        diag_scaler[...,4,4] = mfuncs.divide(pressures, rhos)
        diag_scaler[...,5,5] = 1/(2*gamma*rhos)
        diag_scaler[...,6,6] = mfuncs.divide(pressures, 2*rhos**3)
        diag_scaler[...,7,7] = 1/(2*gamma*rhos)
        right_eigenvectors = right_eigenvectors @ np.sqrt(diag_scaler)

    return right_eigenvectors


# Function for checking the numerical errors when computing the (primitive) Jacobian matrices, characteristic waves (eigenvalues/diagonal matrix), and left and right eigenvectors
def compute_characteristic_errors(grid, sim_variables, axis, check='identity'):
    from functions import numeric

    left_eigenvectors, right_eigenvectors = compute_eigenvectors(grid, sim_variables, axis)
    _axis = tuple(np.arange(-sim_variables.dimensions, 0))

    # Jacobian check: A = R @ λ @ L (stricter)
    if check.lower() == "jacobian":
        characteristics = compute_characteristics(grid, sim_variables, axis)

        i, j = np.diag_indices(characteristics.shape[-1])
        Lambda = np.zeros(sim_variables.cells + [len(i),len(j)])
        Lambda[...,i,j] = characteristics

        jacobian = compute_jacobian(grid, sim_variables, axis=axis)
        jacobian = np.delete(jacobian, 5+axis, axis=-2)
        jacobian = np.delete(jacobian, 5+axis, axis=-1)

        err = np.linalg.norm(jacobian - (right_eigenvectors @ Lambda @ left_eigenvectors), axis=_axis)

    # Identity check: L @ R = I
    elif check.lower() == "identity":
        err = np.linalg.norm((left_eigenvectors @ right_eigenvectors) - np.eye(right_eigenvectors.shape[-1]), axis=_axis)

    return err.max()