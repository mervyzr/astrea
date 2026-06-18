import numpy as np

from physics import constants

##############################################################################
# Conversion values for code units into physical units
##############################################################################

class Constants(object):
    def __init__(self, units):
        try:
            for name, value in constants.__dict__.items():
                if not name.startswith("_"):
                    setattr(self, name, value)
        except Exception:
            for name, value in constants.items():
                setattr(self, name, value)

        if units != "code":
            if units == 'custom':
                # Set up physical scaling (code -> CGS)
                L0 = self.pc
                m0 = self.m_sun
                t0 = self.sec_per_year

                # Set up plot scaling (CGS -> plot)
                length_scale, length_label = self.pc, " [pc]"
                mass_scale, mass_label = self.m_sun, r" [$\mathrm{M}_\odot$]"
                time_scale, time_label = self.sec_per_year, " yr"

                density_scale, density_label = 1, r" [$\mathrm{g}/\mathrm{cm}^3$]"
                velocity_scale, velocity_label = 1e3 * self.kms, " [$10^3$ km/s]"
                momentum_scale, momentum_label = 1, r" [$\mathrm{g}/(\mathrm{cm} \mathrm{s})$]"

                pressure_scale, pressure_label = .1, " [Pa]"
                energy_scale, energy_label = 1, " [erg]"
                energy_density_scale, energy_density_label = 1, r" [$\mathrm{erg}/\mathrm{cm}^3$]"

                bfield_scale, bfield_label = 1e-6, r" [$\mu\mathrm{G}$]"
                divergence_scale, divergence_label = 1e-6, r" [$\mu\mathrm{G}/\mathrm{cm}$]"

            else:
                if units == 'stellar':
                    # Set up physical scaling (code -> CGS)
                    L0 = self.r_sun
                    m0 = self.m_sun
                    t0 = self.sec_per_year

                    # Set up plot scaling (CGS -> plot)
                    length_scale, length_label = self.au, " [au]"
                    time_scale, time_label = self.sec_per_year, " yr"

                elif units == 'cluster':
                    L0 = self.pc
                    m0 = 10 * self.m_sun
                    t0 = self.Myr

                    length_scale, length_label = self.pc, " [pc]"
                    time_scale, time_label = self.Myr, " Myr"

                elif units == 'galactic':
                    L0 = self.kpc
                    m0 = 1e7 * self.m_sun
                    t0 = 10 * self.Myr

                    length_scale, length_label = self.kpc, " [kpc]"
                    time_scale, time_label = self.Myr, " Myr"

                mass_scale, mass_label = self.m_sun, r" [$\mathrm{M}_\odot$]"

                density_scale, density_label = 1, r" [$\mathrm{g}/\mathrm{cm}^3$]"
                velocity_scale, velocity_label = self.kms, " [km/s]"
                momentum_scale, momentum_label = 1, r" [$\mathrm{g}/(\mathrm{cm} \mathrm{s})$]"

                pressure_scale, pressure_label = .1, " [Pa]"
                energy_scale, energy_label = 1, " [erg]"
                energy_density_scale, energy_density_label = 1, r" [$\mathrm{erg}/\mathrm{cm}^3$]"

                bfield_scale, bfield_label = 1e-6, r" [$\mu\mathrm{G}$]"
                divergence_scale, divergence_label = 1e-6, r" [$\mu\mathrm{G}/\mathrm{cm}$]"

            # Compute physical scaling (CGS) for other derived quantities
            rho0 = m0/L0**3
            v0 = L0/t0
            mom0 = rho0 * v0 * L0
            P0 = rho0 * v0**2
            e0 = P0
            E0 = e0 * L0**3

            if self.mu_0 != 1:
                B0 = v0 * np.sqrt(self.mu_0*rho0)
            else:
                B0 = np.sqrt(4*np.pi*rho0 * v0**2 * L0**3)

            # Save plot scaling values and scale labels
            self.plot_scales = {
                "length":           L0 / length_scale,          # code -> cm -> au/pc/kpc
                "mass":             m0 / mass_scale,            # code -> g -> M_sun
                "time":             t0 / time_scale,            # code -> s -> s/yr/Myr
                "density":          rho0 / density_scale,       # code -> g/cm3 -> g/cm3
                "velocity":         v0 / velocity_scale,        # code -> cm/s -> km/s
                "momentum":         mom0 / momentum_scale,      # code -> g/(cm s) -> g/(cm s)
                "pressure":         P0 / pressure_scale,        # code -> dyn/cm3 -> Pa
                "energy":           E0 / energy_scale,          # code -> erg -> erg
                "energy density":   e0 / energy_density_scale,  # code -> erg/cm3 -> erg/cm3
                "Bfield":           B0 / bfield_scale,          # code -> G -> uG
                "divergence":       B0/L0 / divergence_scale,   # code -> G/cm -> uG/cm
                "Mach":             1,                          # unitless
            }

            self.scale_labels = {
                "length":           length_label,           # cm/au/pc/kpc
                "time":             time_label,             # s/yr/Myr
                "velocity":         velocity_label,         # km/s
                "mass":             mass_label,             # M_sun
                "density":          density_label,          # g/cm3
                "momentum":         momentum_label,         # g/(cm s)
                "pressure":         pressure_label,         # Pa
                "energy":           energy_label,           # erg
                "energy density":   energy_density_label,   # erg/cm3
                "Bfield":           bfield_label,           # uG
                "divergence":       divergence_label,       # uG/cm
                "Mach":             "",                     # unitless
            }