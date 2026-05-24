import numpy as np

##############################################################################
# Conversion values for code units into physical units
##############################################################################

class Constants(object):
    def __init__(self, obj, units):
        try:
            for name, value in obj.__dict__.items():
                if not name.startswith("_"):
                    setattr(self, name, value)
        except Exception:
            for name, value in obj.items():
                setattr(self, name, value)

        # Set up scaling for physical units (CGS)
        if units != "code":
            if units == 'custom':
                L0 = 1
                rho0 = 1
                v0 = 1
                length_scale = 1
                length_label = " [pc]"
                time_scale = 1
                time_label = " yr"
            elif units == 'stellar':
                L0 = self.r_sun
                rho0 = self.m_sun/self.au**3
                v0 = self.kms
                length_scale = self.au
                length_label = " [au]"
                time_scale = self.sec_per_year
                time_label = " yr"
            elif units == 'cluster':
                L0 = self.pc
                rho0 = 10 * (self.m_sun/self.pc**3)
                v0 = self.kms
                length_scale = self.pc
                length_label = " [pc]"
                time_scale = self.Myr
                time_label = " Myr"
            elif units == 'galactic':
                L0 = 1e3 * self.pc
                rho0 = 1e11 * (self.m_sun/(1e4 * self.pc**3))
                v0 = 10 * self.kms
                length_scale = 1e3 * self.pc
                length_label = " [kpc]"
                time_scale = self.Myr
                time_label = " Myr"

            m0 = rho0 * L0**3
            if self.mu_0 != 1:
                B0 = v0 * np.sqrt(self.mu_0*rho0)
            else:
                B0 = np.sqrt(4*np.pi*rho0 * v0**2 * L0**3)

            # Scale quantities to plot units
            self.plot_scales = {
                "length":           L0 / length_scale,      # code -> cm -> au/pc/kpc (length_label)
                "time":             (L0/v0) / time_scale,   # code -> s -> s/yr/Myr (time_label)
                "density":          rho0,                   # code -> g/cm3 -> g/cm3
                "velocity":         v0 * 1e-5,              # code -> cm/s -> km/s
                "mass":             m0/self.m_sun,          # code -> g -> M_sun
                "momentum":         rho0 * v0,              # code -> g/(cm2 s) -> g/(cm2 s)
                "pressure":         10 * rho0 * v0**2,      # code -> dyn/cm3 -> Pa
                "energy":           rho0 * v0**2 * L0**3,   # code -> erg -> erg
                "energy density":   rho0 * v0**2,           # code -> erg/cm3 -> erg/cm3
                "Bfield":           1e6 * B0,               # code -> G -> uG
                "divergence":       1e6 * B0/L0,            # code -> G/cm -> uG/cm
                "Mach":             1,                      # unitless
            }

            # Set plot units
            self.scale_labels = {
                "length":           length_label,                                   # cm/au/pc/kpc
                "time":             time_label,                                     # s/yr/Myr
                "density":          r" [$\mathrm{g}/\mathrm{cm}^3$]",               # g/cm3
                "velocity":         r" [$\mathrm{km}/\mathrm{s}$]",                 # km/s
                "mass":             r" [$\mathrm{M}_\odot$]",                       # M_sun
                "momentum":         r" [$\mathrm{g}/(\mathrm{cm}^2 \mathrm{s})$]",  # g/(cm2 s)
                "pressure":         r" [$\mathrm{Pa}$]",                            # Pa
                "energy":           r" [$\mathrm{erg}$]",                           # erg
                "energy density":   r" [$\mathrm{erg}/\mathrm{cm}^3$]",             # erg/cm3
                "Bfield":           r" [$\mu\mathrm{G}$]",                          # uG
                "divergence":       r" [$\mu\mathrm{G}/\mathrm{cm}$]",              # uG/cm
                "Mach":             "",                                             # unitless
            }