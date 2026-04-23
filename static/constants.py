# =================================================================
# Constants and conversions
# =================================================================

#   physical:
c = 2.99792458e+10          # speed of light [cm s^-1]
sigma = 5.670374419e-5      # Stefan-Boltzmann constant [erg s^-1 cm^-2 K^-4]
k_B = 1.380649e-16          # Boltzmann's constant [erg K^-1]
m_p = 1.67262192e-24        # proton mass [g]
amu = 1.66054e-24           # atomic mass unit [g]
h = 6.6260755e-27           # Planck's constant [erg s]
m_H = 1.00784*amu           # mass of hydrogen atom [g]
mu = 2.381                  # mean molecular mass
R = 8.3145e+7               # Gas constant [erg K^-1 mol^-1]
N_A = 6.02214076e-23        # Avogadro constant [mol^-1]
arad = 4.0 * sigma/c        # radiation constant [erg cm^-3 K^-4]
arad2 = (h*c) / k_B         # second radiation constant [cm K]
mu_0 = 1.                   # vacuum magnetic permeability [g cm s^-2 A^-2]
eps_0 = 1.                  # vacuum electric permittivity [A^2 s^4 g^-1 cm^-3]

#   astronomical:
au = 1.49598e+13            # astronomical unit [cm]
pc = 3.0856776e+18          # parsec [cm]
G = 6.67259e-8              # gravitational constant [cm^3 g^-1 s^-2]
m_sun = 1.98892e+33         # solar mass [g]
r_sun = 6.9598e+10          # solar radii [cm]
l_sun = 3.839e+33           # luminosity of the Sun [erg s^-1]
m_earth = 5.972e+27         # Earth mass [g]
r_earth = 6.371e+8          # Earth radii [cm]

#   conversion factors:
eV_to_K = 1.1604505e+9      # electron volts to Kelvin
Habing = 1.6e-3             # CGS units of flux [erg cm^-3 s^-1] to Habing units
sec_per_year = 3.154e+7     # seconds to years
Myr = 3.156e+13             # seconds to Myr
kms = 1e+5                  # centimeters per second
sun_earths = 332980         # Sun in earth masses


# characteristic scales
scales = {
    'code': {
        'L': 1,
        't': 1,
        'rho': 1
    },
    'galactic': {
        'L': 1e5,  # pc
        't': 300,  # Myr
        'rho': 1e-23  # g/cm3
    },
    'cluster': {
        'L': 1,  # pc
        't': 1,  # Myr
        'rho': 1e-23  # g/cm3
    },
    'stellar': {
        'L': .1,  # pc
        't': .1,  # Myr
        'rho': 1e-17  # g/cm3
    }
}

"""x0 = pc  # cm
rho0 = m_sun/x0**3  # g/cm3
t0 = Myr  # s
v0 = (x0/t0) # cm/s
P0 = rho0 * (x0**2)/(t0**2) * .1  # Pa
B0 = v0 * sqrt(rho0 * mu_0)  # Gauss

x0 = 1  # pc
rho0 = m_sun/pc**3  # g/cm3
t0 = 1  # Myr
v0 = (pc/kms)/Myr # km/s
P0 = rho0 * (pc**2)/(Myr**2) * .1  # Pa
B0 = v0 * sqrt(rho0) * 1e5 # Gauss"""