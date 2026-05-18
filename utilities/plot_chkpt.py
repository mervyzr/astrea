import os
import argparse
import concurrent.futures
from itertools import repeat

import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

##############################################################################
# Functions for plotting checkpoint files
# Usage:
# ```bash
# ~$ python3 plot_chkpt.py --file=/path/to/checkpoint_file
# ```
##############################################################################


RHO, PRESSURE, VELS, BFIELDS = 0, 4, slice(1,4), slice(5,8)
CELLS_TO_STR = lambda size: rf"$N = {str(size).strip('[]').replace(' ','').replace(',','x')}$"

SAVE_AS_PDF = False
PLOT_OPTIONS = ['density', 'pressure', 'total energy', 'vx', 'vy', 'vz', 'Bx', 'By', 'Bz']


def run(save=False, title=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '--chkpt_file', dest='chkpt_file', metavar='', type=str, default=argparse.SUPPRESS, help='input path to astrea checkpoint file')
    parser.add_argument('--plot_options', metavar='', nargs="*", type=str.lower, default=argparse.SUPPRESS, help='simulation variable to plot (appendable)')
    parser.add_argument('-s', '--save', dest='save', default=argparse.SUPPRESS, help=argparse.SUPPRESS, action='store_true')
    args = parser.parse_args()

    try:
        hdf5 = args.chkpt_file
    except Exception as e:
        print(f"Error: {e}")
    else:
        try:
            plot_options = args.plot_options
        except Exception as e:
            plot_options = PLOT_OPTIONS
        finally:
            invalid = []
            try:
                if isinstance(plot_options, str):
                    plot_options = plot_options.replace(' ','').replace('-',',').replace('/',',').replace('|',',').split(',')
                for option in plot_options:
                    option = option.replace(' ','').replace('-','')
                    if option.lower() not in ACCEPTED_PLOT_OPTIONS:
                        invalid.append(option)
                        plot_options.remove(option)
                plot_options = [i.lower() for i in plot_options]
                _ = plot_options[0]
            except (IndexError, TypeError):
                print("Error with plot options")
                pass
            finally:
                if invalid != []:
                    print(f"Invalid plot options: {invalid}")


            try:
                save_plot = args.save
            except Exception as e:
                save_plot = False


            with h5py.File(hdf5, "r") as f:
                try:
                    code = f.attrs['code']
                except Exception as e:
                    print("Checkpoint file not created by astrea..")
                else:
                    if code != 'astrea':
                        print("Checkpoint file not created by astrea..")
                    else:
                        time = float(f.attrs['time'])
                        grid = f['grid'][:]

                        config = f.attrs['config']
                        cells = f.attrs['cells']
                        gamma = float(f.attrs['gamma'])
                        dimensions = int(f.attrs['dimensions'])
                        subgrid = f.attrs['subgrid']
                        time_evo = f.attrs['time_evo']
                        solver = f.attrs['solver']
                        units = f.attrs['units']
                        boundary = f.attrs['boundary']

                        coordinates = {ax:axis_coord for ax, axis_coord in enumerate(f.attrs['coordinates'])}
                        box_lengths = {ax:start_end for ax, start_end in enumerate(f.attrs['box_lengths'])}
                        ds = {ax: np.abs(np.diff(coordinates[ax]))/cells[ax] for ax in range(len(cells))}
                        box_volume = np.prod([np.diff(_) for _ in coordinates.values()])

                        constants = Constants(constant_values, units)
                        permeability = constants.mu_0

                        if units != "code":
                            plot_scales, scale_labels = constants.plot_scales, constants.scale_labels
                            length_label = scale_labels['length']
                            time_scale = plot_scales['time']
                            time_label = scale_labels['time']

                        if dimensions > 1:
                            if dimensions > 2:
                                slice_axis = 2  # z-axis
                                slice_3d = int(cells[slice_axis]/2)

                                extent = [item for key, values in box_lengths.items() if key != slice_axis for item in values]
                                x_label, y_label = [values for key, values in {0:r"$x$", 1:r"$y$", 2:r"$z$"}.items() if key != slice_axis]
                            else:
                                extent = [item for values in box_lengths.values() for item in values]
                                x_label, y_label = r"$x$", r"$y$"
                        else:
                            left, right = box_lengths[0]
                            x_label = r'$x$'

                        if units != "code":
                            fig, ax, plot_ = make_figure(plot_options, units, dimensions, coordinates, scale_labels=scale_labels)
                            data = make_data(plot_options, grid, dimensions, gamma, permeability, boundary, ds, units, box_volume, plot_scales=plot_scales)
                        else:
                            fig, ax, plot_ = make_figure(plot_options, units, dimensions, coordinates)
                            data = make_data(plot_options, grid, dimensions, gamma, permeability, boundary, ds, units, box_volume)

                        def assign_plots(idx, ij):
                            _i, _j = ij
                            y = data[idx]

                            if dimensions > 1:
                                graph = ax[_i,_j].imshow(y, interpolation="nearest", cmap=plot_['colours']['2d'][idx], origin="lower", extent=extent)
                                divider = make_axes_locatable(ax[_i,_j])
                                cax = divider.append_axes(position='right', size='5%', pad=0.05)
                                fig.colorbar(graph, cax=cax, orientation='vertical')
                            else:
                                x = np.linspace(left, right, cells[0])
                                ax[_i,_j].plot(x, y, color=plot_['colours']['1d'][idx])

                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            executor.map(assign_plots, range(len(plot_['indexes'])), plot_['indexes'])

                        if title:
                            if units != "code":
                                time *= time_scale
                                plt.suptitle(rf"Grid variables $\mathbf{{u}}$ at $t = {round(time,4)}${time_label} ({CELLS_TO_STR(cells)})")
                            else:
                                plt.suptitle(rf"Grid variables $\mathbf{{u}}$ at $t = {round(time,4)}$ ({CELLS_TO_STR(cells)})")

                        plt.tight_layout()

                        if units != "code":
                            x_label += length_label
                            y_label += length_label
                        fig.text(0.5, 0.04, x_label, ha='center')
                        fig.subplots_adjust(bottom=0.1)
                        if dimensions > 1:
                            fig.text(0.04, 0.5, y_label, ha='center')

                        if save or save_plot:
                            if SAVE_AS_PDF:
                                extension = backend = "pdf"
                            else:
                                extension, backend = "png", "cairo"
                            plt.savefig(f"{os.getcwd()}/varPlot_{dimensions}D_{config}_{subgrid}_{time_evo}_{solver}_{'%.4f' % round(time,4)}.{extension}", bbox_inches='tight', backend=backend)
                        else:
                            plt.show()

                        plt.show()

                        plt.cla()
                        plt.clf()
                        plt.close()
    pass





def divide(dividend, divisor):
    return np.divide(dividend, divisor, out=np.zeros_like(dividend), where=divisor!=0)

def norm(arr):
    return np.linalg.norm(arr, axis=-1)

def slice_(grid, axis, start=0, end=None, step=1, *args):
    slc = [slice(None)] * grid.ndim

    if args and (2 <= len(args) <= 3):
        try:
            start, end, step = args
        except ValueError:
            start, end = args

    if end == None:
        end = grid.shape[axis]

    slc[axis] = slice(start, end, step)
    return grid[tuple(slc)]

def add_boundary(grid, mode, stencil=1, axis=0):
    padding = [(0,0)] * grid.ndim
    padding[axis] = (stencil,stencil)
    return np.pad(grid, padding, mode=mode)

# pressure -> (total) energy density
def convert_pressure(grid, gamma, permeability):
    return grid[...,PRESSURE]/(gamma-1) + .5*(grid[...,RHO]*norm(grid[...,VELS])**2) + .5*(norm(grid[...,BFIELDS])**2)/permeability

# Make figures and axes for plotting
def make_figure(options, units, dimensions, coordinates, scale_labels=None):
    if 0 < len(options) < 13:
        # Set up colours
        colours = plt.rcParams['axes.prop_cycle'].by_key()['color'] * 2
        cmap_colours = {
            "density": "viridis",
            "pressure": "plasma",
            "magnetic pressure": "inferno",
            "total energy": "cividis",
            "internal energy": "PuBuGn",
            "vels": {"x":"berlin", "y":"managua", "z":"vanimo"},
            "momentums": {"x":"RdYlBu", "y":"PuOr", "z":"PRGn"},
            "Bfields": {"x":"RdBu", "y":"BrBG", "z":"PiYG"},
            "Mach": "bone",
            "divergence": "magma",
            "mass": "pink",
            "schlieren": "binary",
        }

        assign_unit = lambda _unit: scale_labels[_unit] if units != "code" else " [arb. units]"

        # Set up labels and axes names
        def assign_plots(_option):
            _option = _option.lower()

            if "energy" in _option or "temp" in _option or _option.startswith("e"):
                if "int" in _option:
                    name = "Internal energy"
                    twod_colour = cmap_colours['internal energy']
                    if "density" in _option:
                        name += ' density'
                        label = r"$e_\mathrm{int}$"
                        unit = assign_unit('energy density')
                    else:
                        label = r"$E_\mathrm{int}$"
                        unit = assign_unit('energy')
                else:
                    name = "Total energy"
                    twod_colour = cmap_colours['total energy']
                    if "density" in _option:
                        name += ' density'
                        label = r"$e_\mathrm{tot}$"
                        unit = assign_unit('energy density')
                    else:
                        label = r"$E_\mathrm{tot}$"
                        unit = assign_unit('energy')

            elif "mom" in _option:
                name = "Momentum"
                twod_colour = cmap_colours['momentums'][_option[-1]]
                label = rf"$p_{_option[-1]}$"
                unit = assign_unit('momentum')

            elif "mass" in _option:
                name = "Mass"
                twod_colour = cmap_colours['mass']
                label = r"$m$"
                unit = assign_unit('mass')

            elif "mach" in _option:
                name = "Mach number"
                twod_colour = cmap_colours['Mach']
                label = r"$\mathcal{M}$"
                unit = assign_unit('Mach')

            elif _option.startswith("p"):
                name = "Pressure"
                twod_colour = cmap_colours['pressure']
                label = r"$P$"
                unit = assign_unit('pressure')

            elif _option.startswith("v"):
                name = "Velocity"
                twod_colour = cmap_colours['vels'][_option[-1]]
                label = rf"$v_{_option[-1]}$"
                unit = assign_unit('velocity')

            elif _option.startswith("b") or _option.startswith("mag"):
                if "p" in _option:
                    name = "Mag. pressure"
                    twod_colour = cmap_colours['magnetic pressure']
                    label = r"$P_B$"
                    unit = assign_unit('pressure')
                else:
                    name = "Mag. field"
                    twod_colour = cmap_colours['Bfields'][_option[-1]]
                    label = rf"$B_{_option[-1]}$"
                    unit = assign_unit('Bfield')

            elif 'div' in _option or 'db' in _option:
                name = "divergence"
                twod_colour = cmap_colours['divergence']
                unit = assign_unit('divergence')
                if _option[-1] == 'b':
                    label = r"$\nabla \cdot B$"
                else:
                    label = rf"$\nabla \cdot B_{_option[-1]}$"

            else:
                name = "Density"
                twod_colour = cmap_colours['density']
                label = r"$\rho$"
                unit = assign_unit('density')

            return f"{name} {label}", rf"{label}{unit}", twod_colour
        
        names, labels, twod_colours = [], [], []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            jobs = executor.map(assign_plots, options)

            for (name, label, twod_colour) in jobs:
                names.append(name)
                labels.append(label)
                twod_colours.append(twod_colour)

        # Set up rows and columns
        indexes = []
        if len(options) < 4:
            rows = 1
        elif len(options) <= 8:
            rows = 2
        else:
            rows = 3
        cols = len(options)//rows + len(options)%rows
        for row in range(rows):
            for col in range(cols):
                indexes.append([row,col])
        indexes = indexes[:len(options)]

        # Set up figure
        mpl.rcParams['text.usetex'] = True
        fig, ax = plt.figure(figsize=(4*cols, 4*rows)), np.full((rows, cols), None)
        plt.rcParams['text.latex.preamble'] = r"\usepackage{lmodern}"
        params = {
            'font.size': 12,
            'font.family': 'DejaVuSans',
            'axes.labelsize': 12,
            'axes.titlesize': 12,
            'legend.fontsize': 12,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,

            'figure.dpi': 300,
            'savefig.dpi': 300,

            'lines.linewidth': 1.0,
            'lines.dashed_pattern': [3, 2]
        }
        plt.rcParams.update(params)

        spec = gridspec.GridSpec(rows, cols*2, figure=fig)

        for _i in range(len(options)):
            row, col = divmod(_i, cols)
            if row < len(options)//cols:
                if dimensions > 2:
                    ax[row,col] = fig.add_subplot(spec[row, 2*col:2*(col+1)], projection="3d")
                else:
                    ax[row,col] = fig.add_subplot(spec[row, 2*col:2*(col+1)])
            else:
                extra = cols - len(options) % cols
                if dimensions > 2:
                    ax[row,col] = fig.add_subplot(spec[row, 2*col+extra:2*(col+1)+extra], projection="3d")
                else:
                    ax[row,col] = fig.add_subplot(spec[row, 2*col+extra:2*(col+1)+extra])

        fig.subplots_adjust(wspace=0.75, hspace=0.25)

        for idx, (_i,_j) in enumerate(indexes):
            ax[_i,_j].tick_params(axis='both', which='major')
            ax[_i,_j].tick_params(axis='both', which='minor')

            if dimensions > 1:
                ax[_i,_j].set_title(labels[idx])
            else:
                ax[_i,_j].set_ylabel(labels[idx])

            if dimensions < 2:
                ax[_i,_j].set_xlim(coordinates[0])
                ax[_i,_j].grid(linestyle="--", linewidth=0.5)

        return fig, ax, {'indexes':indexes, 'names':names, 'labels':labels, 'colours': {'theo':'black', '1d':colours, '2d':twod_colours}}
    else:
        raise IndexError('Number of variables to plot should be < 13')



def make_data(options, grid, dimensions, gamma, permeability, boundary, ds, units, box_volume, plot_scales=None):
    axes = lambda op: {"x":0, "y":1, "z":2}[op[-1]]

    def option_checker(_option, _box_volume, scaling=None):
        _option = _option.lower()

        if "energy" in _option or "temp" in _option or _option.startswith("e"):
            scaler = 'energy'
            if "int" in _option:
                quantity = divide(grid[...,PRESSURE], grid[...,RHO] * (gamma-1))
            else:
                quantity = divide(convert_pressure(grid, gamma, permeability), grid[...,RHO])
            if "density" in _option:
                quantity *= grid[...,RHO]
                scaler += ' density'
        elif _option.startswith("p"):
            quantity = grid[...,PRESSURE]
            scaler = 'pressure'
        elif _option.startswith("v") or "mom" in _option:
            axis = axes(_option)
            quantity = grid[...,1+axis]
            scaler = 'velocity'
            if "mom" in _option:
                quantity *= grid[...,RHO]
                scaler = 'momentum'
        elif "mass" in _option:
            quantity = grid[...,RHO] * _box_volume
            scaler = 'mass'
        elif _option.startswith("b") or _option.startswith("mag"):
            if "p" in _option:
                quantity = .5 * norm(grid[...,BFIELDS])**2
                scaler = 'pressure'
            else:
                axis = axes(_option)
                quantity = grid[...,5+axis]
                scaler = 'Bfield'
        elif 'div' in _option or 'db' in _option:
            div_along_axis = lambda ax: slice_(np.diff(add_boundary(grid[...,5+ax], boundary, axis=ax), axis=ax), axis=ax, end=-1)/ds[ax]
            scaler = 'divergence'
            if _option[-1] == 'b':
                if dimensions > 1:
                    quantity = div_along_axis(0) + div_along_axis(1)
                    if dimensions > 2:
                        quantity += div_along_axis(2)
                    #quantity = np.log10(quantity)
                    #exponent = np.floor(quantity)
                else:
                    quantity = np.zeros_like(grid[...,5])
            else:
                quantity = div_along_axis(axes(_option))
        elif "mach" in _option:
            quantity = np.sqrt(divide(norm(grid[...,VELS])**2, divide(gamma*grid[...,PRESSURE], grid[...,RHO])))
            scaler = 'Mach'
        else:
            quantity = grid[...,RHO]
            scaler = 'density'

        if scaling:
            return scaling[scaler] * quantity.T
        else:
            # pyplot.imshow transposes the 2d plots (might be a column-major relic)
            return quantity.T

    if units != "code":
        get_option = lambda _option, _box_volume: option_checker(_option, _box_volume, scaling=plot_scales)
    else:
        get_option = lambda _option, _box_volume: option_checker(_option, _box_volume)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        jobs = executor.map(get_option, options, repeat(box_volume))

    return [job for job in jobs]








constant_values = {
    'c': 2.99792458e+10,
    'sigma': 5.670374419e-5,
    'k_B': 1.380649e-16,
    'm_p': 1.67262192e-24,
    'amu': 1.66054e-24,
    'h': 6.6260755e-27,
    'm_H': 1.00784*1.66054e-24,
    'mu': 2.381,
    'R': 8.3145e+7,
    'N_A': 6.02214076e-23,
    'arad': 4.0 * 5.670374419e-5/2.99792458e+10,
    'arad2': (6.6260755e-27*2.99792458e+10) / 1.380649e-16,
    'mu_0': 1.,
    'eps_0': 1.,
    'au': 1.49598e+13,
    'pc': 3.0856776e+18,
    'G': 6.67259e-8,
    'm_sun': 1.98892e+33,
    'r_sun': 6.9598e+10,
    'l_sun': 3.839e+33,
    'm_earth': 5.972e+27,
    'r_earth': 6.371e+8,
    'eV_to_K': 1.1604505e+9,
    'Habing': 1.6e-3,
    'sec_per_year': 3.154e+7,
    'Myr': 3.156e+13,
    'kms': 1e+5,
    'sun_earths': 332980,
}


class Constants(object):
    def __init__(self, obj, units):
        for name, value in obj.items():
            setattr(self, name, value)

        # Set up scaling for physical units (CGS)
        if units != "code":
            if units == 'custom':
                L0 = self.pc
                rho0 = self.m_sun/(self.pc**3)
                v0 = self.pc/self.sec_per_year
                length_scale = self.pc
                length_label = " [pc]"
                time_scale = self.sec_per_year
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
                "length":           L0 / length_scale,      # code -> cm -> au/pc/kpc
                "time":             (L0/v0) / time_scale,   # code -> s -> s/yr/Myr
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
                "length":           length_label,                                      # cm/au/pc/kpc
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


ACCEPTED_PLOT_OPTIONS = [
    "density",
    "rho",
    "d",
    "pressure",
    "p",
    "magneticpressure",
    "magpressure",
    "magneticp",
    "magp",
    "velocityx",
    "velocityy",
    "velocityz",
    "vx",
    "vy",
    "vz",
    "bfieldx",
    "bfieldy",
    "bfieldz",
    "bx",
    "by",
    "bz",
    "divbx",
    "divby",
    "divbz",
    "divb",
    "momentumx",
    "momentumy",
    "momentumz",
    "momx",
    "momy",
    "momz",
    "e",
    "totalenergy",
    "etotal",
    "etot",
    "totalenergydensity",
    "etotaldensity",
    "internalenergy",
    "einternal",
    "eint",
    "internalenergydensity",
    "einternaldensity",
    "temp",
    "temperature",
    "internaltemp",
    "internaltemperature",
    "mach",
    "machnumber",
]









if __name__ == "__main__":
    run()