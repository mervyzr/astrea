import os
import argparse
import concurrent.futures

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

SAVE_AS_PDF = False

RHO, PRESSURE, VELS, BFIELDS = 0, 4, slice(1,4), slice(5,8)


def run(save=False, title=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '--chkpt_file', dest='chkpt_file', metavar='', type=str, default=argparse.SUPPRESS, help='input path to astrea checkpoint file')
    parser.add_argument('--plot_options', metavar='', type=str, default=argparse.SUPPRESS, help='simulation variables to plot')
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
            plot_options = ['density', 'pressure', 'total energy', 'vx', 'vy', 'vz', 'Bx', 'By', 'Bz']
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
                        aspect_ratio = f.attrs['aspect_ratio']

                        coordinates = {axis:coord for axis, coord in enumerate(f.attrs['coordinates'])}
                        ds = {ax: np.abs(np.diff(coordinates[ax]))/cells[ax] for ax in range(len(cells))}

                        constants = Constants(constant_values, units)
                        permeability = constants.mu_0
                        plot_scales = constants.plot_scales

                        box_label = plot_scales["box_label"]
                        if units != "code":
                            half_box = plot_scales['length']/2
                            scaled_half_box = half_box/plot_scales['box_scale']
                            box_lengths = {ax: [-ratio*scaled_half_box, ratio*scaled_half_box] for ax, ratio in enumerate(aspect_ratio)}
                        else:
                            box_lengths = coordinates

                        if dimensions > 1:
                            if dimensions > 2:
                                (left, right), (bottom, top), (backwards, forward) = box_lengths.values()
                            else:
                                (left, right), (bottom, top) = box_lengths.values()
                        else:
                            [(left, right)] = box_lengths.values()

                        time_scale = plot_scales['time']/plot_scales['time_scale']
                        time_label = plot_scales['time_label']

                        fig, ax, plot_ = make_figure(plot_options, units, dimensions, coordinates)
                        data = make_data(plot_options, grid, dimensions, gamma, permeability, boundary, ds, constants)

                        def assign_plots(idx, ij):
                            _i, _j = ij
                            y = data[idx]

                            if dimensions > 1:
                                if dimensions > 2:
                                    X, Y, Z = np.meshgrid(
                                        np.linspace(left, right, y.shape[0]), 
                                        np.linspace(bottom, top, y.shape[1]), 
                                        np.linspace(backwards, forward, y.shape[2])
                                        )

                                    plot_3d = np.full_like(y, np.nan)
                                    values, counts = np.unique(y.ravel(), return_counts=True)
                                    background = values[counts.argmax()]
                                    plot_3d[y > background] = y[y > background]

                                    ax[_i,_j].scatter3D(X, Y, Z, c=plot_3d, alpha=.05, marker='.', linewidth=0, cmap=plot_['colours']['2d'][idx])

                                    ax[_i,_j].set_xlabel(f'$x${box_label}')
                                    ax[_i,_j].set_ylabel(f'$y${box_label}')
                                    ax[_i,_j].set_zlabel(f'$z${box_label}')
                                    ax[_i,_j].set_box_aspect(aspect=None, zoom=0.8)
                                else:
                                    graph = ax[_i,_j].imshow(y, interpolation="nearest", cmap=plot_['colours']['2d'][idx], origin="lower", extent=[left,right,bottom,top])
                                    divider = make_axes_locatable(ax[_i,_j])
                                    cax = divider.append_axes(position='right', size='5%', pad=0.05)
                                    fig.colorbar(graph, cax=cax, orientation='vertical')
                            else:
                                x = np.linspace(left, right, cells[0])
                                ax[_i,_j].plot(x, y, color=plot_['colours']['1d'][idx])

                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            executor.map(assign_plots, range(len(plot_['indexes'])), plot_['indexes'])

                        if title:
                            if dimensions > 2:
                                grid_axes = "$(x,y,z)$"
                            elif dimensions > 1:
                                grid_axes = "$(x,y)$"
                            else:
                                grid_axes = "$x$"
                            plt.suptitle(rf"Grid variables $\mathbf{{u}}$ against cell indices {grid_axes} at $t = {round(time*time_scale,3)}${time_label} ($N = {str(cells).strip('[]').replace(' ','').replace(',','x')}$)")

                        plt.tight_layout()

                        if dimensions < 3:
                            fig.text(0.5, 0.04, f'$x${box_label}', ha='center')
                            fig.subplots_adjust(bottom=0.1)
                            if dimensions > 1:
                                fig.text(0.04, 0.5, f'$y${box_label}', ha='center', rotation='vertical')
                                fig.subplots_adjust(left=0.1)

                        if save or save_plot:
                            if SAVE_AS_PDF:
                                extension = backend = "pdf"
                            else:
                                extension, backend = "png", "cairo"
                            plt.savefig(f"{os.getcwd()}/varPlot_{dimensions}D_{config}_{subgrid}_{time_evo}_{solver}_{'%.4f' % round(time*time_scale,4)}.{extension}", bbox_inches='tight', backend=backend)
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
def make_figure(options, units, dimensions, coordinates):
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
        scale_labels = {
            "density":          r" [$\mathrm{g}/\mathrm{cm}^3$]",               # g/cm3
            "velocity":         r" [$\mathrm{km}/\mathrm{s}$]",                 # cm/s
            "mass":             r" [$\mathrm{M}_\odot$]",                       # M_sun
            "momentum":         r" [$\mathrm{g}/(\mathrm{cm}^2 \mathrm{s})$]",  # g/(cm2 s)
            "pressure":         r" [$\mathrm{dyn}/\mathrm{cm}^3$]",             # dyn/cm3
            "energy":           r" [$\mathrm{erg}$]",                           # erg
            "energy density":   r" [$\mathrm{erg}/\mathrm{cm}^3$]",             # erg/cm3
            "Bfield":           r" [$\mu\mathrm{G}$]",                          # uG
            "divergence":       r" [$\mu\mathrm{G}/\mathrm{cm}$]",              # uG/cm
            "Mach":             "",                                             # unitless
        }

        assign_unit = lambda _unit: scale_labels[_unit] if units != "code" else " [arb. units]"

        # Set up labels and axes names
        def assign_plots(_option):
            _option = _option.lower()

            if "energy" in _option or "temp" in _option or _option.startswith("e"):
                unit = assign_unit('energy')
                if "int" in _option:
                    name = "Internal energy"
                    twod_colour = cmap_colours['internal energy']
                    if "density" in _option:
                        name += ' density'
                        label = r"$e_\mathrm{int}$"
                        unit = assign_unit('energy density')
                    else:
                        label = r"$E_\mathrm{int}$"
                else:
                    name = "Total energy"
                    twod_colour = cmap_colours['total energy']
                    if "density" in _option:
                        name += ' density'
                        label = r"$e_\mathrm{tot}$"
                        unit = assign_unit('energy density')
                    else:
                        label = r"$E_\mathrm{tot}$"

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


# Create list of data plots; accepts primitive grid
def make_data(options, grid, dimensions, gamma, permeability, boundary, ds, constants):
    axes = lambda op: {"x":0, "y":1, "z":2}[op[-1]]
    plot_scales = constants.plot_scales

    def option_checker(_option):
        _option = _option.lower()

        if "energy" in _option or "temp" in _option or _option.startswith("e"):
            if "int" in _option:
                quantity = plot_scales['energy'] * divide(grid[...,PRESSURE], grid[...,RHO] * (gamma-1))
            else:
                quantity = plot_scales['energy'] * divide(convert_pressure(grid, gamma, permeability), grid[...,RHO])
            if "density" in _option:
                quantity *= grid[...,RHO]/(plot_scales['length']**3)
        elif _option.startswith("p"):
            quantity = plot_scales['pressure'] * grid[...,PRESSURE]
        elif _option.startswith("v") or "mom" in _option:
            axis = axes(_option)
            quantity = plot_scales['velocity']/constants.kms * grid[...,1+axis]
            if "mom" in _option:
                quantity *= (plot_scales['density']*constants.kms) * grid[...,RHO]
        elif _option.startswith("b") or _option.startswith("mag"):
            if "p" in _option:
                quantity = plot_scales['pressure'] * .5 * norm(grid[...,BFIELDS])**2
            else:
                axis = axes(_option)
                quantity = plot_scales['Bfield'] * grid[...,5+axis]
        elif 'div' in _option or 'db' in _option:
            div_along_axis = lambda ax: slice_(np.diff(add_boundary(grid[...,5+ax], boundary, axis=ax), axis=ax), axis=ax, end=-1)/ds[ax]
            if _option[-1] == 'b':
                if dimensions > 1:
                    quantity = div_along_axis(0) + div_along_axis(1)
                    if dimensions > 2:
                        quantity += div_along_axis(2)
                else:
                    quantity = np.zeros_like(grid[...,5])
            else:
                quantity = div_along_axis(axes(_option))
            quantity *= plot_scales['divergence']
        elif "mach" in _option:
            quantity = np.sqrt(divide(norm(grid[...,VELS])**2, divide(gamma*grid[...,PRESSURE], grid[...,RHO])))
        else:
            quantity = plot_scales['density'] * grid[...,RHO]

        # pyplot.imshow transposes the 2d plots (might be a column-major relic)
        return quantity.T

    with concurrent.futures.ThreadPoolExecutor() as executor:
        jobs = executor.map(option_checker, options)

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
    def __init__(self, obj, unit):
        for name, value in obj.items():
            setattr(self, name, value)

        # Set up scaling (CGS)
        if unit == 'code':
            L0 = 1
            rho0 = 1
            v0 = 1
            m0 = 1
            B0 = 1
            box_scale = 1
            time_scale = 1
            box_label = ""
            time_label = ""
        else:
            if unit == 'stellar':
                L0 = self.r_sun
                rho0 = 1.5
                v0 = 10 * self.kms
                box_scale = self.au  # normalise box size to this scale
                box_label = " [au]"
                time_scale = self.sec_per_year
                time_label = " yr"
            elif unit == 'cluster':
                L0 = 5e4 * self.au
                rho0 = 1e-19
                v0 = self.kms
                box_scale = self.pc
                box_label = " [pc]"
                time_scale = self.Myr
                time_label = " Myr"
            elif unit == 'galactic':
                L0 = 1e3 * self.pc
                rho0 = 1e-24
                v0 = 100 * self.kms
                box_scale = 1e3 * self.pc
                box_label = " [kpc]"
                time_scale = self.Myr
                time_label = " Myr"

            m0 = 1/self.m_sun
            if self.mu_0 != 1:
                B0 = v0 * np.sqrt(self.mu_0*rho0) * 1e6
            else:
                B0 = np.sqrt(4*np.pi*rho0 * v0**2 * L0**3) * 1e6

        self.plot_scales = {
            "length":       L0,                     # cm
            "density":      rho0,                   # g/cm3
            "velocity":     v0,                     # cm/s
            "mass":         m0,                     # M_sun
            "time":         L0/v0,                  # s
            "momentum":     rho0 * v0,              # g/(cm2 s)
            "pressure":     rho0 * v0**2,           # erg/cm3
            "energy":       rho0 * v0**2 * L0**3,   # erg
            "Bfield":       B0,                     # uG
            "divergence":   B0/L0,                  # uG/cm
            "Mach":         1,                      # unitless
            "box_scale":    box_scale,
            "box_label":    box_label,
            "time_scale":   time_scale,
            "time_label":   time_label,
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