import os
import argparse

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


def plot(save=False, title=False):
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

                        _const = {}
                        for name, value in f.attrs.items():
                            if name.startswith('constants-'):
                                _const[name.split('constants-')[-1]] = value
                        constants = Constants(_const)

                        coordinates = {axis:coord for axis, coord in enumerate(f.attrs['axis_coord'])}
                        permeability = constants.mu_0
                        ds = {ax: np.abs(np.diff(coordinates[ax]))/cells[ax] for ax in range(len(cells))}

                        fig, ax, plot_ = make_figure(plot_options, dimensions, coordinates)
                        data = make_data(plot_options, grid, dimensions, gamma, permeability, ds)

                        for idx, (_i,_j) in enumerate(plot_['indexes']):
                            y = data[idx]

                            if dimensions > 1:
                                if dimensions > 2:
                                    X, Y, Z = np.meshgrid(
                                        np.linspace(coordinates[0][0], coordinates[0][1], y.shape[0]), 
                                        np.linspace(coordinates[1][0], coordinates[1][1], y.shape[1]), 
                                        np.linspace(coordinates[2][0], coordinates[2][1], y.shape[2])
                                        )

                                    plot_3d = np.full_like(y, np.nan)
                                    values, counts = np.unique(y.ravel(), return_counts=True)
                                    background = values[counts.argmax()]
                                    plot_3d[y > background] = y[y > background]

                                    ax[_i,_j].scatter3D(X, Y, Z, c=plot_3d, alpha=.05, marker='.', linewidth=0, cmap=plot_['colours']['2d'][idx])

                                    ax[_i,_j].set_xlabel('$x$')
                                    ax[_i,_j].set_ylabel('$y$')
                                    ax[_i,_j].set_zlabel('$z$')
                                    ax[_i,_j].set_box_aspect(aspect=None, zoom=0.8)
                                else:
                                    graph = ax[_i,_j].imshow(y, interpolation="nearest", cmap=plot_['colours']['2d'][idx], origin="lower")
                                    divider = make_axes_locatable(ax[_i,_j])
                                    cax = divider.append_axes(position='right', size='5%', pad=0.05)
                                    fig.colorbar(graph, cax=cax, orientation='vertical')
                            else:
                                x = np.linspace(coordinates[0][0], coordinates[0][1], cells[0])
                                ax[_i,_j].plot(x, y, color=plot_['colours']['1d'][idx])

                        if title:
                            if dimensions > 2:
                                grid_axes = "$(x,y,z)$"
                            elif dimensions > 1:
                                grid_axes = "$(x,y)$"
                            else:
                                grid_axes = "$x$"
                            plt.suptitle(rf"Grid variables $\mathbf{{u}}$ against cell indices {grid_axes} at $t = {round(time,3)}$ ($N = {str(cells).strip('[]').replace(' ','').replace(',','x')}$)")

                        plt.tight_layout()

                        if dimensions < 3:
                            fig.text(0.5, 0.04, r"$x$", ha='center')
                            fig.subplots_adjust(bottom=0.1)
                            if dimensions > 1:
                                fig.text(0.04, 0.5, r"$y$", ha='center', rotation='vertical')
                                fig.subplots_adjust(left=0.1)

                        if save or save_plot:
                            if SAVE_AS_PDF:
                                extension = backend = "pdf"
                            else:
                                extension, backend = "png", "cairo"
                            plt.savefig(f"{os.getcwd()}/varPlot_{dimensions}D_{config}_{subgrid}_{time_evo}_{solver}_{'%.3f' % round(time,3)}.{extension}", bbox_inches='tight', backend=backend)
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
            try:
                start, end = args
                step = 1
            except ValueError:
                start, end, step = 0, grid.shape[axis], 1

    if not end:
        end = grid.shape[axis]

    slc[axis] = slice(start, end, step)
    return grid[tuple(slc)]

def add_boundary(grid, stencil=1, axis=0):
    arr = np.copy(grid)
    padding = [(0,0)] * grid.ndim
    padding[axis] = (stencil,stencil)
    return np.pad(arr, padding, mode='wrap')

def convert_variable(grid, gamma, permeability):
    rho, pressure, vels, Bfields = 0, 4, slice(1,4), slice(5,8)
    return grid[...,pressure]/(gamma-1) + .5 * (grid[...,rho]*norm(grid[...,vels])**2 + (norm(grid[...,Bfields])**2)/permeability)

class Constants(object):
    def __init__(self, obj):
        try:
            for name, value in obj.__dict__.items():
                if not name.startswith("_"):
                    setattr(self, name, value)
        except Exception:
            for name, value in obj.items():
                setattr(self, name, value)

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






# Make figures and axes for plotting
def make_figure(options, dimensions, axis_coord):
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
            "divergence": "binary",
            "mass": "pink",
        }

        # Set up labels and axes names
        names, labels, twod_colours = [], [], []
        for option in options:
            option = option.lower()

            if "energy" in option or "temp" in option or option.startswith("e"):
                if "int" in option:
                    name = "Internal energy"
                    twod_colour = cmap_colours['internal energy']
                    if "density" in option:
                        name += ' density'
                        label = r"$e_\mathrm{int}$"
                    else:
                        label = r"$E_\mathrm{int}$"
                else:
                    name = "Total energy"
                    twod_colour = cmap_colours['total energy']
                    if "density" in option:
                        name += ' density'
                        label = r"$e_\mathrm{tot}$"
                    else:
                        label = r"$E_\mathrm{tot}$"

            elif "mom" in option:
                name = "Momentum"
                twod_colour = cmap_colours['momentums'][option[-1]]
                label = rf"$p_{option[-1]}$"

            elif "mass" in option:
                name = "Mass"
                twod_colour = cmap_colours['mass']
                label = r"$m$"

            elif "mach" in option:
                name = "Mach number"
                twod_colour = cmap_colours['Mach']
                label = r"$\mathcal{M}$"

            elif option.startswith("p"):
                name = "Pressure"
                twod_colour = cmap_colours['pressure']
                label = r"$P$"

            elif option.startswith("v"):
                name = "Velocity"
                twod_colour = cmap_colours['vels'][option[-1]]
                label = rf"$v_{option[-1]}$"

            elif option.startswith("b") or option.startswith("mag"):
                if "p" in option:
                    name = "Mag. pressure"
                    twod_colour = cmap_colours['magnetic pressure']
                    label = r"$P_B$"
                else:
                    name = "Mag. field"
                    twod_colour = cmap_colours['Bfields'][option[-1]]
                    label = rf"$B_{option[-1]}$"

            elif 'div' in option or 'db' in option:
                name = "divergence"
                twod_colour = cmap_colours['divergence']
                if option[-1] == 'b':
                    label = r"$\nabla \cdot B$"
                else:
                    label = rf"$\nabla \cdot B_{option[-1]}$"

            else:
                name = "Density"
                twod_colour = cmap_colours['density']
                label = r"$\rho$"

            names.append(f"{name} {label}")
            labels.append(rf"{label} [arb. units]")
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
                ax[_i,_j].set_xlim(axis_coord[0])
                ax[_i,_j].grid(linestyle="--", linewidth=0.5)

        return fig, ax, {'indexes':indexes, 'names':names, 'labels':labels, 'colours': {'theo':'black', '1d':colours, '2d':twod_colours}}
    else:
        raise IndexError('Number of variables to plot should be < 13')


# Create list of data plots; accepts primitive grid
def make_data(options, grid, dimensions, gamma, permeability, ds):
    rho, pressure, vels, Bfields = 0, 4, slice(1,4), slice(5,8)
    axes = lambda op: {"x":0, "y":1, "z":2}[op[-1]]
    quantities = []

    for option in options:
        option = option.lower()

        if "energy" in option or "temp" in option or option.startswith("e"):
            if "int" in option:
                quantity = divide(grid[...,pressure], grid[...,rho] * (gamma-1))
            else:
                quantity = divide(convert_variable(grid, gamma, permeability), grid[...,rho])
            if "density" in option:
                quantity *= grid[...,rho]
        elif option.startswith("p"):
            quantity = grid[...,pressure]
        elif option.startswith("v") or "mom" in option:
            axis = axes(option)
            quantity = grid[...,1+axis]
            if "mom" in option:
                quantity *= grid[...,rho]
        elif option.startswith("b") or option.startswith("mag"):
            if "p" in option:
                quantity = .5 * norm(grid[...,Bfields])**2
            else:
                axis = axes(option)
                quantity = grid[...,5+axis]
        elif 'div' in option or 'db' in option:
            div_along_axis = lambda ax: slice_(np.diff(add_boundary(grid[...,5+ax], axis=ax), axis=ax), axis=ax, end=-1)/ds[ax]
            if option[-1] == 'b':
                if dimensions > 1:
                    quantity = div_along_axis(0) + div_along_axis(1)
                    if dimensions > 2:
                        quantity += div_along_axis(2)
                else:
                    quantity = np.zeros_like(grid[...,5])
            else:
                quantity = div_along_axis(axes(option))
        elif "mach" in option:
            quantity = np.sqrt(divide(norm(grid[...,vels])**2, divide(gamma*grid[...,pressure], grid[...,rho])))
        else:
            quantity = grid[...,rho]

        # pyplot.imshow transposes the 2d plots (might be a column-major relic)
        quantities.append(quantity.T)
    return quantities


if __name__ == "__main__":
    plot()