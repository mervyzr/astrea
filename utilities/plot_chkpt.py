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


RHO, PRESSURE, VELS, BFIELDS = 0, 4, slice(1,4), slice(5,8)
CELLS_TO_STR = lambda size: rf"$N = {str(size).strip('[]').replace(' ','').replace(',','x')}$"

SAVE_AS_PDF = False
QUIVER_ON = True
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

                        constants = Constants(units)
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
                            data = make_data(plot_options, grid, dimensions, gamma, permeability, boundary, ds, units, box_volume, slice_axis, slice_3d, plot_scales=plot_scales)
                        else:
                            fig, ax, plot_ = make_figure(plot_options, units, dimensions, coordinates)
                            data = make_data(plot_options, grid, dimensions, gamma, permeability, boundary, ds, units, box_volume, slice_axis, slice_3d)

                        def assign_plots(idx, ij):
                            _i, _j = ij
                            y = data[idx]

                            if dimensions > 1:
                                if y.ndim > 2:
                                    graph = ax[_i,_j].imshow(norm(y.T), interpolation="nearest", cmap=plot_['colours']['2d'][idx], origin="lower", extent=extent)
                                    quiver(ax[_i,_j], y, cells, dimensions, coordinates, slice_3d, slice_axis)
                                else:
                                    graph = ax[_i,_j].imshow(y, interpolation="nearest", cmap=plot_['colours']['2d'][idx], origin="lower", extent=extent)
                                divider = make_axes_locatable(ax[_i,_j])
                                cax = divider.append_axes(position='right', size='5%', pad=0.05)
                                fig.colorbar(graph, cax=cax, orientation='vertical')
                            else:
                                x = np.linspace(left, right, cells[0])
                                ax[_i,_j].plot(x, y, color=plot_['colours']['1d'][idx])

                        for idx_ij in enumerate(plot_['indexes']):
                            assign_plots(*idx_ij)

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

                        plt.cla()
                        plt.clf()
                        plt.close()
    pass





def divide(dividend, divisor):
    return np.divide(dividend, divisor, out=np.zeros_like(dividend), where=divisor!=0)

def norm(arr):
    return np.linalg.norm(arr, axis=-1)

def norm2(arr):
    return np.linalg.norm(arr, axis=-1) ** 2

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
    return grid[...,PRESSURE]/(gamma-1) + .5*(grid[...,RHO]*norm2(grid[...,VELS])) + .5*(norm2(grid[...,BFIELDS]))/permeability

# Quiver overplot for vectors
def quiver(ax, vectors, cells, dimensions, coordinates, slice_3d, slice_axis):
    phy_grid = lambda _ax: make_physical_grid(coordinates, cells, _ax)[1]
    blockview = lambda arr: blockwise_view(arr, nrows=8, ncols=8)
    if dimensions > 2:
        mesh = np.meshgrid(phy_grid(0), phy_grid(1), phy_grid(2), indexing='ij')
        X, Y = np.take(mesh, slice_3d, axis=slice_axis)
    else:
        X, Y = np.meshgrid(phy_grid(0), phy_grid(1), indexing='ij')
    ax.quiver(blockview(X), blockview(Y), blockview(vectors[0]), blockview(vectors[1]), width=.01, linewidth=.01)

def make_physical_grid(coordinates, cells, idx):
    start_pos, end_pos = coordinates[idx]
    dh = np.abs(np.diff(coordinates[idx])[0])/cells[idx]
    half_cell = .5 * dh
    return np.average(coordinates[idx]), np.linspace(start_pos-half_cell, end_pos+half_cell, cells[idx]+2)[1:-1]

# Average the submatrices of size (nrows, ncols) in a (h, w) 2D array
def blockwise_view(arr, nrows, ncols):
    h, w = arr.shape
    assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    block_grid = arr.reshape(h//nrows, nrows, -1, ncols).swapaxes(1,2).reshape(-1, nrows, ncols)
    return np.average(block_grid, axis=(1,2)).reshape(h//nrows, w//ncols)

# Make figures and axes for plotting
def make_figure(options, units, dimensions, coordinates, scale_labels=None):
    if 0 < len(options) < 15:
        # Set up colours
        colours = plt.rcParams['axes.prop_cycle'].by_key()['color'] * 2
        cmap_colours = {
            "density": "viridis",
            "pressure": "plasma",
            "magnetic pressure": "inferno",
            "total energy": "cividis",
            "internal energy": "PuBuGn",
            "velocities": "seismic",
            "momentums": "bwr",
            "Bfields": "Spectral",
            "divergence": "coolwarm",
            "Mach": "magma",
            "mass": "pink",
            "schlieren": "bone",
            "velocity": {"x":"berlin", "y":"managua", "z":"vanimo"},
            "momentum": {"x":"RdYlBu", "y":"PuOr", "z":"PRGn"},
            "Bfield": {"x":"RdBu", "y":"BrBG", "z":"PiYG"},
        }

        def make_outputs(name, symbol, unit, colour):
            assign_unit = lambda _unit: scale_labels[_unit] if units != "code" else " [arb. units]"
            return f"{name} {symbol}", rf"{symbol}{assign_unit(unit)}", colour

        # Set up labels and axes names
        def assign_plots(_option):
            _option = _option.lower()

            # Energies
            if "energy" in _option or "temp" in _option or _option.startswith("e"):
                internal = "int" in _option
                colour = cmap_colours["internal energy" if internal else "total energy"]

                prefix = "Internal" if internal else "Total"
                shortform = "int" if internal else "tot"

                name = f"{prefix} energy"
                if "density" in _option:
                    name += " density"
                    symbol = rf"$e_\mathrm{{{shortform}}}$"
                    unit = "energy density"
                else:
                    symbol = rf"$E_\mathrm{{{shortform}}}$"
                    unit = "energy"

                return make_outputs(name, symbol, unit, colour)

            # Momentums
            elif "mom" in _option:
                if _option.endswith("s"):
                    return make_outputs("Momentum", r"$\| \vec{p} \|$", "momentum", cmap_colours["momentums"])
                else:
                    return make_outputs("Momentum", rf"$p_{_option[-1]}$", "momentum", cmap_colours["momentum"][_option[-1]])

            # Mass
            elif "mass" in _option:
                return make_outputs("Mass", r"$m$", "mass", cmap_colours["mass"])

            # Mach
            elif "mach" in _option:
                return make_outputs("Mach number", r"$\mathcal{M}$", "Mach", cmap_colours["Mach"])

            # Pressure
            elif _option.startswith("p"):
                if "b" in _option:
                    return make_outputs("Mag. pressure", r"$P_B$", "pressure", cmap_colours["magnetic pressure"])
                else:
                    return make_outputs("Pressure", r"$P$", "pressure", cmap_colours["pressure"])

            # Velocities
            elif "vel" in _option:
                if _option.endswith("s"):
                    return make_outputs("Velocity", r"$\| \vec{v} \|$", "velocity", cmap_colours["velocities"])
                else:
                    return make_outputs("Velocity", rf"$v_{_option[-1]}$", "velocity", cmap_colours["velocity"][_option[-1]])

            # Magnetic field/pressure
            elif _option.startswith(("b", "mag")):
                if "p" in _option:
                    return make_outputs("Mag. pressure", r"$P_B$", "pressure", cmap_colours["magnetic pressure"])

                if _option.endswith("s"):
                    return make_outputs("Mag. field", r"$\| \vec{B} \|$", "Bfield", cmap_colours["Bfields"])
                else:
                    return make_outputs("Mag. field", rf"$B_{_option[-1]}$", "Bfield", cmap_colours["Bfield"][_option[-1]])

            # Divergence
            elif 'div' in _option or 'db' in _option:
                if _option[-1] == "b":
                    symbol = r"$\nabla \cdot B$"
                    colour = cmap_colours["divergence"]
                else:
                    symbol = rf"$\nabla \cdot B_{_option[-1]}$"
                    colour = cmap_colours["Bfields"][_option[-1]]

                return make_outputs("Divergence", symbol, "divergence", colour)

            # Density
            return make_outputs("Density", r"$\rho$", "density", cmap_colours["density"])

        outputs = [assign_plots(option) for option in options]
        names, labels, twod_colours = map(list, zip(*outputs))

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
                ax[row,col] = fig.add_subplot(spec[row, 2*col:2*(col+1)])
            else:
                extra = cols - len(options) % cols
                ax[row,col] = fig.add_subplot(spec[row, 2*col+extra:2*(col+1)+extra])

        fig.subplots_adjust(wspace=0.75, hspace=0.25)

        for idx, (_i,_j) in enumerate(indexes):
            ax[_i,_j].tick_params(axis='both', which='major')
            ax[_i,_j].tick_params(axis='both', which='minor')

            if dimensions > 1:
                ax[_i,_j].set_title(labels[idx])
            else:
                ax[_i,_j].set_ylabel(labels[idx])
                ax[_i,_j].set_xlim(coordinates[0])
                ax[_i,_j].grid(linestyle="--", linewidth=0.5)

        return fig, ax, {'indexes':indexes, 'names':names, 'labels':labels, 'colours': {'theo':'black', '1d':colours, '2d':twod_colours}}
    else:
        raise IndexError('Number of variables to plot should be < 15')



def make_data(options, grid, dimensions, gamma, permeability, boundary, ds, units, box_volume, slice_axis, slice_3d, plot_scales=None):
    get_axis = lambda option: {"x":0, "y":1, "z":2}[option[-1]]
    axes = np.array(range(dimensions))

    def option_checker(_option, _box_volume, scaling=None):
        _option = _option.lower()

        # Energies
        if "energy" in _option or "temp" in _option or _option.startswith("e"):
            scaler = 'energy'
            if "int" in _option:
                quantity = divide(grid[...,PRESSURE], grid[...,RHO] * (gamma-1))
            else:
                quantity = divide(convert_pressure(grid, gamma, permeability), grid[...,RHO])
            if "density" in _option:
                quantity *= grid[...,RHO]
                scaler += ' density'

        # Pressure
        elif _option.startswith("p"):
            scaler = 'pressure'
            if "p" in _option:
                quantity = .5 * norm2(grid[...,BFIELDS])
            else:
                quantity = grid[...,PRESSURE]

        # Velocity and momentums
        elif _option.startswith("v") or "mom" in _option:
            scaler = 'velocity'
            if _option.endswith("s"):
                quantity = grid[...,VELS]
                if "mom" in _option:
                    quantity *= grid[...,RHO][...,None]
                    scaler = 'momentum'
                if QUIVER_ON:
                    quantity = norm(quantity)
            else:
                axis = get_axis(_option)
                quantity = grid[...,1+axis]
                if "mom" in _option:
                    quantity *= grid[...,RHO]
                    scaler = 'momentum'

        # Mass
        elif "mass" in _option:
            quantity = grid[...,RHO] * _box_volume
            scaler = 'mass'

        # Bfields and magnetic pressure
        elif _option.startswith("b") or _option.startswith("mag"):
            if "p" in _option:
                quantity = .5 * norm2(grid[...,BFIELDS])
                scaler = 'pressure'
            else:
                scaler = 'Bfield'
                if _option.endswith("s"):
                    quantity = grid[...,BFIELDS]
                    if QUIVER_ON:
                        quantity = norm(quantity)
                else:
                    axis = get_axis(_option)
                    quantity = grid[...,5+axis]

        # Divergence
        elif 'div' in _option or 'db' in _option:
            div_along_axis = lambda ax: slice_(np.diff(add_boundary(grid[...,5+ax], boundary, axis=ax), axis=ax), axis=ax, end=-1)/ds[ax]
            scaler = 'divergence'
            if _option[-1] == 'b':
                quantity = sum([div_along_axis(i) for i in axes])
                #quantity = np.log10(quantity)
                #exponent = np.floor(quantity)
            else:
                quantity = div_along_axis(get_axis(_option))

        # Mach number
        elif "mach" in _option:
            quantity = np.sqrt(divide(norm2(grid[...,VELS]), divide(gamma*grid[...,PRESSURE], grid[...,RHO])))
            scaler = 'Mach'

        # Density
        else:
            quantity = grid[...,RHO]
            scaler = 'density'

        if dimensions > 2:
            quantity = np.take(quantity, slice_3d, axis=slice_axis)

        if scaling:
            return scaling[scaler] * quantity.T
        else:
            # pyplot.imshow transposes the 2d plots (might be a column-major relic)
            return quantity.T

    if units != "code":
        get_option = lambda _option, _box_volume: option_checker(_option, _box_volume, scaling=plot_scales)
    else:
        get_option = lambda _option, _box_volume: option_checker(_option, _box_volume)

    return [get_option(i, box_volume) for i in options]








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
    def __init__(self, units):
        for name, value in constant_values.items():
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
                momentum_scale, momentum_label = 1, r" [$\mathrm{g}/(\mathrm{cm}^2 \mathrm{s})$]"

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
                momentum_scale, momentum_label = 1, r" [$\mathrm{g}/(\mathrm{cm}^2 \mathrm{s})$]"

                pressure_scale, pressure_label = .1, " [Pa]"
                energy_scale, energy_label = 1, " [erg]"
                energy_density_scale, energy_density_label = 1, r" [$\mathrm{erg}/\mathrm{cm}^3$]"

                bfield_scale, bfield_label = 1e-6, r" [$\mu\mathrm{G}$]"
                divergence_scale, divergence_label = 1e-6, r" [$\mu\mathrm{G}/\mathrm{cm}$]"

            # Compute physical scaling (CGS) for other derived quantities
            rho0 = m0/L0**3
            v0 = L0/t0
            mom0 = rho0 * v0
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
                "momentum":         mom0 / momentum_scale,      # code -> g/(cm2 s) -> g/(cm2 s)
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
                "momentum":         momentum_label,         # g/(cm2 s)
                "pressure":         pressure_label,         # Pa
                "energy":           energy_label,           # erg
                "energy density":   energy_density_label,   # erg/cm3
                "Bfield":           bfield_label,           # uG
                "divergence":       divergence_label,       # uG/cm
                "Mach":             "",                     # unitless
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