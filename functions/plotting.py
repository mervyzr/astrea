import os
import re
import shutil
import subprocess

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable

from functions import grid as gutils
from functions import analytic
from functions import math as mfuncs
from functions.generic import BColours

##############################################################################
# Plotting functions and media handling
##############################################################################

CELLS_TO_STR = lambda size: rf"$N = {str(size).strip('[]').replace(' ','').replace(',','x')}$"


# Make figures and axes for plotting
def make_figure(options, sim_variables, variable="normal"):
    if 0 < len(options) < 15:
        # Set up colours
        try:
            plt.style.use(sim_variables.plot_style)
        except Exception as e:
            print('Unrecognised plot style')
            plt.style.use('default')

        if sim_variables.plot_style == "dark_background":
            theo_colour = "white"
        else:
            theo_colour = "black"

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
            "divergence": "coolwarm",
            "Mach": "magma",
            "mass": "pink",
            "schlieren": "bone",
        }

        def make_outputs(name, symbol, unit, colour):
            # Grab the characteristic scales and set up the values based on 'option'
            assign_unit = lambda u: sim_variables.constants.scale_labels[u] if sim_variables.units != "code" else " [arb. units]"
            return f"{name} {symbol}", rf"{symbol}{assign_unit(unit)}", rf"$\epsilon_N({symbol[1:-1]})${assign_unit(unit)}", rf"TV({symbol}){assign_unit(unit)}", colour

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
                axis = _option[-1]
                return make_outputs("Momentum", rf"$p_{axis}$", "momentum", cmap_colours["momentums"][axis])

            # Mass
            elif "mass" in _option:
                return make_outputs("Mass", r"$m$", "mass", cmap_colours["mass"])

            # Mach
            elif "mach" in _option:
                return make_outputs("Mach number", r"$\mathcal{M}$", "Mach", cmap_colours["Mach"])

            # Pressure
            elif _option.startswith("p"):
                return make_outputs("Pressure", r"$P$", "pressure", cmap_colours["pressure"])

            # Velocities
            elif _option.startswith("v"):
                axis = _option[-1]
                return make_outputs("Velocity", rf"$v_{axis}$", "velocity", cmap_colours["vels"][axis])

            # Magnetic field/pressure
            elif _option.startswith(("b", "mag")):
                if "p" in _option:
                    return make_outputs("Mag. pressure", r"$P_B$", "pressure", cmap_colours["magnetic pressure"])

                axis = _option[-1]
                return make_outputs("Mag. field", rf"$B_{axis}$", "Bfield", cmap_colours["Bfields"][axis])

            # Divergence
            elif 'div' in _option or 'db' in _option:
                axis = _option[-1]
                if axis == "b":
                    symbol = r"$\nabla \cdot B$"
                    colour = cmap_colours["divergence"]
                else:
                    symbol = rf"$\nabla \cdot B_{axis}$"
                    colour = cmap_colours["Bfields"][axis]

                return make_outputs("Divergence", symbol, "divergence", colour)

            # Density
            return make_outputs("Density", r"$\rho$", "density", cmap_colours["density"])

        outputs = [assign_plots(option) for option in options]
        names, labels, errors, tvs, twod_colours = map(list, zip(*outputs))

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
        if sim_variables.live_plot:
            fig, ax = plt.figure(figsize=[13,8]), np.full((rows, cols), None)
        else:
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

            if "error" in variable:
                ax[_i,_j].set_ylabel(errors[idx])
                ax[_i,_j].grid(linestyle="--", linewidth=0.5)
            elif "tv" in variable:
                ax[_i,_j].set_ylabel(tvs[idx])
                if not sim_variables.multidimensional:
                    ax[_i,_j].grid(linestyle="--", linewidth=0.5)
            else:
                if sim_variables.multidimensional:
                    if not sim_variables.live_plot:
                        ax[_i,_j].set_title(labels[idx])
                else:
                    if not sim_variables.live_plot:
                        ax[_i,_j].set_ylabel(labels[idx])
                    ax[_i,_j].grid(linestyle="--", linewidth=0.5)

        return fig, ax, {'indexes':indexes, 'names':names, 'labels':labels, 'errors':errors, 'tvs':tvs, 'colours': {'theo':theo_colour, '1d':colours, '2d':twod_colours}}
    else:
        raise IndexError('Number of variables to plot should be < 15')


# Create list of data plots; accepts primitive grid
def make_data(options, grid, sim_variables):
    rho, pressure, vels, Bfields = sim_variables.rho, sim_variables.pressure, sim_variables.vels, sim_variables.Bfields
    units, box_volume = sim_variables.units, sim_variables.box_volume
    axes = lambda op: {"x":0, "y":1, "z":2}[op[-1]]

    def option_checker(_option, scaling=None):
        _option = _option.lower()

        if "energy" in _option or "temp" in _option or _option.startswith("e"):
            scaler = 'energy'
            if "int" in _option:
                quantity = mfuncs.divide(grid[...,pressure], grid[...,rho] * (sim_variables.gamma-1))
            else:
                quantity = mfuncs.divide(gutils.convert_thermo_variable('pressure', grid, sim_variables), grid[...,rho])
            if "density" in _option:
                quantity *= grid[...,rho]
                scaler += ' density'
        elif _option.startswith("p"):
            quantity = grid[...,pressure]
            scaler = 'pressure'
        elif _option.startswith("v") or "mom" in _option:
            axis = axes(_option)
            quantity = grid[...,1+axis]
            scaler = 'velocity'
            if "mom" in _option:
                quantity *= grid[...,rho]
                scaler = 'momentum'
        elif "mass" in _option:
            quantity = grid[...,rho] * box_volume
            scaler = 'mass'
        elif _option.startswith("b") or _option.startswith("mag"):
            if "p" in _option:
                quantity = .5 * mfuncs.norm2(grid[...,Bfields])
                scaler = 'pressure'
            else:
                axis = axes(_option)
                quantity = grid[...,5+axis]
                scaler = 'Bfield'
        elif 'div' in _option or 'db' in _option:
            div_along_axis = lambda ax: gutils.slice_(np.diff(gutils.add_boundary(grid[...,5+ax], sim_variables, axis=ax), axis=ax), axis=ax, end=-1)/sim_variables.ds[ax]
            scaler = 'divergence'
            if _option[-1] == 'b':
                quantity = sum([div_along_axis(i) for i in sim_variables.axes])
                #quantity = np.log10(quantity)
                #exponent = np.floor(quantity)
            else:
                quantity = div_along_axis(axes(_option))
        elif "mach" in _option:
            quantity = np.sqrt(mfuncs.divide(mfuncs.norm2(grid[...,vels]), mfuncs.divide(sim_variables.gamma*grid[...,pressure], grid[...,rho])))
            scaler = 'Mach'
        else:
            quantity = grid[...,rho]
            scaler = 'density'

        if sim_variables.dimensions > 2:
            quantity = np.take(quantity, sim_variables.slice_3d, axis=sim_variables.slice_axis)

        if scaling:
            return scaling[scaler] * quantity.T
        else:
            # pyplot.imshow transposes the 2d plots (might be a column-major relic)
            return quantity.T

    if units != "code":
        get_option = lambda _option: option_checker(_option, scaling=sim_variables.constants.plot_scales)
    else:
        get_option = lambda _option: option_checker(_option)

    return [get_option(i) for i in options]


# Initiate the live plot feature
def initiate_live_plot(sim_variables, title=False):
    cells, dimensions, multidimensional = sim_variables.cells, sim_variables.dimensions, sim_variables.multidimensional
    options, units, box_lengths = sim_variables.plot_options, sim_variables.units, sim_variables.box_lengths

    plt.ion()

    fig, ax, plot_ = make_figure(options, sim_variables)

    if multidimensional:
        if dimensions > 2:
            extent = [item for key, values in box_lengths.items() if key != sim_variables.slice_axis for item in values]
        else:
            extent = [item for values in box_lengths.values() for item in values]
    else:
        left, right = box_lengths[0]

    def assign_plots(idx, ij):
        _i, _j = ij
        ax[_i,_j].set_title(plot_['names'][idx], fontsize=20)
        if multidimensional:
            # pyplot.imshow transposes the 2d plots (might be a column-major relic)
            if dimensions > 2:
                ortho_axes = np.delete(np.arange(3), sim_variables.slice_axis)
                _ = np.array(cells)[ortho_axes][::-1]
            else:
                _ = cells[::-1]
            graph = ax[_i,_j].imshow(np.zeros(_), interpolation="nearest", cmap=plot_['colours']['2d'][idx], origin="lower", extent=extent)
            divider = make_axes_locatable(ax[_i,_j])
            cax = divider.append_axes(position='right', size='5%', pad=0.05)
            fig.colorbar(graph, cax=cax, orientation='vertical')
        else:
            ax[_i,_j].set_xlim(left, right)
            ax[_i,_j].grid(linestyle='--', linewidth=0.5)
            graph, = ax[_i,_j].plot(np.linspace(left, right, cells[0]), np.linspace(left, right, cells[0]), color=plot_['colours']['1d'][idx])
        return graph

    graphs = [assign_plots(*idx_ij) for idx_ij in zip(range(len(plot_['indexes'])), plot_['indexes'])]

    if title:
        time_label = r'$t = 0.0000$'
        if units != "code":
            time_label += sim_variables.constants.scale_labels['time']
        plt.suptitle(rf"Grid variables $\mathbf{{u}}$ at {time_label}", fontsize=24)

    plt.tight_layout()

    return fig, ax, graphs


# Update live plot
def update_plot(grid_snapshot, t, sim_variables, fig, ax, graphs):
    options, units = sim_variables.plot_options, sim_variables.units
    plot_data = make_data(options, grid_snapshot, sim_variables)

    if sim_variables.multidimensional:
        for idx, graph in enumerate(graphs):
            graph.set_data(plot_data[idx])
            graph.set_clim([np.min(plot_data[idx]), np.max(plot_data[idx])])
    else:
        for idx, _ax in enumerate(ax.ravel()):
            graphs[idx].set_ydata(plot_data[idx])
            _ax.relim()
            _ax.autoscale_view()

    try:
        fig._suptitle.get_text()
    except AttributeError:
        pass
    else:
        if units != "code":
            t *= sim_variables.constants.plot_scales['time']
            time_label = sim_variables.constants.scale_labels['time']
            plt.suptitle(rf"Grid variables $\mathbf{{u}}$ at $t = {round(t,4)}${time_label}", fontsize=24)
        else:
            plt.suptitle(rf"Grid variables $\mathbf{{u}}$ at $t = {round(t,4)}$", fontsize=24)

    fig.canvas.draw()
    fig.canvas.flush_events()
    pass


# Function for plotting a snapshot of the grid
def plot_snapshot(grid_snapshot, t, sim_variables, title=False):
    config, cells, dimensions, multidimensional, subgrid, time_evo, solver = sim_variables.config, sim_variables.cells, sim_variables.dimensions, sim_variables.multidimensional, sim_variables.subgrid, sim_variables.time_evo, sim_variables.solver
    options, units, box_lengths = sim_variables.plot_options, sim_variables.units, sim_variables.box_lengths

    fig, ax, plot_ = make_figure(options, sim_variables)
    y_data = make_data(options, grid_snapshot, sim_variables)

    if sim_variables.save_as_pdf:
        extension = backend = "pdf"
    else:
        extension, backend = "png", "cairo"

    if units != "code":
        length_label = sim_variables.constants.scale_labels['length']
        time_scale = sim_variables.constants.plot_scales['time']
        time_label = sim_variables.constants.scale_labels['time']

    if multidimensional:
        if dimensions > 2:
            extent = [item for key, values in box_lengths.items() if key != sim_variables.slice_axis for item in values]
            x_label, y_label = [values for key, values in {0:r"$x$", 1:r"$y$", 2:r"$z$"}.items() if key != sim_variables.slice_axis]
        else:
            extent = [item for values in box_lengths.values() for item in values]
            x_label, y_label = r"$x$", r"$y$"
    else:
        left, right = box_lengths[0]
        x_label = r'$x$'

    def assign_plots(idx, ij):
        _i, _j = ij
        y = y_data[idx]

        if multidimensional:
            graph = ax[_i,_j].imshow(y, interpolation="nearest", cmap=plot_['colours']['2d'][idx], origin="lower", extent=extent)
            divider = make_axes_locatable(ax[_i,_j])
            cax = divider.append_axes(position='right', size='5%', pad=0.05)
            fig.colorbar(graph, cax=cax, orientation='vertical')
        else:
            x = np.linspace(left, right, cells[0])
            if sim_variables.beautify_1d_plots:
                gradient_plot([x, y], [_i,_j], ax, color=plot_['colours']['1d'][idx])
            else:
                ax[_i,_j].plot(x, y, color=plot_['colours']['1d'][idx])

    for idx_ij in enumerate(plot_['indexes']):
        assign_plots(*idx_ij)

    if title:
        if units != "code":
            t *= time_scale
            plt.suptitle(rf"Grid variables $\mathbf{{u}}$ at $t = {round(t,4)}${time_label} ({CELLS_TO_STR(cells)})")
        else:
            plt.suptitle(rf"Grid variables $\mathbf{{u}}$ at $t = {round(t,4)}$ ({CELLS_TO_STR(cells)})")

    plt.tight_layout()

    if units != "code":
        x_label += length_label
        y_label += length_label
    fig.text(0.5, 0.04, x_label, ha='center')
    fig.subplots_adjust(bottom=0.1)
    if multidimensional:
        fig.subplots_adjust(left=0.1)
        fig.text(0.04, 0.5, y_label, ha='center')

    plt.savefig(f"{sim_variables.save_path}/snapshots/varPlot_{dimensions}D_{config}_{subgrid}_{time_evo}_{solver}_{'%.4f' % round(t,4)}.{extension}", bbox_inches='tight', backend=backend)

    plt.cla()
    plt.clf()
    plt.close()
    pass


# Generic plot of simulation variables        
def plot_quantities(hdf5, sim_variables, title=False):
    config, dimensions, multidimensional, subgrid, time_evo, solver = sim_variables.config, sim_variables.dimensions, sim_variables.multidimensional, sim_variables.subgrid, sim_variables.time_evo, sim_variables.solver
    t_end, checkpoints = sim_variables.t_end, sim_variables.checkpoints
    options, units, box_lengths = sim_variables.plot_options, sim_variables.units, sim_variables.box_lengths

    if sim_variables.save_as_pdf:
        extension = backend = "pdf"
    else:
        extension, backend = "png", "cairo"

    if units != "code":
        length_label = sim_variables.constants.scale_labels['length']
        time_scale = sim_variables.constants.plot_scales['time']
        time_label = sim_variables.constants.scale_labels['time']

    if multidimensional:
        if dimensions > 2:
            extent = [item for key, values in box_lengths.items() if key != sim_variables.slice_axis for item in values]
            x_label, y_label = [values for key, values in {0:r"$x$", 1:r"$y$", 2:r"$z$"}.items() if key != sim_variables.slice_axis]
        else:
            extent = [item for values in box_lengths.values() for item in values]
            x_label, y_label = r"$x$", r"$y$"
    else:
        left, right = box_lengths[0]
        x_label = r'$x$'


    # hdf5 keys are datetime strings; each datetime represents a simulation run
    datetimes = [datetime for datetime in hdf5.keys()]
    datetimes.sort()

    # Separate the timings based on the number of checkpoints; returns a dict of arrays with the timing intervals for each group/simulation
    plot_timings = np.linspace(0, t_end, checkpoints+1)

    # Get the reference timing for analytical plots; uses the highest resolution for better accuracy
    ref_datetime = max(hdf5, key=lambda dt: np.prod(hdf5[dt].attrs['cells']))

    # The legends can be switched on for these configurations: mostly 1D analytical solutions
    if not multidimensional and ((len(hdf5) != 1) or ("sod" in config or "sedov" in config) or (sim_variables.config_category == "smooth")):
        legends_on = True
    else:
        legends_on = False

    # Assign data values to each subplot
    def assign_plots(idx, ij, _y_data, _cells):
        _i, _j = ij
        y = _y_data[idx]

        if multidimensional:
            # For single or multiple simulation runs; doesn't make sense to overplot 2D/3D runs over each other, but just do it anyway
            graph = ax[_i,_j].imshow(y, interpolation="nearest", cmap=plot_['colours']['2d'][idx], origin="lower", extent=extent)
            divider = make_axes_locatable(ax[_i,_j])
            cax = divider.append_axes(position='right', size='5%', pad=0.05)
            fig.colorbar(graph, cax=cax, orientation='vertical')
        else:
            x = np.linspace(left, right, _cells[0])
            if len(hdf5) != 1:
                # Multiple simulation runs in one HDF5 (only when --test option is used); plot all simulation runs
                ax[_i,_j].plot(x, y, label=CELLS_TO_STR(_cells))
            else:
                # Single simulation run
                if sim_variables.beautify_1d_plots:
                    gradient_plot([x, y], [_i,_j], ax, color=plot_['colours']['1d'][idx])
                else:
                    #ax[_i,_j].plot(x, y, linestyle="-", marker="D", ms=4, markerfacecolor=fig.get_facecolor(), markeredgecolor=plot_['colours']['1d'], color=plot_['colours']['1d'])
                    ax[_i,_j].plot(x, y, color=plot_['colours']['1d'][idx])

    # Iterate through the list of timings generated by the number of checkpoints
    for chkpt in range(checkpoints+1):
        fig, ax, plot_ = make_figure(options, sim_variables)

        ref_time = plot_timings[chkpt]
        # Creates one plot (with multiple subplots) for the grids at datetime
        for datetime in datetimes:
            # Get the entire simulation with multiple timesteps
            simulation = hdf5[datetime]

            # Get grid and size from a specific time (at checkpoint)
            timing = str(plot_timings[chkpt])
            grid = simulation[timing]
            cells = simulation.attrs['cells']

            y_data = make_data(options, grid, sim_variables)

            for idx, ij in enumerate(plot_['indexes']):
                assign_plots(idx, ij, y_data, cells)

            if title:
                grid_cells = "" if len(hdf5) != 1 else rf" ({CELLS_TO_STR(cells)})" 

                if units != "code":
                    t = float(timing) * time_scale
                    plt.suptitle(rf"Grid variables $\mathbf{{u}}$ at $t = {round(t,4)}${time_label}{grid_cells}")
                else:
                    plt.suptitle(rf"Grid variables $\mathbf{{u}}$ at $t = {round(float(timing),4)}${grid_cells}")

            plt.tight_layout()

            if units != "code":
                x_label += length_label
                y_label += length_label
            fig.text(0.5, 0.04, x_label, ha='center')
            fig.subplots_adjust(bottom=0.1)
            if multidimensional:
                fig.subplots_adjust(left=0.1)
                fig.text(0.04, 0.5, y_label, ha='center')


        # Add analytical solutions only for 1D, using the highest resolution/grid size
        if not multidimensional and (sim_variables.config_category == "smooth" or ("sod" in config or "sedov" in config)):
            cells = hdf5[ref_datetime].attrs['cells']
            x = np.linspace(left, right, cells[0])

            sim_variables.cells = cells
            sim_variables.ds = {ax: np.abs(np.diff(sim_variables.coordinates[ax]))/cells[ax] for ax in range(len(cells))}

            # Add analytical solution for smooth functions
            if sim_variables.config_category == "smooth":
                if "manufacture" in config or "euler" in config:
                    analytical = analytic.calculate_Euler_analytical(hdf5[ref_datetime][str(ref_time)][:], sim_variables)
                else:
                    analytical = gutils.initialise(sim_variables)
                y_theo = make_data(options, analytical, sim_variables)

                if config.startswith("sin"):
                    plot_label = rf"{config}$_{{theo}}$"
                elif config == "cpaw" or ("manufacture" in config or "euler" in config):
                    plot_label = rf"{config.upper()}$_{{theo}}$"
                else:
                    plot_label = rf"{config.title()}$_{{theo}}$"

            # Add Sod or Sedov analytical solution
            elif "sod" in config or "sedov" in config:
                _grid, _t = hdf5[ref_datetime][str(ref_time)][:], ref_time
                plot_label = rf"{config.title()}$_{{theo}}$"
                try:
                    if "sod" in config:
                        soln = analytic.calculate_Sod_analytical(_grid, _t, sim_variables)
                    elif "sedov" in config:
                        soln = analytic.calculate_Sedov_analytical(_grid, _t, sim_variables)
                except Exception as e:
                    print(f"{BColours.WARNING}Analytic plot error: {e}{BColours.ENDC}")
                    pass
                else:
                    y_theo = make_data(options, soln, sim_variables)

            for idx, (_i,_j) in enumerate(plot_['indexes']):
                ax[_i,_j].plot(x, y_theo[idx], color=plot_['colours']['theo'], linestyle="--", label=plot_label)

        if legends_on:
            def sort_key(label):
                match = re.search(r"N\s*=\s*(\d+)", label)
                return int(match.group(1)) if match else float('inf')

            if len(hdf5) > 5:
                _ncol = 2
            else:
                _ncol = 1
            handles, labels = plt.gca().get_legend_handles_labels()
            handles, labels = zip(*sorted(zip(handles, labels), key=lambda k: sort_key(k[1])))
            fig.legend(handles, labels, ncol=_ncol)

        if units != "code":
            ref_time *= time_scale
        plt.savefig(f"{sim_variables.save_path}/varPlot_{dimensions}D_{config}_{subgrid}_{time_evo}_{solver}_{'%.4f' % round(ref_time,4)}.{extension}", bbox_inches='tight', backend=backend)

        plt.cla()
        plt.clf()
        plt.close()


# Plot solution errors to determine order of convergence of numerical scheme
def plot_solution_errors(hdf5, sim_variables, error_norm=1, title=False):
    show_eoc_max = False

    options = ["density"]
    config, dimensions, subgrid, time_evo, solver = sim_variables.config, sim_variables.dimensions, sim_variables.subgrid, sim_variables.time_evo, sim_variables.solver

    if sim_variables.save_as_pdf:
        extension = backend = "pdf"
    else:
        extension, backend = "png", "cairo"

    # hdf5 keys are datetime strings
    datetimes = [datetime for datetime in hdf5.keys()]
    datetimes.sort()

    ##############################
    # Solution errors plot
    ##############################
    fig, ax, plot_ = make_figure(options, sim_variables, "errors")

    # Create array to store solution errors for each simulation
    # To prevent misallocation of grid sizes to solution errors, an extra slot is reserved for this
    simulations = len(datetimes)
    quantities = 1 + len(options)
    main_array = np.zeros((quantities, simulations), dtype=float)

    for idx, datetime in enumerate(datetimes):
        # Get the entire simulation with multiple timesteps
        simulation = hdf5[datetime]

        # Initialise temp array for this datetime; reserve first slot for grid size and the rest for sol. errors
        cells = simulation.attrs['cells']
        temp_array = [np.prod(cells)**(1/dimensions),]

        sim_variables.cells = cells
        sim_variables.ds = {ax: np.abs(np.diff(sim_variables.coordinates[ax]))/cells[ax] for ax in range(len(cells))}

        # Get last instance of the grid with final time key
        final_key = max([float(t) for t in simulation.keys()])
        solution_error = analytic.calculate_solution_error(simulation[str(final_key)], sim_variables, error_norm)

        # Append solution error to temp solution error array
        for option in options:
            option = option.lower()

            if option == "all":
                # All conserved quantities
                _error = solution_error['density'] + np.sum(solution_error['momentums']) + solution_error['Etot']
            elif "energy" in option or "temp" in option or option.startswith("e"):
                if "int" in option:
                    _error = solution_error['Eint']
                else:
                    _error = solution_error['Etot']
            elif option.startswith("p"):
                _error = solution_error['pressure']
            elif option.startswith("v") or (option.startswith("b") or "field" in option) or "mom" in option:
                axis = {'x':0, 'y':1, 'z':2}[option[-1]]
                if option.startswith("v"):
                    _error = solution_error['vels'][axis]
                elif "mom" in option:
                    _error = solution_error['momentums'][axis]
                else:
                    _error = solution_error['Bfields'][axis]
            else:
                _error = solution_error['density']

            temp_array.append(_error)

        # Append temp array to main array
        main_array[...,idx] = np.asarray(temp_array, dtype=float)

    # Get x & y data for plotting
    resolutions = main_array[0].ravel()
    solution_errors = main_array[1:]
    resolutions.sort()

    def assign_plots(idx, ij):
        _i, _j = ij
        solution_error = solution_errors[idx]

        # Theoretical convergence line plots
        for order in range(1,6):
            ytheo = solution_error[0] * (resolutions/resolutions[0])**-order
            ax[_i,_j].loglog(resolutions, ytheo, color=plot_['colours']['theo'], linestyle="--")
            ax[_i,_j].annotate(rf"$O(\Delta x^{order})$", xy=(resolutions[-1], ytheo[-1]), xytext=(5,-5), textcoords='offset points')

        ax[_i,_j].loglog(resolutions, solution_error, linestyle="-", marker="o", color=plot_['colours']['1d'][idx])
        ax[_i,_j].set_xlim([min(resolutions)/1.5,max(resolutions)*3.5])

        if show_eoc_max:
            eoc = np.diff(np.log(solution_error))/np.diff(np.log(resolutions))
            ax[_i,_j].scatter([], [], s=.5, color=fig.get_facecolor(), label=rf"$|$EOC$_{{max}}|$ = {round(max(np.abs(eoc)), 4)}")
            ax[_i,_j].legend()

    for idx_ij in enumerate(plot_['indexes']):
        assign_plots(*idx_ij)

    if title:
        if config.startswith('sin'):
            label = config
        else:
            label = config.title()
        plt.suptitle(rf"$L_{error_norm}$ error norm $\epsilon_N(\mathbf{{W}})$ against resolution $N$ for {label} test")

    plt.tight_layout()

    fig.text(0.5, 0.04, r"$N$", ha='center')
    fig.subplots_adjust(bottom=0.15)

    plt.savefig(f"{sim_variables.save_path}/solErr_{config}_L{error_norm}_{subgrid}_{time_evo}_{solver}.{extension}", bbox_inches='tight', backend=backend)

    plt.cla()
    plt.clf()
    plt.close()


    ##############################
    # Order of convergence plot
    ##############################
    fig, ax = plt.subplots()

    ax.set_ylabel("Order of convergence", rotation='vertical')
    ax.grid(linestyle="--", linewidth=0.5)

    x_diff = resolutions[1:]
    y_diff = -np.diff(np.log2(solution_errors), axis=-1)

    for idx in range(len(plot_['indexes'])):
        ax.plot(x_diff, y_diff[idx], linestyle="--", marker="o", color=plot_['colours']['1d'][idx], label=plot_['labels'][idx])

    if title:
        plt.suptitle(rf"Order of convergence against resolution $N$ for {label} test")

    plt.tight_layout()

    fig.text(0.5, 0.04, r"$N$", ha='center')
    fig.subplots_adjust(bottom=0.2)
    _xticklabels = [item.get_text() for item in ax.get_xticklabels()]
    _xticklabels = [rf"${int(v)}\rightarrow{int(resolutions[i+1])}$" for i,v in enumerate(resolutions[:-1])]
    ax.set_xticks(x_diff)
    ax.set_xticklabels(_xticklabels, rotation=45, ha="right")
    ax.legend()

    plt.savefig(f"{sim_variables.save_path}/convergenceOrder_{config}_{subgrid}_{time_evo}_{solver}.{extension}", bbox_inches='tight', backend=backend)

    plt.cla()
    plt.clf()
    plt.close()


# Total variation to determine if numerical scheme prevents oscillation
def plot_total_variation(hdf5, sim_variables, title=False):
    config, subgrid, time_evo, solver = sim_variables.config, sim_variables.subgrid, sim_variables.time_evo, sim_variables.solver
    options = sim_variables.plot_options
    time_label = sim_variables.constants.scale_labels['time']

    if sim_variables.save_as_pdf:
        extension = backend = "pdf"
    else:
        extension, backend = "png", "cairo"

    # hdf5 keys are datetime strings
    datetimes = [datetime for datetime in hdf5.keys()]
    datetimes.sort()

    fig, ax, plot_ = make_figure(options, sim_variables, "tv")

    for datetime in datetimes:
        simulation = hdf5[datetime]

        cells = simulation.attrs['cells']

        total_variations = analytic.calculate_TV(simulation, sim_variables)

        x = np.asarray(list(total_variations.keys()))
        x.sort()
        ys = np.asarray(list(total_variations.values()))

        y_data = np.zeros((len(options), len(x)), dtype=float)
        for idx, option in enumerate(options):
            option = option.lower()            
            if "energy" in option or "temp" in option or option.startswith("e"):
                y_data[idx] = ys[...,-1]
            elif option.startswith("p"):
                y_data[idx] = ys[...,4]
            elif option.startswith("v") or (option.startswith("b") or "field" in option):
                axis = {'x':0, 'y':1, 'z':2}[option[-1]]
                if option.startswith("v"):
                    y_data[idx] = ys[...,1+axis]
                else:
                    y_data[idx] = ys[...,5+axis]
            else:
                y_data[idx] = ys[...,0]

        for idx, (_i,_j) in enumerate(plot_['indexes']):
            ax[_i,_j].plot(x, y_data[idx], color=plot_['colours']['1d'][idx])
            ax[_i,_j].set_xlim([min(x), max(x)])

        if title:
            plt.suptitle(rf"Total variation of grid variables TV($\mathbf{{u}}$) against time $t$ for {config.title()} test ({CELLS_TO_STR(cells)})")

        plt.tight_layout()

        fig.text(0.5, 0.04, rf"Time $t${time_label}", ha='center')
        fig.subplots_adjust(bottom=0.1)

        plt.savefig(f"{sim_variables.save_path}/TV_{config}_{subgrid}_{time_evo}_{solver}.{extension}", bbox_inches='tight', backend=backend)

        plt.cla()
        plt.clf()
        plt.close()


# Determines if numerical scheme is conservative to machine precision
def plot_conservation_equations(hdf5, sim_variables, title=False):
    options = ["mass", "momentum_x", "total energy"]
    config, subgrid, time_evo, solver = sim_variables.config, sim_variables.subgrid, sim_variables.time_evo, sim_variables.solver
    time_label = sim_variables.constants.scale_labels['time']

    if sim_variables.save_as_pdf:
        extension = backend = "pdf"
    else:
        extension, backend = "png", "cairo"
    
    # hdf5 keys are datetime strings
    datetimes = [datetime for datetime in hdf5.keys()]
    datetimes.sort()

    fig, ax, plot_ = make_figure(options, sim_variables)

    for datetime in datetimes:
        simulation = hdf5[datetime]

        cells = simulation.attrs['cells']

        conservation: dict = analytic.calculate_conservation(simulation, sim_variables)

        x = np.asarray(list(conservation.keys()))
        x.sort()
        ys = np.asarray(list(conservation.values()))

        y_data = np.zeros((len(options), len(x)), dtype=float)
        for idx, option in enumerate(options):
            option = option.lower()
            if "energy" in option or "temp" in option:
                y_data[idx] = ys[...,4]
            elif "mom" in option or (option.startswith("b") or "field" in option):
                axis = {'x':0, 'y':1, 'z':2}[option[-1]]
                if "mom" in option:
                    y_data[idx] = ys[...,1+axis]
                else:
                    y_data[idx] = ys[...,5+axis]
            else:
                y_data[idx] = ys[...,0]

        for idx, (_i,_j) in enumerate(plot_['indexes']):
            y = y_data[idx]
            ax[_i,_j].plot(x, y_data[idx], color=plot_['colours']['1d'][idx])
            ax[_i,_j].set_xlim([min(x), max(x)])

            # For plot annotation purposes
            y_init, y_final = y[0], y[-1]
            try:
                decimal_point = int(('%e' % abs(y_final-y_init)).split('-')[1])
            except IndexError:
                decimal_point = int(('%e' % abs(y_final-y_init)).split('+')[1])
            ax[_i,_j].annotate(round(y_init, decimal_point), xy=(x[0], y_init), xytext=(0,0), textcoords='offset points')
            ax[_i,_j].annotate(round(y_final, decimal_point), xy=(x[-1], y_final), xytext=(0,0), textcoords='offset points')

        if title:
            plt.suptitle(rf"Conservation of variables ($m, p_x, E_{{tot}}$) against time $t$ for {config.title()} test ({CELLS_TO_STR(cells)})")

        plt.tight_layout()

        fig.text(0.5, 0.04, rf"Time $t${time_label}", ha='center')
        fig.subplots_adjust(bottom=0.1)

        plt.savefig(f"{sim_variables.save_path}/conserveEq_{config}_{subgrid}_{time_evo}_{solver}.{extension}", bbox_inches='tight', backend=backend)

        plt.cla()
        plt.clf()
        plt.close()


# Make a video of entire simulation; video of all plot options or specific variable
def make_video(hdf5, sim_variables, vidpath, variable="all", title=False):
    config, dimensions, multidimensional, subgrid, time_evo, solver = sim_variables.config, sim_variables.dimensions, sim_variables.multidimensional, sim_variables.subgrid, sim_variables.time_evo, sim_variables.solver
    units, box_lengths = sim_variables.units, sim_variables.box_lengths

    if units != "code":
        length_label = sim_variables.constants.scale_labels['length']
        time_scale = sim_variables.constants.plot_scales['time']
        time_label = sim_variables.constants.scale_labels['time']

    if multidimensional:
        if dimensions > 2:
            extent = [item for key, values in box_lengths.items() if key != sim_variables.slice_axis for item in values]
            x_label, y_label = [values for key, values in {0:r"$x$", 1:r"$y$", 2:r"$z$"}.items() if key != sim_variables.slice_axis]
        else:
            extent = [item for values in box_lengths.values() for item in values]
            x_label, y_label = r"$x$", r"$y$"
    else:
        left, right = box_lengths[0]
        x_label = r'$x$'


    # hdf5 keys are datetime strings
    datetimes = [datetime for datetime in hdf5.keys()]
    datetimes.sort()

    for datetime in datetimes:
        simulation = hdf5[datetime]
        cells = simulation.attrs['cells']

        if isinstance(variable, str):
            variable = variable.lower()
            counter, end_count = 0, len(simulation)

            if variable == "all":
                options = sim_variables.plot_options
            else:
                options = [variable]

            for t, grid in simulation.items():
                print(f"Creating {counter+1}/{end_count} ...", end='\r')

                fig, ax, plot_ = make_figure(options, sim_variables)
                y_data = make_data(options, grid, sim_variables)

                if variable == "all":
                    for idx, (_i,_j) in enumerate(plot_['indexes']):
                        y = y_data[idx]

                        if multidimensional:
                            graph = ax[_i,_j].imshow(y, interpolation="nearest", cmap=plot_['colours']['2d'][idx], origin="lower", extent=extent)
                            divider = make_axes_locatable(ax[_i,_j])
                            cax = divider.append_axes(position='right', size='5%', pad=0.05)
                            fig.colorbar(graph, cax=cax, orientation='vertical')
                            #graph.set_clim(0, 1)
                        else:
                            x = np.linspace(left, right, cells[0])
                            if sim_variables.beautify_1d_plots:
                                gradient_plot([x, y], [_i,_j], ax, color=plot_['colours']['1d'][idx])
                            else:
                                ax[_i,_j].plot(x, y, color=plot_['colours']['1d'][idx])

                    if title:
                        grid_cells = rf" ({CELLS_TO_STR(cells)})"

                        if units != "code":
                            timing = float(t) * time_scale
                            plt.suptitle(rf"Grid variables $\mathbf{{u}}$ at $t = {round(timing,4)}${time_label}{grid_cells}")
                        else:
                            plt.suptitle(rf"Grid variables $\mathbf{{u}}$ at $t = {round(float(t),4)}${grid_cells}")

                    plt.tight_layout()

                    if units != "code":
                        x_label += length_label
                        y_label += length_label
                    fig.text(0.5, 0.04, x_label, ha='center')
                    fig.subplots_adjust(bottom=0.1)
                    if multidimensional:
                        fig.subplots_adjust(left=0.1)
                        fig.text(0.04, 0.5, y_label, ha='center')

                    plt.savefig(f"{vidpath}/{str(counter).zfill(5)}.png", bbox_inches='tight', backend='cairo')

                else:
                    idx = 0

                    if multidimensional:
                        plt.axis('off')
                        graph = ax[idx,idx].imshow(y_data[idx], interpolation="nearest", cmap=plot_['colours']['2d'][idx], origin="lower", extent=extent)
                        #graph.set_clim(0, 1)
                    else:
                        x = np.linspace(left, right, cells[0])
                        ax[idx,idx].plot(x, y_data[idx], color=plot_['colours']['1d'][idx])

                    ax[idx,idx].set_title('')

                    plt.savefig(f"{vidpath}/{str(counter).zfill(5)}.png", bbox_inches='tight', pad_inches=0, backend='cairo')

                plt.cla()
                plt.clf()
                plt.close()

                counter += 1

            try:
                print(f"                                                                                ", end='\r')
                print(f"Creating video ... [{variable}]", end='\r')
                subprocess.call(["ffmpeg", "-hide_banner", "-loglevel", "error", "-framerate", "60", "-pattern_type", "glob", "-i", f"{vidpath}/*.png", "-c:v", "libx264", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", f"{sim_variables.save_path}/vid_{config}_{subgrid}_{time_evo}_{solver}_{variable}.mp4"])
            except Exception as e:
                print(f"{BColours.FAIL}Video creation failed{BColours.ENDC}")
                pass

        elif isinstance(variable, list) and all(isinstance(_, str) for _ in variable):
            variables = [_.lower() for _ in variable]
            style_counter = 0

            for _variable in variables:
                counter, end_count = 0, len(simulation)

                for _, grid in simulation.items():
                    print(f"                                                                                ", end='\r')
                    print(f"Creating {counter+1}/{end_count} ... [{_variable}]", end='\r')

                    fig, ax, plot_ = make_figure([_variable], sim_variables)
                    y_data = make_data([_variable], grid, sim_variables)

                    idx = 0

                    if multidimensional:
                        plt.axis('off')
                        graph = ax[idx,idx].imshow(y_data[idx], interpolation="nearest", cmap=plot_['colours']['2d'][0], origin="lower", extent=extent)
                        #graph.set_clim(0, 1)
                    else:
                        x = np.linspace(left, right, cells[0])
                        ax[idx,idx].plot(x, y_data[idx], color=plot_['colours']['1d'][style_counter])

                    ax[idx,idx].set_title('')

                    plt.savefig(f"{vidpath}/{str(counter).zfill(5)}.png", bbox_inches='tight', pad_inches=0, backend='cairo')

                    plt.cla()
                    plt.clf()
                    plt.close()

                    counter += 1

                style_counter += 1

                try:
                    print(f"                                                                                ", end='\r')
                    print(f"Creating video ... [{_variable}]", end='\r')
                    subprocess.call(["ffmpeg", "-hide_banner", "-loglevel", "error", "-framerate", "60", "-pattern_type", "glob", "-i", f"{vidpath}/*.png", "-c:v", "libx264", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", f"{sim_variables.save_path}/vid_{config}_{subgrid}_{time_evo}_{solver}_{_variable}.mp4"])
                except Exception as e:
                    print(f"{BColours.FAIL}Video creation failed: {e}{BColours.ENDC}")
                    pass
                else:
                    for filename in os.listdir(vidpath):
                        filepath = os.path.join(vidpath, filename)
                        if os.path.isfile(filepath) or os.path.islink(filepath):
                            os.remove(filepath)
        shutil.rmtree(vidpath)


# Function for plotting instance of the grid; insert into any part of the code
def plot_this(grid, sim_variables, **kwargs):
    cells, dimensions, multidimensional = sim_variables.cells, sim_variables.dimensions, sim_variables.multidimensional
    options, units, box_lengths = sim_variables.plot_options, sim_variables.units, sim_variables.box_lengths

    if units != "code":
        length_label = sim_variables.constants.scale_labels['length']
        time_scale = sim_variables.constants.plot_scales['time']
        time_label = sim_variables.constants.scale_labels['time']

    if multidimensional:
        if dimensions > 2:
            extent = [item for key, values in box_lengths.items() if key != sim_variables.slice_axis for item in values]
            x_label, y_label = [values for key, values in {0:r"$x$", 1:r"$y$", 2:r"$z$"}.items() if key != sim_variables.slice_axis]
        else:
            extent = [item for values in box_lengths.values() for item in values]
            x_label, y_label = r"$x$", r"$y$"
    else:
        left, right = box_lengths[0]
        x_label = r'$x$'

    try:
        t = kwargs['t']
    except KeyError:
        try:
            text = kwargs['text']
        except KeyError:
            text = ""
    else:
        if units != "code":
            t *= time_scale
            text = rf"at $t = {t}${time_label}"
        else:
            text = rf"at $t = {t}$"

    fig, ax, plot_ = make_figure(options, sim_variables)
    y_data = make_data(options, grid, sim_variables)

    def assign_plots(idx, ij):
        _i, _j = ij
        y = y_data[idx]

        if multidimensional:
            graph = ax[_i,_j].imshow(y, interpolation="nearest", cmap=plot_['colours']['2d'][idx], origin="lower", extent=extent)
            divider = make_axes_locatable(ax[_i,_j])
            cax = divider.append_axes(position='right', size='5%', pad=0.05)
            fig.colorbar(graph, cax=cax, orientation='vertical')
        else:
            x = np.linspace(left, right, cells[0])
            if sim_variables.beautify_1d_plots:
                gradient_plot([x, y], [_i,_j], ax, color=plot_['colours']['1d'][idx])
            else:
                ax[_i,_j].plot(x, y, color=plot_['colours']['1d'][idx])

    for idx_ij in enumerate(plot_['indexes']):
        assign_plots(*idx_ij)

    plt.suptitle(rf"Grid variables $\mathbf{{u}}$ {text}")
    plt.tight_layout()

    if units != "code":
        x_label += length_label
        y_label += length_label
    fig.text(0.5, 0.04, x_label, ha='center')
    fig.subplots_adjust(bottom=0.1)
    if multidimensional:
        fig.subplots_adjust(left=0.1)
        fig.text(0.04, 0.5, y_label, ha='center')

    if not sim_variables.live_plot:
        plt.show(block=True)
    pass


# Plot the power spectrum for turbulence
def plot_turbulence_spectrum(hdf5, sim_variables, bins=8, normalise=True, t=None):
    cells, dimensions, coordinates, ds = sim_variables.cells, sim_variables.dimensions, sim_variables.coordinates, sim_variables.ds
    units = sim_variables.units

    if sim_variables.save_as_pdf:
        extension = backend = "pdf"
    else:
        extension, backend = "png", "cairo"

    # hdf5 keys are datetime strings
    datetimes = [datetime for datetime in hdf5.keys()]
    datetimes.sort()

    mpl.rcParams['text.usetex'] = True
    fig, ax = plt.subplots()
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
    ax.grid(linestyle="--", linewidth=0.5)

    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$P_\mathrm{kin}(k)$")

    for datetime in datetimes:
        simulation = hdf5[datetime]

        if not t:
            t = list(simulation.keys())[-1]
        grid = simulation[str(t)][:]

        density = grid[...,sim_variables.rho]
        vels = grid[...,sim_variables.vels]

        # Kinetic energy in real space
        E_kin = .5 * density * mfuncs.norm2(vels)

        fft_field = np.fft.fft2(E_kin)  # Fourier transform the energy field
        fft_field = np.fft.fftshift(fft_field)  # Shift zero frequency to the center
        power = np.abs(fft_field)**2  # Compute power spectrum (P(k_x, k_y, k_z))

        if normalise:
            power /= np.prod(cells)

        # Compute the wavenumber components
        ks, power_law = {}, -5/3
        for axis in range(dimensions):
            ks[axis] = 2 * np.pi * np.fft.fftfreq(cells[axis], d=ds[axis])
        if len(ks) == 1:
            kx = ks[0]
            k = np.fft.fftshift(kx)
            C_k = .5
        elif len(ks) == 2:
            kx, ky = list(map(ks.get, range(2)))
            KX, KY = np.meshgrid(np.fft.fftshift(kx), np.fft.fftshift(ky), indexing='ij')
            k = np.sqrt(KX**2 + KY**2)
            C_k = .8
        elif len(ks) == 3:
            kx, ky, kz = list(map(ks.get, range(3)))
            KX, KY, KZ = np.meshgrid(np.fft.fftshift(kx), np.fft.fftshift(ky), np.fft.fftshift(kz), indexing='ij')
            k = np.sqrt(KX**2 + KY**2 + KZ**2)
            C_k = 1.5

        # Bin the energy spectrum
        # Define bins
        k_bins = np.linspace(0, k.max(), np.mean(cells)//bins)
        k_bin_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
        power_spectrum = np.zeros_like(k_bin_centers)

        for i in range(len(k_bins) - 1):
            bin_mask = (k >= k_bins[i]) & (k < k_bins[i + 1])
            power_spectrum[i] = power[bin_mask].sum()  # Sum power in each bin

        # Normalize the power spectrum by area and wavenumber bin width
        if normalise:
            bin_widths = np.diff(k_bins)
            power_spectrum /= bin_widths * np.prod(np.diff(list(coordinates.values()), axis=1))

        # Compute the theoretical values (with fitting based on a sliced window of the power spectrum, not the whole spectrum)
        m, c = np.polyfit(np.log10(k_bin_centers[3:10]), np.log10(power_spectrum[3:10]), 1)
        E_theo = (k_bin_centers**power_law) * (10**C_k)
        log_offset = power_spectrum[0] - E_theo[0]

        # Plot the energy spectrum
        if units != "code":
            t *= sim_variables.constants.plot_scales['time']
            time_label = sim_variables.constants.scale_labels['time'].strip(' []')
            label = rf"$t = {round(t,4)}{time_label}$, m = {round(m,3)}$"
        else:
            label = rf"$t = {round(t,4)}$, m = {round(m,3)}$"
        ax.loglog(k_bin_centers, power_spectrum, label=label)

    # Plot the theoretical line
    ax.loglog(k_bin_centers[1:22], (E_theo*log_offset)[1:22], color='black', linestyle='--', label=rf'$k^{{{power_law}}}$')
    ax.legend()

    plt.tight_layout()

    plt.savefig(f"{sim_variables.save_path}/e_spectrum_{t}.{extension}", bbox_inches='tight', backend=backend)

    plt.cla()
    plt.clf()
    plt.close()


# Plot positions of tracer particles
def plot_tracer_particles(tracers, t, sim_variables, title=False):
    dimensions, multidimensional = sim_variables.dimensions, sim_variables.multidimensional
    units, box_lengths = sim_variables.units, sim_variables.box_lengths

    if sim_variables.save_as_pdf:
        extension = backend = "pdf"
    else:
        extension, backend = "png", "cairo"

    try:
        plt.style.use(sim_variables.plot_style)
    except Exception:
        print('Unrecognised plot style')
        plt.style.use('default')

    mpl.rcParams['text.usetex'] = True
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
    fig = plt.figure()

    ax = fig.add_subplot()

    fig_kwargs = {
        's': max(1, int(500/np.average(sim_variables.cells))),
        'color': 'green',
        'marker': 'o',
        'alpha': max(.2, 1.1 - .1*np.log2(np.average(sim_variables.cells))),
        'linewidths': max(.2, 1.1 - .1*np.log2(np.average(sim_variables.cells))),
    }

    if multidimensional:
        if dimensions > 2:
            (left, right), (bottom, top), (backwards, forward) = box_lengths.values()
        else:
            (left, right), (bottom, top) = box_lengths.values()
    else:
        [(left, right)] = box_lengths.values()

    x_label, y_label, z_label = r'$x$', r'$y$', r'$z$'
    if units != "code":
        x_label += sim_variables.constants.scale_labels['length']
        y_label += sim_variables.constants.scale_labels['length']
        z_label += sim_variables.constants.scale_labels['length']

    ax.set_xlabel(x_label)
    ax.set_xlim(left, right)

    if multidimensional:
        ax.set_ylabel(y_label)
        ax.set_ylim(bottom, top)

        if dimensions > 2:
            ax.set_zlabel(z_label)
            ax.set_zlim(backwards, forward)
            ax.scatter(tracers[...,0], tracers[...,1], tracers[...,2], **fig_kwargs)
        else:
            ax.scatter(tracers[...,0], tracers[...,1], **fig_kwargs)
    else:
        ax.scatter(tracers[...,0], **fig_kwargs)

    if title:
        if units != "code":
            t *= sim_variables.constants.plot_scales['time']
            plt.suptitle(rf"Tracer particles' positions at $t = {round(t,4)}${sim_variables.constants.scale_labels['time']}")
        else:
            plt.suptitle(rf"Tracer particles' positions at $t = {round(t,4)}$")

    plt.savefig(f"{sim_variables.save_path}/snapshots/tracers_{'%.3f' % round(t,4)}.{extension}", bbox_inches='tight', backend=backend)

    plt.cla()
    plt.clf()
    plt.close()


# Gradient fill the plots
def gradient_plot(data, plot_index, ax, **kwargs):
    x, y = data
    i, j = plot_index

    line, = ax[i,j].plot(x, y, **kwargs)
    fill_color = line.get_color()

    zorder = line.get_zorder()
    alpha = line.get_alpha()
    alpha = 1.0 if alpha is None else alpha

    z = np.empty((100, 1, 4), dtype=float)
    rgb = mcolors.colorConverter.to_rgb(fill_color)
    z[:,:,:3] = rgb
    z[:,:,-1] = np.linspace(0, alpha, 100)[:,None]

    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    im = ax[i,j].imshow(z, aspect='auto', extent=[xmin, xmax, ymin, ymax], origin='lower', zorder=zorder)

    xy = np.column_stack([x, y])
    xy = np.vstack([[xmin, ymin], xy, [xmax, ymin], [xmin, ymin]])
    clip_path = Polygon(xy, facecolor='none', edgecolor='none', closed=True)
    ax[i,j].add_patch(clip_path)
    im.set_clip_path(clip_path)

    pass


def plot_3d(y_data, ax, plot_, box_lengths):
    (left, right), (bottom, top), (backwards, forward) = box_lengths.values()

    def assign_plots(idx, ij):
        _i, _j = ij
        y = y_data[idx]

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

        x_label, y_label, z_label = r'$x$', r'$y$', r'$z$'
        ax[_i,_j].set_xlabel(x_label)
        ax[_i,_j].set_ylabel(y_label)
        ax[_i,_j].set_zlabel(z_label)
        ax[_i,_j].set_box_aspect(aspect=None, zoom=0.8)

    for idx_ij in enumerate(plot_['indexes']):
        assign_plots(*idx_ij)

    return None


# Schlieren (gradient) plots; works best with 3D data
def schlieren(quantity, scale=[1,1000], normalise=True):
    # Compute gradients
    gradients = np.gradient(quantity)

    # Compute schlieren intensity (gradient magnitude)
    gradient_mag = np.linalg.norm(gradients, axis=0)

    # Log scaling to enhance contrast
    schlieren_log = np.log1p(gradient_mag)

    # Normalise the scales to scale_min and scale_max
    if normalise:
        scale_min, scale_max = scale
        I_min, I_max = schlieren_log.min(), schlieren_log.max()
        schlieren_log = scale_min + (scale_max-1) * (schlieren_log-I_min)/(I_max-I_min)

        # Alternative, normalise to scale_max
        #schlieren_log /= I_max

    return schlieren_log