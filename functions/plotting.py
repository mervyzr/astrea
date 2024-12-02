import shutil
import subprocess

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable

from functions import analytic, constructor, fv, generic

##############################################################################
# Plotting functions and media handling
##############################################################################

STYLE = "default"
BEAUTIFY = False


PLOT_OPTIONS = ["DENSITY", "PRESSURE", "VX", "ENERGY"]
PLOT_INDEXES = [[0,0], [0,1], [1,0], [1,1]]
PLOT_LABELS = [[r"Density $\rho$", r"Pressure $P$"], [r"Velocity $v_x$", r"Specific internal energy $e$"]]

try:
    plt.style.use(STYLE)
except Exception as e:
    plt.style.use("default")
    COLOURS = [["blue", "red"], ["green", "darkviolet"]]
else:
    if STYLE != "default":
        _color = plt.rcParams['axes.prop_cycle'].by_key()['color']
        COLOURS = [_color[:2], _color[2:4]]
    else:
        COLOURS = [["blue", "red"], ["green", "darkviolet"]]
    if STYLE == "dark_background":
        THEO_COLOUR = "white"
    else:
        THEO_COLOUR = "black"
finally:
    TWOD_COLOURS = [["viridis", "hot"], ["cividis", "plasma"]]






# Make figures and axes for plotting
def make_figure(options, sim_variables, variable="normal", style=STYLE):
    if 0 < len(options) < 11:
        # Set up colours
        try:
            plt.style.use(style)
        except Exception as e:
            plt.style.use('default')
        else:
            if style == "dark_background":
                theo_colour = "white"
            else:
                theo_colour = "black"
        finally:
            colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
            twod_colours = ["viridis", "hot", "cividis", "plasma", "inferno", "ocean", "terrain", "cubehelix", "magma", "gist_earth"]

        # Set up labels and axes names
        labels, errors, tvs = [], [], []
        for option in options:
            option = option.lower()

            if "energy" in option or "temp" in option:
                if "internal" in option:
                    name = "Internal energy"
                    label = r"$e$"
                    error = r"$\log{(\epsilon_\nu(e))}$"
                    tv = r"TV($e$)"
                else:
                    name = "Total energy"
                    label = r"$E$"
                    error = r"$\log{(\epsilon_\nu(E))}$"
                    tv = r"TV($E$)"
                if "density" in option:
                    name += ' density'

            elif "mom" in option:
                name = "Momentum"
                label = r"$p_x$"
                error = r"$\log{(\epsilon_\nu(p_x))}$"
                tv = r"TV($p_x$)"

            elif "mass" in option:
                name = "Mass"
                label = r"$m$"
                error = r"$\log{(\epsilon_\nu(m))}$"
                tv = r"TV($m$)"

            elif "pres" in option:
                name = "Pressure"
                label = r"$P$"
                error = r"$\log{(\epsilon_\nu(P))}$"
                tv = r"TV($P$)"

            elif option.startswith("v"):
                name = "Velocity"
                label = rf"$v_{option[-1]}$"
                error = rf"$\log{{(\epsilon_\nu(v_{option[-1]}))}}$"
                tv = rf"TV($v_{option[-1]}$)"

            elif option.startswith("b"):
                name = "Mag. field"
                label = rf"$B_{option[-1]}$"
                error = rf"$\log{{(\epsilon_\nu(B_{option[-1]}))}}$"
                tv = rf"TV($B_{option[-1]}$)"

            else:
                name = "Density"
                label = r"$\rho$"
                error = r"$\log{(\epsilon_\nu(\rho))}$"
                tv = r"TV($\rho$)"

            labels.append(rf"{name.capitalize()} {label}")
            errors.append(rf"{name.capitalize()} {error}")
            tvs.append(rf"{name.capitalize()} {tv}")

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
        fig, ax = plt.figure(figsize=[21,10]), np.full((rows, cols), None)
        spec = gridspec.GridSpec(rows, cols*2, figure=fig)
        for _i in range(len(options)):
            row, col = divmod(_i, cols)
            if row < len(options)//cols:
                ax[row,col] = fig.add_subplot(spec[row, 2*col:2*(col+1)])
            else:
                extra = cols - len(options) % cols
                ax[row,col] = fig.add_subplot(spec[row, 2*col+extra:2*(col+1)+extra])
        fig.subplots_adjust(wspace=0.75)

        for idx, (_i,_j) in enumerate(indexes):
            if "error" in variable:
                ax[_i,_j].set_ylabel(errors[idx], fontsize=18)
            elif "tv" in variable:
                ax[_i,_j].set_ylabel(tvs[idx], fontsize=18)
            else:
                ax[_i,_j].set_ylabel(labels[idx], fontsize=18)

            if sim_variables.dimension < 2:
                ax[_i,_j].set_xlim([sim_variables.start_pos, sim_variables.end_pos])
                ax[_i,_j].grid(linestyle="--", linewidth=0.5)

        return fig, ax, {'indexes':indexes, 'labels':labels, 'errors':errors, 'tvs':tvs, 'colours': {'theo':theo_colour, '1d':colours, '2d':twod_colours}}
    else:
        raise IndexError('Plot options should be < 11')


# Create list of data plots; accepts primitive grid
def make_data(options, grid, sim_variables):
    axes = {"x":0, "y":1, "z":2}
    quantities = []

    for option in options:
        option = option.lower()

        if "energy" in option or "temp" in option:
            if "internal" in option:
                quantity = fv.divide(grid[...,4], grid[...,0] * (sim_variables.gamma-1))
            else:
                quantity = fv.divide(fv.convert_variable("pressure", grid, sim_variables.gamma), grid[...,0])
            if "density" in option:
                quantity *= grid[...,0]
        elif "mom" in option:
            quantity = grid[...,1+axes[option[-1]]] * grid[...,0]
        elif option.startswith("p"):
            quantity = grid[...,4]
        elif option.startswith("v"):
            quantity = grid[...,1+axes[option[-1]]]
        elif option.startswith("b") or "magnetic" in option:
            quantity = grid[...,5+axes[option[-1]]]
        else:
            quantity = grid[...,0]

        quantities.append(quantity)

    return quantities


# Initiate the live plot feature
def initiate_live_plot(sim_variables):
    N, dimension, start_pos, end_pos = sim_variables.cells, sim_variables.dimension, sim_variables.start_pos, sim_variables.end_pos
    plt.ion()

    fig, ax, plot_ = make_figure(PLOT_OPTIONS, sim_variables)

    graphs = []
    for idx, (_i,_j) in enumerate(plot_['indexes']):
        if dimension == 2:
            fig.text(0.5, 0.04, r"Cell index $x$", ha='center', fontsize=18)
            fig.text(0.04, 0.4, r"Cell index $y$", ha='center', fontsize=18, rotation='vertical')
            graph = ax[_i,_j].imshow(np.zeros((N,N)), interpolation="nearest", cmap=plot_['colours']['2d'][idx], origin="lower")
            divider = make_axes_locatable(ax[_i,_j])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(graph, cax=cax, orientation='vertical')
        else:
            fig.text(0.5, 0.04, r"Cell position $x$", ha='center', fontsize=18)
            ax[_i,_j].set_xlim([start_pos, end_pos])
            ax[_i,_j].grid(linestyle='--', linewidth=0.5)
            graph, = ax[_i,_j].plot(np.linspace(start_pos, end_pos, N), np.linspace(start_pos, end_pos, N), linewidth=2, color=plot_['colours']['1d'][idx])
        graphs.append(graph)
    return fig, ax, graphs


# Update live plot
def update_plot(grid_snapshot, t, sim_variables, fig, ax, graphs):
    plot_data = make_data(PLOT_OPTIONS, grid_snapshot, sim_variables)

    if sim_variables.dimension == 2:
        for index, graph in enumerate(graphs):
            graph.set_data(plot_data[index])
            graph.set_clim([np.min(plot_data[index]), np.max(plot_data[index])])

        plt.suptitle(rf"Primitive variables $\vec{{w}}$ against cell indices $x$ & $y$ at $t = {round(t,4)}$", fontsize=24)
    else:
        for index, graph in enumerate(graphs):
            graph.set_ydata(plot_data[index])
            #graphBR.set_ydata(analytic.calculateEntropyDensity(grid_snapshot, 1.4))  # scaled entropy density

        for _ in ax.ravel()[:len(PLOT_OPTIONS)]:
            _.relim()
            _.autoscale_view()

        plt.suptitle(rf"Primitive variables $\vec{{w}}$ against cell position $x$ at $t = {round(t,4)}$", fontsize=24)

    fig.canvas.draw()
    fig.canvas.flush_events()
    pass


# Function for plotting a snapshot of the grid
def plot_snapshot(grid_snapshot, t, sim_variables, **kwargs):
    config, N, dimension, subgrid, timestep, scheme = sim_variables.config, sim_variables.cells, sim_variables.dimension, sim_variables.subgrid, sim_variables.timestep, sim_variables.scheme
    start_pos, end_pos = sim_variables.start_pos, sim_variables.end_pos

    try:
        text = kwargs["text"]
    except KeyError:
        text = ""

    fig, ax, plot_ = make_figure(PLOT_OPTIONS, sim_variables)
    y_data = make_data(PLOT_OPTIONS, grid_snapshot, sim_variables)

    for idx, (_i,_j) in enumerate(plot_['indexes']):
        y = y_data[idx]

        if dimension == 2:
            graph = ax[_i,_j].imshow(y, interpolation="nearest", cmap=plot_['colours']['2d'][idx], origin="lower")
            divider = make_axes_locatable(ax[_i,_j])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(graph, cax=cax, orientation='vertical')
        else:
            x = np.linspace(start_pos, end_pos, N)
            if BEAUTIFY:
                gradient_plot([x, y], [_i,_j], ax, linewidth=2, color=plot_['colours']['1d'][idx])
            else:
                ax[_i,_j].plot(x, y, linewidth=2, color=plot_['colours']['1d'][idx])

    if dimension == 2:
        plt.suptitle(rf"Primitive variables $\vec{{w}}$ against cell indices $x$ & $y$ at $t \approx {round(t,3)}$ ($N = {N}^{dimension}$) {text}", fontsize=24)
        fig.text(0.5, 0.04, r"Cell index $x$", fontsize=18, ha='center')
        fig.text(0.04, 0.4, r"Cell index $y$", fontsize=18, ha='center', rotation='vertical')
    else:
        plt.suptitle(rf"Primitive variables $\vec{{w}}$ against cell position $x$ {text}", fontsize=24)
        fig.text(0.5, 0.04, r"Cell position $x$", fontsize=18, ha='center')

    plt.savefig(f"{kwargs['save_path']}/varPlot_{dimension}D_{config}_{subgrid}_{timestep}_{scheme}_{'%.3f' % round(t,3)}.png", dpi=330)

    plt.cla()
    plt.clf()
    plt.close()
    pass


# Plot snapshots of quantities for multiple runs
def plot_quantities(hdf5, sim_variables, save_path):
    config, dimension, subgrid, timestep = sim_variables.config, sim_variables.dimension, sim_variables.subgrid, sim_variables.timestep
    scheme, precision, snapshots = sim_variables.scheme, sim_variables.precision, sim_variables.snapshots
    start_pos, end_pos = sim_variables.start_pos, sim_variables.end_pos

    # hdf5 keys are datetime strings
    datetimes = sorted(hdf5, key=hdf5.get)

    # Separate the timings based on the number of snapshots; returns a dict of lists with the timing intervals for each N
    plot_timings_for_each_grp = {}
    for datetime in datetimes:
        all_timings = np.fromiter(hdf5[datetime].keys(), dtype=precision)
        all_timings.sort()
        plot_timings_for_each_grp[datetime] = [timing[-1] for timing in np.array_split(all_timings, abs(snapshots))]

    # Get the reference timing for plots; uses the highest resolution for better accuracy
    ref_N = 0
    for datetime, grp in hdf5.items():
        ref_datetime = datetime if grp.attrs['cells'] > ref_N else ref_datetime
        ref_N = grp.attrs['cells'] if grp.attrs['cells'] > ref_N else ref_N
    ref_timings = plot_timings_for_each_grp[ref_datetime]

    # Iterate through the list of timings generated by the number of snapshots
    for snap_index in range(snapshots):
        fig, ax, plot_ = make_figure(PLOT_OPTIONS, sim_variables)

        # Plot each simulation at the specific timing
        ref_time = ref_timings[snap_index]
        for datetime in datetimes:
            simulation = hdf5[datetime]
            N = simulation.attrs['cells']
            timing = str(plot_timings_for_each_grp[datetime][snap_index])

            x = np.linspace(start_pos, end_pos, N)
            y_data = make_data(PLOT_OPTIONS, simulation[timing], sim_variables)

            # density, pressure, vx, specific internal energy
            for idx, (_i,_j) in enumerate(plot_['indexes']):
                y = y_data[idx]

                if len(hdf5) != 1:
                    if dimension < 2:
                        ax[_i,_j].plot(x, y, linewidth=2, label=f"N = {N}")
                        plt.suptitle(rf"Primitive variables $\vec{{w}}$ against cell position $x$ at $t \approx {round(ref_time,3)}$", fontsize=24)
                else:
                    if dimension == 2:
                        graph = ax[_i,_j].imshow(y, interpolation="nearest", cmap=plot_['colours']['2d'][idx], origin="lower")
                        divider = make_axes_locatable(ax[_i,_j])
                        cax = divider.append_axes('right', size='5%', pad=0.05)
                        fig.colorbar(graph, cax=cax, orientation='vertical')
                        plt.suptitle(rf"Primitive variables $\vec{{w}}$ against cell indices $x$ & $y$ at $t \approx {round(ref_time,3)}$ ($N = {N}^{dimension}$)", fontsize=24)
                        fig.text(0.5, 0.04, r"Cell index $x$", fontsize=18, ha='center')
                        fig.text(0.04, 0.4, r"Cell index $y$", fontsize=18, ha='center', rotation='vertical')
                    else:
                        if BEAUTIFY:
                            gradient_plot([x, y], [_i,_j], ax, linewidth=2, color=plot_['colours']['1d'][idx])
                        else:
                            #ax[_i,_j].plot(x, y, linewidth=2, linestyle="-", marker="D", ms=4, markerfacecolor=fig.get_facecolor(), markeredgecolor=plot_['colours']['1d'], color=plot_['colours']['1d'])
                            ax[_i,_j].plot(x, y, linewidth=2, color=plot_['colours']['1d'][idx])
                        plt.suptitle(rf"Primitive variables $\vec{{w}}$ against cell position $x$ at $t \approx {round(ref_time,3)}$ ($N = {N}$)", fontsize=24)

        # Add analytical solutions only for 1D
        if dimension < 2:
            # Add analytical solution for smooth functions, using the highest resolution and timing
            if sim_variables.config_category == "smooth":
                analytical = constructor.initialise(sim_variables)

                y_theo = make_data(PLOT_OPTIONS, analytical, sim_variables)
                for idx, (_i,_j) in enumerate(plot_['indexes']):
                    ax[_i,_j].plot(x, y_theo[idx], linewidth=2, color=plot_['colours']['theo'], linestyle="--", label=rf"{config.title()}$_{{theo}}$")

            # Add Sod or Sedov analytical solution, using the highest resolution and timing
            elif "sod" in config or "sedov" in config:
                _grid, _t = hdf5[ref_datetime][str(ref_time)], ref_time
                try:
                    if "sod" in config:
                        soln = analytic.calculate_Sod_analytical(_grid, _t, sim_variables)
                        plot_label = r"Sod$_{theo}$"
                    elif "sedov" in config:
                        soln = analytic.calculate_Sedov_analytical(_grid, _t, sim_variables)
                        plot_label = r"Sedov$_{theo}$"
                except Exception as e:
                    print(f"Analytic error: {e}")
                    pass
                else:
                    y_theo = make_data(PLOT_OPTIONS, soln, sim_variables)
                    for idx, (_i,_j) in enumerate(plot_['indexes']):
                        ax[_i,_j].plot(x, y_theo[idx], linewidth=2, color=plot_['colours']['theo'], linestyle="--", label=plot_label)

            fig.text(0.5, 0.04, r"Cell position $x$", fontsize=18, ha='center')
            if len(hdf5) != 1 or "sod" in config or "sedov" in config or sim_variables.config_category == "smooth":
                if len(hdf5) > 5:
                    _ncol = 2
                else:
                    _ncol = 1
                handles, labels = plt.gca().get_legend_handles_labels()
                fig.legend(handles, labels, prop={'size': 16}, loc='upper right', ncol=_ncol)

        plt.savefig(f"{save_path}/varPlot_{dimension}D_{config}_{subgrid}_{timestep}_{scheme}_{'%.3f' % round(ref_time,3)}.png", dpi=330)

        plt.cla()
        plt.clf()
        plt.close()


def plot_solution_errors(hdf5, sim_variables, save_path, norm=1):
    config, dimension, subgrid, timestep, scheme = sim_variables.config, sim_variables.dimension, sim_variables.subgrid, sim_variables.timestep, sim_variables.scheme

    # hdf5 keys are datetime strings
    datetimes = sorted(hdf5, key=hdf5.get)

    # Solution errors plot
    fig, ax, plot_ = make_figure(["density", "pressure", "vx", "internal energy"], sim_variables, "errors")
    #fig, ax = plt.subplots(nrows=1, ncols=2, figsize=[21,10])
    #error_labels = [[r"Density $\log{(\epsilon_\nu(\rho))}$", r"Pressure $\log{(\epsilon_\nu(P))}$"], [r"Velocity $\log{(\epsilon_\nu(v_x))}$", r"Specific internal energy $\log{(\epsilon_\nu(e))}$"]]

    array = np.full((len(PLOT_OPTIONS)+1, 1), None)
    x, y1, y2, y3, y4 = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    for datetime in datetimes:
        # Get last instance of the grid with largest time key
        time_key = max([float(t) for t in hdf5[datetime].keys()])
        solution_errors = analytic.calculate_solution_error(hdf5[datetime][str(time_key)], sim_variables, norm)

        for option in PLOT_OPTIONS:
            if 


def make_data(options, grid, sim_variables):
    axes = {"x":0, "y":1, "z":2}
    quantities = []

    for option in options:
        option = option.lower()

        if "energy" in option or "temp" in option:
            if "internal" in option:
                quantity = fv.divide(grid[...,4], grid[...,0] * (sim_variables.gamma-1))
            else:
                quantity = fv.divide(fv.convert_variable("pressure", grid, sim_variables.gamma), grid[...,0])
            if "density" in option:
                quantity *= grid[...,0]
        elif "mom" in option:
            quantity = grid[...,1+axes[option[-1]]] * grid[...,0]
        elif option.startswith("p"):
            quantity = grid[...,4]
        elif option.startswith("v"):
            quantity = grid[...,1+axes[option[-1]]]
        elif option.startswith("b") or "magnetic" in option:
            quantity = grid[...,5+axes[option[-1]]]
        else:
            quantity = grid[...,0]

        quantities.append(quantity)

    return quantities

        x = np.append(x, hdf5[datetime].attrs['cells']**dimension)
        y1 = np.append(y1, solution_errors[0])  # density
        y2 = np.append(y2, solution_errors[4])  # pressure
        y3 = np.append(y3, solution_errors[1])  # vx
        y4 = np.append(y4, solution_errors[-1])  # specific internal energy
    y_data = [y1, y2, y3, y4]

    for _i, _j in PLOT_INDEXES:
        if _i == _j:
            ax[_i].set_ylabel(error_labels[_i][_j], fontsize=18)
            ax[_i].grid(linestyle="--", linewidth=0.5)

            EOC = np.diff(np.log(y_data[_i][_j]))/np.diff(np.log(x))
            idx = np.argmin(np.abs(np.average(EOC)-EOC))
            c = np.log10(y_data[_i][_j][idx]) - EOC[idx]*np.log10(x[idx])

            for order in [1,2,4,5]:
                alpha = 10**c
                ytheo = alpha*x**(-order)
                ax[_j].loglog(x, ytheo, linewidth=2, color=THEO_COLOUR, linestyle="--")
                ax[_j].annotate(rf"$O(N^{order})$", (x[-1], ytheo[-1]), fontsize=12)
            ax[_j].loglog(x, y_data[_i][_j], linewidth=2, linestyle="--", marker="o", color=COLOURS[_i][_j])
            ax[_j].scatter([], [], s=.5, color=fig.get_facecolor(), label=rf"$|\text{{EOC}}_{{max}}|$ = {round(max(np.abs(EOC)), 4)}")
            ax[_j].legend(prop={'size': 14})

    plt.suptitle(rf"$L_{norm}$ solution error norm $\epsilon_\nu(\vec{{w}})$ against resolution $N_\nu$ for {config.title()} test", fontsize=24)
    fig.text(0.5, 0.04, r"Resolution $\log{(N_\nu)}$", fontsize=18, ha='center')

    plt.savefig(f"{save_path}/solErr_L{norm}_{subgrid}_{timestep}_{scheme}.png", dpi=330)

    plt.cla()
    plt.clf()
    plt.close()

    # Order of convergence plot
    fig, ax = plt.subplots(figsize=[21,10])

    ax.set_ylabel("Order of convergence", fontsize=18)
    ax.grid(linestyle="--", linewidth=0.5)

    x_diff = x[1:]
    y_diff = np.array([[np.log2(y1[:-1]/y1[1:]), np.log2(y2[:-1]/y2[1:])], [np.log2(y3[:-1]/y3[1:]), np.log2(y4[:-1]/y4[1:])]])

    if dimension == 2:
        y_diff /= np.log2(4)

    for _i, _j in PLOT_INDEXES:
        if _i == _j:
            ax.plot(x_diff, y_diff[_i][_j], linewidth=2, linestyle="--", marker="o", color=COLOURS[_i][_j], label=PLOT_LABELS[_i][_j])

    plt.suptitle(rf"Order of convergence against resolution $N_\nu$ for {config.title()} test", fontsize=24)
    fig.text(0.5, 0.04, r"Resolution $N$", fontsize=18, ha='center')
    _xticklabels = [item.get_text() for item in ax.get_xticklabels()]
    _xticklabels = [rf"${int(v)}\rightarrow{int(x[i+1])}$" for i,v in enumerate(x[:-1])]
    ax.set_xticks(x_diff)
    ax.set_xticklabels(_xticklabels, rotation=45, fontsize=12, ha="right")
    ax.legend(prop={'size': 18})

    plt.savefig(f"{save_path}/convergenceOrder_{subgrid}_{timestep}_{scheme}.png", dpi=330)

    plt.cla()
    plt.clf()
    plt.close()


def plot_total_variation(hdf5, sim_variables, save_path):
    config, dimension, subgrid, timestep, scheme = sim_variables.config, sim_variables.dimension, sim_variables.subgrid, sim_variables.timestep, sim_variables.scheme

    # hdf5 keys are datetime strings
    datetimes = sorted(hdf5, key=hdf5.get)

    fig, ax, plot_ = make_figure(PLOT_OPTIONS, sim_variables, "tv")

    for datetime in datetimes:
        N = hdf5[datetime].attrs['cells']
        tv_dict = analytic.calculate_tv(hdf5[datetime], sim_variables)

        x = np.asarray(list(tv_dict.keys()))
        x.sort()

        y = np.asarray(list(tv_dict.values()))
        y1 = y[...,0]  # density
        y2 = y[...,4]  # pressure
        y3 = y[...,1]  # vx
        y4 = y[...,-1]  # specific internal energy
        y_data = [y1, y2, y3, y4]

        for idx, (_i,_j) in enumerate(plot_['indexes']):
            ax[_i,_j].plot(x, y_data[idx], linewidth=2, color=plot_['colours']['1d'][idx])

        if dimension >= 2:
            grid_size = rf"${N}^{dimension}$"
        else:
            grid_size = N

        plt.suptitle(rf"Total variation of primitive variables TV($\vec{{w}}$) against time $t$ for {config.title()} test ($N = {grid_size}$)", fontsize=24)
        fig.text(0.5, 0.04, r"Time $t$", fontsize=18, ha='center')

        plt.savefig(f"{save_path}/TV_{config}_{subgrid}_{timestep}_{scheme}_{N}.png", dpi=330)

        plt.cla()
        plt.clf()
        plt.close()


def plot_conservation_equations(hdf5, sim_variables, save_path):
    config, dimension, subgrid, timestep, scheme = sim_variables.config, sim_variables.dimension, sim_variables.subgrid, sim_variables.timestep, sim_variables.scheme

    # hdf5 keys are datetime strings
    datetimes = sorted(hdf5, key=hdf5.get)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=[21,10])
    eq_labels = [r"Mass $m$", r"Momentum $p_x$", r"Energy $E$"]

    for _j in [0,1,2]:
        ax[_j].set_ylabel(eq_labels[_j], fontsize=18)
        ax[_j].grid(linestyle="--", linewidth=0.5)

    for datetime in datetimes:
        N = hdf5[datetime].attrs['cells']
        eq_dict = analytic.calculate_conservation(hdf5[datetime], sim_variables)

        x = np.asarray(list(eq_dict.keys()))
        x.sort()

        y = np.asarray(list(eq_dict.values()))
        y1 = y[...,0]  # mass
        y2 = y[...,4]  # total energy
        y3 = y[...,1]  # momentum_x
        y4 = y[...,5]  # B*vol_x
        y_data = [[y1, y2], [y3, y4]]

        for _i, _j in PLOT_INDEXES:
            y_i, y_f = y_data[_i][_j][0], y_data[_i][_j][-1]
            try:
                decimal_point = int(('%e' % abs(y_f-y_i)).split('-')[1])
            except IndexError:
                decimal_point = int(('%e' % abs(y_f-y_i)).split('+')[1])
            if _i == 0:
                ax[_j].plot(x, y_data[_i][_j], linewidth=2, color=COLOURS[_i][_j])
                ax[_j].annotate(round(y_i, decimal_point), (x[0], y_i), fontsize=12)
                ax[_j].annotate(round(y_f, decimal_point), (x[-1], y_f), fontsize=12)
            elif _i == 1 and _j == 0:
                ax[2].plot(x, y_data[_i][_j], linewidth=2, color=COLOURS[_i][_j])
                ax[2].annotate(round(y_i, decimal_point), (x[0], y_i), fontsize=12)
                ax[2].annotate(round(y_f, decimal_point), (x[-1], y_f), fontsize=12)

        if dimension >= 2:
            grid_size = rf"${N}^{dimension}$"
        else:
            grid_size = N

        plt.suptitle(rf"Conservation of variables ($m, p_x, E_{{tot}}$) against time $t$ for {config.title()} test ($N = {grid_size}$)", fontsize=24)
        fig.text(0.5, 0.04, r"Time $t$", fontsize=18, ha='center')

        plt.savefig(f"{save_path}/conserveEq_{config}_{subgrid}_{timestep}_{scheme}_{N}.png", dpi=330)

        plt.cla()
        plt.clf()
        plt.close()


#DONE,NOT CHECKED
def make_video(hdf5, sim_variables, save_path, vidpath, variable="all"):
    config, dimension, subgrid, timestep, scheme = sim_variables.config, sim_variables.dimension, sim_variables.subgrid, sim_variables.timestep, sim_variables.scheme
    start_pos, end_pos = sim_variables.start_pos, sim_variables.end_pos
    variable = variable.lower()

    # hdf5 keys are datetime strings
    datetimes = sorted(hdf5, key=hdf5.get)

    for datetime in datetimes:
        simulation = hdf5[datetime]
        counter, end_count = 0, len(simulation)
        N = simulation.attrs['cells']

        for t, grid in simulation.items():
            print(f"Creating {counter+1}/{end_count} ...", end='\r')

            if variable == "all":
                fig, ax, plot_ = make_figure(PLOT_OPTIONS, sim_variables)
                y_data = make_data(PLOT_OPTIONS, grid, sim_variables)

                for idx, (_i,_j) in enumerate(plot_['indexes']):
                    y = y_data[idx]

                    if dimension == 2:
                        fig.text(0.5, 0.04, r"Cell index $x$", fontsize=18, ha='center')
                        fig.text(0.04, 0.4, r"Cell index $y$", fontsize=18, ha='center', rotation='vertical')
                        graph = ax[_i,_j].imshow(y, interpolation="nearest", cmap=plot_['colours']['2d'][idx], origin="lower")
                        divider = make_axes_locatable(ax[_i,_j])
                        cax = divider.append_axes('right', size='5%', pad=0.05)
                        fig.colorbar(graph, cax=cax, orientation='vertical')
                        plt.suptitle(rf"Primitive variables $\vec{{w}}$ against cell indices $x$ & $y$ at $t = {round(float(t),4)}$ ($N = {N}^{dimension}$)", fontsize=24)
                        
                    else:
                        x = np.linspace(start_pos, end_pos, N)
                        fig.text(0.5, 0.04, r"Cell position $x$", fontsize=18, ha='center')
                        if BEAUTIFY:
                            gradient_plot([x, y], [_i,_j], ax, linewidth=2, color=plot_['colours']['1d'][idx])
                        else:
                            ax[_i,_j].plot(x, y, linewidth=2, color=plot_['colours']['1d'][idx])
                        plt.suptitle(rf"Primitive variables $\vec{{w}}$ against cell position $x$ at $t = {round(float(t),4)}$ ($N = {N}$)", fontsize=24)

                plt.savefig(f"{vidpath}/{str(counter).zfill(4)}.png", dpi=330)

            else:
                plot_option = [variable]
                fig, ax, plot_ = make_figure(plot_option, sim_variables)
                plt.axis('off')

                y_data = make_data(plot_option, grid, sim_variables)

                if dimension == 2:
                    ax.imshow(y_data, interpolation="nearest", cmap=plot_['colours']['2d'][0], origin="lower")
                else:
                    ax.plot(x, y_data, linewidth=2, color=plot_['colours']['1d'][0])

                plt.savefig(f"{vidpath}/{str(counter).zfill(4)}.png", dpi=330, bbox_inches='tight', pad_inches=0, transparent=True)

            plt.cla()
            plt.clf()
            plt.close()

            counter += 1

        try:
            subprocess.call(["ffmpeg", "-framerate", "60", "-pattern_type", "glob", "-i", f"{vidpath}/*.png", "-c:v", "libx264", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", f"{save_path}/vid_{config}_{subgrid}_{timestep}_{scheme}.mp4"])
        except Exception as e:
            print(f"{generic.BColours.FAIL}Video creation failed{generic.BColours.ENDC}")
            pass
        else:
            shutil.rmtree(vidpath)


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