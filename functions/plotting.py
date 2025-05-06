import os
import shutil
import subprocess

import numpy as np
import matplotlib as mpl
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



# Make figures and axes for plotting
def make_figure(options, sim_variables, variable="normal", style=STYLE):
    if 0 < len(options) < 13:
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
        names, labels, errors, tvs = [], [], [], []
        for option in options:
            option = option.lower()

            if "energy" in option or "temp" in option or option.startswith("e"):
                if "int" in option:
                    name = "Internal energy"
                    if "density" in option:
                        name += ' density'
                        label = r"$e_\mathrm{int}$"
                        error = r"$\epsilon_N(e_\mathrm{int})$"
                        tv = r"TV($e_\mathrm{int}$)"
                    else:
                        label = r"$E_\mathrm{int}$"
                        error = r"$\epsilon_N(E_\mathrm{int})$"
                        tv = r"TV($E_\mathrm{int}$)"
                else:
                    name = "Total energy"
                    if "density" in option:
                        name += ' density'
                        label = r"$e_\mathrm{tot}$"
                        error = r"$\epsilon_N(e_\mathrm{tot})$"
                        tv = r"TV($e_\mathrm{tot}$)"
                    else:
                        label = r"$E_\mathrm{tot}$"
                        error = r"$\epsilon_N(E_\mathrm{tot})$"
                        tv = r"TV($E_\mathrm{tot}$)"

            elif "mom" in option:
                name = "Momentum"
                label = rf"$p_{option[-1]}$"
                error = rf"$\epsilon_N(p_{option[-1]})$"
                tv = rf"TV($p_{option[-1]}$)"

            elif "mass" in option:
                name = "Mass"
                label = r"$m$"
                error = r"$\epsilon_N(m)$"
                tv = r"TV($m$)"

            elif option.startswith("p"):
                name = "Pressure"
                label = r"$P$"
                error = r"$\epsilon_N(P)$"
                tv = r"TV($P$)"

            elif option.startswith("v"):
                name = "Velocity"
                label = rf"$v_{option[-1]}$"
                error = rf"$\epsilon_N(v_{option[-1]})$"
                tv = rf"TV($v_{option[-1]}$)"

            elif option.startswith("b") or option.startswith("mag"):
                if "p" in option:
                    name = "Mag. pressure"
                    label = r"$P_B$"
                    error = r"$\epsilon_N(P_B)$"
                    tv = r"TV($P_B$)"
                else:
                    name = "Mag. field"
                    label = rf"$B_{option[-1]}$"
                    error = rf"$\epsilon_N(B_{option[-1]})$"
                    tv = rf"TV($B_{option[-1]}$)"

            else:
                name = "Density"
                label = r"$\rho$"
                error = r"$\epsilon_N(\rho)$"
                tv = r"TV($\rho$)"

            names.append(f"{name} {label}")
            labels.append(rf"{label} [arb. units]")
            errors.append(rf"{error} [arb. units]")
            tvs.append(rf"{tv} [arb. units]")

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
        fig, ax = plt.figure(), np.full((rows, cols), None)
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
        #fig.subplots_adjust(wspace=0, hspace=0)
        fig.subplots_adjust(wspace=0.75)

        for idx, (_i,_j) in enumerate(indexes):
            ax[_i,_j].set_title(names[idx])
            ax[_i,_j].tick_params(axis='both', which='major')
            ax[_i,_j].tick_params(axis='both', which='minor')
            if "error" in variable:
                ax[_i,_j].set_ylabel(errors[idx])
            elif "tv" in variable:
                ax[_i,_j].set_ylabel(tvs[idx])
            else:
                ax[_i,_j].set_ylabel(labels[idx])

            if sim_variables.dimension < 2:
                ax[_i,_j].set_xlim([sim_variables.start_pos, sim_variables.end_pos])
                ax[_i,_j].grid(linestyle="--", linewidth=0.5)
            else:
                ax[_i,_j].yaxis.set_label_position("right")
                ax[_i,_j].yaxis.labelpad = 80

        return fig, ax, {'indexes':indexes, 'labels':labels, 'errors':errors, 'tvs':tvs, 'colours': {'theo':theo_colour, '1d':colours, '2d':twod_colours}}
    else:
        raise IndexError('Number of variables to plot should be < 13')


# Create list of data plots; accepts primitive grid
def make_data(options, grid, sim_variables):
    quantities = []

    for option in options:
        option = option.lower()

        if "energy" in option or "temp" in option or option.startswith("e"):
            if "int" in option:
                quantity = fv.divide(grid[...,4], grid[...,0] * (sim_variables.gamma-1))
            else:
                quantity = fv.divide(fv.convert_variable("pressure", grid, sim_variables.gamma), grid[...,0])
            if "density" in option:
                quantity *= grid[...,0]
        elif option.startswith("p"):
            quantity = grid[...,4]
        elif option.startswith("v") or "mom" in option:
            axis = {"x":0, "y":1, "z":2}[option[-1]]
            quantity = grid[...,1+axis]
            if "mom" in option:
                quantity *= grid[...,0]
        elif option.startswith("b") or option.startswith("mag"):
            if "p" in option:
                quantity = .5 * fv.norm(grid[...,5:8])**2
            else:
                axis = {"x":0, "y":1, "z":2}[option[-1]]
                quantity = grid[...,5+axis]
        else:
            quantity = grid[...,0]

        quantities.append(quantity)
    return quantities


# Initiate the live plot feature
def initiate_live_plot(sim_variables):
    N, dimension, start_pos, end_pos = sim_variables.cells, sim_variables.dimension, sim_variables.start_pos, sim_variables.end_pos
    options = sim_variables.plot_options
    plt.ion()

    fig, ax, plot_ = make_figure(options, sim_variables)

    #plt.tight_layout()

    graphs = []
    for idx, (_i,_j) in enumerate(plot_['indexes']):
        if dimension == 2:
            fig.text(0.5, 0.04, r"$x$", ha='center')
            fig.text(0.04, 0.4, r"$y$", ha='center', rotation='vertical')
            graph = ax[_i,_j].imshow(np.zeros((N,N)), interpolation="nearest", cmap=plot_['colours']['2d'][idx], origin="lower")
            divider = make_axes_locatable(ax[_i,_j])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(graph, cax=cax, orientation='vertical')
        else:
            fig.text(0.5, 0.04, r"$x$", ha='center')
            ax[_i,_j].set_xlim([start_pos, end_pos])
            ax[_i,_j].grid(linestyle='--', linewidth=0.5)
            graph, = ax[_i,_j].plot(np.linspace(start_pos, end_pos, N), np.linspace(start_pos, end_pos, N), color=plot_['colours']['1d'][idx])
        graphs.append(graph)
    return fig, ax, graphs


# Update live plot
def update_plot(grid_snapshot, t, sim_variables, fig, ax, graphs):
    options = sim_variables.plot_options
    plot_data = make_data(options, grid_snapshot, sim_variables)

    if sim_variables.dimension == 2:
        for index, graph in enumerate(graphs):
            graph.set_data(plot_data[index])
            graph.set_clim([np.min(plot_data[index]), np.max(plot_data[index])])

        #plt.suptitle(rf"Grid variables $\boldsymbol{{u}}$ against cell indices $x$ & $y$ at $t = {round(t,4)}$", fontsize=24)
    else:
        for index, graph in enumerate(graphs):
            graph.set_ydata(plot_data[index])
            #graphBR.set_ydata(analytic.calculateEntropyDensity(grid_snapshot, 1.4))  # scaled entropy density

        for _ in ax.ravel()[:len(options)]:
            _.relim()
            _.autoscale_view()

        #plt.suptitle(rf"Grid variables $\boldsymbol{{u}}$ against cell position $x$ at $t = {round(t,4)}$", fontsize=24)

    fig.canvas.draw()
    fig.canvas.flush_events()
    pass


# Function for plotting a snapshot of the grid
def plot_snapshot(grid_snapshot, t, sim_variables, **kwargs):
    config, N, dimension, subgrid, timestep, solver = sim_variables.config, sim_variables.cells, sim_variables.dimension, sim_variables.subgrid, sim_variables.timestep, sim_variables.solver
    start_pos, end_pos = sim_variables.start_pos, sim_variables.end_pos
    options = sim_variables.plot_options

    try:
        text = kwargs["text"]
    except KeyError:
        text = ""

    fig, ax, plot_ = make_figure(options, sim_variables)
    y_data = make_data(options, grid_snapshot, sim_variables)

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
                gradient_plot([x, y], [_i,_j], ax, color=plot_['colours']['1d'][idx])
            else:
                ax[_i,_j].plot(x, y, color=plot_['colours']['1d'][idx])

    if dimension == 2:
        #plt.suptitle(rf"Grid variables $\boldsymbol{{u}}$ against cell indices $x$ & $y$ at $t = {round(t,3)}$ ($N = {N}^{dimension}$) {text}", fontsize=30)
        fig.text(0.5, 0.04, r"$x$", ha='center')
        fig.text(0.04, 0.4, r"$y$", ha='center', rotation='vertical')
    else:
        #plt.suptitle(rf"Grid variables $\boldsymbol{{u}}$ against cell position $x$ {text}", fontsize=30)
        fig.text(0.5, 0.04, r"$x$", ha='center')

    plt.savefig(f"{kwargs['save_path']}/varPlot_{dimension}D_{config}_{subgrid}_{timestep}_{solver}_{'%.3f' % round(t,3)}.png", bbox_inches='tight')

    plt.cla()
    plt.clf()
    plt.close()
    pass


# Plot snapshots of quantities for multiple runs
def plot_quantities(hdf5, sim_variables, save_path):
    config, dimension, subgrid, timestep, solver = sim_variables.config, sim_variables.dimension, sim_variables.subgrid, sim_variables.timestep, sim_variables.solver
    precision, snapshots = sim_variables.precision, sim_variables.snapshots
    start_pos, end_pos = sim_variables.start_pos, sim_variables.end_pos
    options = sim_variables.plot_options

    # hdf5 keys are datetime strings
    datetimes = [datetime for datetime in hdf5.keys()]
    datetimes.sort()

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
        fig, ax, plot_ = make_figure(options, sim_variables)
        legends_on = False

        # Plot each simulation at the specific timing
        ref_time = ref_timings[snap_index]
        for datetime in datetimes:
            simulation = hdf5[datetime]
            N = simulation.attrs['cells']
            timing = str(plot_timings_for_each_grp[datetime][snap_index])

            x = np.linspace(start_pos, end_pos, N)
            y_data = make_data(options, simulation[timing], sim_variables)

            for idx, (_i,_j) in enumerate(plot_['indexes']):
                y = y_data[idx]

                if len(hdf5) != 1:
                    if dimension < 2:
                        ax[_i,_j].plot(x, y, label=f"N = {N}")
                        fig.text(0.5, 0.04, r"$x$", ha='center')
                        #plt.suptitle(rf"Grid variables $\boldsymbol{{u}}$ against cell position $x$ at $t = {round(ref_time,3)}$", fontsize=30)
                        legends_on = True
                else:
                    if dimension == 2:
                        graph = ax[_i,_j].imshow(y, interpolation="nearest", cmap=plot_['colours']['2d'][idx], origin="lower")
                        divider = make_axes_locatable(ax[_i,_j])
                        cax = divider.append_axes('right', size='5%', pad=0.05)
                        fig.colorbar(graph, cax=cax, orientation='vertical')
                        #plt.suptitle(rf"Grid variables $\boldsymbol{{u}}$ against cell indices $x$ & $y$ at $t = {round(ref_time,3)}$ ($N = {N}^{dimension}$)", fontsize=30)
                        fig.text(0.5, 0.04, r"$x$", ha='center')
                        fig.text(0.04, 0.4, r"$y$", ha='center', rotation='vertical')
                    else:
                        if BEAUTIFY:
                            gradient_plot([x, y], [_i,_j], ax, color=plot_['colours']['1d'][idx])
                        else:
                            #ax[_i,_j].plot(x, y, linewidth=2, linestyle="-", marker="D", ms=4, markerfacecolor=fig.get_facecolor(), markeredgecolor=plot_['colours']['1d'], color=plot_['colours']['1d'])
                            ax[_i,_j].plot(x, y, color=plot_['colours']['1d'][idx])
                        fig.text(0.5, 0.04, r"$x$", ha='center')
                        #plt.suptitle(rf"Grid variables $\boldsymbol{{u}}$ against cell position $x$ at $t = {round(ref_time,3)}$ ($N = {N}$)", fontsize=24)

        # Add analytical solutions only for 1D
        if dimension < 2:
            # Add analytical solution for smooth functions, using the highest resolution and timing
            if sim_variables.config_category == "smooth":
                analytical = constructor.initialise(sim_variables)

                y_theo = make_data(options, analytical, sim_variables)
                for idx, (_i,_j) in enumerate(plot_['indexes']):
                    ax[_i,_j].plot(x, y_theo[idx], color=plot_['colours']['theo'], linestyle="--", label=rf"{config.title()}$_{{theo}}$")
                legends_on = True

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
                    print(f"{generic.BColours.WARNING}Analytic plot error: {e}{generic.BColours.ENDC}")
                    pass
                else:
                    y_theo = make_data(options, soln, sim_variables)
                    for idx, (_i,_j) in enumerate(plot_['indexes']):
                        ax[_i,_j].plot(x, y_theo[idx], color=plot_['colours']['theo'], linestyle="--", label=plot_label)
                    legends_on = True

        if legends_on:
            if len(hdf5) > 5:
                _ncol = 2
            else:
                _ncol = 1
            handles, labels = plt.gca().get_legend_handles_labels()
            fig.legend(handles, labels, ncol=_ncol)

        plt.savefig(f"{save_path}/varPlot_{dimension}D_{config}_{subgrid}_{timestep}_{solver}_{'%.3f' % round(ref_time,3)}.png", dpi=330)

        plt.cla()
        plt.clf()
        plt.close()


# Plot solution errors to determine order of convergence of numerical scheme
def plot_solution_errors(hdf5, sim_variables, save_path, error_norm):
    options = ["density", "total energy"]
    config, dimension, subgrid, timestep, solver = sim_variables.config, sim_variables.dimension, sim_variables.subgrid, sim_variables.timestep, sim_variables.solver

    # hdf5 keys are datetime strings
    datetimes = [datetime for datetime in hdf5.keys()]
    datetimes.sort()

    # Solution errors plot
    fig, ax, plot_ = make_figure(options, sim_variables, "errors")

    array = np.full((1+len(options), len(datetimes)), 0., dtype=sim_variables.precision)
    for idx, datetime in enumerate(datetimes):
        _arr = [hdf5[datetime].attrs['cells']**dimension]

        # Get last instance of the grid with largest time key
        time_key = max([float(t) for t in hdf5[datetime].keys()])
        solution_errors: np.array = analytic.calculate_solution_error(hdf5[datetime][str(time_key)], sim_variables, error_norm)

        for option in options:
            option = option.lower()
            if "energy" in option or "temp" in option or option.startswith("e"):
                if "int" in option:
                    _arr.append(solution_errors[-1])
                else:
                    _arr.append(solution_errors[-2])
            elif option.startswith("p"):
                _arr.append(solution_errors[4])
            elif option.startswith("v") or (option.startswith("b") or "field" in option):
                axis = {'x':0, 'y':1, 'z':2}[option[-1]]
                if option.startswith("v"):
                    _arr.append(solution_errors[1+axis])
                else:
                    _arr.append(solution_errors[5+axis])
            else:
                _arr.append(solution_errors[0])

        array[...,idx] = np.asarray(_arr, dtype=sim_variables.precision)
    x, y_data = array[:1].ravel(), array[1:]
    x.sort()

    for idx, (_i,_j) in enumerate(plot_['indexes']):
        y = y_data[idx]

        EOC = np.diff(np.log(y))/np.diff(np.log(x))
        _idx = np.argmin(np.abs(np.average(EOC)-EOC))
        c = np.log10(y[_idx]) - EOC[_idx]*np.log10(x[_idx])

        for order in [1,2,4,5]:
            alpha = 10**c
            ytheo = alpha*x**(-order)
            ax[_i,_j].loglog(x, ytheo, color=plot_['colours']['theo'], linestyle="--")
            ax[_i,_j].annotate(rf"$O(N^{order})$", xy=(x[-1], ytheo[-1]), xytext=(5,-5), textcoords='offset points')
        ax[_i,_j].loglog(x, y, linestyle="-", marker="o", color=plot_['colours']['1d'][idx])
        ax[_i,_j].scatter([], [], s=.5, color=fig.get_facecolor(), label=rf"$|$EOC$_{{max}}|$ = {round(max(np.abs(EOC)), 4)}")
        ax[_i,_j].legend()
        ax[_i,_j].set_xlim([min(x)/1.5,max(x)*1.5])

    #plt.suptitle(rf"$L_{error_norm}$ error norm $\epsilon_N(\boldsymbol{{W}})$ against resolution $N$ for {config.title()} test", fontsize=30)
    fig.text(0.5, 0.04, r"Resolution $N$", ha='center')

    plt.savefig(f"{save_path}/solErr_L{error_norm}_{subgrid}_{timestep}_{solver}.png", bbox_inches='tight')

    plt.cla()
    plt.clf()
    plt.close()

    # Order of convergence plot
    fig, ax = plt.subplots()

    ax.set_ylabel("Order of convergence", rotation='vertical')
    ax.grid(linestyle="--", linewidth=0.5)

    x_diff = x[1:]
    y_diff = np.log2(y_data[...,:-1]/y_data[...,1:])

    if dimension == 2:
        y_diff /= np.log2(4)

    for idx in range(len(plot_['indexes'])):
        ax.plot(x_diff, y_diff[idx], linestyle="--", marker="o", color=plot_['colours']['1d'][idx], label=plot_['labels'][idx])

    #plt.suptitle(rf"Order of convergence against resolution $N$ for {config.title()} test", fontsize=30)
    fig.text(0.5, 0.04, r"Resolution $N$", ha='center')
    _xticklabels = [item.get_text() for item in ax.get_xticklabels()]
    _xticklabels = [rf"${int(v)}\rightarrow{int(x[i+1])}$" for i,v in enumerate(x[:-1])]
    ax.set_xticks(x_diff)
    ax.set_xticklabels(_xticklabels, rotation=45, ha="right")
    ax.legend()

    plt.savefig(f"{save_path}/convergenceOrder_{subgrid}_{timestep}_{solver}.png", bbox_inches='tight')

    plt.cla()
    plt.clf()
    plt.close()


# Total variation to determine if numerical scheme prevents oscillation
def plot_total_variation(hdf5, sim_variables, save_path):
    config, dimension, subgrid, timestep, solver = sim_variables.config, sim_variables.dimension, sim_variables.subgrid, sim_variables.timestep, sim_variables.solver
    options = sim_variables.plot_options

    # hdf5 keys are datetime strings
    datetimes = [datetime for datetime in hdf5.keys()]
    datetimes.sort()

    fig, ax, plot_ = make_figure(options, sim_variables, "tv")

    for datetime in datetimes:
        N = hdf5[datetime].attrs['cells']
        total_variations: dict = analytic.calculate_TV(hdf5[datetime], sim_variables)

        x = np.asarray(list(total_variations.keys()))
        x.sort()
        ys = np.asarray(list(total_variations.values()))

        y_data = np.full((len(options), len(x)), 0., dtype=sim_variables.precision)
        for idx, option in enumerate(options):
            option = option.lower()            
            if "energy" in option or "temp" in option or option.startswith("e"):
                if "int" in option:
                    y_data[idx] = ys[...,-1]
                else:
                    y_data[idx] = ys[...,-2]
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

        if dimension >= 2:
            grid_size = f"{N}^{dimension}"
        else:
            grid_size = N

        #plt.suptitle(rf"Total variation of grid variables TV($\boldsymbol{{u}}$) against time $t$ for {config.title()} test ($N = {grid_size}$)", fontsize=30)
        fig.text(0.5, 0.04, rf"Time $t$ [arb. units]", ha='center')

        plt.savefig(f"{save_path}/TV_{config}_{subgrid}_{timestep}_{solver}_{N}.png", bbox_inches='tight')

        plt.cla()
        plt.clf()
        plt.close()


# Determines if numerical scheme is conservative to machine precision
def plot_conservation_equations(hdf5, sim_variables, save_path):
    options = ["mass", "momentum_x", "energy"]
    config, dimension, subgrid, timestep, solver = sim_variables.config, sim_variables.dimension, sim_variables.subgrid, sim_variables.timestep, sim_variables.solver
    
    # hdf5 keys are datetime strings
    datetimes = [datetime for datetime in hdf5.keys()]
    datetimes.sort()

    fig, ax, plot_ = make_figure(options, sim_variables)

    for datetime in datetimes:
        N = hdf5[datetime].attrs['cells']
        conservation: dict = analytic.calculate_conservation(hdf5[datetime], sim_variables)

        x = np.asarray(list(conservation.keys()))
        x.sort()
        ys = np.asarray(list(conservation.values()))

        y_data = np.full((len(options), len(x)), 0., dtype=sim_variables.precision)
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

        if dimension >= 2:
            grid_size = f"{N}^{dimension}"
        else:
            grid_size = N

        #plt.suptitle(rf"Conservation of variables ($m, p_x, E_{{tot}}$) against time $t$ for {config.title()} test ($N = {grid_size}$)", fontsize=30)
        fig.text(0.5, 0.04, rf"Time $t$ [arb. units]", ha='center')

        plt.savefig(f"{save_path}/conserveEq_{config}_{subgrid}_{timestep}_{solver}_{N}.png", bbox_inches='tight')

        plt.cla()
        plt.clf()
        plt.close()


# Make a video of entire simulation; video of all plot options or specific variable
def make_video(hdf5, sim_variables, save_path, vidpath, variable="all"):
    config, dimension, subgrid, timestep, solver = sim_variables.config, sim_variables.dimension, sim_variables.subgrid, sim_variables.timestep, sim_variables.solver
    start_pos, end_pos = sim_variables.start_pos, sim_variables.end_pos

    # hdf5 keys are datetime strings
    datetimes = [datetime for datetime in hdf5.keys()]
    datetimes.sort()

    for datetime in datetimes:
        simulation = hdf5[datetime]
        N = simulation.attrs['cells']

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
                x = np.linspace(start_pos, end_pos, N)
                y_data = make_data(options, grid, sim_variables)

                if variable == "all":
                    for idx, (_i,_j) in enumerate(plot_['indexes']):
                        y = y_data[idx]

                        if dimension == 2:
                            fig.text(0.5, 0.04, r"$x$", ha='center')
                            fig.text(0.04, 0.4, r"$y$", ha='center', rotation='vertical')
                            graph = ax[_i,_j].imshow(y, interpolation="nearest", cmap=plot_['colours']['2d'][idx], origin="lower")
                            divider = make_axes_locatable(ax[_i,_j])
                            cax = divider.append_axes('right', size='5%', pad=0.05)
                            fig.colorbar(graph, cax=cax, orientation='vertical')
                            #plt.suptitle(rf"Grid variables $\boldsymbol{{u}}$ against cell indices $x$ & $y$ at $t = {round(float(t),4)}$ ($N = {N}^{dimension}$)", fontsize=30)

                        else:
                            fig.text(0.5, 0.04, r"$x$", ha='center')
                            if BEAUTIFY:
                                gradient_plot([x, y], [_i,_j], ax, color=plot_['colours']['1d'][idx])
                            else:
                                ax[_i,_j].plot(x, y, color=plot_['colours']['1d'][idx])
                            #plt.suptitle(rf"Grid variables $\boldsymbol{{u}}$ against cell position $x$ at $t = {round(float(t),4)}$ ($N = {N}$)", fontsize=30)

                    plt.savefig(f"{vidpath}/{str(counter).zfill(5)}.png", bbox_inches='tight')

                else:
                    idx = 0
                    plt.axis('off')

                    if dimension == 2:
                        ax[idx,idx].imshow(y_data[idx], interpolation="nearest", cmap=plot_['colours']['2d'][idx], origin="lower")
                    else:
                        ax[idx,idx].plot(x, y_data[idx], color=plot_['colours']['1d'][idx])

                    plt.savefig(f"{vidpath}/{str(counter).zfill(5)}.png", bbox_inches='tight', pad_inches=0, transparent=True)

                plt.cla()
                plt.clf()
                plt.close()

                counter += 1

            try:
                subprocess.call(["ffmpeg", "-hide_banner", "-loglevel", "error", "-framerate", "60", "-pattern_type", "glob", "-i", f"{vidpath}/*.png", "-c:v", "libx264", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", f"{save_path}/vid_{config}_{subgrid}_{timestep}_{solver}_{variable}.mp4"])
            except Exception as e:
                print(f"{generic.BColours.FAIL}Video creation failed{generic.BColours.ENDC}")
                pass

        elif isinstance(variable, list) and all(isinstance(_, str) for _ in variable):
            variables = [_.lower() for _ in variable]
            style_counter = 0

            for _variable in variables:
                counter, end_count = 0, len(simulation)

                for t, grid in simulation.items():
                    print(f"Creating {counter+1}/{end_count} ... [{_variable}]", end='\r')

                    fig, ax, plot_ = make_figure([_variable], sim_variables)
                    y_data = make_data([_variable], grid, sim_variables)

                    idx = 0
                    plt.axis('off')

                    if dimension == 2:
                        ax[idx,idx].imshow(y_data[idx], interpolation="nearest", cmap=plot_['colours']['2d'][style_counter], origin="lower")
                    else:
                        ax[idx,idx].plot(x, y_data[idx], color=plot_['colours']['1d'][style_counter])

                    plt.savefig(f"{vidpath}/{str(counter).zfill(5)}.png", bbox_inches='tight', pad_inches=0, transparent=True)

                    plt.cla()
                    plt.clf()
                    plt.close()

                    counter += 1

                style_counter += 1

                try:
                    subprocess.call(["ffmpeg", "-hide_banner", "-loglevel", "error", "-framerate", "60", "-pattern_type", "glob", "-i", f"{vidpath}/*.png", "-c:v", "libx264", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", f"{save_path}/vid_{config}_{subgrid}_{timestep}_{solver}_{_variable}.mp4"])
                except Exception as e:
                    print(f"{generic.BColours.FAIL}Video creation failed{generic.BColours.ENDC}")
                    pass
                else:
                    for filename in os.listdir(vidpath):
                        filepath = os.path.join(vidpath, filename)
                        if os.path.isfile(filepath) or os.path.islink(filepath):
                            os.remove(filepath)
        shutil.rmtree(vidpath)


# Function for plotting instance of the grid; insert into any part of the code
def plot_this(grid, sim_variables, **kwargs):
    options = ['density', 'pressure', 'total energy', 'vx', 'vy', 'vz', 'Bx', 'By', 'Bz']

    try:
        t = kwargs['t']
    except KeyError:
        try:
            text = kwargs['text']
        except KeyError:
            text = ""
    else:
        text = rf"at $t = {t}$"

    fig, ax, plot_ = make_figure(options, sim_variables)
    y_data = make_data(options, grid, sim_variables)

    for idx, (_i,_j) in enumerate(plot_['indexes']):
        y = y_data[idx]

        if sim_variables.dimension == 2:
            graph = ax[_i,_j].imshow(y, interpolation="nearest", cmap=plot_['colours']['2d'][idx], origin="lower")
            divider = make_axes_locatable(ax[_i,_j])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(graph, cax=cax, orientation='vertical')
        else:
            x = np.linspace(sim_variables.start_pos, sim_variables.end_pos, len(y))
            if BEAUTIFY:
                gradient_plot([x, y], [_i,_j], ax, color=plot_['colours']['1d'][idx])
            else:
                ax[_i,_j].plot(x, y, color=plot_['colours']['1d'][idx])

    if sim_variables.dimension == 2:
        #plt.suptitle(rf"Grid variables $\boldsymbol{{u}}$ against cell indices $x$ & $y$ {text}", fontsize=30)
        fig.text(0.5, 0.04, r"$x$", ha='center')
        fig.text(0.04, 0.4, r"$y$", ha='center', rotation='vertical')
    else:
        #plt.suptitle(rf"Grid variables $\boldsymbol{{u}}$ against cell position $x$ {text}", fontsize=30)
        fig.text(0.5, 0.04, r"$x$", ha='center')

    if not sim_variables.live_plot:
        plt.show(block=True)
    pass


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