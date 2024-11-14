import shutil
import platform
import subprocess

import numpy as np
if platform.system() == "Darwin":
    import matplotlib
    matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable

from functions import analytic, constructor, fv, generic

##############################################################################
# Plotting functions and media handling
##############################################################################

STYLE = "default"
BEAUTIFY = False


PLOT_INDEXES = [[0,0], [0,1], [1,0], [1,1]]
PLOT_LABELS = [[r"Density $\rho$", r"Pressure $P$"], [r"Velocity $v_x$", r"Specific thermal energy $\frac{P}{\rho}$"]]
TWOD_COLOURS = [["viridis", "hot"], ["cividis", "plasma"]]
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


# Initiate the live plot feature
def initiate_live_plot(sim_variables):
    N, dimension, start_pos, end_pos = sim_variables.cells, sim_variables.dimension, sim_variables.start_pos, sim_variables.end_pos
    plt.ion()

    fig, ax = plt.subplots(nrows=2, ncols=2)
    plt.subplots_adjust(wspace=.2)

    graphs = []
    for _i, _j in PLOT_INDEXES:
        ax[_i,_j].set_ylabel(PLOT_LABELS[_i][_j])
        if dimension == 2:
            fig.text(0.5, 0.04, r"Cell index $x$", ha='center')
            fig.text(0.04, 0.4, r"Cell index $y$", ha='center', rotation='vertical')
            if _j == 1:
                ax[_i,_j].yaxis.set_label_position("right")
                ax[_i,_j].yaxis.labelpad = 55
            graph = ax[_i,_j].imshow(np.zeros((N,N)), interpolation="nearest", cmap=TWOD_COLOURS[_i][_j], origin="lower")
            divider = make_axes_locatable(ax[_i,_j])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(graph, cax=cax, orientation='vertical')
        else:
            fig.text(0.5, 0.04, r"Cell position $x$", ha='center')
            if _j == 1:
                ax[_i,_j].yaxis.tick_right()
                ax[_i,_j].yaxis.set_label_position("right")
            ax[_i,_j].set_xlim([start_pos, end_pos])
            ax[_i,_j].grid(linestyle='--', linewidth=0.5)
            graph, = ax[_i,_j].plot(np.linspace(start_pos, end_pos, N), np.linspace(start_pos, end_pos, N), linewidth=2, color=COLOURS[_i][_j])
        graphs.append(graph)
    return fig, ax, graphs


# Update live plot
def update_plot(grid_snapshot, t, dimension, fig, ax, graphs):
    # top-left: density, top-right: pressure, bottom-left: velocity_x, bottom-right: specific thermal energy
    plot_data = [grid_snapshot[...,0], grid_snapshot[...,4], grid_snapshot[...,1], fv.divide(grid_snapshot[...,4], grid_snapshot[...,0])]

    if dimension == 2:
        for index, graph in enumerate(graphs):
            graph.set_data(plot_data[index])
            graph.set_clim([np.min(plot_data[index]), np.max(plot_data[index])])

        plt.suptitle(rf"Primitive variables $\vec{{w}}$ against cell indices $x$ & $y$ at $t = {round(t,4)}$")
    else:
        for index, graph in enumerate(graphs):
            graph.set_ydata(plot_data[index])
            #graphBR.set_ydata(analytic.calculateEntropyDensity(grid_snapshot, 1.4))  # scaled entropy density

        for _i, _j in PLOT_INDEXES:
            ax[_i,_j].relim()
            ax[_i,_j].autoscale_view()

        plt.suptitle(rf"Primitive variables $\vec{{w}}$ against cell position $x$ at $t = {round(t,4)}$")

    fig.canvas.draw()
    fig.canvas.flush_events()
    pass


# Function for plotting a snapshot of the grid
def plot_snapshot(grid_snapshot, t, sim_variables, **kwargs):
    config, N, dimension, subgrid, timestep, scheme = sim_variables.config, sim_variables.cells, sim_variables.dimension, sim_variables.subgrid, sim_variables.timestep, sim_variables.scheme
    start_pos, end_pos = sim_variables.start_pos, sim_variables.end_pos

    try:
        text = kwargs["text"]
    except Exception as e:
        text = ""

    if dimension == 2:
        figsize = [15, 10]
    else:
        figsize = [21, 10]

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=figsize)

    for _i, _j in PLOT_INDEXES:
        ax[_i,_j].set_ylabel(PLOT_LABELS[_i][_j], fontsize=18)
        if dimension < 2:
            ax[_i,_j].set_xlim([start_pos, end_pos])
            ax[_i,_j].grid(linestyle="--", linewidth=0.5)

    x = np.linspace(start_pos, end_pos, N)
    y1 = grid_snapshot[...,0]  # density
    y2 = grid_snapshot[...,4]  # pressure
    y3 = grid_snapshot[...,1]  # vx
    y4 = y2/y1  # specific thermal energy
    y_data = [[y1, y2], [y3, y4]]

    for _i, _j in PLOT_INDEXES:
        y = y_data[_i][_j]

        if dimension == 2:
            graph = ax[_i,_j].imshow(y, interpolation="nearest", cmap=TWOD_COLOURS[_i][_j], origin="lower")
            divider = make_axes_locatable(ax[_i,_j])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(graph, cax=cax, orientation='vertical')
        else:
            if BEAUTIFY:
                gradient_plot([x, y], [_i,_j], ax, linewidth=2, color=COLOURS[_i][_j])
            else:
                ax[_i,_j].plot(x, y, linewidth=2, color=COLOURS[_i][_j])

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
def plot_quantities(f, sim_variables, save_path):
    config, dimension, subgrid, timestep = sim_variables.config, sim_variables.dimension, sim_variables.subgrid, sim_variables.timestep
    scheme, precision, snapshots = sim_variables.scheme, sim_variables.precision, sim_variables.snapshots
    start_pos, end_pos, initial_left = sim_variables.start_pos, sim_variables.end_pos, sim_variables.initial_left

    # hdf5 keys are string; need to convert back to int and sort again
    n_list = [int(n) for n in f.keys()]
    n_list.sort()

    if dimension == 2:
        figsize = [15, 10]
    else:
        figsize = [21, 10]

    # Separate the timings based on the number of snapshots; returns a dict of lists with the timing intervals for each N
    timings = {}
    for N in n_list:
        _timings = np.fromiter(f[str(N)].keys(), dtype=precision)
        _timings.sort()
        timings[N] = [timing[-1] for timing in np.array_split(_timings, abs(snapshots))]

    # Iterate through the list of timings generated by the number of snapshots
    for time_index in range(snapshots):
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=figsize)

        # Set up figure
        for _i, _j in PLOT_INDEXES:
            ax[_i,_j].set_ylabel(PLOT_LABELS[_i][_j], fontsize=18)
            if dimension < 2:
                ax[_i,_j].set_xlim([start_pos, end_pos])
                ax[_i,_j].grid(linestyle="--", linewidth=0.5)

        # Plot each simulation at the specific timing
        for N in n_list:
            time_key = str(timings[N][time_index])
            y1 = f[str(N)][time_key][...,0]   # density
            y2 = f[str(N)][time_key][...,4]   # pressure
            y3 = f[str(N)][time_key][...,1]   # vx
            y4 = y2/y1  # specific thermal energy
            x = np.linspace(start_pos, end_pos, N)
            y_data = [[y1, y2], [y3, y4]]

            # density, pressure, vx, thermal energy
            for _i, _j in PLOT_INDEXES:
                y = y_data[_i][_j]

                if len(f) != 1:
                    if dimension < 2:
                        ax[_i,_j].plot(x, y, linewidth=2, label=f"N = {N}")
                        plt.suptitle(rf"Primitive variables $\vec{{w}}$ against cell position $x$ at $t \approx {round(timings[max(n_list)][time_index],3)}$", fontsize=24)
                else:
                    if dimension == 2:
                        graph = ax[_i,_j].imshow(y, interpolation="nearest", cmap=TWOD_COLOURS[_i][_j], origin="lower")
                        divider = make_axes_locatable(ax[_i,_j])
                        cax = divider.append_axes('right', size='5%', pad=0.05)
                        fig.colorbar(graph, cax=cax, orientation='vertical')
                        plt.suptitle(rf"Primitive variables $\vec{{w}}$ against cell indices $x$ & $y$ at $t \approx {round(timings[max(n_list)][time_index],3)}$ ($N = {N}^{dimension}$)", fontsize=24)
                        fig.text(0.5, 0.04, r"Cell index $x$", fontsize=18, ha='center')
                        fig.text(0.04, 0.4, r"Cell index $y$", fontsize=18, ha='center', rotation='vertical')
                    else:
                        if BEAUTIFY:
                            gradient_plot([x, y], [_i,_j], ax, linewidth=2, color=COLOURS[_i][_j])
                        else:
                            #ax[_i,_j].plot(x, y, linewidth=2, linestyle="-", marker="D", ms=4, markerfacecolor=fig.get_facecolor(), markeredgecolor=COLOURS[_i][_j], color=COLOURS[_i][_j])
                            ax[_i,_j].plot(x, y, linewidth=2, color=COLOURS[_i][_j])
                        plt.suptitle(rf"Primitive variables $\vec{{w}}$ against cell position $x$ at $t \approx {round(timings[max(n_list)][time_index],3)}$ ($N = {N}$)", fontsize=24)

        # Add analytical solutions only for 1D
        if dimension < 2:
            # Add analytical solution for smooth functions, using the highest resolution and timing
            if sim_variables.config_category == "smooth":
                analytical = constructor.initialise(sim_variables)

                # Adjust ylim and tolerances for Gaussian and sin-wave tests
                if config.startswith("gauss"):
                    P_tol = 5e-7
                else:
                    P_tol = .005
                P_range = np.linspace(initial_left[4]-P_tol, initial_left[4]+P_tol, 9)
                v_range = np.linspace(initial_left[1]-.005, initial_left[1]+.005, 9)
                ax[0,1].set_yticks(P_range)
                ax[1,0].set_yticks(v_range)
                ax[0,1].set_ylim([initial_left[4]-P_tol, initial_left[4]+P_tol])
                ax[1,0].set_ylim([initial_left[1]-.005, initial_left[1]+.005])

                y_theo = [[analytical[...,0], analytical[...,4]], [analytical[...,1], analytical[...,4]/analytical[...,0]]]
                for _i, _j in PLOT_INDEXES:
                    ax[_i,_j].plot(x, y_theo[_i][_j], linewidth=2, color=THEO_COLOUR, linestyle="--", label=rf"{config.title()}$_{{theo}}$")

            # Add Sod analytical solution, using the highest resolution and timing
            elif "sod" in config:
                tube, _t = f[str(max(n_list))][str(timings[max(n_list)][time_index])], timings[max(n_list)][time_index]
                Sod = analytic.calculate_Sod_analytical(tube, _t, sim_variables)

                y_theo = [[Sod[...,0], Sod[...,4]], [Sod[...,1], Sod[...,4]/Sod[...,0]]]
                for _i, _j in PLOT_INDEXES:
                    ax[_i,_j].plot(x, y_theo[_i][_j], linewidth=2, color=THEO_COLOUR, linestyle="--", label=r"Sod$_{theo}$")

            fig.text(0.5, 0.04, r"Cell position $x$", fontsize=18, ha='center')
            if len(f) != 1 or "sod" in config or config.startswith("gauss") or config.startswith("sin"):
                if len(f) > 5:
                    _ncol = 2
                else:
                    _ncol = 1
                handles, labels = plt.gca().get_legend_handles_labels()
                fig.legend(handles, labels, prop={'size': 16}, loc='upper right', ncol=_ncol)

        plt.savefig(f"{save_path}/varPlot_{dimension}D_{config}_{subgrid}_{timestep}_{scheme}_{'%.3f' % round(timings[max(n_list)][time_index],3)}.png", dpi=330)

        plt.cla()
        plt.clf()
        plt.close()
    return None


def plot_solution_errors(f, sim_variables, save_path, coeff, norm=1):
    config, dimension, subgrid, timestep, scheme = sim_variables.config, sim_variables.dimension, sim_variables.subgrid, sim_variables.timestep, sim_variables.scheme

    # hdf5 keys are string; need to convert back to int and sort again
    n_list = [int(n) for n in f.keys()]
    n_list.sort()

    # Solution errors plot
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=[21,10])
    error_labels = [[r"Density $\log{(\epsilon_\nu(\rho))}$", r"Pressure $\log{(\epsilon_\nu(P))}$"], [r"Velocity $\log{(\epsilon_\nu(v_x))}$", r"Thermal energy $\log{(\epsilon_\nu(\frac{P}{\rho}))}$"]]

    x, y1, y2, y3, y4 = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    for N in n_list:
        x = np.append(x, f[str(N)].attrs['cells'])
        solution_errors = analytic.calculate_solution_error(f[str(N)], sim_variables, norm)
        y1 = np.append(y1, solution_errors[0])  # density
        y2 = np.append(y2, solution_errors[4])  # pressure
        y3 = np.append(y3, solution_errors[1])  # vx
        y4 = np.append(y4, solution_errors[-1])  # specific thermal energy
    y_data = [[y1, y2], [y3, y4]]

    for _i, _j in PLOT_INDEXES:
        if _i == _j:
            ax[_i].set_ylabel(error_labels[_i][_j], fontsize=18)
            ax[_i].grid(linestyle="--", linewidth=0.5)

            EOC = np.diff(np.log(y_data[_i][_j]))/np.diff(np.log(x))
            idx = np.random.randint(0, len(EOC))
            c = np.log10(y_data[_i][_j][idx]) - EOC[idx]*np.log10(x[idx])

            for order in [1,2,4,5]:
                alpha = 10**(c + np.log10(coeff))
                ytheo = alpha*x**(-order)
                ax[_j].loglog(x, ytheo, linewidth=2, color=THEO_COLOUR, linestyle="--")
                ax[_j].annotate(rf"$O(N^{order})$", (x[-1], ytheo[-1]), fontsize=12)
            ax[_j].loglog(x, y_data[_i][_j], linewidth=2, linestyle="--", marker="o", color=COLOURS[_i][_j])
            ax[_j].scatter([], [], s=.5, color=fig.get_facecolor(), label=rf"$|\text{{EOC}}_{{max}}|$ = {round(max(np.abs(np.diff(np.log(y_data[_i][_j]))/np.diff(np.log(x)))), 4)}")
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
    return None


def plot_total_variation(f, sim_variables, save_path):
    config, dimension, subgrid, timestep, scheme = sim_variables.config, sim_variables.dimension, sim_variables.subgrid, sim_variables.timestep, sim_variables.scheme

    # hdf5 keys are string; need to convert back to int and sort again
    n_list = [int(n) for n in f.keys()]
    n_list.sort()

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=[21,10])
    tv_labels = [[r"Density TV($\rho$)", r"Pressure TV($P$)"], [r"Velocity TV($v_x$)", r"Thermal energy TV($\frac{P}{\rho}$)"]]

    for _i, _j in PLOT_INDEXES:
        ax[_i,_j].set_ylabel(tv_labels[_i][_j], fontsize=18)
        ax[_i,_j].grid(linestyle="--", linewidth=0.5)

    for N in n_list:
        tv_dict = analytic.calculate_tv(f[str(N)], sim_variables)
        x = np.asarray(list(tv_dict.keys()))
        y = np.asarray(list(tv_dict.values()))
        y1 = y[...,0]  # density
        y2 = y[...,4]  # pressure
        y3 = y[...,1]  # vx
        y4 = y[...,-1]  # specific thermal energy
        y_data = [[y1, y2], [y3, y4]]
        x.sort()

        for _i, _j in PLOT_INDEXES:
            ax[_i,_j].plot(x, y_data[_i][_j], linewidth=2, color=COLOURS[_i][_j])

        plt.suptitle(rf"Total variation of primitive variables TV($\vec{{w}}$) against time $t$ for {config.title()} test ($N = {N}^{dimension}$)", fontsize=24)
        fig.text(0.5, 0.04, r"Time $t$", fontsize=18, ha='center')

        plt.savefig(f"{save_path}/TV_{config}_{subgrid}_{timestep}_{scheme}_{N}.png", dpi=330)

        plt.cla()
        plt.clf()
        plt.close()
    return None


def plot_conservation_equations(f, sim_variables, save_path):
    config, dimension, subgrid, timestep, scheme = sim_variables.config, sim_variables.dimension, sim_variables.subgrid, sim_variables.timestep, sim_variables.scheme

    # hdf5 keys are string; need to convert back to int and sort again
    n_list = [int(n) for n in f.keys()]
    n_list.sort()

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=[21,10])
    eq_labels = [r"Mass ($m$)", r"Momentum ($p_x$)", r"Energy ($E_{tot}$)"]

    for _j in [0,1,2]:
        ax[_j].set_ylabel(eq_labels[_j], fontsize=18)
        ax[_j].grid(linestyle="--", linewidth=0.5)

    for N in n_list:
        eq_dict = analytic.calculate_conservation(f[str(N)], sim_variables)
        x = np.asarray(list(eq_dict.keys()))
        y = np.asarray(list(eq_dict.values()))
        y1 = y[...,0]  # mass
        y2 = y[...,4]  # total energy
        y3 = y[...,1]  # momentum_x
        y4 = y[...,5]  # B*vol_x
        y_data = [[y1, y2], [y3, y4]]
        x.sort()

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

        plt.suptitle(rf"Conservation of variables ($m, p_x, E_{{tot}}$) against time $t$ for {config.title()} test ($N = {N}^{dimension}$)", fontsize=24)
        fig.text(0.5, 0.04, r"Time $t$", fontsize=18, ha='center')

        plt.savefig(f"{save_path}/conserveEq_{config}_{subgrid}_{timestep}_{scheme}_{N}.png", dpi=330)

        plt.cla()
        plt.clf()
        plt.close()
        return None


def make_video(f, sim_variables, save_path, vidpath, variable="all"):
    config, dimension, subgrid, timestep, scheme = sim_variables.config, sim_variables.dimension, sim_variables.subgrid, sim_variables.timestep, sim_variables.scheme
    start_pos, end_pos = sim_variables.start_pos, sim_variables.end_pos
    variable = variable.lower()

    # hdf5 keys are string; need to convert back to int and sort again
    n_list = [int(n) for n in f.keys()]
    n_list.sort()

    if dimension == 2:
        figsize = [15, 10]
    else:
        figsize = [21, 10]

    for N in n_list:
        simulation = f[str(N)]
        counter = 0

        for t, grid in simulation.items():
            x = np.linspace(start_pos, end_pos, N)

            if variable == "all":
                fig, ax = plt.subplots(nrows=2, ncols=2, figsize=figsize)

                for _i, _j in PLOT_INDEXES:
                    ax[_i,_j].set_ylabel(PLOT_LABELS[_i][_j], fontsize=18)
                    if dimension < 2:
                        ax[_i,_j].set_xlim([start_pos, end_pos])
                        ax[_i,_j].grid(linestyle="--", linewidth=0.5)

                y1 = grid[...,0]  # density
                y2 = grid[...,4]  # pressure
                y3 = grid[...,1]  # vx
                y4 = y2/y1  # specific thermal energy
                y_data = [[y1, y2], [y3, y4]]

                for _i, _j in PLOT_INDEXES:
                    y = y_data[_i][_j]

                    if dimension == 2:
                        fig.text(0.5, 0.04, r"Cell index $x$", fontsize=18, ha='center')
                        fig.text(0.04, 0.4, r"Cell index $y$", fontsize=18, ha='center', rotation='vertical')
                        graph = ax[_i,_j].imshow(y, interpolation="nearest", cmap=TWOD_COLOURS[_i][_j], origin="lower")
                        divider = make_axes_locatable(ax[_i,_j])
                        cax = divider.append_axes('right', size='5%', pad=0.05)
                        fig.colorbar(graph, cax=cax, orientation='vertical')
                        plt.suptitle(rf"Primitive variables $\vec{{w}}$ against cell indices $x$ & $y$ at $t = {round(float(t),4)}$ ($N = {N}^{dimension}$)", fontsize=24)
                        
                    else:
                        fig.text(0.5, 0.04, r"Cell position $x$", fontsize=18, ha='center')
                        if BEAUTIFY:
                            gradient_plot([x, y], [_i,_j], ax, linewidth=2, color=COLOURS[_i][_j])
                        else:
                            ax[_i,_j].plot(x, y, linewidth=2, color=COLOURS[_i][_j])
                        plt.suptitle(rf"Primitive variables $\vec{{w}}$ against cell position $x$ at $t = {round(float(t),4)}$ ($N = {N}$)", fontsize=24)

                plt.savefig(f"{vidpath}/{str(counter).zfill(4)}.png", dpi=330)

            else:
                fig, ax = plt.subplots(figsize=figsize)
                plt.axis('off')

                if variable in ["energy", "temperature", "temp", "t"]:
                    y_data = fv.divide(grid[...,4], grid[...,0])
                    colour, colour2 = "darkviolet", "plasma"
                elif variable in ["pressure", "p"]:
                    y_data = grid[...,4]
                    colour, colour2 = "red", "hot"
                elif variable in ["vx", "x"]:
                    y_data = grid[...,1]
                    colour, colour2 = "green", "cividis"
                elif variable in ["vy", "y"]:
                    y_data = grid[...,2]
                    colour, colour2 = "yellow", "magma"
                else:
                    y_data = grid[...,0]
                    colour, colour2 = "blue", "viridis"

                if dimension == 2:
                    ax.imshow(y_data, interpolation="nearest", cmap=colour2, origin="lower")
                else:
                    ax.plot(x, y_data, linewidth=2, color=colour)

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
    return None


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