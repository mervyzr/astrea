import os
import shutil
import subprocess

import numpy as np
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
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



# Make figures and axes for plotting
def make_figure(option, sim_variables, variable="normal", style=STYLE, **kwargs):
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

    # Set up figure
    #mpl.rcParams['text.usetex'] = True
    fig, ax = plt.subplots()
    plt.rcParams['text.latex.preamble'] = r"\usepackage{lmodern}"
    params = {'font.size': 12,
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

    if "error" in variable:
        ax.set_ylabel(rf"{error} [arb. units]")
    elif "tv" in variable:
        ax.set_ylabel(rf"{tv} [arb. units]")
    else:
        ax.set_ylabel(rf"{label} [arb. units]")

    if sim_variables.dimension < 2:
        ax.set_xlim([sim_variables.start_pos, sim_variables.end_pos])
        ax.grid(linestyle="--", linewidth=0.5)
    else:
        ax.yaxis.set_label_position("right")
        ax.yaxis.labelpad = 80

    return fig, ax, {'name':name, 'label':label, 'error':error, 'tv':tv, 'theo_colour':theo_colour, '1d_colours':colours, '2d_colours':twod_colours}


# Create list of data plots; accepts primitive grid
def make_data(option, grid, sim_variables):
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

    return quantity



# Plot snapshots of quantities for multiple runs
def plot_quantities(hdf5, sim_variables, save_path):

    def make_physical_grid(_start_pos, _end_pos, _N):
        dx = abs(_end_pos-_start_pos)/_N
        half_cell = dx/2
        return np.linspace(_start_pos-half_cell, _end_pos+half_cell, _N+2)[1:-1]
    
    #analytical_grid = make_physical_grid(-1, 1, 8192)
    #import h5py
    #with h5py.File('/Users/Mervin/Desktop/mHydyS_shu-osher_ppm_ssprk(5,4)_68821867.hdf5','r') as asdasd:
    #    analytical_config = asdasd['8192']['0.47'][:]


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
    """ref_N = 0
    for datetime, grp in hdf5.items():
        ref_datetime = datetime if grp.attrs['cells'] > ref_N else ref_datetime
        ref_N = grp.attrs['cells'] if grp.attrs['cells'] > ref_N else ref_N
    ref_timings = plot_timings_for_each_grp[ref_datetime]"""

    ref_N = 0
    for datetime, grp in hdf5.items():
        ref_key = datetime if grp.attrs['subgrid'].lower() == "ppm" else datetime
        ref_N = grp.attrs['cells'] if grp.attrs['subgrid'].lower() == "ppm" else ref_N
    ref_timings = plot_timings_for_each_grp[ref_key]

    # Iterate through the list of timings generated by the number of snapshots
    for snap_index in range(snapshots):
        ref_time = ref_timings[snap_index]

        for idx, option in enumerate(options):
            fig, ax, plot_ = make_figure(option, sim_variables)
            legend_on = False
            if option == 'pressure':
                legend_on = True

            for _, datetime in enumerate(datetimes):
                grp = hdf5[datetime]

                cells = grp.attrs['cells']
                config = grp.attrs['config']
                dimension = grp.attrs['dimension']
                subgrid = grp.attrs['subgrid']
                solver = grp.attrs['solver']
                timestep = grp.attrs['timestep']

                timing = str(plot_timings_for_each_grp[datetime][snap_index])

                x = np.linspace(start_pos, end_pos, cells)
                y = make_data(option, grp[timing], sim_variables)

                # runtype was multiple
                if len(hdf5) != 1:
                    if dimension < 2:
                        #ax.plot(x, y, label=f"N = {cells}")
                        ax.plot(x, y, label=subgrid.upper())
                        #ax.plot(x, y, label=solver.upper())
                        #ax.plot(x, y, label=timestep.upper())

                else:
                    if dimension == 2:
                        graph = ax.imshow(y, interpolation="nearest", cmap=plot_['2d_colours'][idx], origin="lower")
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes('right', size='5%', pad=0.05)
                        cbar = plt.colorbar(graph, cax=cax, orientation='vertical')
                        cbar.set_label(plot_['label'], rotation='vertical')
                        ax.set_ylabel(r"$y$", rotation='vertical')

                    else:
                        ax.plot(x, y, color=plot_['1d_colours'][idx])

                ax.set_xlabel(r"$x$")
                if option == 'density' or option == 'pressure' or 'energy' in option:
                    ax.xaxis.tick_top()
                    ax.xaxis.set_label_position('top')
                if 'energy' in option or option == 'pressure':
                    ax.yaxis.tick_right()
                    ax.yaxis.set_label_position('right')
                #print(f"Option: {option}, config: {config}, cells: {cells}, subgrid: {subgrid}, solver: {solver}, timestep: {timestep}, time: {ref_time}\n")


            # Add analytical solutions only for 1D
            if dimension < 2:
                # Add analytical solution for smooth functions, using the highest resolution and timing
                if sim_variables.config_category == "smooth":
                    analytical = constructor.initialise(sim_variables)
                    y_theo = make_data(option, analytical, sim_variables)

                    ax.plot(x, y_theo, color=plot_['theo_colour'], linestyle='--', label=rf"{config.title()}$_{{theo}}$")

                # Add Sod or Sedov analytical solution, using the highest resolution and timing
                elif "sod" in config or "sedov" in config:
                    #_grid, _t = hdf5[ref_datetime][str(ref_time)], ref_time
                    _grid, _t = hdf5[ref_key][str(ref_time)], ref_time
                    
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
                        y_theo = make_data(option, soln, sim_variables)
                        ax.plot(x, y_theo, color=plot_['theo_colour'], linestyle='--', label=plot_label)

            #sim_variables = sim_variables._replace(cells=8192)
            #shu_osher_analytical = make_data(option, analytical_config, sim_variables)
            #ax.plot(analytical_grid, shu_osher_analytical, color='black', linestyle="--", label=r'$N = 8192$')

            if legend_on:
                if len(hdf5) > 5:
                    _ncol = 2
                else:
                    _ncol = 1
                handles, labels = plt.gca().get_legend_handles_labels()
                plt.legend(handles, labels, ncol=_ncol)

            plt.savefig(f"{save_path}/varPlot_{option}_{dimension}D_{config}_{subgrid}_{timestep}_{solver}_{'%.3f' % round(ref_time,3)}.png")

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

    plot_colors = ["red", "blue", "green", "purple"]
    for _,datetime in enumerate(datetimes):
        N = hdf5[datetime].attrs['cells']
        subgrid = hdf5[datetime].attrs['subgrid']
        sim_variables = sim_variables._replace(subgrid=subgrid)
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
            ax[_i,_j].plot(x, y_data[idx], linewidth=2, color=plot_colors[_], label=subgrid.upper())
            ax[_i,_j].set_xlim([min(x), max(x)])

        if dimension >= 2:
            grid_size = f"{N}^{dimension}"
        else:
            grid_size = N

        plt.suptitle(rf"Total variation of grid variables TV($\boldsymbol{{u}}$) against time $t$ ($N = {grid_size}$)", fontsize=30)
        fig.text(0.5, 0.04, rf"Time $t$ [arb. units]", fontsize=24, ha='center')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, prop={'size': 24}, loc='upper right')

    plt.savefig(f"{save_path}/TV_{config}_{subgrid}_{timestep}_{solver}_{N}.png", dpi=330)

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
            ax[_i,_j].plot(x, y_data[idx], linewidth=2, color=plot_['colours']['1d'][idx])
            ax[_i,_j].set_xlim([min(x), max(x)])

            # For plot annotation purposes
            y_init, y_final = y[0], y[-1]
            try:
                decimal_point = int(('%e' % abs(y_final-y_init)).split('-')[1])
            except IndexError:
                decimal_point = int(('%e' % abs(y_final-y_init)).split('+')[1])
            ax[_i,_j].annotate(round(y_init, decimal_point), (x[0], y_init), fontsize=18)
            ax[_i,_j].annotate(round(y_final, decimal_point), (x[-1], y_final), fontsize=18)
            print(y_init, " : ", y_final)

        if dimension >= 2:
            grid_size = f"{N}^{dimension}"
        else:
            grid_size = N

        plt.suptitle(rf"Conservation of variables ($m, p_x, E_{{tot}}$) against time $t$ for {config.title()} test ($N = {grid_size}$)", fontsize=30)
        fig.text(0.5, 0.04, rf"Time $t$ [arb. units]", fontsize=24, ha='center')

        plt.savefig(f"{save_path}/conserveEq_{config}_{subgrid}_{timestep}_{solver}_{N}.png", dpi=330)

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






def plot_solution_errors(folders, sim_variables, error_norm=1):
    import h5py
    mpl.rcParams['text.usetex'] = True

    for _, folder in enumerate(folders):

        # Set up figure
        colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
        fig, ax = plt.subplots(figsize=[7,11])
        plt.rcParams['text.latex.preamble'] = r"\usepackage{lmodern}"
        params = {'font.size': 12,
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

        for index, file in enumerate(os.listdir(f'{os.getcwd()}/{folder}')):
            if not file.startswith('.'):
                hdf5 = h5py.File(f'{os.getcwd()}/{folder}/{file}', 'r')

                datetimes = [datetime for datetime in hdf5.keys()]
                datetimes.sort()

                array = np.full((2, len(datetimes)), 0., dtype=sim_variables.precision)
                for idx, datetime in enumerate(datetimes):
                    config = hdf5[datetime].attrs['config']
                    subgrid = hdf5[datetime].attrs['subgrid']
                    _arr = [hdf5[datetime].attrs['cells']**1]

                    # Get last instance of the grid with largest time key
                    time_key = max([float(t) for t in hdf5[datetime].keys()])
                    sim_variables = sim_variables._replace(config=config)
                    if config.lower() == 'gaussian':
                        sim_variables = sim_variables._replace(start_pos=-1)
                        sim_variables = sim_variables._replace(t_end=2)
                        sim_variables = sim_variables._replace(initial_left=np.array([0,1,1,1,1e-6,0,0,0]))
                        sim_variables = sim_variables._replace(initial_right=np.array([0,1,1,1,1e-6,0,0,0]))
                        sim_variables = sim_variables._replace(misc={'peak_pos':0, 'ampl':.75, 'fwhm':.08, 'y_offset':1})
                    elif config.lower() == 'sine':
                        sim_variables = sim_variables._replace(start_pos=0)
                        sim_variables = sim_variables._replace(t_end=1)
                        sim_variables = sim_variables._replace(initial_left=np.array([0,1,1,1,1,0,0,0]))
                        sim_variables = sim_variables._replace(initial_right=np.array([0,1,1,1,1,0,0,0]))
                        sim_variables = sim_variables._replace(misc={'freq':2, 'ampl':.1, 'y_offset':2})
                    solution_errors = analytic.calculate_solution_error(hdf5[datetime][str(time_key)], sim_variables, error_norm)

                    #array[cells] = solution_errors[0]
                    array[...,idx] = np.asarray(_arr, dtype=sim_variables.precision)


            x = array[:1].ravel()
            y = array[1:]
            x.sort()
            hdf5.close()


            EOC = np.diff(np.log(y))/np.diff(np.log(x))
            _idx = np.argmin(np.abs(np.average(EOC)-EOC))
            c = np.log10(y[_idx]) - EOC[_idx]*np.log10(x[_idx])

            if "ppm" in file:
                for order in [1,2,4,5,7]:
                    alpha = 10**c
                    ytheo = alpha*x**(-order)
                    ax.loglog(x, ytheo, color='black', linestyle="--")
                    ax.annotate(rf"$O(N^{order})$", (x[-1], ytheo[-1]))

            ax.loglog(x, y, linestyle="-", marker="o", color=colours[index], label=rf"{subgrid.upper()}, $|\text{{EOC}}_\text{{max}}|$ = {round(max(np.abs(EOC)), 4)}")


        ax.legend()
        ax.set_xlim([min(x)/1.5,max(x)*2])
        ax.set_xlabel(r"Resolution $N$")
        ax.set_ylabel(r"$\epsilon_N(\rho)$ [arb. units]", rotation='vertical')
        ax.grid(linestyle="--", linewidth=0.5)
        if folder == 'gaussian':
            ax.set_title(rf"Gaussian wave")
        else:
            ax.set_title(rf"sine-wave")
            ax.yaxis.ax.yaxis.tick_right()
            ax.yaxis.set_label_position('right')
    

        plt.savefig(f"solErr_L{error_norm}.png")

        plt.cla()
        plt.clf()
        plt.close()








def dual_plot(paths, sim_variables, error_norm=1):
    options = ["density", "density"]
    plot_ = make_weird_figure(options)
    config, dimension, subgrid, timestep, solver = sim_variables.config, sim_variables.dimension, sim_variables.subgrid, sim_variables.timestep, sim_variables.solver


    #mpl.rcParams['text.usetex'] = True

    plt.rcParams['text.latex.preamble'] = r"\usepackage{lmodern}"
    params = {'font.size': 12, 'font.family': 'DejaVuSans', 'axes.labelsize': 12, 'axes.titlesize': 12, 'legend.fontsize': 12, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'figure.dpi': 300, 'savefig.dpi': 300, 'lines.linewidth': 1.0, 'lines.dashed_pattern': [3, 2]}
    plt.rcParams.update(params)

    fig, ax = plt.subplots(figsize=[11,7], nrows=1, ncols=2)

    import h5py

    for path_i, _ in enumerate(paths):
        hdf5 = h5py.File(f'gaussian/{_}','r')

        for something, something_else in hdf5.items():
            _config = something_else.attrs['subgrid'].upper()

        # hdf5 keys are datetime strings
        datetimes = [datetime for datetime in hdf5.keys()]
        datetimes.sort()

        array = np.full((1+len(options), len(datetimes)), 0., dtype=sim_variables.precision)
        for idx, datetime in enumerate(datetimes):
            _arr = [hdf5[datetime].attrs['cells']**dimension]

            # Get last instance of the grid with largest time key
            time_key = max([float(t) for t in hdf5[datetime].keys()])
            sim_variables = sim_variables._replace(config='gaussian')
            sim_variables = sim_variables._replace(start_pos=-1)
            sim_variables = sim_variables._replace(t_end=2)
            sim_variables = sim_variables._replace(initial_left=np.array([0,1,1,1,1e-6,0,0,0]))
            sim_variables = sim_variables._replace(initial_right=np.array([0,1,1,1,1e-6,0,0,0]))
            sim_variables = sim_variables._replace(misc={'peak_pos':0, 'ampl':.75, 'fwhm':.08, 'y_offset':1})
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
        x = array[:1].ravel()
        y_data1 = array[1:]
        x.sort()
        hdf5.close()


        hdf5 = h5py.File(f'sine/{_}','r')

        for something, something_else in hdf5.items():
            _config = something_else.attrs['subgrid'].upper()

        # hdf5 keys are datetime strings
        datetimes = [datetime for datetime in hdf5.keys()]
        datetimes.sort()

        array = np.full((1+len(options), len(datetimes)), 0., dtype=sim_variables.precision)
        for idx, datetime in enumerate(datetimes):
            _arr = [hdf5[datetime].attrs['cells']**dimension]

            # Get last instance of the grid with largest time key
            time_key = max([float(t) for t in hdf5[datetime].keys()])
            sim_variables = sim_variables._replace(config='sine')
            sim_variables = sim_variables._replace(start_pos=0)
            sim_variables = sim_variables._replace(t_end=1)
            sim_variables = sim_variables._replace(initial_left=np.array([0,1,1,1,1,0,0,0]))
            sim_variables = sim_variables._replace(initial_right=np.array([0,1,1,1,1,0,0,0]))
            sim_variables = sim_variables._replace(misc={'freq':2, 'ampl':.1, 'y_offset':2})
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
        x = array[:1].ravel()
        y_data2 = array[1:]
        x.sort()
        hdf5.close()

        for idx, (_i,_j) in enumerate(plot_['indexes']):
            if _j == 0:
                y = y_data1[idx]
            else:
                y = y_data2[idx]

            EOC = np.diff(np.log(y))/np.diff(np.log(x))
            _idx = np.argmin(np.abs(np.average(EOC)-EOC))
            c = np.log10(y[_idx]) - EOC[_idx]*np.log10(x[_idx])

            if "ppm" in _:
                for order in [1,2,4,5,7]:
                    alpha = 10**c
                    ytheo = alpha*x**(-order)
                    ax[_j].loglog(x, ytheo, color="black", linestyle="--")
                    ax[_j].annotate(rf"$O(N^{order})$", xy=(x[-1], ytheo[-1]), xytext=(5,-5), textcoords='offset points')

            ax[_j].loglog(x, y, linestyle="-", marker="o", color=plot_['colours']['1d'][path_i], label=rf"{_config}, $|$EOC$_\mathrm{{max}}$$|$ = {round(max(np.abs(EOC)), 4)}")
            ax[_j].legend()
            ax[_j].set_xlim([min(x)/1.5,max(x)*2.5])
            ax[_j].set_xlabel(r"Resolution $N$")
            ax[_j].grid(linestyle="--", linewidth=0.5)
            if _j == 0:
                ax[_j].set_title(r"Gaussian function")
                ax[_j].set_ylabel(r"$\epsilon_N(\rho)$ [arb. units]", rotation='vertical')
            else:
                ax[_j].set_title(r"sine-wave")
                ax[_j].yaxis.tick_right()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(f"solErr_L{error_norm}.png", bbox_inches='tight')

    plt.cla()
    plt.clf()
    plt.close()









def make_weird_figure(options, style=STYLE, **kwargs):
    if 0 < len(options) < 13:
        theo_colour = "black"
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

        return {'indexes':indexes, 'labels':labels, 'errors':errors, 'tvs':tvs, 'colours': {'theo':theo_colour, '1d':colours, '2d':twod_colours}}
    else:
        raise IndexError('Number of variables to plot should be < 13')