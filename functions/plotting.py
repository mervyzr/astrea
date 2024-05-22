import os
import shutil
import platform
import subprocess

import numpy as np
import moviepy.video.io.ImageSequenceClip
if platform.system() == "Darwin":
    if platform.machine() == "arm64" and platform.mac_ver()[0] > '10.15.7':
        import matplotlib
        from settings import savePlots, saveVideo
        if savePlots or saveVideo:
            matplotlib.use('Agg')
        else:
            matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon

from functions import generic, analytic

##############################################################################

_style = "default"
beautify = False


plotIndexes = [[0,0], [0,1], [1,0], [1,1]]
plotLabels = [[r"Density $\rho$", r"Pressure $P$"], [r"Velocity $v_x$", r"Specific thermal energy $\frac{P}{\rho}$"]]
try:
    plt.style.use(_style)
except Exception as e:
    plt.style.use("default")
    colours = [["blue", "red"], ["green", "darkviolet"]]
    pass
else:
    if _style != "default":
        _color = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colours = [_color[:2], _color[2:4]]
    else:
        colours = [["blue", "red"], ["green", "darkviolet"]]


# Initiate the live plot feature
def initiateLivePlot(simVariables):
    startPos, endPos, N = simVariables.startPos, simVariables.endPos, simVariables.cells

    plt.ion()
    fig, ax = plt.subplots(nrows=2, ncols=2)

    plot_x = np.linspace(startPos, endPos, N)

    ax[0,1].yaxis.set_label_position("right")
    ax[1,1].yaxis.set_label_position("right")

    ax[0,1].yaxis.tick_right()
    ax[1,1].yaxis.tick_right()

    graphs = []
    for _i, _j in plotIndexes:
        ax[_i,_j].set_ylabel(plotLabels[_i][_j])
        ax[_i,_j].set_xlim([startPos, endPos])
        ax[_i,_j].grid(linestyle='--', linewidth=0.5)
        graph, = ax[_i,_j].plot(plot_x, plot_x, linewidth=2, color=colours[_i][_j])
        graphs.append(graph)

    return fig, ax, graphs


# Update live plot
def updatePlot(arr, t, fig, ax, graphs):
    graphTL, graphTR, graphBL, graphBR = graphs

    graphTL.set_ydata(arr[:,0])  # density
    graphTR.set_ydata(arr[:,4])  # pressure
    graphBL.set_ydata(arr[:,1])  # vx
    graphBR.set_ydata(arr[:,4]/arr[:,0])  # specific thermal energy
    #graphBR.set_ydata(analytic.calculateEntropyDensity(arr, 1.4))  # scaled entropy density

    for _i, _j in plotIndexes:
        ax[_i,_j].relim()
        ax[_i,_j].autoscale_view()

    plt.suptitle(rf"Primitive variables $\vec{{w}}$ against cell position $x$ at $t = {round(t,4)}$")
    fig.text(0.5, 0.04, r"Cell position $x$", ha='center')
    fig.canvas.draw()
    fig.canvas.flush_events()
    pass


# Plot snapshots of quantities for multiple runs
def plotQuantities(f, simVariables, savepath):
    config, subgrid, timestep, precision, snapshots = simVariables.config, simVariables.subgrid, simVariables.timestep, simVariables.precision, simVariables.snapshots
    startPos, endPos, freq, initialLeft = simVariables.startPos, simVariables.endPos, simVariables.freq, simVariables.initialLeft

    # hdf5 keys are string; need to convert back to int and sort again
    nList = [int(n) for n in f.keys()]
    nList.sort()

    # Separate the timings based on the number of snapshots; returns a list of lists with the timing intervals for each simulation
    indexes = []
    for i, N in enumerate(nList):
        timings = np.fromiter(f[str(N)].keys(), dtype=precision)
        timings.sort()
        indexes.append([timing[-1] for timing in np.array_split(timings, abs(int(snapshots)))])

    # Iterate through the timings; the last set of timings refer to the highest resolution
    for i in range(len(indexes[-1])):
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=[21, 10])

        # Set up figure
        for _i, _j in plotIndexes:
            ax[_i,_j].set_ylabel(plotLabels[_i][_j], fontsize=18)
            ax[_i,_j].set_xlim([startPos, endPos])
            ax[_i,_j].grid(linestyle="--", linewidth=0.5)

        # Plot each simulation at the i-th timing
        for j, N in enumerate(nList):
            time_key = str(indexes[j][i])
            y1 = f[str(N)][time_key][:,0]   # density
            y2 = f[str(N)][time_key][:,4]   # pressure
            y3 = f[str(N)][time_key][:,1]   # vx
            y4 = y2/y1  # specific thermal energy
            x = np.linspace(startPos, endPos, N)
            y_data = [[y1, y2], [y3, y4]]

            # density, pressure, vx, thermal energy
            for _i, _j in plotIndexes:
                if len(f) != 1:
                    ax[_i,_j].plot(x, y_data[_i][_j], linewidth=2, label=f"N = {N}")
                    plt.suptitle(rf"Primitive variables $\vec{{w}}$ against cell position $x$ at $t \approx {round(indexes[-1][i],3)}$", fontsize=24)
                else:
                    if beautify:
                        gradient_plot([x, y_data[_i][_j]], [_i,_j], ax, linewidth=2, color=colours[_i][_j])
                    else:
                        #ax[_i,_j].plot(x, y_data[_i][_j], linewidth=2, linestyle="-", marker="D", ms=4, markerfacecolor=fig.get_facecolor(), markeredgecolor=colours[_i][_j], color=colours[_i][_j])
                        ax[_i,_j].plot(x, y_data[_i][_j], linewidth=2, color=colours[_i][_j])
                    plt.suptitle(rf"Primitive variables $\vec{{w}}$ against cell position $x$ at $t \approx {round(indexes[-1][i],3)}$ ($N = {N}$)", fontsize=24)

        # Adjust ylim and plot analytical solutions for Gaussian, sin-wave and sinc-wave tests
        if config.startswith("sin") or config.startswith("gaussian"):
            last_sim = f[list(f.keys())[-1]]
            first_config = last_sim[list(last_sim.keys())[0]][0]
            midpoint = (endPos+startPos)/2

            analytical = np.zeros((N, len(initialLeft)), dtype=precision)
            analytical[:] = initialLeft
            if config.startswith("gaussian"):
                analytical[:,0] = 1e-3 + (1-1e-3) * np.exp(-(x-midpoint)**2/.01)
                Ptol = 5e-7
            else:
                Ptol = .005
                if config == "sinc":
                    analytical[:,0] = np.sinc(x * freq/np.pi) + 1
                else:
                    analytical[:,0] = 1 + (.1 * np.sin(freq*np.pi*x))

            Prange = np.linspace(initialLeft[4]-Ptol, initialLeft[4]+Ptol, 9)
            vrange = np.linspace(initialLeft[1]-.005, initialLeft[1]+.005, 9)
            ax[0,1].set_yticks(Prange)
            ax[1,0].set_yticks(vrange)
            ax[0,1].set_ylim([initialLeft[4]-Ptol, initialLeft[4]+Ptol])
            ax[1,0].set_ylim([initialLeft[1]-.005, initialLeft[1]+.005])

            y_theo = [[analytical[:, 0], analytical[:, 4]], [analytical[:, 1], analytical[:, 4]/analytical[:, 0]]]
            for _i, _j in plotIndexes:
                ax[_i,_j].plot(x, y_theo[_i][_j], linewidth=1, color="black", linestyle="--", label="Analytical solution")

        # Add Sod analytical solution, using the highest resolution and timing
        elif config == "sod":
            tube, _t = f[str(nList[-1])][str(indexes[-1][i])], indexes[-1][i]
            Sod = analytic.calculateSodAnalytical(tube, _t, simVariables)

            y_theo = [[Sod[:, 0], Sod[:, 4]], [Sod[:, 1], Sod[:, 4]/Sod[:, 0]]]
            for _i, _j in plotIndexes:
                ax[_i,_j].plot(x, y_theo[_i][_j], linewidth=1, color="black", linestyle="--", label="Analytical solution")

        # Add Sedov analytical solution, using the highest resolution and timing
        elif config == "sedov":
            pass

        fig.text(0.5, 0.04, r"Cell position $x$", fontsize=18, ha='center')
        if len(f) != 1 or config == "sod" or config.startswith("gauss") or config.startswith("sin"):
            handles, labels = plt.gca().get_legend_handles_labels()
            fig.legend(handles, labels, prop={'size': 16}, loc='upper right')

        plt.savefig(f"{savepath}/wPlot_{config}_{subgrid}_{timestep}_{round(indexes[-1][i],3)}.png", dpi=330)

        plt.cla()
        plt.clf()
        plt.close()
    return None


def plotSolutionErrors(f, simVariables, savepath, prop_coeff=10, norm=1):
    config, subgrid, timestep = simVariables.config, simVariables.subgrid, simVariables.timestep
    startPos, endPos, freq = simVariables.startPos, simVariables.endPos, simVariables.freq

    # hdf5 keys are string; need to convert back to int and sort again
    nList = [int(n) for n in f.keys()]
    nList.sort()

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=[21,10])
    errorLabels = [r"Density $\log{(\epsilon_\nu(\rho))}$", r"Thermal energy $\log{(\epsilon_\nu(\frac{P}{\rho}))}$"]

    for _j in [0,1]:
        ax[_j].set_ylabel(errorLabels[_j], fontsize=18)
        ax[_j].grid(linestyle="--", linewidth=0.5)

    x, y1, y2, y3, y4 = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    for N in nList:
        x = np.append(x, f[str(N)].attrs['cells'])
        solutionErrors = analytic.calculateSolutionError(f[str(N)], simVariables, norm)
        y1 = np.append(y1, solutionErrors[0])  # density
        y2 = np.append(y2, solutionErrors[-1])  # specific thermal energy
        y3 = np.append(y3, solutionErrors[4])  # pressure
        y4 = np.append(y4, solutionErrors[1])  # vx
    y_data = [[y1, y2], [y3, y4]]

    for _i, _j in plotIndexes:
        m, c = np.polyfit(np.log10(x), np.log10(y_data[_i][_j]), 1)
        if _i == 0:
            for order in [1,2,4]:
                alpha = 10**(c + np.log10(prop_coeff))
                ytheo = alpha*x**(-order)
                ax[_j].loglog(x, ytheo, linewidth=1, color="black", linestyle="--")
                ax[_j].annotate(rf"$O(N^{order})$", (x[-1], ytheo[-1]), fontsize=12)
            ax[_j].loglog(x, y_data[_i][_j], linewidth=2, linestyle="--", marker="o", color=colours[_i][_j], label=f"grad. = {round(m,4)}")
            ax[_j].scatter([], [], s=.5, color=fig.get_facecolor(), label=rf"$|\text{{EOC}}_{{max}}|$ = {round(max(np.abs(np.diff(np.log(y_data[_i][_j]))/np.diff(np.log(x)))), 4)}")
            ax[_j].legend(prop={'size': 14})

    print(f"{generic.bcolours.OKGREEN}EOC (density){generic.bcolours.ENDC}: {np.diff(np.log(y1))/np.diff(np.log(x))}\n{generic.bcolours.OKGREEN}EOC (pressure){generic.bcolours.ENDC}: {np.diff(np.log(y2))/np.diff(np.log(x))}\n{generic.bcolours.OKGREEN}EOC (vx){generic.bcolours.ENDC}: {np.diff(np.log(y3))/np.diff(np.log(x))}\n{generic.bcolours.OKGREEN}EOC (thermal){generic.bcolours.ENDC}: {np.diff(np.log(y4))/np.diff(np.log(x))}")

    plt.suptitle(rf"$L_{norm}$ solution error norm $\epsilon_\nu(\vec{{w}})$ against resolution $N_\nu$ for {config.title()} test", fontsize=24)
    fig.text(0.5, 0.04, r"Resolution $\log{(N_\nu)}$", fontsize=18, ha='center')

    plt.savefig(f"{savepath}/solErr_L{norm}_{subgrid}_{timestep}.png", dpi=330, facecolor="w")

    plt.cla()
    plt.clf()
    plt.close()
    return None


def plotTotalVariation(f, simVariables, savepath):
    config, subgrid, timestep = simVariables.config, simVariables.subgrid, simVariables.timestep

    # hdf5 keys are string; need to convert back to int and sort again
    nList = [int(n) for n in f.keys()]
    nList.sort()

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=[21,10])
    tvLabels = [[r"Density TV($\rho$)", r"Pressure TV($P$)"], [r"Velocity TV($v_x$)", r"Thermal energy TV($\frac{P}{\rho}$)"]]

    for _i, _j in plotIndexes:
        ax[_i,_j].set_ylabel(tvLabels[_i][_j], fontsize=18)
        ax[_i,_j].grid(linestyle="--", linewidth=0.5)

    for N in nList:
        tvDict = analytic.calculateTV(f[str(N)])
        x = np.asarray(list(tvDict.keys()))
        y = np.asarray(list(tvDict.values()))
        y1 = y[:,0]  # density
        y2 = y[:,4]  # pressure
        y3 = y[:,1]  # vx
        y4 = y[:,-1]  # specific thermal energy
        y_data = [[y1, y2], [y3, y4]]
        x.sort()

        for _i, _j in plotIndexes:
            ax[_i,_j].plot(x, y_data[_i][_j], linewidth=2, color=colours[_i][_j])

        plt.suptitle(rf"Total variation of primitive variables TV($\vec{{w}}$) against time $t$ for {config.title()} test ($N = {N}$)", fontsize=24)
        fig.text(0.5, 0.04, r"Time $t$", fontsize=18, ha='center')

        plt.savefig(f"{savepath}/TV_{config}_{subgrid}_{timestep}_{N}.png", dpi=330)

        plt.cla()
        plt.clf()
        plt.close()
    return None


def plotConservationEquations(f, simVariables, savepath):
    config, gamma, subgrid, timestep = simVariables.config, simVariables.gamma, simVariables.subgrid, simVariables.timestep
    startPos, endPos = simVariables.startPos, simVariables.endPos

    # hdf5 keys are string; need to convert back to int and sort again
    nList = [int(n) for n in f.keys()]
    nList.sort()

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=[21,10])
    eqLabels = [r"Mass ($m$)", r"Momentum ($p_x$)", r"Energy ($E_{tot}$)"]

    for _j in [0,1,2]:
        ax[_j].set_ylabel(eqLabels[_j], fontsize=18)
        ax[_j].grid(linestyle="--", linewidth=0.5)

    for N in nList:
        eqDict = analytic.calculateConservation(f[str(N)], simVariables)
        x = np.asarray(list(eqDict.keys()))
        y = np.asarray(list(eqDict.values()))
        y1 = y[:,0]  # mass
        y2 = y[:,4]  # total energy
        y3 = y[:,1]  # momentum_x
        y4 = y[:,5]  # B*vol_x
        y_data = [[y1, y2], [y3, y4]]
        x.sort()

        for _i, _j in plotIndexes:
            y_i, y_f = y_data[_i][_j][0], y_data[_i][_j][-1]
            try:
                decimalPoint = int(('%e' % abs(y_f-y_i)).split('-')[1])
            except IndexError:
                decimalPoint = int(('%e' % abs(y_f-y_i)).split('+')[1])
            if _i == 0:
                ax[_j].plot(x, y_data[_i][_j], linewidth=2, color=colours[_i][_j])
                ax[_j].annotate(round(y_i, decimalPoint), (x[0], y_i), fontsize=12)
                ax[_j].annotate(round(y_f, decimalPoint), (x[-1], y_f), fontsize=12)
            elif _i == 1 and _j == 0:
                ax[2].plot(x, y_data[_i][_j], linewidth=2, color=colours[_i][_j])
                ax[2].annotate(round(y_i, decimalPoint), (x[0], y_i), fontsize=12)
                ax[2].annotate(round(y_f, decimalPoint), (x[-1], y_f), fontsize=12)

        plt.suptitle(rf"Conservation of variables ($m, p_x, E_{{tot}}$) against time $t$ for {config.title()} test ($N = {N}$)", fontsize=24)
        fig.text(0.5, 0.04, r"Time $t$", fontsize=18, ha='center')

        plt.savefig(f"{savepath}/conserveEq_{config}_{subgrid}_{timestep}_{N}.png", dpi=330)

        plt.cla()
        plt.clf()
        plt.close()
        return None


def makeVideo(f, simVariables, savepath, vidpath):
    config, subgrid, timestep = simVariables.config, simVariables.subgrid, simVariables.timestep
    startPos, endPos = simVariables.startPos, simVariables.endPos

    # hdf5 keys are string; need to convert back to int and sort again
    nList = [int(n) for n in f.keys()]
    nList.sort()

    for N in nList:
        simulation = f[str(N)]
        counter = 0

        for t, domain in simulation.items():
            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=[21, 10])

            for _i, _j in plotIndexes:
                ax[_i,_j].set_ylabel(plotLabels[_i][_j], fontsize=18)
                ax[_i,_j].set_xlim([startPos, endPos])
                ax[_i,_j].grid(linestyle="--", linewidth=0.5)

            y1 = domain[:,0]  # density
            y2 = domain[:,4]  # pressure
            y3 = domain[:,1]  # vx
            y4 = domain[:,4]/domain[:,0]  # specific thermal energy
            x = np.linspace(startPos, endPos, N)
            y_data = [[y1, y2], [y3, y4]]

            for _i, _j in plotIndexes:
                if beautify:
                    gradient_plot([x, y_data[_i][_j]], [_i,_j], ax, linewidth=2, color=colours[_i][_j])
                else:
                    ax[_i,_j].plot(x, y_data[_i][_j], linewidth=2, color=colours[_i][_j])

            plt.suptitle(rf"Primitive variables $\vec{{w}}$ against cell position $x$ at $t = {round(float(t),4)}$ ($N = {N}$)", fontsize=24)
            fig.text(0.5, 0.04, r"Cell position $x$", fontsize=18, ha='center')

            plt.savefig(f"{vidpath}/{str(counter).zfill(4)}.png", dpi=330, facecolor="w")

            plt.cla()
            plt.clf()
            plt.close()

            counter += 1

        try:
            subprocess.call(["ffmpeg", "-framerate", "60", "-pattern_type", "glob", "-i", f"{vidpath}/*.png", "-c:v", "libx264", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", f"{savepath}/vid_{config}_{subgrid}_{timestep}.mp4"])
        except Exception as e:
            print(f"{generic.bcolours.WARNING}ffmpeg failed: {e}{generic.bcolours.ENDC}")
            try:
                images = [os.path.join(vidpath,img) for img in os.listdir(vidpath) if img.endswith(".png")]
                images.sort()

                video = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(images, fps=60)
                video.write_videofile(f"{savepath}/vid_{config}_{subgrid}_{timestep}.mp4")
            except Exception as e:
                print(f"{generic.bcolours.WARNING}moviepy failed: {e}{generic.bcolours.ENDC}")
                pass
                print(f"{generic.bcolours.FAIL}Video creation failed{generic.bcolours.ENDC}")
            else:
                shutil.rmtree(vidpath)
        else:
            shutil.rmtree(vidpath)
    return None


# Useful function for plotting each instance of the domain (livePlot must be switched OFF)
def plotInstance(domain, showPlot=True, text="", startPos=0, endPos=1, **kwargs):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=[21, 10])

    for index, (_i,_j) in enumerate(plotIndexes):
        ax[_i,_j].set_ylabel(plotLabels[index], fontsize=18)
        ax[_i,_j].set_xlim([startPos, endPos])
        ax[_i,_j].grid(linestyle="--", linewidth=0.5)

    y1 = domain[:,0]   # density
    y2 = domain[:,4]   # pressure
    y3 = domain[:,1]   # vx
    y4 = y2/y1  # specific thermal energy
    x = np.linspace(startPos, endPos, len(y1))
    y_data = [[y1, y2], [y3, y4]]

    for _i, _j in plotIndexes:
        if beautify:
            gradient_plot([x, y_data[_i][_j]], [_i,_j], ax, linewidth=2, color=colours[_i][_j])
        else:
            ax[_i,_j].plot(x, y_data[_i][_j], linewidth=2, color=colours[_i][_j])
    plt.suptitle(rf"Primitive variables $\vec{{w}}$ against cell position $x$ {text}", fontsize=24)
    fig.text(0.5, 0.04, r"Cell position $x$", fontsize=18, ha='center')

    if showPlot:
        plt.show(block=True)
    else:
        step = kwargs['step']
        seed = kwargs['seed']
        plt.savefig(f"{seed}_{step}_{text.replace(' ','').title()}.png", dpi=330, facecolor="w")

    plt.cla()
    plt.clf()
    plt.close()
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