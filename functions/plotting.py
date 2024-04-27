import os
import shutil
import subprocess

import matplotlib
from settings import saveVideo, savePlots
if savePlots or saveVideo:
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import moviepy.video.io.ImageSequenceClip
from matplotlib.patches import Polygon

from functions import generic, analytic

##############################################################################

plt.style.use("default")
beautify = False


plotIndexes = [[0,0], [0,1], [1,0], [1,1]]
plotLabels = [r"Density $\rho$", r"Pressure $P$", r"Velocity $v_x$", r"Thermal energy $\frac{P}{\rho}$"]
colours = ["blue", "red", "green", "darkviolet"]


# Initiate the live plot feature
def initiateLivePlot(startPos, endPos, N):
    plt.ion()
    fig, ax = plt.subplots(nrows=2, ncols=2)

    plot_x = np.linspace(startPos, endPos, N)

    ax[0,1].yaxis.set_label_position("right")
    ax[1,1].yaxis.set_label_position("right")

    ax[0,1].yaxis.tick_right()
    ax[1,1].yaxis.tick_right()

    graphs = []
    for index, (_i,_j) in enumerate(plotIndexes):
        ax[_i,_j].set_ylabel(plotLabels[index])
        ax[_i,_j].set_xlim([startPos, endPos])
        ax[_i,_j].grid(linestyle='--', linewidth=0.5)
        graph, = ax[_i,_j].plot(plot_x, plot_x, linewidth=2, color=colours[index])
        graphs.append(graph)

    return fig, ax, graphs


# Update live plot
def updatePlot(arr, t, fig, ax, graphs):
    graphTL, graphTR, graphBL, graphBR = graphs

    graphTL.set_ydata(arr[:,0])  # density
    graphTR.set_ydata(arr[:,4])  # pressure
    graphBL.set_ydata(arr[:,1])  # vx
    graphBR.set_ydata(arr[:,4]/arr[:,0])  # thermal energy

    for _i, _j in plotIndexes:
        ax[_i,_j].relim()
        ax[_i,_j].autoscale_view()

    plt.suptitle(rf"Primitive variables $\vec{{w}}$ against cell position $x$ at $t = {round(t,4)}$")
    fig.text(0.5, 0.04, r"Cell position $x$", ha='center')
    fig.canvas.draw()
    fig.canvas.flush_events()
    pass


# Plot snapshots of quantities for multiple runs
def plotQuantities(f, configVariables, testVariables, savepath):
    config, gamma, solver, timestep = configVariables['config'], configVariables['gamma'], configVariables['solver'], configVariables['timestep']
    startPos, endPos, shockPos = testVariables['startPos'], testVariables['endPos'], testVariables['shockPos']

    # hdf5 keys are string; need to convert back to int and sort again
    nList = [int(n) for n in f.keys()]
    nList.sort()

    # Separate the timings based on the number of snapshots; returns a list of lists with the timing intervals for each simulation
    indexes = []
    for i, N in enumerate(nList):
        timings = np.fromiter(f[str(N)].keys(), dtype=np.float64)
        timings.sort()
        indexes.append([timing[-1] for timing in np.array_split(timings, abs(int(configVariables['snapshots'])))])

    # Iterate through the timings; the last set of timings refer to the highest resolution
    for i in range(len(indexes[-1])):
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=[21, 10])

        # Set up figure
        for index, (_i,_j) in enumerate(plotIndexes):
            ax[_i,_j].set_ylabel(plotLabels[index], fontsize=18)
            ax[_i,_j].set_xlim([startPos, endPos])
            ax[_i,_j].grid(linestyle="--", linewidth=0.5)

        # Plot each simulation at the i-th timing
        for j, N in enumerate(nList):
            time_key = str(indexes[j][i])
            y1 = f[str(N)][time_key][:, 0]   # density
            y2 = f[str(N)][time_key][:, 4]   # pressure
            y3 = f[str(N)][time_key][:, 1]   # vx
            y4 = y2/y1                       # thermal energy
            x = np.linspace(startPos, endPos, N)
            y_data = [y1, y2, y3, y4]

            # density, pressure, vx, thermal energy
            for index, (_i,_j) in enumerate(plotIndexes):
                if len(f) != 1:
                    ax[_i,_j].plot(x, y_data[index], linewidth=2, label=f"N = {N}")
                    plt.suptitle(rf"Primitive variables $\vec{{w}}$ against cell position $x$ at $t \approx {round(indexes[-1][i],3)}$", fontsize=24)
                else:
                    if beautify:
                        gradient_plot([x,y_data[index]], [_i,_j], ax=ax, linewidth=2, color=colours[index])
                    else:
                        ax[_i,_j].plot(x, y_data[index], linewidth=2, color=colours[index])
                    plt.suptitle(rf"Primitive variables $\vec{{w}}$ against cell position $x$ at $t \approx {round(indexes[-1][i],3)}$ ($N = {len(y1)}$)", fontsize=24)

        # Adjust ylim and plot analytical solutions for Gaussian and sin-wave tests
        if config.startswith("sin") or config.startswith("gaussian"):
            last_sim = f[list(f.keys())[-1]]
            first_config = last_sim[list(last_sim.keys())[0]][0]
            initialConfig = testVariables['initialLeft']
            midpoint = (endPos+startPos)/2

            analytical = np.zeros((N, len(initialConfig)), dtype=np.float64)
            analytical[:] = initialConfig
            if config.startswith("gaussian"):
                analytical[:,0] = 1e-3 + (1-1e-3) * np.exp(-(x-midpoint)**2/.01)
                Ptol = 5e-7
            else:
                analytical[:,0] = 1 + (.1 * np.sin(testVariables['freq']*np.pi*x))
                Ptol = .005

            Prange = np.linspace(initialConfig[4]-Ptol, initialConfig[4]+Ptol, 9)
            vrange = np.linspace(initialConfig[1]-.005, initialConfig[1]+.005, 9)
            ax[0,1].set_yticks(Prange)
            ax[1,0].set_yticks(vrange)
            ax[0,1].set_ylim([initialConfig[4]-Ptol, initialConfig[4]+Ptol])
            ax[1,0].set_ylim([initialConfig[1]-.005, initialConfig[1]+.005])

            y_theo = [analytical[:, 0], analytical[:, 4], analytical[:, 1], analytical[:, 4]/analytical[:, 0]]
            for index, (_i,_j) in enumerate(plotIndexes):
                ax[_i,_j].plot(x, y_theo[index], linewidth=1, color="black", linestyle="--", label="Analytical solution")

        # Add Sod analytical solution, using the highest resolution and timing
        elif config == "sod":
            tube, _t = f[str(nList[-1])][str(indexes[-1][i])], indexes[-1][i]
            Sod = analytic.calculateSodAnalytical(tube, _t,  gamma, startPos, endPos, shockPos)

            y_theo = [Sod[:, 0], Sod[:, 4], Sod[:, 1], Sod[:, 4]/Sod[:, 0]]
            for index, (_i,_j) in enumerate(plotIndexes):
                ax[_i,_j].plot(x, y_theo[index], linewidth=1, color="black", linestyle="--", label="Analytical solution")

        # Add Sedov analytical solution, using the highest resolution and timing
        elif config == "sedov":
            pass

        fig.text(0.5, 0.04, r"Cell position $x$", fontsize=18, ha='center')
        if len(f) != 1 or config == "sod" or config.startswith("gauss") or config.startswith("sin"):
            handles, labels = plt.gca().get_legend_handles_labels()
            fig.legend(handles, labels, prop={'size': 16}, loc='upper right')

        plt.savefig(f"{savepath}/wPlot_{config}_{solver}_{timestep}_{round(indexes[-1][i],3)}.png", dpi=330, facecolor="w")

        plt.cla()
        plt.clf()
        plt.close()
    return None


def plotSolutionErrors(f, configVariables, testVariables, savepath):
    config, solver, timestep = configVariables['config'], configVariables['solver'], configVariables['timestep']
    startPos, endPos = testVariables['startPos'], testVariables['endPos']

    # hdf5 keys are string; need to convert back to int and sort again
    nList = [int(n) for n in f.keys()]
    nList.sort()

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=[21, 10])
    errorLabels = [r"Density $\log{(\epsilon_\nu(\rho))}$", r"Pressure $\log{(\epsilon_\nu(P))}$", r"Velocity $\log{(\epsilon_\nu(v_x))}$", r"Thermal energy $\log{(\epsilon_\nu(\frac{P}{\rho}))}$"]

    for index, (_i,_j) in enumerate(plotIndexes):
        ax[_i,_j].set_ylabel(errorLabels[index], fontsize=18)
        ax[_i,_j].grid(linestyle="--", linewidth=0.5)

    x, y1, y2, y3, y4 = [], [], [], [], []
    for N in nList:
        x.append(f[str(N)].attrs['cells'])
        solutionErrors = analytic.calculateSolutionError(f[str(N)], startPos, endPos, config)
        y1.append(solutionErrors[0])  # density
        y2.append(solutionErrors[4])  # pressure
        y3.append(solutionErrors[1])  # vx
        y4.append(solutionErrors[-1])  # thermal energy
    x, y1, y2, y3, y4 = np.asarray(x), np.asarray(y1), np.asarray(y2), np.asarray(y3), np.asarray(y4)
    y_data = [y1, y2, y3, y4]

    for index, (_i,_j) in enumerate(plotIndexes):
        m, c = np.polyfit(np.log10(x), np.log10(y_data[index]), 1)
        ax[_i,_j].loglog(x, y_data[index], linewidth=2, linestyle="--", marker="o", color=colours[index], label=f"grad. = {round(m,4)}")
        ax[_i,_j].legend(prop={'size': 14})

    print(f"{generic.bcolours.OKGREEN}EOC (density){generic.bcolours.ENDC}: {np.diff(np.log(y1))/np.diff(np.log(x))}\n{generic.bcolours.OKGREEN}EOC (pressure){generic.bcolours.ENDC}: {np.diff(np.log(y2))/np.diff(np.log(x))}\n{generic.bcolours.OKGREEN}EOC (vx){generic.bcolours.ENDC}: {np.diff(np.log(y3))/np.diff(np.log(x))}\n{generic.bcolours.OKGREEN}EOC (thermal){generic.bcolours.ENDC}: {np.diff(np.log(y4))/np.diff(np.log(x))}")

    # -------- start theoretical portion --------
    """alpha1, alpha2, alpha3, alpha4 = 10**(c1+1), 10**(c1+2), 10**(c1+3), 10**(c1+4)
    theo_y1, theo_y2, theo_y3, theo_y4 = alpha1/x, alpha2/(x**2), alpha3/(x**3), alpha4/(x**4)
    ax[0,0].loglog(x, theo_y1, linewidth=1, linestyle="--", color="black")
    ax[0,0].loglog(x, theo_y2, linewidth=1, linestyle="--", color="red")
    ax[0,0].loglog(x, theo_y3, linewidth=1, linestyle="--", color="green")
    ax[0,0].loglog(x, theo_y4, linewidth=1, linestyle="--", color="purple")"""
    # -------- end theoretical portion --------

    plt.suptitle(r"Solution errors $\epsilon_\nu(\vec{w})$ against resolution $N_\nu$", fontsize=24)
    fig.text(0.5, 0.04, r"Resolution $\log{(N_\nu)}$", fontsize=18, ha='center')

    plt.savefig(f"{savepath}/solErr_{solver}_{timestep}.png", dpi=330, facecolor="w")

    plt.cla()
    plt.clf()
    plt.close()
    return None


def makeVideo(f, configVariables, testVariables, savepath, vidpath):
    config, solver, timestep = configVariables['config'], configVariables['solver'], configVariables['timestep']
    startPos, endPos = testVariables['startPos'], testVariables['endPos']

    # hdf5 keys are string; need to convert back to int and sort again
    nList = [int(n) for n in f.keys()]
    nList.sort()

    for N in nList:
        simulation = f[str(N)]
        counter = 0

        for t, domain in simulation.items():
            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=[21, 10])

            for index, (_i,_j) in enumerate(plotIndexes):
                ax[_i,_j].set_ylabel(plotLabels[index], fontsize=18)
                ax[_i,_j].set_xlim([startPos, endPos])
                ax[_i,_j].grid(linestyle="--", linewidth=0.5)

            y1 = domain[:, 0]               # density
            y2 = domain[:, 4]               # pressure
            y3 = domain[:, 1]               # vx
            y4 = domain[:, 4]/domain[:, 0]  # thermal energy
            x = np.linspace(startPos, endPos, N)
            y_data = [y1, y2, y3, y4]

            for index, (_i,_j) in enumerate(plotIndexes):
                if beautify:
                    gradient_plot([x,y_data[index]], [_i,_j], ax=ax, linewidth=2, color=colours[index])
                else:
                    ax[_i,_j].plot(x, y_data[index], linewidth=2, color=colours[index])

            plt.suptitle(rf"Primitive variables $\vec{{w}}$ against cell position $x$ at $t = {round(float(t),4)}$ ($N = {len(y1)}$)", fontsize=24)
            fig.text(0.5, 0.04, r"Cell position $x$", fontsize=18, ha='center')

            plt.savefig(f"{vidpath}/{str(counter).zfill(4)}.png", dpi=330, facecolor="w")

            plt.cla()
            plt.clf()
            plt.close()

            counter += 1

        try:
            subprocess.call(["ffmpeg", "-framerate", "60", "-pattern_type", "glob", "-i", f"{vidpath}/*.png", "-c:v", "libx264", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", f"{savepath}/vid_{config}_{solver}_{timestep}.mp4"])
        except Exception as e:
            print(f"{generic.bcolours.WARNING}ffmpeg failed: {e}{generic.bcolours.ENDC}")
            try:
                images = [os.path.join(vidpath,img) for img in os.listdir(vidpath) if img.endswith(".png")]
                images.sort()

                video = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(images, fps=60)
                video.write_videofile(f"{savepath}/vid_{config}_{solver}_{timestep}.mp4")
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

    y1 = domain[:, 0]   # density
    y2 = domain[:, 4]   # pressure
    y3 = domain[:, 1]   # vx
    y4 = y2/y1          # thermal energy
    x = np.linspace(startPos, endPos, len(y1))
    y_data = [y1, y2, y3, y4]

    for index, (_i,_j) in enumerate(plotIndexes):
        if beautify:
            gradient_plot([x,y_data[index]], [_i,_j], ax=ax, linewidth=2, color=colours[index])
        else:
            ax[_i,_j].plot(x, y_data[index], linewidth=2, color=colours[index])
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
def gradient_plot(data, plot_index, ax=None, fill_color=None, **kwargs):
    x, y = data
    i, j = plot_index

    if ax is None:
        ax = plt.gca()

    line, = ax[i,j].plot(x, y, **kwargs)
    if fill_color is None:
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