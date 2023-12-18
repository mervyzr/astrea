import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
import moviepy.video.io.ImageSequenceClip

import functions as fn


save = False


# Initiate the live plot feature
def initiateLivePlot(startPos, endPos, N):
    plot_x = np.linspace(startPos, endPos, N)
    plt.ion()

    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax[0,0].set_ylabel(r"density $\rho$")
    ax[0,1].set_ylabel(r"pressure $P$")
    ax[1,0].set_ylabel(r"velocity $v_x$")
    ax[1,1].set_ylabel(r"thermal energy $P/\rho$")

    ax[0,1].yaxis.set_label_position("right")
    ax[1,1].yaxis.set_label_position("right")

    ax[0,1].yaxis.tick_right()
    ax[1,1].yaxis.tick_right()

    ax[0,0].set_xlim([startPos, endPos])
    ax[0,1].set_xlim([startPos, endPos])
    ax[1,0].set_xlim([startPos, endPos])
    ax[1,1].set_xlim([startPos, endPos])

    ax[0,0].grid(linestyle='--', linewidth=0.5)
    ax[0,1].grid(linestyle='--', linewidth=0.5)
    ax[1,0].grid(linestyle='--', linewidth=0.5)
    ax[1,1].grid(linestyle='--', linewidth=0.5)

    graphTL, = ax[0,0].plot(plot_x, plot_x, linewidth=2, color="blue")  # density
    graphTR, = ax[0,1].plot(plot_x, plot_x, linewidth=2, color="red")  # pressure
    graphBL, = ax[1,0].plot(plot_x, plot_x, linewidth=2, color="green")  # vx
    graphBR, = ax[1,1].plot(plot_x, plot_x, linewidth=2, color="black")  # thermal energy

    return fig, ax, [graphTL, graphTR, graphBL, graphBR]


# Update live plot
def updatePlot(arr, t, fig, ax, plots):
    graphTL, graphTR, graphBL, graphBR = plots

    graphTL.set_ydata(arr[:,0])  # density
    graphTR.set_ydata(arr[:,4])  # pressure
    graphBL.set_ydata(arr[:,1])  # vx
    graphBR.set_ydata(arr[:,4]/arr[:,0])  # thermal energy

    ax[0,0].relim()
    ax[0,0].autoscale_view()
    ax[0,1].relim()
    ax[0,1].autoscale_view()
    ax[1,0].relim()
    ax[1,0].autoscale_view()
    ax[1,1].relim()
    ax[1,1].autoscale_view()

    plt.suptitle(rf"Quantities $q$ against cell position $x$ at $t = {round(t,4)}$")
    fig.text(0.5, 0.04, r"Cell position $x$", ha='center')
    fig.canvas.draw()
    fig.canvas.flush_events()

    pass


# Plot q as a snapshot
def plotQuantities(runs, *args, **kwargs):
    try:
        start, end = kwargs["start"], kwargs["end"]
    except Exception as e:
        start, end = 0, 1
    try:
        index = kwargs["index"]
    except Exception as e:
        index = -1
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=[21, 10])

    ax[0,0].set_ylabel(r"Density $\rho$", fontsize=18)
    ax[0,1].set_ylabel(r"Pressure $P$", fontsize=18)
    ax[1,0].set_ylabel(r"Velocity $v_x$", fontsize=18)
    ax[1,1].set_ylabel(r"Thermal energy $P/\rho$", fontsize=18)
    ax[0,0].set_xlim([start, end])
    ax[0,1].set_xlim([start, end])
    ax[1,0].set_xlim([start, end])
    ax[1,1].set_xlim([start, end])
    ax[0,0].grid(linestyle='--', linewidth=0.5)
    ax[0,1].grid(linestyle='--', linewidth=0.5)
    ax[1,0].grid(linestyle='--', linewidth=0.5)
    ax[1,1].grid(linestyle='--', linewidth=0.5)

    for simulation in runs:
        y1 = simulation[list(simulation.keys())[index]][:, 0]  # density
        y2 = simulation[list(simulation.keys())[index]][:, 4]  # pressure
        y3 = simulation[list(simulation.keys())[index]][:, 1]  # vx
        y4 = y2/y1  # thermal energy
        x = np.linspace(start, end, len(y1))

        ax[0,0].plot(x, y1, linewidth=2, label=f"N = {len(y1)}")  # density
        ax[0,1].plot(x, y2, linewidth=2, label=f"N = {len(y1)}")  # pressure
        ax[1,0].plot(x, y3, linewidth=2, label=f"N = {len(y1)}")  # vx
        ax[1,1].plot(x, y4, linewidth=2, label=f"N = {len(y1)}")  # thermal energy

    plt.suptitle(r"Plot of quantities $q$ against cell position $x$", fontsize=24)
    fig.text(0.5, 0.04, r"Cell position $x$", fontsize=18, ha='center')
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, prop={'size': 16}, loc='upper right')

    if save:
        try:
            kwargs['test']
        except Exception as e:
            plt.savefig(f"{os.getcwd()}/quantitiesPlot.png", dpi=330, facecolor="w")
        else:
            plt.savefig(f"{os.getcwd()}/quantitiesPlot_{kwargs['test']}.png", dpi=330, facecolor="w")
    else:
        plt.show(block=True)

    plt.cla()
    plt.clf()
    plt.close()
    return None


def plotSolutionErrors(runs, *args, **kwargs):
    try:
        start, end = kwargs["start"], kwargs["end"]
    except Exception as e:
        start, end = 0, 1
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=[21, 10])

    ax[0,0].set_ylabel(r"density $\rho$", fontsize=18)  # density
    ax[0,1].set_ylabel(r"pressure $P$", fontsize=18)  # pressure
    ax[1,0].set_ylabel(r"velocity $v_x$", fontsize=18)  # vx
    ax[1,1].set_ylabel(r"thermal energy $\frac{P}{\rho}$", fontsize=18)  # thermal energy
    ax[0,0].grid(linestyle='--', linewidth=0.5)
    ax[0,1].grid(linestyle='--', linewidth=0.5)
    ax[1,0].grid(linestyle='--', linewidth=0.5)
    ax[1,1].grid(linestyle='--', linewidth=0.5)

    x, y1, y2, y3, y4 = [], [], [], [], []
    for simulation in runs:
        solutionErrors = fn.calculateSolutionError(simulation, start, end)
        x.append(len(simulation[0]))
        y1.append(solutionErrors[0])  # density
        y2.append(solutionErrors[4])  # pressure
        y3.append(solutionErrors[1])  # vx
        y4.append(solutionErrors[4]/solutionErrors[0])  # thermal energy
    
    ax[0,0].plot(np.log10(np.asarray(x)), np.log10(np.asarray(y1)), linewidth=2, linestyle="--", marker="o", color="blue")
    ax[0,1].plot(np.log10(np.asarray(x)), np.log10(np.asarray(y2)), linewidth=2, linestyle="--", marker="o", color="red")
    ax[1,0].plot(np.log10(np.asarray(x)), np.log10(np.asarray(y3)), linewidth=2, linestyle="--", marker="o", color="green")
    ax[1,1].plot(np.log10(np.asarray(x)), np.log10(np.asarray(y4)), linewidth=2, linestyle="--", marker="o", color="black")

    plt.suptitle(r"Plot of solution errors $\epsilon_\nu(q)$ against resolution $N_\nu$", fontsize=24)
    fig.text(0.5, 0.04, r"Resolution $\log_{10}{[N_\nu]}$", fontsize=18, ha='center')
    fig.text(0.04, 0.5, r"Solution errors $\log_{10}{[\epsilon_\nu(q)]}$", fontsize=18, va='center', rotation='vertical')

    if save:
        try:
            kwargs['test']
        except Exception as e:
            plt.savefig(f"{os.getcwd()}/solutionErrors.png", dpi=330, facecolor="w")
        else:
            plt.savefig(f"{os.getcwd()}/solutionErrors_{kwargs['test']}.png", dpi=330, facecolor="w")
    else:
        plt.show(block=True)

    plt.cla()
    plt.clf()
    plt.close()
    return None


def writeVideo(image_folder, *args):
    try:
        args[0]
    except Exception as e:
        test = ""
    else:
        test = f"_{args[0]}"

    try:
        os.system(f"ffmpeg -framerate 24 -pattern_type glob -i '{image_folder}/*.png' -c:v libx264 -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -pix_fmt yuv420p ../vid{test}.mp4")
    except Exception as e:
        print(f"ffmpeg failed: {e}")
        try:
            images = [os.path.join(image_folder,img) for img in os.listdir(image_folder) if img.endswith(".png")]
            images.sort()

            video = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(images, fps=24)
            video.write_videofile(f"{image_folder}/vid{test}.mp4")
        except Exception as e:
            print(f"moviepy failed: {e}")
            pass


def makeVideo(runs, *args, **kwargs):
    try:
        start, end = kwargs["start"], kwargs["end"]
    except Exception as e:
        start, end = 0, 1

    for simulation in runs:
        N = len(simulation[0])
        counter = 0

        path = f"{os.getcwd()}/../vidplots"
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

        for t, domain in simulation.items():
            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=[21, 10])

            ax[0,0].set_ylabel(r"Density $\rho$", fontsize=18)
            ax[0,1].set_ylabel(r"Pressure $P$", fontsize=18)
            ax[1,0].set_ylabel(r"Velocity $v_x$", fontsize=18)
            ax[1,1].set_ylabel(r"Thermal energy $P/\rho$", fontsize=18)
            ax[0,0].set_xlim([start, end])
            ax[0,1].set_xlim([start, end])
            ax[1,0].set_xlim([start, end])
            ax[1,1].set_xlim([start, end])
            ax[0,0].grid(linestyle='--', linewidth=0.5)
            ax[0,1].grid(linestyle='--', linewidth=0.5)
            ax[1,0].grid(linestyle='--', linewidth=0.5)
            ax[1,1].grid(linestyle='--', linewidth=0.5)

            y1 = domain[:, 0]  # density
            y2 = domain[:, 4]  # pressure
            y3 = domain[:, 1]  # vx
            y4 = domain[:, 4]/domain[:, 0]  # thermal energy
            x = np.linspace(start, end, len(y1))

            ax[0,0].plot(x, y1, linewidth=2, color="blue")  # density
            ax[0,1].plot(x, y2, linewidth=2, color="red")  # pressure
            ax[1,0].plot(x, y3, linewidth=2, color="green")  # vx
            ax[1,1].plot(x, y4, linewidth=2, color="black")  # thermal energy

            plt.suptitle(rf"Plot of quantities $q$ against cell position $x$ at $t = {round(t,4)}$", fontsize=24)
            fig.text(0.5, 0.04, r"Cell position $x$", fontsize=18, ha='center')
            handles, labels = plt.gca().get_legend_handles_labels()
            fig.legend(handles, labels, prop={'size': 16}, loc='upper right')

            plt.savefig(f"{path}/{str(counter).zfill(4)}.png", dpi=330, facecolor="w")

            plt.cla()
            plt.clf()
            plt.close()

            counter += 1

        try:
            kwargs['test']
        except Exception as e:
            writeVideo(path)
        else:
            writeVideo(path, kwargs['test'])
    return None