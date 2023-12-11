import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
import moviepy.video.io.ImageSequenceClip

import functions as fn


show = True

def plotQuantities(*args, **kwargs):
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

    for simulation in args:
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

    if show:
        plt.show(block=True)
    else:
        try:
            kwargs['test']
        except Exception as e:
            plt.savefig(f"{os.getcwd()}/quantitiesPlot.png", dpi=330, facecolor="w")
        else:
            plt.savefig(f"{os.getcwd()}/quantitiesPlot_{kwargs['test']}.png", dpi=330, facecolor="w")

    plt.cla()
    plt.clf()
    plt.close()
    return None


def plotSolutionErrors(*args, **kwargs):
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
    for simulation in args:
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

    if show:
        plt.show(block=True)
    else:
        try:
            kwargs['test']
        except Exception as e:
            plt.savefig(f"{os.getcwd()}/quantitiesPlot.png", dpi=330, facecolor="w")
        else:
            plt.savefig(f"{os.getcwd()}/quantitiesPlot_{kwargs['test']}.png", dpi=330, facecolor="w")

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
        os.system(f"ffmpeg -framerate 24 -pattern_type glob -i '{image_folder}/*.png' -c:v libx264 -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -pix_fmt yuv420p vid{test}.mp4")
    except Exception as e:
        print(f"ffmpeg failed: {e}")
        try:
            images = [os.path.join(f"{os.getcwd()}/{image_folder}",img) for img in os.listdir(image_folder) if img.endswith(".png")]
            images.sort()

            video = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(images, fps=24)
            video.write_videofile(f"{os.getcwd()}/vid{test}.mp4")
        except Exception as e:
            print(f"moviepy failed: {e}")
            pass


def makeVideo(*args, **kwargs):
    try:
        start, end = kwargs["start"], kwargs["end"]
    except Exception as e:
        start, end = 0, 1

    for simulation in args:
        N = len(simulation[0])
        counter = 0

        path = f"{os.getcwd()}/plots"
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
            writeVideo("plots")
        else:
            writeVideo("plots", kwargs['test'])
    return None