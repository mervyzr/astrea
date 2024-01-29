import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
import moviepy.video.io.ImageSequenceClip

import functions as fn

##############################################################################


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


# Plot snapshots of quantities for multiple runs
def plotQuantities(runs, snapshots, config, gamma, startPos, endPos, shockPos):
    try:
        int(snapshots)
    except Exception as e:
        snapshots = 1
    else:
        if snapshots < 1:
            snapshots = 1

    indexes = []
    for simulation in runs:
        timings = np.fromiter(simulation.keys(), dtype=float)
        indexes.append([timing[-1] for timing in np.array_split(timings, snapshots)])

    for i, timing in enumerate(indexes[0]):
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=[21, 10])

        ax[0,0].set_ylabel(r"Density $\rho$", fontsize=18)
        ax[0,1].set_ylabel(r"Pressure $P$", fontsize=18)
        ax[1,0].set_ylabel(r"Velocity $v_x$", fontsize=18)
        ax[1,1].set_ylabel(r"Thermal energy $P/\rho$", fontsize=18)
        ax[0,0].set_xlim([startPos, endPos])
        ax[0,1].set_xlim([startPos, endPos])
        ax[1,0].set_xlim([startPos, endPos])
        ax[1,1].set_xlim([startPos, endPos])
        ax[0,0].grid(linestyle='--', linewidth=0.5)
        ax[0,1].grid(linestyle='--', linewidth=0.5)
        ax[1,0].grid(linestyle='--', linewidth=0.5)
        ax[1,1].grid(linestyle='--', linewidth=0.5)             

        if len(runs) == 1:
            simulation, time = runs[0], indexes[0]
            y1 = simulation[time[i]][:, 0]  # density
            y2 = simulation[time[i]][:, 4]  # pressure
            y3 = simulation[time[i]][:, 1]  # vx
            y4 = y2/y1                      # thermal energy
            x = np.linspace(startPos, endPos, len(y1))

            ax[0,0].plot(x, y1, linewidth=2, color="blue")   # density
            ax[0,1].plot(x, y2, linewidth=2, color="red")    # pressure
            ax[1,0].plot(x, y3, linewidth=2, color="green")  # vx
            ax[1,1].plot(x, y4, linewidth=2, color="black")  # thermal energy

            if config == "sod":
                Sod = fn.calculateSodAnalytical(simulation[time[i]], time[i], gamma, startPos, endPos, shockPos)
                ax[0,0].plot(x, Sod[:, 0], linewidth=1, color="purple", linestyle="--")
                ax[0,1].plot(x, Sod[:, 4], linewidth=1, color="purple", linestyle="--")
                ax[1,0].plot(x, Sod[:, 1], linewidth=1, color="purple", linestyle="--")
                ax[1,1].plot(x, Sod[:, 4]/Sod[:, 0], linewidth=1, color="purple", linestyle="--")

            plt.suptitle(rf"Plot of quantities $q$ against cell position $x$ at $t \approx {round(timing,3)}$ ($N = {len(y1)}$)", fontsize=24)
            fig.text(0.5, 0.04, r"Cell position $x$", fontsize=18, ha='center')

            plt.savefig(f"{os.getcwd()}/../quantitiesPlot_{config}_{round(timing,3)}.png", dpi=330, facecolor="w")

            plt.cla()
            plt.clf()
            plt.close()
        else:
            for j, simulation in enumerate(runs):
                y1 = simulation[indexes[j][i]][:, 0]  # density
                y2 = simulation[indexes[j][i]][:, 4]  # pressure
                y3 = simulation[indexes[j][i]][:, 1]  # vx
                y4 = y2/y1                            # thermal energy
                x = np.linspace(startPos, endPos, len(y1))

                ax[0,0].plot(x, y1, linewidth=2, label=f"N = {len(y1)}")  # density
                ax[0,1].plot(x, y2, linewidth=2, label=f"N = {len(y1)}")  # pressure
                ax[1,0].plot(x, y3, linewidth=2, label=f"N = {len(y1)}")  # vx
                ax[1,1].plot(x, y4, linewidth=2, label=f"N = {len(y1)}")  # thermal energy

                if config == "sod":
                    Sod = fn.calculateSodAnalytical(simulation[indexes[j][i]], indexes[j][i], gamma, startPos, endPos, shockPos)
                    ax[0,0].plot(x, Sod[:, 0], linewidth=1, color="purple", linestyle="--")
                    ax[0,1].plot(x, Sod[:, 4], linewidth=1, color="purple", linestyle="--")
                    ax[1,0].plot(x, Sod[:, 1], linewidth=1, color="purple", linestyle="--")
                    ax[1,1].plot(x, Sod[:, 4]/Sod[:, 0], linewidth=1, color="purple", linestyle="--")

            plt.suptitle(rf"Plot of quantities $q$ against cell position $x$ at $t \approx {round(timing,3)}$", fontsize=24)
            fig.text(0.5, 0.04, r"Cell position $x$", fontsize=18, ha='center')
            handles, labels = plt.gca().get_legend_handles_labels()
            fig.legend(handles, labels, prop={'size': 16}, loc='upper right')

            plt.savefig(f"{os.getcwd()}/../quantitiesPlot_{config}_{round(timing,3)}.png", dpi=330, facecolor="w")

            plt.cla()
            plt.clf()
            plt.close()
    return None


def plotSolutionErrors(runs, config, startPos, endPos):
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
        solutionErrors = fn.calculateSolutionError(simulation, startPos, endPos)
        x.append(len(simulation[0]))
        y1.append(solutionErrors[0])                    # density
        y2.append(solutionErrors[4])                    # pressure
        y3.append(solutionErrors[1])                    # vx
        y4.append(solutionErrors[4]/solutionErrors[0])  # thermal energy
    
    ax[0,0].plot(np.log10(np.asarray(x)), np.log10(np.asarray(y1)), linewidth=2, linestyle="--", marker="o", color="blue")
    ax[0,1].plot(np.log10(np.asarray(x)), np.log10(np.asarray(y2)), linewidth=2, linestyle="--", marker="o", color="red")
    ax[1,0].plot(np.log10(np.asarray(x)), np.log10(np.asarray(y3)), linewidth=2, linestyle="--", marker="o", color="green")
    ax[1,1].plot(np.log10(np.asarray(x)), np.log10(np.asarray(y4)), linewidth=2, linestyle="--", marker="o", color="black")

    plt.suptitle(r"Plot of solution errors $\epsilon_\nu(q)$ against resolution $N_\nu$", fontsize=24)
    fig.text(0.5, 0.04, r"Resolution $\log_{10}{[N_\nu]}$", fontsize=18, ha='center')
    fig.text(0.04, 0.5, r"Solution errors $\log_{10}{[\epsilon_\nu(q)]}$", fontsize=18, va='center', rotation='vertical')

    plt.savefig(f"{os.getcwd()}/../solutionErrors_{config}.png", dpi=330, facecolor="w")

    plt.cla()
    plt.clf()
    plt.close()
    return None


def makeVideo(runs, config, startPos, endPos):
    for simulation in runs:
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
            ax[0,0].set_xlim([startPos, endPos])
            ax[0,1].set_xlim([startPos, endPos])
            ax[1,0].set_xlim([startPos, endPos])
            ax[1,1].set_xlim([startPos, endPos])
            ax[0,0].grid(linestyle='--', linewidth=0.5)
            ax[0,1].grid(linestyle='--', linewidth=0.5)
            ax[1,0].grid(linestyle='--', linewidth=0.5)
            ax[1,1].grid(linestyle='--', linewidth=0.5)

            y1 = domain[:, 0]               # density
            y2 = domain[:, 4]               # pressure
            y3 = domain[:, 1]               # vx
            y4 = domain[:, 4]/domain[:, 0]  # thermal energy
            x = np.linspace(startPos, endPos, len(y1))

            ax[0,0].plot(x, y1, linewidth=2, color="blue")   # density
            ax[0,1].plot(x, y2, linewidth=2, color="red")    # pressure
            ax[1,0].plot(x, y3, linewidth=2, color="green")  # vx
            ax[1,1].plot(x, y4, linewidth=2, color="black")  # thermal energy

            plt.suptitle(rf"Plot of quantities $q$ against cell position $x$ at $t = {round(t,4)}$ ($N = {len(y1)}$)", fontsize=24)
            fig.text(0.5, 0.04, r"Cell position $x$", fontsize=18, ha='center')

            plt.savefig(f"{path}/{str(counter).zfill(4)}.png", dpi=330, facecolor="w")

            plt.cla()
            plt.clf()
            plt.close()

            counter += 1

        try:
            os.system(f"ffmpeg -framerate 30 -pattern_type glob -i '{path}/*.png' -c:v libx264 -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -pix_fmt yuv420p ../vid{config}.mp4")
        except Exception as e:
            print(f"ffmpeg failed: {e}")
            try:
                images = [os.path.join(path,img) for img in os.listdir(path) if img.endswith(".png")]
                images.sort()

                video = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(images, fps=30)
                video.write_videofile(f"{path}/vid{config}.mp4")
            except Exception as e:
                print(f"moviepy failed: {e}")
                pass
    return None