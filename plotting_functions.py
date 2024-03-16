import os
import shutil
from datetime import datetime

import matplotlib
from settings import saveFile, saveVideo
if saveFile or saveVideo:
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import moviepy.video.io.ImageSequenceClip

from functions import generic, analytic

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
    graphBR, = ax[1,1].plot(plot_x, plot_x, linewidth=2, color="darkviolet")  # thermal energy

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
def plotQuantities(f, snapshots, plotVariables):
    config, gamma, solver, timestep, startPos, endPos, shockPos = plotVariables
    now = datetime.now().strftime("%Y%m%d%H%M")

    # Separate the timings based on the number of snapshots; returns a list of lists with the timing intervals for each simulation
    indexes = []
    for group, simulation in f.items():
        timings = np.fromiter(simulation.keys(), dtype=float)
        indexes.append([timing[-1] for timing in np.array_split(timings, abs(int(snapshots)))])
    
    # Iterate through the timings; the last set of timings refer to the highest resolution
    for i in range(len(indexes[-1])):
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

        # Plot each simulation at the i-th timing
        for j, group in enumerate(f):
            time_key = str(indexes[j][i])
            y1 = f[group][time_key][:, 0]   # density
            y2 = f[group][time_key][:, 4]   # pressure
            y3 = f[group][time_key][:, 1]   # vx
            y4 = y2/y1                      # thermal energy
            x = np.linspace(startPos, endPos, len(y1))

            if len(f) != 1:
                ax[0,0].plot(x, y1, linewidth=2, label=f"N = {len(y1)}")  # density
                ax[0,1].plot(x, y2, linewidth=2, label=f"N = {len(y1)}")  # pressure
                ax[1,0].plot(x, y3, linewidth=2, label=f"N = {len(y1)}")  # vx
                ax[1,1].plot(x, y4, linewidth=2, label=f"N = {len(y1)}")  # thermal energy
                plt.suptitle(rf"Plot of quantities $q$ against cell position $x$ at $t \approx {round(indexes[-1][i],3)}$", fontsize=24)
            else:
                ax[0,0].plot(x, y1, linewidth=2, color="blue")        # density
                ax[0,1].plot(x, y2, linewidth=2, color="red")         # pressure
                ax[1,0].plot(x, y3, linewidth=2, color="green")       # vx
                ax[1,1].plot(x, y4, linewidth=2, color="darkviolet")  # thermal energy
                plt.suptitle(rf"Plot of quantities $q$ against cell position $x$ at $t \approx {round(indexes[-1][i],3)}$ ($N = {len(y1)}$)", fontsize=24)
        
        # Adjust ylim for sin-wave test
        if config == "sin":
            last_sim = f[list(f.keys())[-1]]
            first_config = last_sim[list(last_sim.keys())[0]][0]
            initial_rho, initial_vx, initial_vy, initial_vz, initial_P = first_config
            vrange = np.linspace(initial_vx-.005, initial_vx+.005, 9)
            Prange = np.linspace(initial_P-.005, initial_P+.005, 9)
            ax[0,1].set_yticks(Prange)
            ax[1,0].set_yticks(vrange)
            ax[0,1].set_ylim([initial_P-.005, initial_P+.005])
            ax[1,0].set_ylim([initial_vx-.005, initial_vx+.005])

        # Add Sod analytical solution, using the highest resolution and timing
        elif config == "sod":
            Sod = analytic.calculateSodAnalytical(f[group][str(indexes[-1][i])], indexes[-1][i], gamma, startPos, endPos, shockPos)
            ax[0,0].plot(x, Sod[:, 0], linewidth=1, color="black", linestyle="--", label="Analytical solution")
            ax[0,1].plot(x, Sod[:, 4], linewidth=1, color="black", linestyle="--", label="Analytical solution")
            ax[1,0].plot(x, Sod[:, 1], linewidth=1, color="black", linestyle="--", label="Analytical solution")
            ax[1,1].plot(x, Sod[:, 4]/Sod[:, 0], linewidth=1, color="black", linestyle="--", label="Analytical solution")
        
        # Add Sedov analytical solution, using the highest resolution and timing
        elif config == "sedov":
            pass

        fig.text(0.5, 0.04, r"Cell position $x$", fontsize=18, ha='center')
        if len(f) != 1 or config == "sod":
            handles, labels = plt.gca().get_legend_handles_labels()
            fig.legend(handles, labels, prop={'size': 16}, loc='upper right')

        plt.savefig(f"{os.getcwd()}/plots/qPlot_{config}_{solver}_{timestep}_{round(indexes[-1][i],3)}.png", dpi=330, facecolor="w")

        plt.cla()
        plt.clf()
        plt.close()
    return None


def plotSolutionErrors(f, plotVariables):
    config, solver, timestep, startPos, endPos = plotVariables
    now = datetime.now().strftime("%Y%m%d%H%M")

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=[21, 10])

    ax[0,0].set_ylabel(r"Density $\log{(\epsilon_\nu(\rho))}$", fontsize=18)  # density
    ax[0,1].set_ylabel(r"Pressure $\log{(\epsilon_\nu(P))}$", fontsize=18)  # pressure
    ax[1,0].set_ylabel(r"Velocity $\log{(\epsilon_\nu(v_x))}$", fontsize=18)  # vx
    ax[1,1].set_ylabel(r"Thermal energy $\log{(\epsilon_\nu(\frac{P}{\rho}))}$", fontsize=18)  # thermal energy
    ax[0,0].grid(linestyle='--', linewidth=0.5)
    ax[0,1].grid(linestyle='--', linewidth=0.5)
    ax[1,0].grid(linestyle='--', linewidth=0.5)
    ax[1,1].grid(linestyle='--', linewidth=0.5)

    x, y1, y2, y3, y4 = [], [], [], [], []
    for group, simulation in f.items():
        x.append(len(simulation['0']))
        solutionErrors = analytic.calculateSolutionError(simulation)
        y1.append(solutionErrors[0])  # density
        y2.append(solutionErrors[4])  # pressure
        y3.append(solutionErrors[1])  # vx
        y4.append(solutionErrors[5])  # thermal energy
    x, y1, y2, y3, y4 = np.asarray(x), np.asarray(y1), np.asarray(y2), np.asarray(y3), np.asarray(y4)

    m1, c1 = np.polyfit(np.log10(x), np.log10(y1), 1)
    m2, c2 = np.polyfit(np.log10(x), np.log10(y2), 1)
    m3, c3 = np.polyfit(np.log10(x), np.log10(y3), 1)
    m4, c4 = np.polyfit(np.log10(x), np.log10(y4), 1)

    ax[0,0].loglog(x, y1, linewidth=2, linestyle="--", marker="o", color="blue", label=f"grad. = {round(m1,4)}")
    ax[0,1].loglog(x, y2, linewidth=2, linestyle="--", marker="o", color="red", label=f"grad. = {round(m2,4)}")
    ax[1,0].loglog(x, y3, linewidth=2, linestyle="--", marker="o", color="green", label=f"grad. = {round(m3,4)}")
    ax[1,1].loglog(x, y4, linewidth=2, linestyle="--", marker="o", color="darkviolet", label=f"grad. = {round(m4,4)}")

    ax[0,0].legend(prop={'size': 14})
    ax[0,1].legend(prop={'size': 14})
    ax[1,0].legend(prop={'size': 14})
    ax[1,1].legend(prop={'size': 14})

    print(f"{generic.bcolours.OKGREEN}EOC (density){generic.bcolours.ENDC}: {np.diff(np.log(y1))/np.diff(np.log(x))}\n{generic.bcolours.OKGREEN}EOC (pressure){generic.bcolours.ENDC}: {np.diff(np.log(y2))/np.diff(np.log(x))}\n{generic.bcolours.OKGREEN}EOC (vx){generic.bcolours.ENDC}: {np.diff(np.log(y3))/np.diff(np.log(x))}\n{generic.bcolours.OKGREEN}EOC (thermal){generic.bcolours.ENDC}: {np.diff(np.log(y4))/np.diff(np.log(x))}")

    # -------- start theoretical portion --------
    """alpha1, alpha2, alpha3, alpha4 = 10**(c1+1), 10**(c1+2), 10**(c1+3), 10**(c1+4)
    theo_y1, theo_y2, theo_y3, theo_y4 = alpha1/x, alpha2/(x**2), alpha3/(x**3), alpha4/(x**4)
    ax[0,0].loglog(x, theo_y1, linewidth=1, linestyle="--", color="black")
    ax[0,0].loglog(x, theo_y2, linewidth=1, linestyle="--", color="red")
    ax[0,0].loglog(x, theo_y3, linewidth=1, linestyle="--", color="green")
    ax[0,0].loglog(x, theo_y4, linewidth=1, linestyle="--", color="purple")"""
    # -------- end theoretical portion --------

    plt.suptitle(r"Plot of solution errors $\epsilon_\nu(q)$ against resolution $N_\nu$", fontsize=24)
    fig.text(0.5, 0.04, r"Resolution $\log{(N_\nu)}$", fontsize=18, ha='center')

    plt.savefig(f"{os.getcwd()}/plots/solErr_{solver}_{timestep}.png", dpi=330, facecolor="w")

    plt.cla()
    plt.clf()
    plt.close()
    return None


def makeVideo(f, videoVariables):
    config, solver, timestep, startPos, endPos = videoVariables
    now = datetime.now().strftime("%Y%m%d%H%M")

    for group, simulation in f.items():
        counter = 0

        path = f"{os.getcwd()}/../vidplots"
        if not os.path.exists(path):
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
            ax[1,1].plot(x, y4, linewidth=2, color="darkviolet")  # thermal energy

            plt.suptitle(rf"Plot of quantities $q$ against cell position $x$ at $t = {round(t,4)}$ ($N = {len(y1)}$)", fontsize=24)
            fig.text(0.5, 0.04, r"Cell position $x$", fontsize=18, ha='center')

            plt.savefig(f"{path}/{str(counter).zfill(4)}.png", dpi=330, facecolor="w")

            plt.cla()
            plt.clf()
            plt.close()

            counter += 1

        try:
            os.system(f"ffmpeg -framerate 60 -pattern_type glob -i '{path}/*.png' -c:v libx264 -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -pix_fmt yuv420p ../vid{config}_{solver}_{timestep}_{now}.mp4")
        except Exception as e:
            print(f"ffmpeg failed: {e}")
            try:
                images = [os.path.join(path,img) for img in os.listdir(path) if img.endswith(".png")]
                images.sort()

                video = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(images, fps=60)
                video.write_videofile(f"../vid{config}_{solver}_{timestep}_{now}.mp4")
            except Exception as e:
                print(f"moviepy failed: {e}")
                pass
            else:
                shutil.rmtree(path)
        else:
            shutil.rmtree(path)
    return None