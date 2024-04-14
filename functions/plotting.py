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
import moviepy.video.io.ImageSequenceClip

from functions import generic, analytic

##############################################################################

# Initiate the live plot feature
def initiateLivePlot(startPos, endPos, N):
    plot_x = np.linspace(startPos, endPos, N)
    plt.ion()

    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax[0,0].set_ylabel(r"Density $\rho$")
    ax[0,1].set_ylabel(r"Pressure $P$")
    ax[1,0].set_ylabel(r"Velocity $v_x$")
    ax[1,1].set_ylabel(r"Thermal energy $\frac{P}{\rho}$")

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

        ax[0,0].set_ylabel(r"Density $\rho$", fontsize=18)
        ax[0,1].set_ylabel(r"Pressure $P$", fontsize=18)
        ax[1,0].set_ylabel(r"Velocity $v_x$", fontsize=18)
        ax[1,1].set_ylabel(r"Thermal energy $\frac{P}{\rho}$", fontsize=18)
        ax[0,0].set_xlim([startPos, endPos])
        ax[0,1].set_xlim([startPos, endPos])
        ax[1,0].set_xlim([startPos, endPos])
        ax[1,1].set_xlim([startPos, endPos])
        ax[0,0].grid(linestyle='--', linewidth=0.5)
        ax[0,1].grid(linestyle='--', linewidth=0.5)
        ax[1,0].grid(linestyle='--', linewidth=0.5)
        ax[1,1].grid(linestyle='--', linewidth=0.5)

        # Plot each simulation at the i-th timing
        for j, N in enumerate(nList):
            time_key = str(indexes[j][i])
            y1 = f[str(N)][time_key][:, 0]   # density
            y2 = f[str(N)][time_key][:, 4]   # pressure
            y3 = f[str(N)][time_key][:, 1]   # vx
            y4 = y2/y1                      # thermal energy
            x = np.linspace(startPos, endPos, len(y1))

            if len(f) != 1:
                ax[0,0].plot(x, y1, linewidth=2, label=f"N = {len(y1)}")  # density
                ax[0,1].plot(x, y2, linewidth=2, label=f"N = {len(y1)}")  # pressure
                ax[1,0].plot(x, y3, linewidth=2, label=f"N = {len(y1)}")  # vx
                ax[1,1].plot(x, y4, linewidth=2, label=f"N = {len(y1)}")  # thermal energy
                plt.suptitle(rf"Primitive variables $\vec{{w}}$ against cell position $x$ at $t \approx {round(indexes[-1][i],3)}$", fontsize=24)
            else:
                ax[0,0].plot(x, y1, linewidth=2, color="blue")        # density
                ax[0,1].plot(x, y2, linewidth=2, color="red")         # pressure
                ax[1,0].plot(x, y3, linewidth=2, color="green")       # vx
                ax[1,1].plot(x, y4, linewidth=2, color="darkviolet")  # thermal energy
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

            ax[0,0].plot(x, analytical[:, 0], linewidth=1, color="black", linestyle="--", label="Analytical solution")
            ax[0,1].plot(x, analytical[:, 4], linewidth=1, color="black", linestyle="--", label="Analytical solution")
            ax[1,0].plot(x, analytical[:, 1], linewidth=1, color="black", linestyle="--", label="Analytical solution")
            ax[1,1].plot(x, analytical[:, 4]/analytical[:, 0], linewidth=1, color="black", linestyle="--", label="Analytical solution")

        # Add Sod analytical solution, using the highest resolution and timing
        elif config == "sod":
            tube, _t = f[str(nList[-1])][str(indexes[-1][i])], indexes[-1][i]
            Sod = analytic.calculateSodAnalytical(tube, _t,  gamma, startPos, endPos, shockPos)
            ax[0,0].plot(x, Sod[:, 0], linewidth=1, color="black", linestyle="--", label="Analytical solution")
            ax[0,1].plot(x, Sod[:, 4], linewidth=1, color="black", linestyle="--", label="Analytical solution")
            ax[1,0].plot(x, Sod[:, 1], linewidth=1, color="black", linestyle="--", label="Analytical solution")
            ax[1,1].plot(x, Sod[:, 4]/Sod[:, 0], linewidth=1, color="black", linestyle="--", label="Analytical solution")

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

    ax[0,0].set_ylabel(r"Density $\log{(\epsilon_\nu(\rho))}$", fontsize=18)  # density
    ax[0,1].set_ylabel(r"Pressure $\log{(\epsilon_\nu(P))}$", fontsize=18)  # pressure
    ax[1,0].set_ylabel(r"Velocity $\log{(\epsilon_\nu(v_x))}$", fontsize=18)  # vx
    ax[1,1].set_ylabel(r"Thermal energy $\log{(\epsilon_\nu(\frac{P}{\rho}))}$", fontsize=18)  # thermal energy
    ax[0,0].grid(linestyle='--', linewidth=0.5)
    ax[0,1].grid(linestyle='--', linewidth=0.5)
    ax[1,0].grid(linestyle='--', linewidth=0.5)
    ax[1,1].grid(linestyle='--', linewidth=0.5)

    x, y1, y2, y3, y4 = [], [], [], [], []
    for N in nList:
        x.append(f[str(N)].attrs['cells'])
        solutionErrors = analytic.calculateSolutionError(f[str(N)], startPos, endPos, config)
        y1.append(solutionErrors[0])  # density
        y2.append(solutionErrors[4])  # pressure
        y3.append(solutionErrors[1])  # vx
        y4.append(solutionErrors[-1])  # thermal energy
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

            ax[0,0].set_ylabel(r"Density $\rho$", fontsize=18)
            ax[0,1].set_ylabel(r"Pressure $P$", fontsize=18)
            ax[1,0].set_ylabel(r"Velocity $v_x$", fontsize=18)
            ax[1,1].set_ylabel(r"Thermal energy $\frac{P}{\rho}$", fontsize=18)
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