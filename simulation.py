import time

import numpy as np
import matplotlib.pyplot as plt

import configs as cfg
import functions as fn
import solvers as solver
import plotting_functions as plotter

##############################################################################

config = "sod"
cells = 100
cfl = .8
gamma = 1.4

livePlot = True

##############################################################################

if config == "sin":
    startPos = 0
    endPos = 1
    shockPos = 1
    tEnd = 2
elif config == "sedov":
    startPos = -10
    endPos = 10
    shockPos = 1
    tEnd = .6
else:
    startPos = 0
    endPos = 1
    shockPos = .5
    tEnd = .2


# Main code
def runSimulation(N, _config=config, _cfl=cfl, _gamma=gamma, _startPos=startPos, _endPos=endPos, _shockPos=shockPos, _tEnd=tEnd):
    simulation = {}
    N += (N%2)  # Make N into an even number
    domain = cfg.initialise(N, _config, _gamma, _startPos, _endPos, _shockPos)
    
    # Compute dx and set t = 0
    dx = abs(_endPos-_startPos)/N
    t = 0

    if livePlot:
        plot_x = np.linspace(_startPos, _endPos, N)
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
        ax[0,0].set_xlim([_startPos, _endPos])
        ax[0,1].set_xlim([_startPos, _endPos])
        ax[1,0].set_xlim([_startPos, _endPos])
        ax[1,1].set_xlim([_startPos, _endPos])
        ax[0,0].grid(linestyle='--', linewidth=0.5)
        ax[0,1].grid(linestyle='--', linewidth=0.5)
        ax[1,0].grid(linestyle='--', linewidth=0.5)
        ax[1,1].grid(linestyle='--', linewidth=0.5)

        graphTL, = ax[0,0].plot(plot_x, plot_x, linewidth=2, color="blue")  # density
        graphTR, = ax[0,1].plot(plot_x, plot_x, linewidth=2, color="red")  # pressure
        graphBL, = ax[1,0].plot(plot_x, plot_x, linewidth=2, color="green")  # vx
        graphBR, = ax[1,1].plot(plot_x, plot_x, linewidth=2, color="black")  # thermal energy

    while t <= _tEnd:
        # Saves each instance of the system at time t
        tube = fn.convertConservative(domain, _gamma)
        simulation[t] = np.copy(tube)

        if livePlot:
            graphTL.set_ydata(tube[:,0])  # density
            graphTR.set_ydata(tube[:,4])  # pressure
            graphBL.set_ydata(tube[:,1])  # vx
            graphBR.set_ydata(tube[:,4]/tube[:,0])  # thermal energy
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

        # Compute the numerical fluxes at each interface
        if _config == "sin":
            # Use periodic boundary for edge cells
            qLs, qRs = np.concatenate(([domain[-1]],domain)), np.concatenate((domain,[domain[0]]))
        else:
            # Use outflow boundary for edge cells
            qLs, qRs = np.concatenate(([domain[0]],domain)), np.concatenate((domain,[domain[-1]]))
        hydroTube = solver.LFSolver(_gamma)
        fluxes = hydroTube.calculateRiemannFlux(qLs, qRs)

        # Compute new time step
        dt = _cfl * dx/hydroTube.eigmax

        # Update the new solution with the computed time step and the (numerical fluxes?)
        domain -= ((dt/dx) * np.diff(fluxes, axis=0))
        t += dt
    return simulation

##############################################################################

lap = time.time()
run = runSimulation(cells, config)
print(time.time() - lap, len(run))

#plotter.plotQuantities(run, index=-1, start=startPos, end=endPos)
#plotter.plotSolutionErrors(run, start=startPos, end=endPos)
#plotter.makeVideo(run, start=startPos, end=endPos)