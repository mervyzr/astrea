import sys
import signal

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def graceful_exit(sig, frame):
    sys.stdout.write('\033[2K\033[1G')
    print("Simulation end by SIGINT; exiting gracefully..")
    sys.exit(0)

def initiate_live_plot(arr, axes=False):
    plt.ion()
    fig, ax = plt.subplots(1)

    if axes:
        fig.text(0.5, 0.04, r"$x$", ha='center')
        fig.text(0.04, 0.5, r"$y$", ha='center', rotation='vertical')
    else:
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    graph = ax.imshow(arr, interpolation="None", cmap=colors.ListedColormap(['#3e4a56', '#cec6b3']))  # [background_colour, object_colour]

    if axes:
        plt.gca().invert_yaxis()
        ax.set_xticks(np.linspace(0,len(arr),5))
        ax.set_yticks(np.linspace(0,len(arr),5))
    else:
        ax.axis('tight')
        ax.axis('off')
    return fig, graph

def update_plot(arr, t, *plot_functions):
    fig, graph = plot_functions

    graph.set_data(arr)

    if fig.texts == []:
        fig.text(0, 0, f't={t}', fontsize=20, color='white')
    else:
        fig.texts[0].set_text(f't={t}')

    fig.canvas.draw()
    fig.canvas.flush_events()
    pass


def run(N, config='random', t_end='inf', tol=.5):
    original_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, graceful_exit)

    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="Conway's Game of Life simulation")

        parser.add_argument('--size', '--N', '--n', '--number', dest='N', metavar='', type=int, default=argparse.SUPPRESS, help='length of the simulation box')
        parser.add_argument('--config', metavar='', type=str.lower, default=argparse.SUPPRESS, help='initial conditions', choices=['random','glider'])
        parser.add_argument('--t_end', metavar='', default=argparse.SUPPRESS, help='end time for simulation')
        parser.add_argument('--tol', metavar='', type=float, default=argparse.SUPPRESS, help='tolerance for populating random configuration')

        args = vars(parser.parse_args())

        if args.get('N'):
            N = args['N']
        if args.get('config'):
            config = args['config']
        if args.get('t_end'):
            t_end = args['t_end']
        if args.get('tol'):
            tol = args['tol']

    if t_end == 'inf':
        t_end = 31540000

    # Generate initial condition
    board = np.zeros((N,N))
    if config == 'glider':
        board[N-1,:3] = 1
        board[N-2,2] = 1
        board[N-3,1] = 1
    else:
        mask = np.random.rand(N,N) >= tol
        board[mask] = 1

    plot_functions = initiate_live_plot(board)
    t, dt = 0, 1

    inf_count, inf_tolerance = 0, 30
    states = np.array(3 * [np.zeros_like(board)])

    while (board.any() and t <= t_end):
        update_plot(board, t, *plot_functions)

        # Save previous states for checks for infinite loop
        if t <= 2:
            states[t] = np.copy(board)
        else:
            states = np.roll(states, shift=-1, axis=0)
            states[-1] = np.copy(board)

        # Check for infinite loop
        if (states[0] == states[-1]).all():
            if 'first_count' not in locals():
                first_count = t
            inf_count += 1
            if inf_count >= inf_tolerance:
                print(f"Simulation only left with period-2 oscillators, ending game at t={first_count}")
                break
        else:
            # Create boundary conditions
            padded_board = np.pad(board, ((1,1), (1,1)), mode="wrap")

            # Check neighbours (L-R-U-D-TL-TR-BL-BR)
            neigbours = (
                padded_board[1:-1,:-2]
                + padded_board[1:-1,2:]
                + padded_board[:-2,1:-1]
                + padded_board[2:,1:-1]
                + padded_board[:-2,:-2]
                + padded_board[:-2,2:]
                + padded_board[2:,:-2]
                + padded_board[2:,2:]
                )

            # Update step
            board[(board == 1) & ((neigbours < 2) | (neigbours > 3))] = 0
            board[(board == 0) & (neigbours == 3)] = 1

            t += dt
            if t%10 == 0:
                print(f"step = {t}", end="\r")

    signal.signal(signal.SIGINT, original_sigint_handler)

run(500)