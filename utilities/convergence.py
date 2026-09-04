#!/usr/bin/env python3
"""Convergence-order check for the optimisation work.

Advances a smooth configuration to exactly t_end on a sequence of grid sizes and
reports the L-norm solution error together with the observed order of accuracy.
For the periodic smooth tests (sine, Gaussian) the exact solution at t_end is the
initial condition, which is what functions.analytic.calculate_solution_error
compares against.

This is the acceptance criterion for the refactors that are not bit-identical:
the observed order and the absolute error constants must both be preserved.

    python3 utilities/convergence.py --config=sine --dimensions=1 --cells 16 32 64 128
    python3 utilities/convergence.py --config=sine --dimensions=2 --cells 16 32 64
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from functions import analytic, ginit
from functions import grid as gutils
from numkit import c_transport as ct
from spatial.spatial import evolve as spatial_evolve
from temporal.temporal import evolve as temporal_evolve
from utilities.validate import build_variables


def run_to_end(sim_variables):
    """Advance to exactly t_end, clipping the last step so the final time is exact."""
    initial = ginit.initialise(sim_variables)
    convert = ct.convert if sim_variables.magnetic else gutils.convert
    grid = convert("primitive", initial, sim_variables)

    t, t_end, nsteps = 0.0, float(sim_variables.t_end), 0
    while t < t_end:
        fluxes, eigmax = spatial_evolve(grid, sim_variables, first_stage=True)
        dt = float(sim_variables.cfl * eigmax)
        if t + dt > t_end:
            dt = t_end - t
        grid = temporal_evolve(spatial_evolve, grid, fluxes, dt, sim_variables)
        sim_variables.axes = np.roll(sim_variables.axes, shift=-1)
        t += dt
        nsteps += 1
        if nsteps > 100000:
            raise RuntimeError("step limit reached; dt is probably collapsing")
    return convert("conservative", grid, sim_variables), nsteps


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="sine")
    parser.add_argument("--dimensions", type=int, default=1)
    parser.add_argument("--cells", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--norm", type=int, default=1, help="1 = L1, 2 = L2, >5 = Linf")
    parser.add_argument("--subgrid", default=None)
    parser.add_argument("--solver", default=None)
    parser.add_argument("--time_evo", default=None)
    parser.add_argument("--quantity", default="density",
                        choices=["density", "pressure", "Etot"])
    args = parser.parse_args()

    print(f"config={args.config} dims={args.dimensions} norm=L{args.norm} "
          f"quantity={args.quantity}")
    print(f"{'N':>6} {'steps':>7} {'error':>14} {'order':>8}")

    errors = []
    for cells in args.cells:
        sim_variables = build_variables(args.config, args.dimensions, cells,
                                        args.subgrid, args.solver, args.time_evo)
        grid, nsteps = run_to_end(sim_variables)
        err = analytic.calculate_solution_error(grid, sim_variables, args.norm)
        value = float(np.ravel(err[args.quantity])[0])
        order = np.log2(errors[-1]/value) if errors and value > 0 else float("nan")
        errors.append(value)
        print(f"{cells:>6} {nsteps:>7} {value:>14.6e} {order:>8.3f}")

    if len(errors) > 1:
        orders = [np.log2(errors[i]/errors[i+1]) for i in range(len(errors)-1)]
        print(f"observed orders: {[f'{o:.3f}' for o in orders]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
