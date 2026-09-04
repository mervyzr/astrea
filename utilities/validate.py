#!/usr/bin/env python3
"""Reference-capture and comparison harness for the optimisation work.

Runs a fixed number of timesteps of a chosen configuration and records the
per-step dt sequence together with the final grid, so that a refactor can be
checked against a previously captured baseline.

    # capture a baseline (do this before editing anything)
    python3 utilities/validate.py capture ref.npz --config=sedov --dimensions=2 --cells=32

    # after a refactor, compare against it
    python3 utilities/validate.py compare ref.npz --config=sedov --dimensions=2 --cells=32

`compare` reports the max absolute/relative difference of the final grid and of
the dt sequence, and whether the two runs are bit-identical. Bit-identity is the
acceptance criterion for the refactors that are meant to be exact; the ones that
only change summation order are expected to differ at the last ulp.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from functions import ginit
from functions import grid as gutils
from iokit import handler, simulation
from numkit import c_transport as ct
from spatial.spatial import evolve as spatial_evolve
from static import tests
from temporal.temporal import evolve as temporal_evolve

ROOT = Path(__file__).resolve().parent.parent


def build_variables(config, dimensions, cells, subgrid=None, solver=None, time_evo=None):
    """Assemble a Variables instance without going through the CLI or the plotting stack."""
    # handler.allocate parses sys.argv through iokit.cli_funcs; hide our own flags from it
    saved_argv, sys.argv = sys.argv, sys.argv[:1]
    try:
        config_variables = handler.allocate(
            1234, ROOT, ROOT/"static"/".db.json",
            Path("/tmp/astrea_validate"), Path("/tmp/.astrea_validate_h5"),
        )
    finally:
        sys.argv = saved_argv
    config_variables["config"] = config
    config_variables["dimensions"] = dimensions
    config_variables["cells"] = np.array([cells]*dimensions)
    for key, value in (("subgrid", subgrid), ("solver", solver), ("time_evo", time_evo)):
        if value is not None:
            config_variables[key] = value

    test_variables = tests.generate_test_conditions(config_variables)
    sim_variables = simulation.Variables(config_variables, test_variables)

    # Everything that would otherwise pull in matplotlib or write to disk
    sim_variables.live_plot = False
    sim_variables.record_all_steps = False
    sim_variables.save_snaps = sim_variables.save_plots = False
    sim_variables.save_video = sim_variables.save_file = False
    sim_variables.write_chkpt = False
    sim_variables.quiet = True

    sim_variables.dimensions = dimensions
    sim_variables.multidimensional = dimensions >= 2
    sim_variables.cells = np.array([cells]*dimensions)
    sim_variables.axes = np.array(range(dimensions))
    sim_variables.ds = {
        ax: np.abs(np.diff(sim_variables.coordinates[ax]))/cells for ax in range(dimensions)
    }
    for attr, default in (("magnetic", False), ("tracers", False), ("turbulence", False),
                          ("chemistry", False), ("gravity", False), ("ext_gravity", False)):
        if not hasattr(sim_variables, attr):
            setattr(sim_variables, attr, default)
    return sim_variables


def run_steps(sim_variables, nsteps):
    """Advance the solution nsteps times, returning the dt sequence and the final grid."""
    # ginit.initialise is what sets sim_variables.magnetic (from the initial B field)
    initial = ginit.initialise(sim_variables)
    convert = ct.convert if sim_variables.magnetic else gutils.convert
    grid = convert("primitive", initial, sim_variables)

    dts = []
    for _ in range(nsteps):
        fluxes, eigmax = spatial_evolve(grid, sim_variables, first_stage=True)
        dt = sim_variables.cfl * eigmax
        dts.append(float(dt))
        grid = temporal_evolve(spatial_evolve, grid, fluxes, dt, sim_variables)
        sim_variables.axes = np.roll(sim_variables.axes, shift=-1)
    return np.array(dts), grid


def describe(sim_variables, nsteps):
    return (f"config={sim_variables.config} subgrid={sim_variables.subgrid} "
            f"solver={sim_variables.solver} time_evo={sim_variables.time_evo} "
            f"dims={sim_variables.dimensions} cells={sim_variables.cells[0]} "
            f"magnetic={sim_variables.magnetic} steps={nsteps}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["capture", "compare"])
    parser.add_argument("path", help="reference .npz to write (capture) or read (compare)")
    parser.add_argument("--config", default="sedov")
    parser.add_argument("--dimensions", type=int, default=2)
    parser.add_argument("--cells", type=int, default=32)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--subgrid", default=None)
    parser.add_argument("--solver", default=None)
    parser.add_argument("--time_evo", default=None)
    parser.add_argument("--rtol", type=float, default=1e-13,
                        help="tolerance for the 'close' verdict in compare mode")
    args = parser.parse_args()

    sim_variables = build_variables(args.config, args.dimensions, args.cells,
                                    args.subgrid, args.solver, args.time_evo)
    dts, grid = run_steps(sim_variables, args.steps)
    print(describe(sim_variables, args.steps))

    if args.mode == "capture":
        np.savez_compressed(args.path, dts=dts, grid=grid)
        print(f"captured -> {args.path}  (grid {grid.shape} {grid.dtype})")
        print(f"  dt[0]={dts[0]:.17g}  dt[-1]={dts[-1]:.17g}  sum|grid|={np.abs(grid).sum():.17g}")
        return 0

    ref = np.load(args.path)
    ref_dts, ref_grid = ref["dts"], ref["grid"]
    if ref_grid.shape != grid.shape:
        print(f"SHAPE MISMATCH: reference {ref_grid.shape} vs current {grid.shape}")
        return 2

    exact_dt = np.array_equal(ref_dts, dts)
    exact_grid = np.array_equal(ref_grid, grid)
    scale = np.max(np.abs(ref_grid)) or 1.0
    dgrid = np.max(np.abs(ref_grid - grid))
    ddt = np.max(np.abs(ref_dts - dts))
    dt_scale = np.max(np.abs(ref_dts)) or 1.0

    print(f"  dt   : bit-identical={exact_dt}  max abs diff={ddt:.3e}  rel={ddt/dt_scale:.3e}")
    print(f"  grid : bit-identical={exact_grid}  max abs diff={dgrid:.3e}  rel={dgrid/scale:.3e}")

    if exact_dt and exact_grid:
        print("VERDICT: bit-identical")
        return 0
    if dgrid/scale < args.rtol and ddt/dt_scale < args.rtol:
        print(f"VERDICT: close (within rtol={args.rtol:g}) but NOT bit-identical")
        return 0
    print(f"VERDICT: DIFFERS beyond rtol={args.rtol:g}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
