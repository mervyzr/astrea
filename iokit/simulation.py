import os
from pathlib import Path
from collections import namedtuple

import numpy as np
from tinydb import TinyDB, Query

from functions import generic
from functions.generic import BColours
from physics.krome import krome_funcs
from physics.conversions import Constants

##############################################################################
# I/O functions for simulation variables
##############################################################################

# Plucking function for creating namedtuple
def plucker(obj, *args):
    attrs = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            attrs.extend(arg)
        else:
            attrs.append(arg)
    Container = namedtuple('Container', attrs)
    return Container(*(getattr(obj, attr) for attr in attrs))


class Variables(object):
    __slots__ = [
        '__dict__',
        'rho', 'vx', 'vy', 'vz', 'pressure', 'Bx', 'By', 'Bz', 'gx', 'gy', 'gz', 'energy', 'vels', 'Bfields', 'momentums',
        'config', 'cells', 'cfl', 'gamma', 'gravity', 'self_gravity', 'ext_gravity', 'dimensions', 'subgrid', 'time_evo', 'solver',
        'coordinates', 'shock_pos', 't_end', 'boundary', 'guards', 'trim', 'test_specifics', 'init_cond', 'ambient', 'ds',
        'checkpoints', 'live_plot', 'save_snaps', 'save_plots', 'save_video', 'save_file', 'plot_style', 'plot_options',
        'axes', 'magnetic', 'convert', 'roots', 'weights', 'higher_order', 'grid_interpolate', 'multidimensional', 'config_category', 'subgrid_category', 'solver_category',
        'seed', 'now', 'elapsed', 'access_key', 'datetime', 'eps', 'home', 'save_path', 'db_path', 'hdf5', 'timesteps', 'print_status',
        'record_all_steps', 'write_chkpt', 'chkpt_file', 'quiet', 'verbose', 'test',
        'units', 'constants', 'chemistry', 'network', 'pykrome', 'species', 'abundances', 'tracers', 'nvars',
    ]

    def __init__(self, config_variables, test_variables):
        db, params = TinyDB(config_variables['db_path']), Query()

        # Declare physical variables and their index in the array: [density, vx/px, vy/py, vz/pz, pressure/energy, Bx, By, Bz]
        self.nvars = 8
        self.IDX = self.rho, self.vx, self.vy, self.vz, self.pressure, self.Bx, self.By, self.Bz = tuple(range(self.nvars))
        self.vels, self.Bfields = slice(1,4), slice(5,8)
        self.energy, self.momentums = self.pressure, self.vels

        # Parse configuration variables into the class
        for key in config_variables:
            setattr(self, key, config_variables[key])

        # Parse tests variables into the class
        for key in test_variables:
            setattr(self, key, test_variables[key])

        # Parse additional variables into the class
        self.now = None
        self.elapsed = None
        self.access_key = None
        self.timesteps = 0

        self.constants = Constants(self.units)

        self.config_category = db.get(params.accepted.any([self.config]))['category']
        self.subgrid_category = db.get(params.accepted.any([self.subgrid]))['category']
        self.solver_category = db.get(params.accepted.any([self.solver]))['category']

        # 5th-order Gauss-Legendre quadrature with interval [0,1] for OS solver
        if self.solver_category == "complete" and not self.solver.startswith("e"):
            roots, weights = np.array(list(np.polynomial.legendre.leggauss(5)))/2
            self.roots = roots + .5
            self.weights = weights

        # Generate guard zones based on the subgrid
        self.guards = 2
        if self.subgrid_category == "plm":
            self.guards = 1
        elif self.subgrid_category == "weno":
            try:
                weno_order = int(self.subgrid.replace('-','')[-1])
            except:
                pass
            else:
                if weno_order > 5:
                    self.guards = 3
        elif self.subgrid_category == "eno":
            self.guards = 3
        self.trim = (slice(self.guards,-self.guards),)*self.dimensions + (slice(None),)

        # Higher-order method options
        self.higher_order = self.grid_interpolate = False
        if self.subgrid_category in ["ppm", "eno", "weno"]:
            self.higher_order = self.grid_interpolate = True

            # WENO-Z can use point representation
            if self.subgrid_category == "weno" and (self.subgrid.endswith("z")):
                self.grid_interpolate = False

            # PPM-specific options
            if self.subgrid_category == "ppm":
                self.ppm_author = os.getenv("PPM_AUTHOR", "MC:2011")  # [McCorquodale & Colella, 2011 (MC:2011); Colella et al., 2011 (C+:2011); Peterson & Hammett, 2008 (PH:2008)]
                self.ppm_dissipate = os.getenv("PPM_DISSIPATE", False)

        # CT-specific options
        self.ct_dissipative = os.getenv("CT_DISSIPATIVE", False)

        # Axes options
        self.multidimensional = self.dimensions >= 2
        self.axes = np.array(range(self.dimensions))
        if self.dimensions > 2:
            self.slice_axis = 2  # z-axis
            self.slice_3d = int(self.cells[self.slice_axis]/2)

        # Gravity set-up
        self.self_gravity = self.ext_gravity = False
        if self.gravity:
            if self.gravity == "self":
                self.self_gravity = True
            elif self.gravity in ("ext", "external"):
                self.ext_gravity = True
            else:
                self.self_gravity = self.ext_gravity = True
        self.gravity = True if (self.self_gravity or self.ext_gravity) else False

        if self.ext_gravity:
            self.gx, self.gy, self.gz = range(3)

        # Turbulence set-up
        self.turbulence = True if "turb" in self.config else False

        # Chemistry network set-up; check if folder for chemical code exists
        if self.chemistry:
            try:
                chem_on = bool(int(self.chemistry))
            except ValueError:
                if self.chemistry not in ['krome', 'chimes', 'pychem']:
                    self.chemistry = False
            else:
                if chem_on:
                    self.chemistry = 'krome'
            self.chem_path = Path(self.home, 'physics', self.chemistry)

            if not Path.is_dir(self.chem_path):
                print(f"{BColours.WARNING}Chemistry switched on but physics/{self.chemistry} folder cannot be found. Switching off chemistry..{BColours.ENDC}")
                self.chemistry = False
                self.chem_path = ''

        # Printer functions
        if self.verbose:
            self.print_status = generic.print_verbose
        else:
            self.print_status = generic.print_simple

        # Set up boxes for plotting
        self.box_volume = np.prod([np.diff(_) for _ in self.coordinates.values()])
        if self.units != "code":
            length_scale = self.constants.plot_scales['length']
            try:
                semi = self.test_specifics['mode'].lower().startswith(('o','q'))
            except Exception:
                semi = False

            if semi:
                self.box_lengths = {ax: [start_pos, length_scale*end_pos] for ax, (start_pos, end_pos) in self.coordinates.items()}
            else:
                centres = {ax: np.average(axis_coord) for ax, axis_coord in self.coordinates.items()}
                self.box_lengths = {ax: [length_scale*(start_pos-centres[ax]), length_scale*(end_pos-centres[ax])] for ax, (start_pos, end_pos) in self.coordinates.items()}
        else:
            self.box_lengths = self.coordinates

        # Media options
        if self.test:
            self.save_plots = True
            if (self.live_plot or self.save_snaps or self.save_video):
                self.live_plot = self.save_snaps = self.save_video = False

        if (self.save_snaps or self.save_plots or self.save_video) and self.live_plot:
            print(f"{BColours.WARNING}Live plot can only be switched on when NOT saving media files because live_plot interferes with matplotlib.pyplot.savefig..{BColours.ENDC}")
            self.live_plot = False

        self.beautify_1d_plots = os.getenv("BEAUTIFY_1D_PLOTS", False)
        self.save_as_pdf = os.getenv("SAVE_AS_PDF", False)

        self.record_all_steps = True if (self.save_plots or self.save_video or self.save_file) else False