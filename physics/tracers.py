import numpy as np

from functions import grid as gutils

##############################################################################
# Functions for tracer particles, including initialisation and updating
##############################################################################

# Initialise the discrete POINTWISE tracer particles at the centre of each cell; returns a (N x dim) x 3 array
def initialise(sim_variables):
    cells, dimensions, multidimensional, coordinates = sim_variables.cells, sim_variables.dimensions, sim_variables.multidimensional, sim_variables.coordinates

    _, physical_grid_x = gutils.make_physical_grid(coordinates, cells, 0)

    if multidimensional:
        _, physical_grid_y = gutils.make_physical_grid(coordinates, cells, 1)
        if dimensions > 2:
            _, physical_grid_z = gutils.make_physical_grid(coordinates, cells, 2)

            grid = np.meshgrid(physical_grid_x, physical_grid_y, physical_grid_z, indexing='ij')
        else:
            grid = np.meshgrid(physical_grid_x, physical_grid_y, indexing='ij')
    else:
        grid = physical_grid_x

    return np.stack(grid, axis=-1)


# Updating tracer particles positions based on velocities with forward Euler
def update(tracers, grid, dt, sim_variables):
    tracers += dt * grid[...,1+(sim_variables.axes)]

    # Enforce periodic boundaries
    if sim_variables.boundary == "wrap":
        for axis, axis_coord in sim_variables.coordinates.items():
            tracers[...,axis] = axis_coord[0] + (
                (tracers[...,axis] - axis_coord[0]) % np.abs(np.diff(axis_coord)[0])
            )

    return tracers