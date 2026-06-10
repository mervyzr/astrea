import concurrent.futures
from itertools import repeat

import numpy as np

from functions import grid as gutils
from numkit import c_transport as ct
from spatial import pcm, plm, ppm, weno, cweno, wenoz, teno

##############################################################################
# Collates and controls space evolution
##############################################################################

def evolve(grid, sim_variables, first_stage=False):
    multidimensional, subgrid, subgrid_category, axes, magnetic = sim_variables.multidimensional, sim_variables.subgrid, sim_variables.subgrid_category, sim_variables.axes, sim_variables.magnetic

    # Convert to primitive variables
    primitive = ct.convert("conservative", grid, sim_variables) if sim_variables.magnetic else gutils.convert("conservative", grid, sim_variables)


    # Hydrodynamics computation (with fluxes and eigmax)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        if subgrid_category == "weno":
            if subgrid.startswith("c"):
                jobs = executor.map(cweno.run, repeat(primitive), repeat(sim_variables), axes)
            elif subgrid.endswith("z"):
                jobs = executor.map(wenoz.run, repeat(primitive), repeat(sim_variables), axes)
            else:
                jobs = executor.map(weno.run, repeat(primitive), repeat(sim_variables), axes)

        elif subgrid_category == "eno":
            jobs = executor.map(teno.run, repeat(primitive), repeat(sim_variables), axes)

        elif subgrid_category == "ppm":
            # Compute additional dissipation for PPM, if active
            if sim_variables.ppm_dissipate:
                eta = ppm.get_flattening_coeff(primitive, sim_variables)
                jobs = executor.map(ppm.run, repeat(primitive), repeat(sim_variables), axes, repeat(eta))
            else:
                jobs = executor.map(ppm.run, repeat(primitive), repeat(sim_variables), axes)

        elif subgrid_category == "plm":
            jobs = executor.map(plm.run, repeat(primitive), repeat(sim_variables), axes)

        else:
            jobs = executor.map(pcm.run, repeat(primitive), repeat(sim_variables), axes)

        data = dict(zip(axes, jobs))


    # Extract specific values needed from data dict
    extract = lambda variable: {axis: data[axis][variable] for axis in axes}
    fluxes = extract('fluxes')

    # Magnetohydrodynamics computation
    if magnetic and multidimensional:
        alphas = extract('alphas')

        reconstruct_transverse = ct.reconstruct_transverse
        compute_emf = ct.compute_emf
        compute_flux = ct.compute_ct_flux

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Magnetic transverse interfaces reconstructed along orthogonal axis/axes; use the averaged (+) & (-) values
            ortho_interfaces = dict(enumerate(executor.map(reconstruct_transverse, repeat(data), repeat(sim_variables), range(3))))

            # The proper assignment of the corners is important for directional updates, so the dict keys are used for this assignment
            if sim_variables.ct_dissipative:
                compute_emf = lambda _ortho_interfaces, _alphas, _axis: ct.compute_emf(_ortho_interfaces, _alphas, _axis, dissipative=True)
            emfs = dict(enumerate(executor.map(compute_emf, repeat(ortho_interfaces), repeat(alphas), range(3))))

            # Update fluxes with CT implementation
            ct_fluxes = executor.map(compute_flux, map(fluxes.get, axes), repeat(emfs), repeat(sim_variables), axes)
            fluxes.update(dict(zip(axes, ct_fluxes)))

    # Calculate the total fluxes through all upwind surfaces [F(i+1/2,j,k) - F(i-1/2,j,k)]/dx, [G(i,j+1/2,k) - G(i,j-1/2,k)]/dy, [H(i,j,k+1/2) - H(i,j,k-1/2)]/dz
    total_flux = -np.sum(list(fluxes.values()), axis=0)

    if first_stage:
        # Compute the maximum eigenvalues from each axis for determining the full time step
        eigmaxes = extract('eigmax')

        return total_flux, np.min(list(eigmaxes.values()))
    else:
        return total_flux