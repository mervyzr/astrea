import concurrent.futures
from itertools import repeat

import numpy as np

from functions import numeric
from functions import grid as gutils
from numkit import c_transport as ct
from spatial import pcm, plm, ppm, weno, cweno, wenoz, teno

##############################################################################
# Collates and controls space evolution
##############################################################################

def evolve(grid, sim_variables, first_stage=False):
    subgrid, subgrid_category, magnetic = sim_variables.subgrid, sim_variables.subgrid_category, sim_variables.magnetic
    multidimensional, axes, ds = sim_variables.multidimensional, sim_variables.axes, sim_variables.ds

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

    fluxes, characteristics, interfaces = zip(*jobs)


    # Magnetohydrodynamics computation
    if magnetic and multidimensional:
        # Compute alphas for CT computation
        alphas = {ax:ct.compute_alphas(characteristics[idx], axis=ax) for idx, ax in enumerate(axes)}

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Magnetic transverse interfaces reconstructed along orthogonal axis/axes; use the averaged (+) & (-) values
            normal_interfaces = dict(enumerate(executor.map(ct.reconstruct_transverse, repeat(dict(zip(axes, interfaces))), repeat(sim_variables), range(3))))

            # The proper assignment of the corners is important for directional updates, so the dict keys are used for this assignment
            compute_emf = ct.compute_emf
            if sim_variables.ct_dissipative:
                compute_emf = lambda *args: ct.compute_emf(*args, dissipative=True)
            emfs = dict(enumerate(executor.map(compute_emf, repeat(normal_interfaces), repeat(alphas), range(3))))

            # Update fluxes with CT implementation
            fluxes = tuple(executor.map(ct.compute_ct_flux, fluxes, repeat(emfs), repeat(sim_variables), axes))

    # Calculate the total fluxes through all upwind surfaces [F(i+1/2,j,k) - F(i-1/2,j,k)]/dx, [G(i,j+1/2,k) - G(i,j-1/2,k)]/dy, [H(i,j,k+1/2) - H(i,j,k-1/2)]/dz
    # Accumulated in place: np.sum over a tuple of arrays first stacks them into a
    # (naxes,N,N,N,8) temporary, 3 GiB at 256^3, and then allocates the negation on top.
    # Negation is exact so ((-a)-b)-c is bit-identical to -((a+b)+c)
    total_flux = -fluxes[0]
    for flux in fluxes[1:]:
        total_flux -= flux

    if first_stage:
        # Compute the maximum eigenvalues from each axis for determining the full time step
        eigmax = np.min([ds[ax]/numeric.compute_eigmax(characteristics[idx], axis=ax) for idx, ax in enumerate(axes)])
        return total_flux, eigmax
    else:
        return total_flux