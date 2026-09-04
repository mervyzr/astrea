import numpy as np

from functions import numeric
from functions import grid as gutils
from numkit import c_transport as ct
from spatial import pcm, plm, ppm, weno, cweno, wenoz, teno

##############################################################################
# Collates and controls space evolution
##############################################################################
# The axis sweeps used to run concurrently on a ThreadPoolExecutor. They are serial now, for
# two reasons. Their temporaries were co-resident, which set the peak footprint at roughly
# three times one sweep's working set -- measured 74x the state array with three threads
# against 34x with one, i.e. the difference between 256^3 fitting in 64 GB and not. And the
# reconstruction now calls numba kernels, whose threading layer here is workqueue, which is
# not thread safe and aborts the process if a parallel kernel is entered from more than one
# Python thread. The parallelism moved inside the kernels, where it is over cells rather than
# over three axes, so there is more of it available than there was.


# Pick the per-axis reconstruction routine for the configured subgrid model
def get_runner(sim_variables):
    subgrid, subgrid_category = sim_variables.subgrid, sim_variables.subgrid_category

    if subgrid_category == "weno":
        if subgrid.startswith("c"):
            return cweno.run
        elif subgrid.endswith("z"):
            return wenoz.run
        return weno.run
    elif subgrid_category == "eno":
        return teno.run
    elif subgrid_category == "ppm":
        return ppm.run
    elif subgrid_category == "plm":
        return plm.run
    return pcm.run


def evolve(grid, sim_variables, first_stage=False):
    magnetic = sim_variables.magnetic
    multidimensional, axes, ds = sim_variables.multidimensional, sim_variables.axes, sim_variables.ds

    # Convert to primitive variables
    primitive = ct.convert("conservative", grid, sim_variables) if sim_variables.magnetic else gutils.convert("conservative", grid, sim_variables)


    # Hydrodynamics computation (with fluxes and eigmax)
    runner = get_runner(sim_variables)
    run_args = ()
    if sim_variables.subgrid_category == "ppm" and sim_variables.ppm_dissipate:
        # Compute additional dissipation for PPM, if active
        run_args = (ppm.get_flattening_coeff(primitive, sim_variables),)

    fluxes, characteristics, interfaces = zip(*(
        runner(primitive, sim_variables, axis, *run_args) for axis in axes
    ))


    # Magnetohydrodynamics computation
    if magnetic and multidimensional:
        # Compute alphas for CT computation
        alphas = {ax:ct.compute_alphas(characteristics[idx], axis=ax) for idx, ax in enumerate(axes)}

        # Magnetic transverse interfaces reconstructed along orthogonal axis/axes; use the averaged (+) & (-) values
        axis_interfaces = dict(zip(axes, interfaces))
        normal_interfaces = {
            idx: ct.reconstruct_transverse(axis_interfaces, sim_variables, idx) for idx in range(3)
        }

        # The proper assignment of the corners is important for directional updates, so the dict keys are used for this assignment
        if sim_variables.ct_dissipative:
            emfs = {idx: ct.compute_emf(normal_interfaces, alphas, idx, dissipative=True) for idx in range(3)}
        else:
            emfs = {idx: ct.compute_emf(normal_interfaces, alphas, idx) for idx in range(3)}

        # Update fluxes with CT implementation
        fluxes = tuple(
            ct.compute_ct_flux(flux, emfs, sim_variables, axis) for flux, axis in zip(fluxes, axes)
        )

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
