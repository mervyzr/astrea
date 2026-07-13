import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


twod = False


case1 = {
    'name': "rotation-0-0",
    'pcm': "/Users/mervin/Desktop/momentum-check/astrea_blob_pcm_euler_85842006.hdf5",
    'cweno': "/Users/mervin/Desktop/momentum-check/astrea_blob_cweno_ssprk54_19381888.hdf5",
    'wenoz': "/Users/mervin/Desktop/momentum-check/astrea_blob_wenoz_ssprk54_62002417.hdf5",
    'cells': [100] * 3,
    'angles': [0,0],
    't_limit': 0.15,  # time limit before expansion crosses the boundary (unrealistic effects and thus gives unrealistic results)
}

case2 = {
    'name': "rotation-90-180",
    'pcm': "/Users/mervin/Desktop/momentum-check/astrea_blob_pcm_euler_72761695.hdf5",
    'cweno': "/Users/mervin/Desktop/momentum-check/astrea_blob_cweno_ssprk54_8770989.hdf5",
    'cells': [100] * 3,
    'angles': [90,180],
    't_limit': 0.15,
}

case3 = {
    'name': "rotation-45-135",
    'pcm': "/Users/mervin/Desktop/momentum-check/astrea_blob_pcm_euler_33348612.hdf5",
    'plm': "/Users/mervin/Desktop/momentum-check/astrea_blob_plm_euler_92195069.hdf5",
    'ppm': "/Users/mervin/Desktop/momentum-check/astrea_blob_ppm_ssprk54_38987345.hdf5",
    'cweno': "/Users/mervin/Desktop/momentum-check/astrea_blob_cweno_ssprk54_21654850.hdf5",
    'wenoz': "/Users/mervin/Desktop/momentum-check/astrea_blob_wenoz_ssprk54_77214132.hdf5",
    'cells': [64] * 3,
    'angles': [45,135],
    't_limit': 0.1,
}

case4 = {
    'name': "rotation-73-298",
    'pcm': "/Users/mervin/Desktop/momentum-check/astrea_blob_pcm_euler_66799515.hdf5",
    'plm': "/Users/mervin/Desktop/momentum-check/astrea_blob_plm_euler_66122450.hdf5",
    'cweno': "/Users/mervin/Desktop/momentum-check/astrea_blob_cweno_ssprk54_60332680.hdf5",
    'wenoz': "/Users/mervin/Desktop/momentum-check/astrea_blob_wenoz_ssprk54_18849282.hdf5",
    'cells': [128] * 3,
    'angles': [73,298],
    't_limit': 0.03,
}

case5 = {
    'name': "rotation-122-298",
    'pcm': "/Users/mervin/Desktop/momentum-check/astrea_blob_pcm_euler_20384003.hdf5",
    'cweno': "/Users/mervin/Desktop/momentum-check/astrea_blob_cweno_ssprk54_74571890.hdf5",
    'wenoz': "/Users/mervin/Desktop/momentum-check/astrea_blob_wenoz_ssprk54_98804939.hdf5",
    'cells': [100] * 3,
    'angles': [122,298],
    't_limit': 0.03,
}


axes = np.array(range(3))
axis_coord = [-.5, .5]
shock_pos = .05
gamma = 1.4
omega = 50

# interpolated stuff for CWENO
def convert(grid):
    base = np.copy(grid)
    for idx, expansion in enumerate([laplacian(grid, axis) for axis in axes]):
        base += (ds[axes[idx]]**2)/24 * expansion
    return base

def laplacian(grid, axis):
    padded_grid = add_boundary(grid, axis=axis)
    return 1/(ds[axis]**2) * (np.diff(slice_(padded_grid, axis, start=1), axis=axis) - np.diff(slice_(padded_grid, axis, end=-1), axis=axis))

def add_boundary(grid, stencil=1, axis=0):
    padding = [(0,0)] * grid.ndim
    padding[axis] = (stencil,stencil)
    return np.pad(grid, padding, mode='wrap')

def slice_(grid, axis, start=0, end=None, step=1, *args):
    slc = [slice(None)] * grid.ndim
    if args and (2 <= len(args) <= 3):
        try:
            start, end, step = args
        except ValueError:
            start, end = args
    if end == None:
        end = grid.shape[axis]
    slc[axis] = slice(start, end, step)
    return grid[tuple(slc)]

def make_physical_grid(coordinates, cells, idx):
    start_pos, end_pos = coordinates[idx]
    dh = np.abs(np.diff(coordinates[idx])[0])/cells[idx]
    half_cell = .5 * dh
    return np.average(coordinates[idx]), np.linspace(start_pos-half_cell, end_pos+half_cell, cells[idx]+2)[1:-1]

def initiate(interpolate):
    density = gamma**2/10 + smoothing(gamma**2)
    pressure = gamma/10 + smoothing(.25 * gamma**2/10 * (omega*r0)**2)
    vx = smoothing(omega) * (omega_hat[1]*z - omega_hat[2]*y)
    vy = smoothing(omega) * -(omega_hat[0]*z - omega_hat[2]*x)
    vz = smoothing(omega) * (omega_hat[0]*y - omega_hat[1]*x)

    if interpolate:
        density = convert(density)
        pressure = convert(pressure)
        vx = convert(vx)
        vy = convert(vy)
        vz = convert(vz)

    Lx0 = (y * density*vz - z * density*vy) * dV
    Ly0 = -(x * density*vz - z * density*vx) * dV
    Lz0 = (x * density*vy - y * density*vx) * dV

    m0 = np.sum(density) * V
    p0 = np.sum(density*vx + density*vy + density*vz)
    E0 = np.sum(pressure/(density * (gamma-1)) + .5*(vx**2 + vy**2 + vz**2))

    L0 = np.stack([Lx0, Ly0, Lz0], axis=-1)
    L0_mag = np.sum(np.sqrt(Lx0**2 + Ly0**2 + Lz0**2))
    L0_hat = L0/L0_mag
    theta0 = np.degrees(np.arccos(L0_hat[...,2]))
    phi0 = np.degrees(np.arctan2(L0_hat[...,1], L0_hat[...,0])) % 360

    return m0, p0, E0, L0, L0_mag, L0_hat, theta0, phi0

def plucker(hdf5, t_limit, interpolate=False):
    timings = []
    total_mass, total_momentum, total_energy = [], [], []
    total_angular, total_error, total_theta, total_phi = [], [], [], []
    thetas, phis = [], []

    full_simulation = hdf5[list(hdf5.keys())[0]]

    m0, p0, E0, L0, L0_mag, L0_hat, theta0, phi0 = initiate(interpolate=interpolate)

    for timestamp in full_simulation.keys():
        if float(timestamp) <= t_limit:
            timings.append(float(timestamp))
            grid = full_simulation[timestamp][:]

            px = grid[...,0] * grid[...,1]
            py = grid[...,0] * grid[...,2]
            pz = grid[...,0] * grid[...,3]

            # L = r x p
            Lx = (y * pz - z * py) * dV
            Ly = -(x * pz - z * px) * dV
            Lz = (x * py - y * px) * dV

            L = np.stack([Lx, Ly, Lz], axis=-1)
            L_mag = np.sum(np.sqrt(Lx**2 + Ly**2 + Lz**2))
            L_hat = L/L_mag

            # Conservative elements
            total_mass.append(np.sum(grid[...,0]*V) - m0)
            total_momentum.append(np.sum(px + py + pz) - p0)
            total_energy.append(np.sum(grid[...,4]/(grid[...,0] * (gamma-1)) + .5*(grid[...,1]**2 + grid[...,2]**2 + grid[...,3]**2)) - E0)

            # Total angular momentum
            total_angular.append(np.sum((L_mag - L0_mag) / L0_mag))

            # Relative error
            relative_error = np.sum(np.linalg.norm(L-L0, axis=-1)/np.linalg.norm(L0, axis=-1))
            total_error.append(relative_error)

            # Theta deviation
            theta_current = np.degrees(np.arccos(L_hat[...,2]))
            total_theta.append(np.average(theta_current - theta0))
            thetas.append(theta_current - theta0)

            # Phi deviation
            phi_current = np.degrees(np.arctan2(L_hat[...,1], L_hat[...,0])) % 360
            total_phi.append(np.average(phi_current - phi0))
            phis.append(phi_current - phi0)
        else:
            break

    return timings, total_angular, total_error, total_theta, total_phi, np.sum(thetas, axis=0), np.sum(phis, axis=0)







# start
cases = [case3, case4, case5]

mpl.rcParams['text.usetex'] = True
fig, ax = plt.subplots(1,len(cases))
plt.rcParams['text.latex.preamble'] = r"\usepackage{lmodern}"
params = {
    'font.size': 12,
    'font.family': 'DejaVuSans',
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,

    'figure.dpi': 300,
    'savefig.dpi': 300,

    'lines.linewidth': 1.0,
    'lines.dashed_pattern': [3, 2]
}
plt.rcParams.update(params)
plt.suptitle(r"$\Delta L_\mathrm{total} = \mathrm{sum} \left( \frac{ \| \vec{L}_t \| - \| \vec{L}_0 \| }{ \| \vec{L}_0 \| } \right)$")


for idx, case in enumerate(cases):
    cells = case['cells']
    theta, phi = case['angles']
    t_limit = case['t_limit']

    coordinates = {ax: axis_coord for ax in range(len(cells))}
    ds = {ax: np.abs(np.diff(coordinates[ax]))/cells[ax] for ax in range(len(cells))}
    dV = np.prod(list(ds.values()))
    V = np.diff(axis_coord)**3

    x_centre, physical_grid_x = make_physical_grid(coordinates, cells, 0)
    y_centre, physical_grid_y = make_physical_grid(coordinates, cells, 1)
    z_centre, physical_grid_z = make_physical_grid(coordinates, cells, 2)

    x, y, z = np.meshgrid(physical_grid_x, physical_grid_y, physical_grid_z, indexing='ij')
    x0, y0, z0 = x - x_centre, y - y_centre, z - z_centre
    r = np.sqrt(x0**2 + y0**2 + z0**2)
    r0 = np.sqrt((shock_pos-x_centre)**2 + (shock_pos-y_centre)**2 + (shock_pos-z_centre)**2)

    omega_hat = np.array([np.sin(theta*np.pi/180)*np.cos(phi*np.pi/180), np.sin(theta*np.pi/180)*np.sin(phi*np.pi/180), np.cos(theta*np.pi/180)])

    ndotr = np.dot(np.stack([x,y,z], axis=-1), omega_hat)
    R = np.sqrt(r**2 - ndotr**2)
    smoothing = lambda q: q * np.exp(-.5 * (R/r0)**2)

    # Collate data for each case
    rthetas, rphis, plts_2d = [], [], []
    try:
        with h5py.File(case['cweno'], 'r') as hdf5:
            results = plucker(hdf5, t_limit, interpolate=True)
    except:
        pass
    else:
        if twod:
            rthetas.append(results[-2])
            rphis.append(results[-1])
            plts_2d.append('CWENO')
        else:
            ax[idx].plot(np.asarray(results[0]), np.asarray(results[1]), color='blue', label='CWENO')

    try:
        with h5py.File(case['wenoz'], "r") as hdf5:
            results = plucker(hdf5, t_limit)
    except:
        pass
    else:
        if twod:
            rthetas.append(results[-2])
            rphis.append(results[-1])
            plts_2d.append('WENO-Z')
        else:
            ax[idx].plot(np.asarray(results[0]), np.asarray(results[1]), color='green', label='WENO-Z')

    try:
        with h5py.File(case['ppm'], "r") as hdf5:
            results = plucker(hdf5, t_limit, interpolate=True)
    except:
        pass
    else:
        if twod:
            rthetas.append(results[-2])
            rphis.append(results[-1])
            plts_2d.append('PPM')
        else:
            ax[idx].plot(np.asarray(results[0]), np.asarray(results[1]), color='purple', label='PPM')

    try:
        with h5py.File(case['plm'], "r") as hdf5:
            results = plucker(hdf5, t_limit)
    except:
        pass
    else:
        if twod:
            rthetas.append(results[-2])
            rphis.append(results[-1])
            plts_2d.append('PLM')
        else:
            ax[idx].plot(np.asarray(results[0]), np.asarray(results[1]), color='black', label='PLM')

    try:
        with h5py.File(case['pcm'], "r") as hdf5:
            results = plucker(hdf5, t_limit)
    except:
        pass
    else:
        if twod:
            rthetas.append(results[-2])
            rphis.append(results[-1])
            plts_2d.append('PCM')
        else:
            ax[idx].plot(np.asarray(results[0]), np.asarray(results[1]), color='red', label='PCM')

    ax[idx].grid(linestyle="--", linewidth=0.5)
    ax[idx].set_title(rf"$\theta = {theta}^\circ, \phi = {phi}^\circ$")
    ax[idx].set_xlabel(r"$t$")
    #ax[idx].set_ylabel(r"$\Delta L_\mathrm{total}$")

    leg = ax[idx].legend()
    for line in leg.get_lines():
        line.set_linewidth(3)
    """for text in leg.get_texts():
        text.set_fontsize('x-large')"""

plt.show()
#plt.savefig(f"/Users/mervin/Desktop/momentum-check/{name}.png", bbox_inches='tight')

plt.clf()
plt.cla()
plt.close()




if twod:
    indexes = []
    if 2 * len(plts_2d) < 2:
        rows = 1
    elif 2 * len(plts_2d) <= 10:
        rows = 2
    else:
        rows = 3
    cols = (2*len(plts_2d))//rows
    for row in range(rows):
        for col in range(cols):
            indexes.append([row,col])
    mpl.rcParams['text.usetex'] = True
    fig, ax = plt.figure(figsize=(4*cols, 4*rows)), np.full((rows, cols), None)
    plt.rcParams['text.latex.preamble'] = r"\usepackage{lmodern}"
    params = {
        'font.size': 12,
        'font.family': 'DejaVuSans',
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,

        'figure.dpi': 300,
        'savefig.dpi': 300,

        'lines.linewidth': 1.0,
        'lines.dashed_pattern': [3, 2]
    }
    plt.rcParams.update(params)
    spec = gridspec.GridSpec(rows, cols*2, figure=fig)

    for _i in range(2*len(plts_2d)):
        row, col = divmod(_i, cols)
        if row < (2*len(plts_2d))//cols:
            ax[row,col] = fig.add_subplot(spec[row, 2*col:2*(col+1)])
        else:
            extra = cols - (2*len(plts_2d)) % cols
            ax[row,col] = fig.add_subplot(spec[row, 2*col+extra:2*(col+1)+extra])


    grab = lambda q, ax=2: np.sqrt(np.take(q, int(cells[ax]/2), axis=ax)**2)
    #grab = lambda q, ax=2: np.sum(q, axis=ax)

    for j in range(len(plts_2d)):
        for i in range(2):
            if i == 0:
                data = grab(rthetas[j])
                label = r"$\mid \Delta \theta \mid = \sqrt{(\theta_{\hat{L}} - \theta_{\hat{L}_0})^2}$"
                colour = "viridis"
            elif i == 1:
                data = grab(rphis[j])
                label = r"$\mid \Delta \phi \mid = \sqrt{(\phi_{\hat{L}} - \phi_{\hat{L}_0})^2}$"
                colour = "plasma"
            graph = ax[i,j].imshow(data, interpolation="nearest", cmap=colour, origin="lower")
            divider = make_axes_locatable(ax[i,j])
            cax = divider.append_axes(position='right', size='5%', pad=0.05)
            fig.colorbar(graph, cax=cax, orientation='vertical')
            ax[i,j].set_title(plts_2d[j])
            if j == 0:
                ax[i,j].set_ylabel(label, ha='center', rotation='vertical')

    plt.tight_layout()
    fig.subplots_adjust(left=0.1)

    plt.show()

    plt.clf()
    plt.cla()
    plt.close()