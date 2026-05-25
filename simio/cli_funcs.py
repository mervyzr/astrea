import random
import argparse

from tinydb import TinyDB, Query

from functions.generic import BColours

##############################################################################
# I/O functions for CLI inputs
##############################################################################

def parse_CLI(db_path):

    def bool_handler(value):
        return (value.lower() == 'true' or value.lower() == '1')

    db, params = TinyDB(db_path), Query()

    bool_choices = ['true','false','True','False',1,0]
    accepted_values = lambda _type: [value for category in db.search(params.type == _type) for value in category['accepted']]
    quotes = db.get(params.type == 'quotes')['name']

    parser = argparse.ArgumentParser(description='Astrea is a multi-dimensional magnetohydrodynamics simulation written in Python 3. Refer to the README for more information.', 
                                     epilog=f"--- {BColours.ITALIC}{quotes[random.randint(0,len(quotes)-1)]}{BColours.ENDC} ---", 
                                     formatter_class=argparse.RawTextHelpFormatter, 
                                     usage=argparse.SUPPRESS)

    parser.add_argument('-v', '--verbose', dest='verbose', help='switch on verbose description of simulation', action='store_true')
    parser.add_argument('-q', '--quiet', dest='quiet', help='switch off printing to screen', action='store_true')
    parser.add_argument('-w', '--write', dest='write_chkpt', help='switch on checkpoint file saving', action='store_true')
    parser.add_argument('-t', '--test', dest='test', help='run the tests for astrea (convergence, conservation, etc.)', action='store_true')

    parser.add_argument('--config', metavar='', type=str.lower, default=argparse.SUPPRESS, help='configuration to run in the simulation', choices=accepted_values('config'))
    parser.add_argument('--cells', '--grid', dest='cells', metavar='', default=argparse.SUPPRESS, help='number of cells in the grid')
    parser.add_argument('--cfl', metavar='', type=float, default=argparse.SUPPRESS, help='Courant number in the Courant-Friedrichs-Lewy stability condition')
    parser.add_argument('--gamma', metavar='', type=float, default=argparse.SUPPRESS, help='adiabatic index')
    parser.add_argument('--dimensions', type=int, metavar='', default=argparse.SUPPRESS, help='dimensionality of the simulation', choices=db.get(params.type == 'dimensions')['accepted'])
    parser.add_argument('--gravity', metavar='', type=str.lower, default=argparse.SUPPRESS, help='set gravity in the simulation', choices=db.get(params.type == 'gravity')['accepted'])
    parser.add_argument('--units', metavar='', type=str.lower, default=argparse.SUPPRESS, help='set units/scale of the simulation', choices=db.get(params.type == 'units')['accepted'])

    parser.add_argument('--subgrid', metavar='', type=str.lower, default=argparse.SUPPRESS, help='subgrid model used for reconstruction within grid cells', choices=accepted_values('subgrid'))
    parser.add_argument('--time_evo', metavar='', type=str.lower, default=argparse.SUPPRESS, help='time integration method used for temporal evolution', choices=accepted_values('time_evo'))
    parser.add_argument('--solver', metavar='', type=str.lower, default=argparse.SUPPRESS, help='solver method for the Riemann problem', choices=accepted_values('solver'))

    parser.add_argument('--checkpoints', metavar='', type=int, default=argparse.SUPPRESS, help='number of checkpoints in simulation')

    parser.add_argument('--live_plot', metavar='', type=bool_handler, default=argparse.SUPPRESS, help='toggle the live plotting function', choices=bool_choices)
    parser.add_argument('--save_snaps', metavar='', type=bool_handler, default=argparse.SUPPRESS, help='toggle saving snapshots of the simulation', choices=bool_choices)
    parser.add_argument('--save_plots', metavar='', type=bool_handler, default=argparse.SUPPRESS, help='toggle saving quantitative plots of the simulation', choices=bool_choices)
    parser.add_argument('--save_video', metavar='', type=bool_handler, default=argparse.SUPPRESS, help='toggle saving a video of the simulation', choices=bool_choices)
    parser.add_argument('--save_file', metavar='', type=bool_handler, default=argparse.SUPPRESS, help='toggle saving the simulation data file (.hdf5)', choices=bool_choices)
    parser.add_argument('--plot_style', metavar='', type=str.lower, default=argparse.SUPPRESS, help='plot styles (based on matplotlib style sheets)')
    parser.add_argument('--plot_options', metavar='', nargs="*", type=str.lower, default=argparse.SUPPRESS, help='simulation variable to plot (appendable)')

    parser.add_argument('--file', dest='chkpt_file', metavar='', type=str, default='', help='(absolute) path to astrea checkpoint file')
    parser.add_argument('--tracers', help='switch on tracer particles in the simulation', action='store_true')

    parser.add_argument('--chemistry', help='switch on chemical network in simulation', action='store_true')
    parser.add_argument('--network', metavar='', type=str.lower, default='', help='(absolute) path to chemical network file')
    parser.add_argument('--abundances', metavar='', type=str.lower, default='', help='(absolute) path to (.yml) file for initial abundances of chemical species')

    parser.add_argument('--init', default=argparse.SUPPRESS, help=argparse.SUPPRESS, action='store_true')

    args = parser.parse_args()

    return vars(args)