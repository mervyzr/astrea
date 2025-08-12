from datetime import timedelta

##############################################################################
# Generic functions not specific to the finite volume method
##############################################################################

# Colours for printing to terminal
class BColours:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'


# Print progress status to Terminal
def print_progress(t, sim_variables):
    _seed = f"{BColours.OKBLUE}{sim_variables.seed}{BColours.ENDC}"
    _config = f"{BColours.OKCYAN}{sim_variables.config.upper()}{BColours.ENDC}"
    _cells = f"{BColours.OKCYAN}{str(sim_variables.cells).strip('[]').replace(' ','').replace(',','x')}{BColours.ENDC}"
    _subgrid = f"{BColours.OKCYAN}{sim_variables.subgrid.upper()}{BColours.ENDC}"
    _timestep = f"{BColours.OKCYAN}{sim_variables.timestep.upper()}{BColours.ENDC}"
    _solver = f"{BColours.OKCYAN}{sim_variables.solver.upper()}{BColours.ENDC}"
    _cfl = f"{BColours.OKCYAN}{sim_variables.cfl}{BColours.ENDC}"
    _dimension = f"{BColours.OKCYAN}{BColours.BOLD}({sim_variables.dimension}D){BColours.ENDC}"
    _instance = f"{BColours.WARNING}{'%.6f'%t} / {'%.2f'%sim_variables.t_end}{BColours.ENDC}"

    print(f"[{sim_variables.now.strftime('%Y-%m-%d %H:%M:%S')} | {_seed}] {_dimension} CONFIG={_config}, CELLS={_cells}, CFL={_cfl}, SUBGRID={_subgrid}, SOLVER={_solver}, TIMESTEP={_timestep} || {_instance}", end='\r')
    pass


# Print final status to Terminal
def print_final(sim_variables, timestep_count):
    _seed = f"{BColours.OKBLUE}{sim_variables.seed}{BColours.ENDC}"
    _config = f"{BColours.OKCYAN}{sim_variables.config.upper()}{BColours.ENDC}"
    _cells = f"{BColours.OKCYAN}{str(sim_variables.cells).strip('[]').replace(' ','').replace(',','x')}{BColours.ENDC}"
    _subgrid = f"{BColours.OKCYAN}{sim_variables.subgrid.upper()}{BColours.ENDC}"
    _timestep = f"{BColours.OKCYAN}{sim_variables.timestep.upper()}{BColours.ENDC}"
    _solver = f"{BColours.OKCYAN}{sim_variables.solver.upper()}{BColours.ENDC}"
    _cfl = f"{BColours.OKCYAN}{sim_variables.cfl}{BColours.ENDC}"
    _dimension = f"{BColours.OKCYAN}{BColours.BOLD}({sim_variables.dimension}D){BColours.ENDC}"
    #_performance = f"{BColours.OKGREEN}{round(sim_variables.elapsed*1e6/(np.prod(sim_variables.cells)*timestep_count), 3)} \u03BCs/(dt*cells){BColours.ENDC}"

    if sim_variables.elapsed >= 60*60:
        _elapsed = f"{BColours.FAIL}{str(timedelta(seconds=sim_variables.elapsed))}s{BColours.ENDC}"
    elif 60*60 > sim_variables.elapsed >= 30*60:
        _elapsed = f"{BColours.WARNING}{str(timedelta(seconds=sim_variables.elapsed))}s{BColours.ENDC}"
    else:
        _elapsed = f"{BColours.OKGREEN}{str(timedelta(seconds=sim_variables.elapsed))}s{BColours.ENDC}"

    print(f"[{sim_variables.now.strftime('%Y-%m-%d %H:%M:%S')} | {_seed}] {_dimension} CONFIG={_config}, CELLS={_cells}, CFL={_cfl}, SUBGRID={_subgrid}, SOLVER={_solver}, TIMESTEP={_timestep} || Elapsed: {_elapsed} ({timestep_count})", flush=True)
    pass