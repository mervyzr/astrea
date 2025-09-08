import os
import sys
import platform
from time import perf_counter
from datetime import timedelta

import psutil
import GPUtil
import numpy as np
from tabulate import tabulate
from pygit2 import Repository

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


def get_size(_bytes):
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if _bytes < factor:
            return f"{_bytes:.2f}{unit}B"
        _bytes /= factor


def verbose_timer(func):
    def wrapper(*args, **kwargs):
        for arg in args:
            try:
                verbose = arg.verbose
            except Exception:
                verbose = False
            else:
                break
        start = perf_counter()
        result = func(*args, **kwargs)
        if verbose:
            print(f' {func.__name__!r}           {perf_counter() - start:.5f} s')
        return result
    return wrapper


# Print progress status to Terminal
def print_simple(sim_variables, t=None, status=''):
    _seed = f"{BColours.OKBLUE}{sim_variables.seed}{BColours.ENDC}"
    _config = f"{BColours.OKCYAN}{sim_variables.config.upper()}{BColours.ENDC}"
    _cells = f"{BColours.OKCYAN}{str(sim_variables.cells).strip('[]').replace(' ','').replace(',','x')}{BColours.ENDC}"
    _subgrid = f"{BColours.OKCYAN}{sim_variables.subgrid.upper()}{BColours.ENDC}"
    _time_evo = f"{BColours.OKCYAN}{sim_variables.time_evo.upper()}{BColours.ENDC}"
    _solver = f"{BColours.OKCYAN}{sim_variables.solver.upper()}{BColours.ENDC}"
    _cfl = f"{BColours.OKCYAN}{sim_variables.cfl}{BColours.ENDC}"
    _dimension = f"{BColours.OKCYAN}{BColours.BOLD}({sim_variables.dimension}D){BColours.ENDC}"
    #_performance = f"{BColours.OKGREEN}{round(sim_variables.elapsed*1e6/(np.prod(sim_variables.cells)*sim_variables.timesteps), 3)} \u03BCs/(dt*cells){BColours.ENDC}"

    if status.lower() == 'final':
        if sim_variables.elapsed >= 60*60:
            _elapsed = f"{BColours.FAIL}{str(timedelta(seconds=sim_variables.elapsed))}s{BColours.ENDC}"
        elif 60*60 > sim_variables.elapsed >= 30*60:
            _elapsed = f"{BColours.WARNING}{str(timedelta(seconds=sim_variables.elapsed))}s{BColours.ENDC}"
        else:
            _elapsed = f"{BColours.OKGREEN}{str(timedelta(seconds=sim_variables.elapsed))}s{BColours.ENDC}"

        print(f"[{sim_variables.now.strftime('%Y-%m-%d %H:%M:%S')} | {_seed}] {_dimension} CONFIG={_config}, CELLS={_cells}, CFL={_cfl}, SUBGRID={_subgrid}, SOLVER={_solver}, TIME_EVO={_time_evo} || Elapsed: {_elapsed} ({sim_variables.timesteps})", flush=True)
        pass
    elif status.lower() == 'init':
        pass
    else:
        _instance = f"{BColours.WARNING}{'%.6f'%t} / {'%.2f'%sim_variables.t_end}{BColours.ENDC}"
        print(f"[{sim_variables.now.strftime('%Y-%m-%d %H:%M:%S')} | {_seed}] {_dimension} CONFIG={_config}, CELLS={_cells}, CFL={_cfl}, SUBGRID={_subgrid}, SOLVER={_solver}, TIME_EVO={_time_evo} || {_instance}", end='\r')
        pass


# Print verbose status to Terminal
def print_verbose(sim_variables, t=None, status=''):
    if status.lower() == "init":
        print('', '='*80)

        print(f' astrea code, branch:remotes/origin/{Repository(".").head.shorthand}')
        print(f' Python version {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}-{sys.version_info.releaselevel}')
        print(f' PYTHON_PATH={os.environ["_"]}')

        print('', '='*30, 'System Information', '='*30)
        uname = platform.uname()
        print(f' System : {"macOS" if uname.system == "Darwin" else uname.system}')
        print(f' Node : {uname.node}')
        print(f' Release : {uname.release}')
        print(f' Version : {uname.version}')
        print(f' Architecture : {uname.machine}')
        print(f' Processor : {uname.processor}')

        print('', '='*30, 'CPU Information', '='*30)
        print(f' Physical cores : {psutil.cpu_count(logical=False)}')
        print(f' Total cores : {psutil.cpu_count(logical=True)}')
        print(f' Threads per core : {psutil.cpu_count()/psutil.cpu_count(logical=False)}')

        print('', '='*30, 'Memory Information', '='*30)
        svmem = psutil.virtual_memory()
        print(f' Total : {get_size(svmem.total)}')
        print(f' Available : {get_size(svmem.available)}')
        print(f' Used : {get_size(svmem.used)}')

        print('', '='*15, 'SWAP', '='*15)
        swap = psutil.swap_memory()
        print(f' Total : {get_size(swap.total)}')
        print(f' Free : {get_size(swap.free)}')
        print(f' Used : {get_size(swap.used)}')

        print('', '='*30, 'GPU Information', '='*30)
        list_gpus = [(gpu.id, gpu.name, f'{gpu.load*100}%', f'{gpu.memoryFree}MB', f'{gpu.memoryUsed}MB', f'{gpu.memoryTotal}MB', f'{gpu.temperature} C', gpu.uuid) for gpu in GPUtil.getGPUs()]
        print(tabulate(list_gpus, headers=('id', 'name', 'load', 'free memory', 'used memory', 'total memory', 'temperature', 'uuid')))

        print(f'', '='*30, 'Disk Information', '='*30)
        for partition in psutil.disk_partitions():
            print(f' === Device: {partition.device} ===')
            print(f'    Mountpoint: {partition.mountpoint}')
            print(f'    File system type: {partition.fstype}')
            try:
                partition_usage = psutil.disk_usage(partition.mountpoint)
            except PermissionError:
                continue
            else:
                print(f'    Total size: {get_size(partition_usage.total)}')
                print(f'    Used: {get_size(partition_usage.used)}')
                print(f'    Free: {get_size(partition_usage.free)}')
                print(f'    Percentage: {partition_usage.percent}%')
        disk_io = psutil.disk_io_counters()
        print(f' Total read : {get_size(disk_io.read_bytes)}')
        print(f' Total write : {get_size(disk_io.write_bytes)}')
    
        print(f'', '='*30, 'Sim. Information', '='*30)
        print(f' Boot time : {sim_variables.now.strftime("%Y-%m-%d %H:%M:%S")}')
        print(f' Output directory : {sim_variables.save_path}')
        print(f'')
        print(f' OPTS={sys.argv[1:]}')
        print(f'')
        print(f' Seed : {sim_variables.seed}')
        print(f' Config : {sim_variables.config.title()}')
        print(f' Subgrid : {sim_variables.subgrid.upper()}')
        print(f' Time evolution : {sim_variables.time_evo.upper()}')
        print(f' Solver : {sim_variables.solver.upper()}')
        print('')
        print(f' Dimension : {sim_variables.dimension}D')
        print(f' Cells : {str(sim_variables.cells).strip("[]").replace(" ","").replace(","," x ")}')
        print(f' Boundary condition : {sim_variables.boundary.title()}')
        print(f' CFL number : {sim_variables.cfl}')
        print(f' Adiabatic index : {sim_variables.gamma}')
        print(f' Permeability : {sim_variables.permeability}')
        print('')
        print(f' Sim. end (code unit) : {sim_variables.t_end}')
        print(f' Checkpoints : {sim_variables.checkpoints}')

        print('')
        print('', '-'*30, 'Executing simulation', '-'*30)
        #print(f' step       time            CPU usage       RAM usage       Swap usage      Avg. GPU usage')
        #print(f' -----      -----           ---------       ---------       ---------       --------------')


    elif status.lower() == 'final':
        print('')
        print('Total elapsed time (HH:MM:SS):', str(timedelta(seconds=sim_variables.elapsed)), f'({sim_variables.timesteps} steps)')


    else:
        try:
            gpus_load = [[gpu.load*100] for gpu in GPUtil.getGPUs()]
        except Exception as e:
            gpu_load = '--'
        else:
            if gpus_load:
                gpu_load = np.average(gpus_load)
            else:
                gpu_load = '--'
        #print(f' {sim_variables.timesteps}          {t:.6f}        {psutil.cpu_percent()}%           {psutil.virtual_memory().percent}%           {psutil.swap_memory().percent}%           {gpu_load}%')
        print('')
        print('\n', '', tabulate([(sim_variables.timesteps, '%.6f'%t, f'{psutil.cpu_percent()}%', f'{psutil.virtual_memory().percent}%', f'{psutil.swap_memory().percent}%', f'{gpu_load}%')], headers=('step', 'time', 'CPU usage', 'RAM usage', 'Swap usage', 'Avg. GPU usage')))
        print('')
    pass