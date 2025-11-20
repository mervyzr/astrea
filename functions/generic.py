import os
import sys
import datetime
import platform
from time import perf_counter
from datetime import datetime, timedelta

import git
import psutil
import pynvml
import numpy as np
from tabulate import tabulate

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


def get_gpu_info():
    pynvml.nvmlInit()

    list_gpus = []
    for i in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        uuid = pynvml.nvmlDeviceGetUUID(handle)
        name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")

        list_gpus.append((
            i,                      # GPU id
            name,                   # GPU name
            f'{util.gpu}%',         # GPU load
            get_size(mem_info.free),   # memory free
            get_size(mem_info.used),   # memory used
            get_size(mem_info.total),  # memory total
            f'{temp} C',            # temperature
            uuid                     # UUID
        ))

    pynvml.nvmlShutdown()
    return list_gpus


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
            print(f'    {func.__name__!r:>20} :     {perf_counter() - start:<10.5f} s')
        return result
    return wrapper


# Print progress status to Terminal
def print_simple(sim_variables, t=None, status=''):
    cyan_text = lambda text: f"{BColours.OKCYAN}{text}{BColours.ENDC}"

    _seed = f"{BColours.OKBLUE}{sim_variables.seed}{BColours.ENDC}"
    _dimension = cyan_text(f"({sim_variables.dimensions}D)")
    _config = cyan_text(sim_variables.config.upper())
    _cells = cyan_text(str(sim_variables.cells).strip('[]').replace(' ','').replace(',','x'))
    _cfl = cyan_text(sim_variables.cfl)
    _subgrid = cyan_text(sim_variables.subgrid.upper())
    _solver = cyan_text(sim_variables.solver.upper())
    _time_evo = cyan_text(sim_variables.time_evo.upper())

    if status.lower() == 'final':
        if sim_variables.elapsed >= 60*60:
            colour = lambda text: f"{BColours.FAIL}{text}{BColours.ENDC}"
        elif 60*60 > sim_variables.elapsed >= 30*60:
            colour = lambda text: f"{BColours.WARNING}{text}{BColours.ENDC}"
        else:
            colour = lambda text: f"{BColours.OKGREEN}{text}{BColours.ENDC}"
        _performance = (
            colour(f"Elapsed: {str(timedelta(seconds=sim_variables.elapsed))}s") 
            + " | " 
            + colour(f"{sim_variables.timesteps} steps") 
            + " | "
            + colour(f"{1e-3 * (np.prod(sim_variables.cells)*sim_variables.timesteps)/sim_variables.cpu_elapsed:.2f} kCUPS")
        )

        print(f"[{sim_variables.now.strftime('%Y-%m-%d %H:%M:%S')} | {_seed}] {_dimension} CONFIG={_config}, CELLS={_cells}, CFL={_cfl}, SUBGRID={_subgrid}, SOLVER={_solver}, TIME_EVO={_time_evo} || {_performance} ||", flush=True)
        pass
    elif status.lower() == 'init':
        pass
    else:
        _instance = f"{BColours.WARNING}{t:.6f} / {sim_variables.t_end:.2f}{BColours.ENDC}"
        print(f"[{sim_variables.now.strftime('%Y-%m-%d %H:%M:%S')} | {_seed}] {_dimension} CONFIG={_config}, CELLS={_cells}, CFL={_cfl}, SUBGRID={_subgrid}, SOLVER={_solver}, TIME_EVO={_time_evo} || {_instance}", end='\r')
        pass


# Print verbose status to Terminal
def print_verbose(sim_variables, t=None, status=''):
    # Toggle which data to print
    sys_info = True
    cpu_info = True
    mem_info = True
    gpu_info = True
    dsk_info = False
    sim_info = True

    if status.lower() == "init":
        print('='*80)

        print(f'  astrea code, branch:remotes/origin/{git.Repo(sim_variables.home).active_branch.name}')
        print(f'  Python version {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}-{sys.version_info.releaselevel}')
        print(f'  PYTHON_PATH={os.environ["_"]}')

        if sys_info:
            print('='*30, 'System Information', '='*30)
            uname = platform.uname()
            print(f'{"System":>15} :    {"macOS" if uname.system == "Darwin" else uname.system}')
            print(f'{"Node":>15} :    {uname.node}')
            print(f'{"Release":>15} :    {uname.release}')
            print(f'{"Version":>15} :    {uname.version}')
            print(f'{"Architecture":>15} :    {uname.machine}')
            print(f'{"Processor":>15} :    {uname.processor}')

        if cpu_info:
            print('='*30, 'CPU Information', '='*30)
            print(f'{"Physical cores":>15} :    {psutil.cpu_count(logical=False)}')
            print(f'{"Total cores":>15} :    {psutil.cpu_count(logical=True)}')
            #print(f' {"Threads per core":>20} :    {psutil.cpu_count()/psutil.cpu_count(logical=False)}')

        if mem_info:
            print('='*30, 'Memory Information', '='*30)
            svmem = psutil.virtual_memory()
            print(f'{"Total":>10} :    {get_size(svmem.total)}')
            print(f'{"Available":>10} :    {get_size(svmem.available)}')
            print(f'{"Used":>10} :    {get_size(svmem.used)}')

            print(f'{"":>4}======= Swap =======')
            swap = psutil.swap_memory()
            print(f'{"Total":>10} :    {get_size(swap.total)}')
            print(f'{"Free":>10} :    {get_size(swap.free)}')
            print(f'{"Used":>10} :    {get_size(swap.used)}')

        if gpu_info:
            print('='*30, 'GPU Information', '='*30)
            try:
                list_gpus = get_gpu_info()
            except Exception:
                print("Unable to obtain GPU information")
            else:
                print(tabulate(list_gpus, headers=('id', 'name', 'load', 'free memory', 'used memory', 'total memory', 'temperature', 'uuid')))

        if dsk_info:
            print('='*30, 'Disk Information', '='*30)
            disk_io = psutil.disk_io_counters()
            print(f'{"Total read":>15} :    {get_size(disk_io.read_bytes)}')
            print(f'{"Total write":>15} :    {get_size(disk_io.write_bytes)}')
            for partition in psutil.disk_partitions():
                print(f'{"":>4}=== Device: {partition.device} ===')
                print(f'{"":>8}Mountpoint: {partition.mountpoint}')
                print(f'{"":>8}File system: {partition.fstype}')
                try:
                    partition_usage = psutil.disk_usage(partition.mountpoint)
                except PermissionError:
                    continue
                else:
                    print(f'{"":>8}Total size: {get_size(partition_usage.total)}')
                    print(f'{"":>8}Used: {get_size(partition_usage.used)}')
                    print(f'{"":>8}Free: {get_size(partition_usage.free)}')
                    print(f'{"":>8}Percentage: {partition_usage.percent}%')

        if sim_info:
            print('='*30, 'Sim. Information', '='*30)
            print(f'{"Seed":>16} :    {sim_variables.seed}')
            print(f'{"Boot time":>16} :    {sim_variables.now.strftime("%Y-%m-%d %H:%M:%S")}')
            print(f'{"Output directory":>16} :    {sim_variables.save_path}')
            print(f'')
            print(f'{"":>5}OPTS={sys.argv[1:]}')
            print(f'')
            if sim_variables.chemistry:
                print(f'{"":>5}SPECIES={sim_variables.species}')
                print(f'')
            print(f'{"Config.":>15} :    {sim_variables.config.upper()}')
            print(f'{"Subgrid":>15} :    {sim_variables.subgrid.upper()}')
            print(f'{"Time evo.":>15} :    {sim_variables.time_evo.upper()}')
            print(f'{"Solver":>15} :    {sim_variables.solver.upper()}')
            print('')
            print(f'{"Dimension":>15} :    {sim_variables.dimensions}D')
            print(f'{"Cells":>15} :    {str(sim_variables.cells).strip("[]").replace(" ","").replace(","," x ")}')
            print(f'{"Boundary":>15} :    {'PERIODIC' if sim_variables.boundary.lower() == 'wrap' else 'OUTFLOW'}')
            print(f'{"CFL number":>15} :    {sim_variables.cfl}')
            print(f'{"Adiabatic index":>15} :    {sim_variables.gamma}')
            print(f'{"Permeability":>15} :    {sim_variables.permeability}')
            print('')
            print(f'{"End time":>15} :    {sim_variables.t_end}')
            print(f'{"Checkpoints":>15} :    {sim_variables.checkpoints}')

        print('')
        print('-'*30, 'Executing simulation', '-'*30)


    elif status.lower() == 'final':
        print('')
        print('='*82)
        print('-'*30, f'{BColours.OKGREEN}SIMULATION COMPLETED{BColours.ENDC}', '-'*30)
        print('='*82)
        print('')
        print(tabulate([(
            sim_variables.seed, 
            sim_variables.now.strftime('%Y-%m-%d %H:%M:%S'), 
            str(timedelta(seconds=sim_variables.elapsed)), 
            sim_variables.timesteps, 
            f"{1e-3 * (np.prod(sim_variables.cells)*sim_variables.timesteps)/sim_variables.cpu_elapsed:.2f}", 
            '|', 
            sim_variables.config, 
            str(sim_variables.cells).strip('[]').replace(' ','').replace(',',' x '), 
            sim_variables.cfl, 
            sim_variables.subgrid, 
            sim_variables.solver, 
            sim_variables.time_evo, 
            )], headers=(
                'seed', 
                'start time', 
                'elapsed', 
                'steps', 
                'kCUPS', 
                '|', 
                'config', 
                'cells', 
                'cfl', 
                'subgrid', 
                'solver', 
                'time_evo', 
                )))
        print('')


    else:
        try:
            pynvml.nvmlInit()
            gpu_load = np.average([[pynvml.nvmlDeviceGetUtilizationRates(pynvml.nvmlDeviceGetHandleByIndex(i)).gpu] for i in range(pynvml.nvmlDeviceGetCount())])
            pynvml.nvmlShutdown()
        except Exception:
            gpu_load = '--'
        print('\n')
        print(tabulate([(
            sim_variables.seed, 
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
            sim_variables.timesteps, 
            f'{t:.6f}', 
            f'{psutil.cpu_percent()}%', 
            f'{psutil.virtual_memory().percent}%', 
            f'{psutil.swap_memory().percent}%', 
            f'{gpu_load}%', 
            )], headers=(
                'seed', 
                'datetime', 
                'step', 
                'sim. time', 
                'CPU usage', 
                'RAM usage', 
                'Swap usage', 
                'Avg. GPU usage', 
                )))
        print('')
    pass