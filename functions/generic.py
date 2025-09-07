import os
import sys
import platform
from datetime import timedelta

import psutil
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


# Print progress status to Terminal
def print_status(sim_variables, t=None, final=False):
    _seed = f"{BColours.OKBLUE}{sim_variables.seed}{BColours.ENDC}"
    _config = f"{BColours.OKCYAN}{sim_variables.config.upper()}{BColours.ENDC}"
    _cells = f"{BColours.OKCYAN}{str(sim_variables.cells).strip('[]').replace(' ','').replace(',','x')}{BColours.ENDC}"
    _subgrid = f"{BColours.OKCYAN}{sim_variables.subgrid.upper()}{BColours.ENDC}"
    _time_evo = f"{BColours.OKCYAN}{sim_variables.time_evo.upper()}{BColours.ENDC}"
    _solver = f"{BColours.OKCYAN}{sim_variables.solver.upper()}{BColours.ENDC}"
    _cfl = f"{BColours.OKCYAN}{sim_variables.cfl}{BColours.ENDC}"
    _dimension = f"{BColours.OKCYAN}{BColours.BOLD}({sim_variables.dimension}D){BColours.ENDC}"
    #_performance = f"{BColours.OKGREEN}{round(sim_variables.elapsed*1e6/(np.prod(sim_variables.cells)*sim_variables.timesteps), 3)} \u03BCs/(dt*cells){BColours.ENDC}"

    if not final:
        _instance = f"{BColours.WARNING}{'%.6f'%t} / {'%.2f'%sim_variables.t_end}{BColours.ENDC}"
        print(f"[{sim_variables.now.strftime('%Y-%m-%d %H:%M:%S')} | {_seed}] {_dimension} CONFIG={_config}, CELLS={_cells}, CFL={_cfl}, SUBGRID={_subgrid}, SOLVER={_solver}, TIME_EVO={_time_evo} || {_instance}", end='\r')
        pass
    else:
        if sim_variables.elapsed >= 60*60:
            _elapsed = f"{BColours.FAIL}{str(timedelta(seconds=sim_variables.elapsed))}s{BColours.ENDC}"
        elif 60*60 > sim_variables.elapsed >= 30*60:
            _elapsed = f"{BColours.WARNING}{str(timedelta(seconds=sim_variables.elapsed))}s{BColours.ENDC}"
        else:
            _elapsed = f"{BColours.OKGREEN}{str(timedelta(seconds=sim_variables.elapsed))}s{BColours.ENDC}"

        print(f"[{sim_variables.now.strftime('%Y-%m-%d %H:%M:%S')} | {_seed}] {_dimension} CONFIG={_config}, CELLS={_cells}, CFL={_cfl}, SUBGRID={_subgrid}, SOLVER={_solver}, TIME_EVO={_time_evo} || Elapsed: {_elapsed} ({sim_variables.timesteps})", flush=True)
        pass


#https://thepythoncode.com/article/get-hardware-system-information-python#System_Information

uname = platform.uname()


info = f"""
===============================================================================
astrea code, branch:remotes/origin/{Repository('.').head.shorthand}
Python version {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}
python_path={os.environ['_']}
=============================== machine info ==================================
                  OS =  {'macOS' if uname.system == "Darwin" else uname.system} {uname.release} ({uname.machine})
         total_cores =  {psutil.cpu_count(logical=True)}

    threads_per_core =  {psutil.cpu_count()/psutil.cpu_count(logical=False)}

"""


# number of cores
print("Physical cores:", psutil.cpu_count(logical=False))
print("Total cores:", psutil.cpu_count(logical=True))
# CPU frequencies
cpufreq = psutil.cpu_freq()
print(f"Max Frequency: {cpufreq.max:.2f}Mhz")
print(f"Min Frequency: {cpufreq.min:.2f}Mhz")
print(f"Current Frequency: {cpufreq.current:.2f}Mhz")
# CPU usage
print("CPU Usage Per Core:")
for i, percentage in enumerate(psutil.cpu_percent(percpu=True, interval=1)):
    print(f"Core {i}: {percentage}%")
print(f"Total CPU Usage: {psutil.cpu_percent()}%")



""" ===============================================================================
 DISPATCH code framework, branch:remotes/origin/25-08-19-ppm, hash:4fb425a4f
 SOLVER=AN/weno OPTS=debug, CPU=
 GCC version 15.1.0
 ================================ mpi_t%info ====================================       0.002
          n_ranks =   1
        n_sockets =   1
          n_cores =   1
        n_threads =  14
 threads_per_core =   1
         n_places =   0
 This version was compiled with default real KIND = 4
 ================================= io_t%init ====================================       0.003
 parameters from: weno128.nml
 output directory: data/weno128/
          16          16          16
          16          16          16          16
          16          16          16          16          16
          16          16          16          16          16          16
          32          32          32
          32          32          32          32
          32          32          32          32          32
          32          32          32          32          32          32
 ============================== scaling_t%init ==================================       0.004
  CODE UNITS:              (CGS)             (ASTRO)
                length:   1.234E+19            4.00 pc
                  time:   6.858E+14           21.73 Myr
                  mass:   5.994E+36         3013.80 M_Sun
              velocity:   1.800E+04
               density:   3.187E-21
              pressure:   1.033E-12
  energy per unit mass:   3.240E+08
 entropy per unit mass:   8.254E+07
 magnetic flux density:   3.602E-06
               gravity:   1.000E+02
           temperature:   3.925E+00
 ====================== microphysics/eos/ideal/eos_mod ==========================       0.004
 =============================== refine_t%init ==================================       0.004
 number of AMR criteria:   2
 =============== cartesian_t%init: Cartesian patch arrangement ==================       0.005
 hash_table%reset: new sizes,MB,B/entry,ms=    524288   131072     60.50  121
 patch_t%pre_init: etype=thermal, kind=AN_weno
 extras%pre_init: SNe allocated
 ============================ sink initialization ===============================       0.034
 ------------------------- evolution initialization -----------------------------       0.034
 ../../data/stellar_evolution/Ekstrom.dat
 ------------------------------------ SNe ---------------------------------------       0.051
 SN life-times: Schallerlife.tbl
         mass                    time
    Myr         code        M_sun      code
    7.00       2.323E-03    47.5        2.19    
    9.00       2.986E-03    29.0        1.33    
    12.0       3.982E-03    17.6       0.809    
    15.0       4.977E-03    12.6       0.580    
    20.0       6.636E-03    8.86       0.408    
    25.0       8.295E-03    7.02       0.323    
    40.0       1.327E-02    4.77       0.219    
    60.0       1.991E-02    3.86       0.177    
    85.0       2.820E-02    3.21       0.148    
    120.       3.982E-02    2.98       0.137    
 SNe_t%test: mass(Msun),life(Myr)=   6.0   1.50E+04
 SNe_t%test: mass(Msun),life(Myr)=   8.5    33.    
 SNe_t%test: mass(Msun),life(Myr)=  12.0    18.    
 SNe_t%test: mass(Msun),life(Myr)=  17.0    10.    
 SNe_t%test: mass(Msun),life(Myr)=  24.0    7.2    
 SNe_t%test: mass(Msun),life(Myr)=  33.9    5.1    
 SNe_t%test: mass(Msun),life(Myr)=  48.0    4.1    
 SNe_t%test: mass(Msun),life(Myr)=  67.9    3.6    
 SNe_t%test: mass(Msun),life(Myr)=  96.0    3.1    
         512         512 tasks generated
 AMR levels:  min,root,max =    8   8   8
 ------------------- cartesian_t%init: preparing execution ----------------------       0.054
         512 tasks to generate nbor lists for
 init_nbors: debug,n,hash_min=           0         512           1 T
         512 nbor lists done
 list_t%check_nbor_consistency: n=    512      0      0  init_all_nbors
 list_t%check_nbor_consistency: n=    512      0      0
 spline_test: errors,cost=  0.000000  0.086762  0.111633   0.0 ns/pt
 spline_test: errors,cost=  0.000000  0.000000  0.000000  25.0 ns/pt
 SOLVER=AN_weno          10
 memory shape:  40  40  40   6   5   1
 variable indices:   d:1   p1:2   p2:3   p3:4   e:5 
      2184 bytes per cell
   133.307 MB per task
                               procedure       calls          time       time(%)      s/call  locks(%)
                           solver_t%init       56.0           114.          96.6    2.036055       0.0  
                force_t%turbulence_start       112.           3.56           3.0    0.031793       0.0  
            TOTAL updates, thread time           1.00         118.         100.      0.00E+00 advance/s     0.00 core-mus/cell-upd       10 wall sec
 MPI recv:      0.0 MB/s     0.000 MB/mesg  mean latency: 0.000  max: 0.000  nq_send_max:   0  f_unpk: 0.00  f_mem: 0.00  f_que: 0.00  f_cheap: 0.00
 lock waiting time, lists:   0.0    links:   0.0    state:   0.0    mem:   0.0    heap:   0.0    total:   0.0   core-s =   0.0 %
 lock     log time =  0.00 core-s
                               procedure       calls          time       time(%)      s/call  locks(%)
                           solver_t%init       70.0           140.          96.8    2.000178       0.0  
                force_t%turbulence_start       140.           4.47           3.1    0.031910       0.0  
            TOTAL updates, thread time           1.00         145.         100.      0.00E+00 advance/s     0.00 core-mus/cell-upd       20 wall sec
 MPI recv:      0.0 MB/s     0.000 MB/mesg  mean latency: 0.000  max: 0.000  nq_send_max:   0  f_unpk: 0.00  f_mem: 0.00  f_que: 0.00  f_cheap: 0.00
 lock waiting time, lists:   0.0    links:   0.0    state:   0.0    mem:   0.0    heap:   0.0    total:   0.0   core-s =   0.0 %
 lock     log time =  0.00 core-s"""