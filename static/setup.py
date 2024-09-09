#!/usr/bin/env python3

import os
import shutil
import subprocess


currentdir = os.getcwd()
version = "3.10"


if not os.path.exists(f"{currentdir}/settings.py") or not os.path.isdir(f"{currentdir}/.venv"):
    print("Initialising setup..")

    if not os.path.exists(f"{currentdir}/settings.py"):
        shutil.copy2(f"{currentdir}/static/.default.py", f"{currentdir}/settings.py")

    if not os.path.isdir(f"{currentdir}/.venv"):
        print("Creating Python venv for simulation..", end='\r')
        venv_dir = os.path.join(currentdir, ".venv")
        subprocess.run(f"python{version} -m venv {venv_dir} && {venv_dir}/bin/pip3 install -q -r {currentdir}/static/requirements.txt", shell=True)
        subprocess.run(f"chmod +x {currentdir}/simulate.py", shell=True)
        print("\033[92mSetup complete!\033[0m")
else:
    print("Nothing to set up.")