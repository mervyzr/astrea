#!/usr/bin/env python

import os
import shutil
import subprocess

import venv


currentdir = os.getcwd()
home = os.path.expanduser("~")
status = 0

if not (os.path.exists(f"{currentdir}/settings.py") or os.path.isdir(f"{home}/.shock_venv")):
    print("Initialising setup..")

    if not os.path.exists(f"{currentdir}/settings.py"):
        shutil.copy2(f"{currentdir}/functions/.default.py", f"{currentdir}/settings.py")

    if not os.path.isdir(f"{home}/.shock_venv"):
        print("Creating Python venv for simulation..")
        venv_dir = os.path.join(home, ".shock_venv")
        venv.create(venv_dir, with_pip=True)
        subprocess.run(["bin/pip", "install", "-q", "-r", os.path.abspath("requirements.txt")], cwd=venv_dir)

    if not os.path.exists(f"{currentdir}/savedData"):
        os.makedirs(f"{currentdir}/savedData")

    print("Setup complete!")
else:
    print("Nothing to set up")