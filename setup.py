#!/usr/bin/env python

import os
import shutil
import subprocess
from distutils.core import setup

import venv


currentdir = os.getcwd()
home = os.path.expanduser("~")

print("Initialising setup..")

if not os.path.exists(f"{currentdir}/settings.py"):
    print(f"Copying settings file from default..")
    shutil.copy2(f"{currentdir}/functions/.default.py", f"{currentdir}/settings.py")

if not os.path.isdir(f"{home}/.mHydyS_venv"):
    print("Creating venv for simulation..")
    venv_dir = os.path.join(home, ".mHydyS_venv")
    venv.create(venv_dir, with_pip=True)
    subprocess.run(["bin/pip", "install", "-q", "-r", os.path.abspath("requirements.txt")], cwd=venv_dir)

print("Setup complete!")