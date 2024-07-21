#!/usr/bin/env python3

import os
import shutil
import subprocess

import venv


currentdir = os.getcwd()

if not os.path.exists(f"{currentdir}/settings.py") or not os.path.isdir(f"{currentdir}/.venv"):
    print("Initialising setup..")

    if not os.path.exists(f"{currentdir}/settings.py"):
        shutil.copy2(f"{currentdir}/static/.default.py", f"{currentdir}/settings.py")

    if not os.path.isdir(f"{currentdir}/.venv"):
        print("Creating Python venv for simulation..")
        venv_dir = os.path.join(currentdir, ".venv")
        venv.create(venv_dir, with_pip=True)
        subprocess.run(["bin/pip", "install", "-q", "-r", os.path.abspath("static/requirements.txt")], cwd=venv_dir)

    print("Setup complete!")
else:
    print("Nothing to set up")