#!/usr/bin/env python3

import os
import shutil
import subprocess


currentdir = os.getcwd()
python_version = "python3.12"

if not os.path.exists(f"{currentdir}/settings.py") or not os.path.isdir(f"{currentdir}/.venv"):
    print("Initialising setup..")

    if not os.path.exists(f"{currentdir}/settings.py"):
        shutil.copy2(f"{currentdir}/static/.default.py", f"{currentdir}/settings.py")

    if not os.path.isdir(f"{currentdir}/.venv"):
        print("Creating Python venv for simulation..")
        python_path = subprocess.run(["which", python_version], capture_output=True).stdout.decode("utf-8").rstrip()
        venv_dir = os.path.join(currentdir, ".venv")
        subprocess.run([python_path, "-m", "venv", venv_dir])
        subprocess.run(["source", f"{currentdir}/.venv/bin/activate"])
        subprocess.run(["pip3", "install", "-q", "-r", os.path.abspath("static/requirements.txt")], cwd=venv_dir)
        subprocess.run(["deactivate"])

    print("Setup complete!")
else:
    print("Nothing to set up.")