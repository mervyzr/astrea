#!/usr/bin/env python3

import os
import shutil
import subprocess
from setuptools import setup, find_packages
from setuptools.command.install import install


CURRENTDIR = os.getcwd()

with open(f"{CURRENTDIR}/README.md", "r") as f:
    long_description = f.read()


class CustomInstallCommand(install):
    """Customised setuptools install command"""
    def run(self):
        if not os.path.exists(f"{CURRENTDIR}/settings.yml"):
            shutil.copy2(f"{CURRENTDIR}/static/.default.yml", f"{CURRENTDIR}/settings.yml")
        subprocess.run(f"chmod +x {CURRENTDIR}/simulate.py", shell=True)
        install.run(self)


setup(
    name="mHydyS",
    version="1.1.1",
    author="Mervin Yap",
    author_email="myap@ph1.uni-koeln.de",
    description="(Magneto-)hydrodynamics code for simulating shocks in the ISM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="<https://github.com/mervyzr/mHydyS>",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "h5py",
        "matplotlib",
        "pyyaml",
        "tinydb",
        "python-dotenv",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GPLv3",
        "Operating System :: Linux :: macOS",
    ],
    python_requires='>=3.10',
    cmdclass={
        "install": CustomInstallCommand,
    },
)