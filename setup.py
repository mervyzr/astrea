#!/usr/bin/env python3

import os
import shutil
import subprocess
from setuptools import setup, find_packages
from setuptools.command.install import install


CURRENTDIR = os.getcwd()

class CustomInstallCommand(install):
    """Customised setuptools install command"""
    def run(self):
        if not os.path.exists(f"{CURRENTDIR}/parameters.yml"):
            shutil.copy2(f"{CURRENTDIR}/static/.default.yml", f"{CURRENTDIR}/parameters.yml")
        subprocess.run(f"chmod +x {CURRENTDIR}/astrea.py", shell=True)
        install.run(self)


setup(
    name="astrea",
    version="3.1.3",
    author="Mervin Yap",
    author_email="myap@ph1.uni-koeln.de",
    description="Astrophysical Shockwave and Turbulence REsearch for interstellar Applications: Multi-dimensional magnetohydrodynamics code for modelling shockwaves & turbulence in the interstellar medium",
    url="<https://github.com/mervyzr/astrea>",
    packages=find_packages(
        exclude=[
            'saved_data', 
            '.vidplots',
            'krome',
            ],
    ),
    install_requires=[
        "numpy>=2.0.0",
        "h5py>=3.7",
        "scipy",
        "matplotlib",
        "pyyaml",
        "tinydb",
        "python-dotenv",
        "pygit2",
        "psutil",
        "gputil",
        "tabulate",
        "wheel",
    ],
    keywords=[
        'astrophysics',
        'computational astrophysics',
        'interstellar medium',
        'computational fluid dynamics',
        'finite volume method',
        'Riemann solver',
        'numerical simulation'
    ],
    classifiers=[
        "Development Status :: 5 - Stable",
        "Programming Language :: Python :: 3",
        "License :: GPLv3",
        "Operating System :: Linux :: macOS (ARM/Intel)",
    ],
    python_requires='>=3.10',
    cmdclass={
        "install": CustomInstallCommand,
    },
)