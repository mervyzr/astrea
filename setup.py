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
        subprocess.run(f"chmod +x {CURRENTDIR}/mhydys.py", shell=True)
        install.run(self)


setup(
    name="mhydys",
    version="1.2.1",
    author="Mervin Yap",
    author_email="myap@ph1.uni-koeln.de",
    description="Magnetohydrodynamics code for simulating shocks in the ISM",
    url="<https://github.com/mervyzr/mHydyS>",
    packages=find_packages(
        exclude=['savedData','.vidplots'],
    ),
    install_requires=[
        "wheel",
        "numpy>=2.1.0",
        "scipy>=1.13",
        "h5py>=3.6",
        "matplotlib",
        "pyyaml",
        "tinydb",
        "python-dotenv",
    ],
    keywords=[
        'astrophysics',
        'computational astrophysics',
        'interstellar medium',
        'fluid dynamics',
        'computational fluid dynamics',
        'finite volume method',
        'Riemann solver',
        'numerical simulation'
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: GPLv3",
        "Operating System :: Linux :: macOS",
    ],
    python_requires='>=3.10',
    cmdclass={
        "install": CustomInstallCommand,
    },
)