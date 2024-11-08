import os
import shutil
import subprocess
import setuptools
from setuptools.command.install import install


CURRENTDIR = os.getcwd()

class CustomInstallCommand(install):
    """Customised setuptools install command"""
    def run(self):
        if not os.path.exists(f"{CURRENTDIR}/settings.yml"):
            shutil.copy2(f"{CURRENTDIR}/static/.default.yml", f"{CURRENTDIR}/settings.yml")
        subprocess.run(f"chmod +x {CURRENTDIR}/simulate.py", shell=True)
        install.run(self)

setuptools.setup(cmdclass={"install": CustomInstallCommand,})