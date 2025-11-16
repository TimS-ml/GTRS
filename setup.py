"""
Setup configuration for NavSim package.

This script handles the installation and configuration of the NavSim package,
a data-driven non-reactive autonomous vehicle simulation and benchmarking framework.
It defines package metadata, dependencies, and installation requirements.
"""

import os

import setuptools

# Change directory to the script's location to allow installation from anywhere
# This ensures relative paths work correctly regardless of where setup.py is invoked
script_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_folder)

# Read package dependencies from requirements.txt file
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Configure and install the NavSim package
setuptools.setup(
    name="navsim",
    version="2.0.0",
    author="University of Tuebingen",
    author_email="kashyap.chitta@uni-tuebingen.de",
    description="NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking",
    url="https://github.com/autonomousvision/navsim",
    python_requires=">=3.9",
    packages=setuptools.find_packages(script_folder),
    package_dir={"": "."},
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "License :: Free for non-commercial use",
    ],
    license="apache-2.0",
    install_requires=requirements,
)
