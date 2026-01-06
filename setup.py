"""Setup script for fuzzylogic_cartpole package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="fuzzylogic_cartpole",
    version="0.1.0",
    author="fuzzylogic_cartpole",
    description="A simple example on how to use fuzzy logic to control the CartPole gymnasium environment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gjaeger1/fuzzylogic_cartpole",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "gymnasium>=0.29.0",
        "fuzzylogic>=1.2.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
        ],
    },
)
