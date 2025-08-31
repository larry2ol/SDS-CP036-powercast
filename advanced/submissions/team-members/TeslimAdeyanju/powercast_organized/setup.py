#!/usr/bin/env python3
"""
PowerCast: Deep Learning for Time-Series Power Consumption Forecasting
Setup configuration for package installation
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="powercast",
    version="1.0.0",
    author="Teslim Uthman Adeyanju",
    author_email="info@adeyanjuteslim.co.uk",
    description="Deep Learning for Time-Series Power Consumption Forecasting",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/TeslimAdeyanju/powercast",
    project_urls={
        "Bug Tracker": "https://github.com/TeslimAdeyanju/powercast/issues",
        "Documentation": "https://github.com/TeslimAdeyanju/powercast/docs",
        "Source Code": "https://github.com/TeslimAdeyanju/powercast",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.812",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
            "myst-parser>=0.15",
        ],
        "notebook": [
            "jupyter>=1.0",
            "jupyterlab>=3.0",
            "ipywidgets>=7.6",
        ],
    },
    entry_points={
        "console_scripts": [
            "powercast-train=powercast.cli:train_models",
            "powercast-evaluate=powercast.cli:evaluate_models",
        ],
    },
    include_package_data=True,
    package_data={
        "powercast": [
            "config/*.yml",
            "data/*.csv",
        ],
    },
    zip_safe=False,
    keywords=[
        "time-series",
        "forecasting", 
        "deep-learning",
        "lstm",
        "gru",
        "power-consumption",
        "neural-networks",
        "tensorflow",
        "machine-learning",
    ],
)
