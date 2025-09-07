"""
PowerCast: Deep Learning for Time-Series Power Consumption Forecasting
Author: Teslim Uthman Adeyanju
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="powercast",
    version="1.0.0",
    author="Teslim Uthman Adeyanju",
    author_email="info@adeyanjuteslim.co.uk",
    description="Deep Learning for Time-Series Power Consumption Forecasting in Tetouan City",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TeslimAdeyanju/powercast",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
            "myst-parser>=0.15",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    entry_points={
        "console_scripts": [
            "powercast-train=modules.week3_neural_networks:main",
            "powercast-predict=modules.utils:predict_cli",
        ],
    },
    keywords="time-series forecasting deep-learning power-consumption neural-networks",
    project_urls={
        "Bug Reports": "https://github.com/TeslimAdeyanju/powercast/issues",
        "Source": "https://github.com/TeslimAdeyanju/powercast",
        "Documentation": "https://github.com/TeslimAdeyanju/powercast/docs",
    },
)
