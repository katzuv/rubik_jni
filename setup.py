#!/usr/bin/env python3
"""
Setup script for rubik_py Python bindings.

Note: This is a minimal setup.py that assumes the native module has already been built
using CMake. For full build instructions, see README.md.
"""

from setuptools import setup, find_packages
import os
import sys

# Read version from file if it exists
version = "0.1.0"

setup(
    name="rubik_py",
    version=version,
    author="Photon Vision",
    description="Python bindings for Rubik object detection using TensorFlow Lite",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/katzuv/rubik_jni",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
