#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 14:47:08 2025

@author: lbremaud
"""

from setuptools import setup, find_packages

setup(
    name="loompy",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pyvista",
        "matplotlib",
    ],
    author="DEBONDIUM",
    description="Weaving mesh generation and visualization toolkit",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/DEBONDIUM/loompy",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
