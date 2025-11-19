#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="mdrunner",
    version="0.0.1",
    description="Describe Your Cool Project",
    author="Alex Tong",
    author_email="alexandertongdev@gmail.com",
    url="https://github.com/atong01/md-runner",
    install_requires=["hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "generate = src.generate_md:main",
        ]
    },
)
