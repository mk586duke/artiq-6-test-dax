#!/usr/bin/env python3

# Copyright 2020 Drew Risinger, Chris Monroe Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Install the EURIQA packages for extending ARTIQ.

Includes :mod:`.euriqabackend` for lab- and hardware-specific code,
and :mod:`.euriqafrontend` for GUI code, experiments to run, etc.
"""
import setuptools

if __name__ == "__main__":
    # These are the main requirements. They're reliable, low likelihood of causing version
    # conflicts/issues, and used in experiments. They can be moved out of the main as needed
    # Loosely, these are the requirements to run experiments as artiq_master, except ARTIQ
    # itself (b/c it has strict dependencies)
    base_requirements = [
        "click",
#        "dax",
        "h5py",
        "more_itertools",
        "numdifftools",
        "numpy>=1.15.0",
        "pandas",
        "prettytable",
#        "pulsecompiler",
        "python-box",
        "pyyaml",
        "setuptools",
        "sipyco",
        "scipy",
        "statsmodels",
        "tables",
        "uncertainties",
    ]

    extra_requirements = {
        "dev": ["pre-commit", "sphinx", "pygraphviz"],
        "doc": ["sphinx"],
        "circuits": [
            # For reading and writing Quantum circuits
            "cirq",
            "networkx",
            "qiskit-terra",
        ],
        "hardware": [
            # Use this for interfacing with hardware devices, for controllers etc
            # "ok", # required, but needs to be installed manually (OpalKelly FrontPanel)
            "networkx",
            "pyvisa",
            "pyvisa-py",
        ],
        "gateware": [
            "progress",  # progress spinner for build_artiq script
            # misoc & migen are required to build, but should be installed by the
            # conda environment for correct versioning w/ ARTIQ
            # misoc", "migen"
        ],
        "applets": [
            # For using/viewing applets
            "pint",  # Sandia Voltage Control applet only
            "ply",  # Sandia Voltage control applet only
            "pyparsing",  # Sandia Voltage control applet only
            "pyqtgraph",  # Qt5-compatible version, from M-Labs. Handled in Nix build.
        ],
    }

    setuptools.setup(
        name="euriqabackend",
        version="0.1",
        # Automatically detect packages in specified/current directory
        packages=setuptools.find_packages(),
        # entry points to extend and add custom commands
        entry_points={
            # add python setup.py command_name
            # "distutils.commands": [
            # "command_name = euriqabackend:MyClass"
            # ],
            # fmt: off
            "console_scripts": [
                # custom scripts
                "build_artiq = euriqabackend.utilities.build_artiq:cli",
                "reflash_dac = euriqabackend.devices.sandia_dac.reflash_dac:cli",
                "h5analyzer = euriqafrontend.utilities.h5analyzer.__main__:main",
                "find_rid = euriqafrontend.utilities.commandline.find_rid:main",
                "tweak_values = euriqafrontend.utilities.commandline.tweak_values:cli",

                # controllers
                "aqctl_sandia_dac_100x = "
                "euriqabackend.controllers.aqctl_sandia_dac_100x:main",
                "aqctl_globalaom_harris = "
                "euriqabackend.controllers.aqctl_globalaom_harris:main",
                "aqctl_multiaom_harris = "
                "euriqabackend.controllers.aqctl_multiaom_harris:main",
                "aqctl_conex = euriqabackend.controllers.aqctl_conex:main",
                "aqctl_oven = euriqabackend.controllers.aqctl_oven:main",
                "aqctl_rfcompiler = euriqabackend.controllers.aqctl_rfcompiler:main",
                "aqctl_n6700b = euriqabackend.controllers.aqctl_keysight_psu_n6700b:main",
                "aqctl_psu_n6700b = euriqabackend.controllers.aqctl_keysight_psu_n6700b:main",

                # applets
                "100x_dac_gui = euriqafrontend.applets.control_100x_dac:main",
            ]
            # fmt: on
        },
        author="UMD Trapped Ion Group, JQI",
        author_email="",
        description="Backend code for EURIQA program at University of Maryland",
        # Packages required to use this. Can specify application name,
        # specific version, or range. Use PIP syntax
        install_requires=base_requirements,
        # Extra requirements are another amazing feature of
        # setuptools, it allows people to install extra
        # dependencies if you are interested. In this example
        # doing a "pip install name[all]" would install the
        # python-utils package as well.
        extras_require=extra_requirements,
        # Packages required to install this package, not just for
        # running it but for the actual install. These will not be
        # installed but only downloaded so they can be used during
        # the install. The pytest-runner is a useful example:
        # setup_requires=["pytest-runner"],
        # The requirements for the test command. Regular testing
        # is possible through: python setup.py test The Pytest
        # module installs a different command though: python
        # setup.py pytest
        tests_require=["pytest"],
        # The package_data, include_package_data and
        # exclude_package_data arguments are used to specify which
        # non-python files should be included in the package. An
        # example would be documentation files. More about this
        # in the next paragraph
        package_data={
            # Include (restructured text, markdown, settings) files from
            # any directory
            "": ["*.rst", "*.md", "*.hash", "*.ui", "*.qrc", "*.json", "*.pyon"],
            # Include text files from the eggs package:
            "eggs": ["*.txt"],
        },
        # If a package is zip_safe the package will be installed
        # as a zip file. This can be faster but it generally
        # doesn't make too much of a difference and breaks
        # packages if they need access to either the source or the
        # data files. When this flag is omitted setuptools will
        # try to autodetect based on the existence of datafiles
        # and C extensions. If either exists it will not install
        # the package as a zip. Generally omitting this parameter
        # is the best option but if you have strange problems with
        # missing files, try disabling zip_safe.
        # zip_safe=False,
        # For this parameter I would recommend including the
        # README.rst
        long_description=open("README.md", encoding="utf-8").read(),
        long_description_content_type="text/markdown",
        # The license should be one of the standard open source
        # licenses: https://opensource.org/licenses/alphabetical
        license="Apache 2",
        # Homepage url for the package
        # url='https://wol.ph/',
    )
