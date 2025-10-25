import os
from glob import glob
from setuptools import find_packages, setup

package_name = "quadcopter"
tests_require = ["pytest"]

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(where="src", exclude=["test"]),
    package_dir={"": "src"},
    data_files=[
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*launch.[pxy][yma]*")),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="tianxdai",
    maintainer_email="tianxdai@gmail.com",
    description="TODO: Package description",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "simulator = quadcopter.nodes.simulator:main",
            "visualizer = quadcopter.nodes.visualizer:main",
            "controller = quadcopter.nodes.controller:main",
        ],
    },
)
