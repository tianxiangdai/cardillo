import os
from glob import glob
from setuptools import find_packages, setup

package_name = "cardillo_ros"
tests_require = ["pytest"]

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*launch.[pxy][yma]*")),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="cardillo",
    maintainer_email="cardillo@xxxx.xxxx",
    description="TODO: Package description",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "simulator = nodes.simulator:main",
            "visualizer = nodes.visualizer:main",
            "controller = nodes.controller:main",
        ],
    },
)
