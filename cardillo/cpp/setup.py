from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "algebra_cpp",
        ["cardillo/cpp/algebra.cpp"],
        include_dirs=["/usr/include/eigen3"],
    ),
]

setup(
    name="algebra_cpp",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
