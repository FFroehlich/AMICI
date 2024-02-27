"""Setuptools file for creating AMICI module

This file is based on setuptools alone and does not require CMake.
All sources are compiled anew.

This file expects to be run from within its directory.

Non-python-package requirements:
- swig>=3.0
- Optional: hdf5 libraries and headers
"""
import os
import sys
from pathlib import Path

from cmake_build_extension import CMakeExtension
from setuptools import setup

# Add containing directory to path, as we need some modules from the AMICI
# package already for installation
sys.path.insert(0, os.path.dirname(__file__))

from amici.custom_commands import (
    AmiciBuildCMakeExtension,
    AmiciBuildPy,
    AmiciDevelop,
    AmiciInstall,
    AmiciInstallLib,
    AmiciSDist,
)


def get_extensions():
    """Get required C(++) extensions for build_ext"""
    # CMake prefix path for finding FindXXX.cmake to find SuiteSparse
    #  components
    install_dir = (Path(__file__).parent / "amici").absolute()
    prefix_path = install_dir
    AmiciBuildCMakeExtension.extend_cmake_prefix_path(str(prefix_path))

    # Used by all extensions
    global_cmake_configure_options = [
        "-DCMAKE_VERBOSE_MAKEFILE=ON",
    ]

    # SuiteSparse Config
    suitesparse_config = CMakeExtension(
        name="SuiteSparse_config",
        install_prefix="amici",
        source_dir="amici/ThirdParty/SuiteSparse/SuiteSparse_config",
        cmake_configure_options=[
            *global_cmake_configure_options,
            "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
            # Building SuiteSparse_config does not require a BLAS
            #  we just set BLAS_LIBRARIES to skip the search,
            #  the value is not used
            # "-DBLA_VENDOR=All",
            "-DBLAS_LIBRARIES=dummy",
            "-DSUITESPARSE_USE_64BIT_BLAS=ON",
            "-DSUITESPARSE_USE_CUDA=OFF",
            "-DSUITESPARSE_USE_FORTRAN=OFF",
        ],
    )
    # SuiteSparse AMD
    amd = CMakeExtension(
        name="amd",
        install_prefix="amici",
        source_dir="amici/ThirdParty/SuiteSparse/AMD",
        cmake_configure_options=[
            *global_cmake_configure_options,
            "-DSUITESPARSE_USE_FORTRAN=OFF",
        ],
    )
    # SuiteSparse BTF
    btf = CMakeExtension(
        name="btf",
        install_prefix="amici",
        source_dir="amici/ThirdParty/SuiteSparse/BTF",
        cmake_configure_options=[
            *global_cmake_configure_options,
            "-DSUITESPARSE_USE_FORTRAN=OFF",
        ],
    )
    # SuiteSparse COLAMD
    colamd = CMakeExtension(
        name="colamd",
        install_prefix="amici",
        source_dir="amici/ThirdParty/SuiteSparse/COLAMD",
        cmake_configure_options=[
            *global_cmake_configure_options,
            "-DSUITESPARSE_USE_FORTRAN=OFF",
        ],
    )
    # SuiteSparse KLU
    klu = CMakeExtension(
        name="klu",
        install_prefix="amici",
        source_dir="amici/ThirdParty/SuiteSparse/KLU",
        cmake_configure_options=[
            *global_cmake_configure_options,
            "-DKLU_USE_CHOLMOD=OFF",
            "-DSUITESPARSE_USE_CUDA=OFF",
            "-DSUITESPARSE_USE_FORTRAN=OFF",
        ],
    )
    # SUNDIALS
    sundials = CMakeExtension(
        name="sundials",
        install_prefix="amici",
        source_dir="amici/ThirdParty/sundials",
        cmake_configure_options=[
            *global_cmake_configure_options,
            "-DBUILD_ARKODE=OFF",
            "-DBUILD_CVODE=OFF",
            "-DBUILD_IDA=OFF",
            "-DBUILD_KINSOL=OFF",
            "-DBUILD_SHARED_LIBS=OFF",
            "-DBUILD_STATIC_LIBS=ON",
            "-DBUILD_NVECTOR_MANYVECTOR=OFF",
            "-DBUILD_SUNNONLINSOL_PETSCSNES=OFF",
            "-DEXAMPLES_ENABLE_C=OFF",
            "-DEXAMPLES_INSTALL=OFF",
            "-DENABLE_KLU=ON",
            # We need the potentially temporary and unpredictable build path
            #  to use artifacts from other extensions here. `${build_dir}` will
            #  be replaced by the actual path by `AmiciBuildCMakeExtension`
            #  before being passed to CMake.
            "-DKLU_LIBRARY_DIR='${build_dir}/amici/lib'",
            "-DKLU_INCLUDE_DIR='${build_dir}/amici/include/suitesparse'",
        ],
    )
    # AMICI
    amici_ext = CMakeExtension(
        name="amici",
        install_prefix="amici",
        source_dir="amici",
        cmake_configure_options=[
            *global_cmake_configure_options,
            "-Werror=dev"
            if "GITHUB_ACTIONS" in os.environ
            else "-Wno-error=dev",
            "-DAMICI_PYTHON_BUILD_EXT_ONLY=ON",
            f"-DPython3_EXECUTABLE={Path(sys.executable).as_posix()}",
        ],
    )
    # Order matters!
    return [suitesparse_config, amd, btf, colamd, klu, sundials, amici_ext]


def main():
    # Readme as long package description to go on PyPi
    # (https://pypi.org/project/amici/)
    with open(
        os.path.join(os.path.dirname(__file__), "README.md"),
        encoding="utf-8",
    ) as fh:
        long_description = fh.read()

    ext_modules = get_extensions()

    # handle parallel building
    # Note: can be empty to use all hardware threads
    if (parallel_jobs := os.environ.get("AMICI_PARALLEL_COMPILE")) is not None:
        os.environ["CMAKE_BUILD_PARALLEL_LEVEL"] = parallel_jobs
    else:
        os.environ["CMAKE_BUILD_PARALLEL_LEVEL"] = "1"

    # Install
    setup(
        cmdclass={
            "install": AmiciInstall,
            "sdist": AmiciSDist,
            "build_ext": AmiciBuildCMakeExtension,
            "install_lib": AmiciInstallLib,
            "develop": AmiciDevelop,
            "build_py": AmiciBuildPy,
        },
        long_description=long_description,
        long_description_content_type="text/markdown",
        ext_modules=ext_modules,
    )


if __name__ == "__main__":
    main()
