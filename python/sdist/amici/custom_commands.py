"""Custom setuptools commands for AMICI installation"""

import os
import subprocess
import sys
from pathlib import Path

from amici.swig import fix_typehints
from cmake_build_extension import BuildExtension, CMakeExtension
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.command.install_lib import install_lib
from setuptools.command.sdist import sdist


class AmiciInstall(install):
    """Custom `install` command to handle extra arguments"""

    print("running AmiciInstall")

    # Passing --no-clibs allows to install the Python-only part of AMICI
    user_options = install.user_options + [
        ("no-clibs", None, "Don't build AMICI C++ extension"),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.no_clibs = False

    def finalize_options(self):
        if self.no_clibs:
            self.no_clibs = True
        super().finalize_options()


class AmiciDevelop(develop):
    """Custom develop to build clibs"""

    # Passing --no-clibs allows to install the Python-only part of AMICI
    user_options = develop.user_options + [
        ("no-clibs", None, "Don't build AMICI C++ extension"),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.no_clibs = False

    def finalize_options(self):
        if self.no_clibs:
            self.no_clibs = True
        super().finalize_options()


class AmiciInstallLib(install_lib):
    """Custom install to allow preserving of debug symbols"""

    def run(self):
        """strip debug symbols

        Returns:

        """
        print("running AmiciInstallLib")

        if (
            os.environ.get("ENABLE_AMICI_DEBUGGING") == "TRUE"
            and sys.platform == "darwin"
        ):
            search_dir = os.path.join(os.getcwd(), self.build_dir, "amici")
            for file in os.listdir(search_dir):
                if file.endswith(".so"):
                    subprocess.run(
                        [
                            "dsymutil",
                            os.path.join(search_dir, file),
                            "-o",
                            os.path.join(search_dir, f"{file}.dSYM"),
                        ]
                    )

        # Continue with the actual installation
        super().run()


class AmiciSDist(sdist):
    """Customized creation of source distribution"""

    def run(self):
        """Setuptools entry-point"""
        print(f"running {self.__class__.__name__}")

        save_git_version()

        super().run()


def save_git_version():
    """Create file with extended version string

    This requires git. We assume that whoever creates the sdist will work
    inside a valid git repository.
    """
    with open(os.path.join("amici", "git_version.txt"), "w") as f:
        try:
            cmd = [
                "git",
                "describe",
                "--abbrev=4",
                "--dirty=-dirty",
                "--always",
                "--tags",
            ]
            subprocess.run(cmd, stdout=f)
        except Exception as e:
            print(e)


class AmiciBuildPy(build_py):
    def run(self):
        print(f"running {self.__class__.__name__}")
        # We need build_ext before build_py, that all artifacts will be
        # copied from the build dir
        self.run_command("build_ext")
        return super().run()


class AmiciBuildCMakeExtension(BuildExtension):
    def finalize_options(self):
        # Allow overriding the - since setuptools version 64 randomly named -
        #  setuptools/distutils temporary build directory via environment variable.
        # This is useful for CI builds where we need the files in this directory
        #  for code coverage analysis.
        if os.getenv("AMICI_BUILD_TEMP"):
            self.build_temp = os.getenv("AMICI_BUILD_TEMP")

        super().finalize_options()

    def run(self):
        """Copy the generated clibs to the extensions folder to be included in
        the wheel
        """
        print(f"running {self.__class__.__name__}")

        # custom flag to build without extensions
        no_clibs = (
            "develop" in self.distribution.command_obj
            and self.get_finalized_command("develop").no_clibs
        )
        no_clibs |= (
            "install" in self.distribution.command_obj
            and self.get_finalized_command("install").no_clibs
        )

        if no_clibs:
            # Nothing to build
            return

        # Continue with the actual extension building
        result = super().run()

        if not self.dry_run:
            # Fix SWIG-generated typehints
            build_dir = self.build_lib if self.inplace == 0 else os.getcwd()
            swig_py_module_path = Path(build_dir, "amici", "amici.py")
            # this is class is used for the amici core extension, and any model
            #  extensions. if amici.py is present, this is the core extension.
            if swig_py_module_path.is_file():
                print("updating typehints")
                fix_typehints(swig_py_module_path, swig_py_module_path)

        return result

    def build_extension(self, ext: CMakeExtension) -> None:
        # put some structure into CMake output
        print("-" * 30, ext.name, "-" * 30, file=sys.stderr)

        # Some hack to be able to use distutils' potentially temporary build
        # directory in CMake options:
        # Any occurrence of `${build_dir}` will be replaced by said path.
        build_dir = self.build_lib if self.inplace == 0 else os.getcwd()
        build_dir = Path(build_dir).absolute().as_posix()
        ext.cmake_configure_options = [
            x.replace("${build_dir}", build_dir)
            for x in ext.cmake_configure_options
        ]

        super().build_extension(ext)

        print("-" * 30, ext.name, "-" * 30, file=sys.stderr)
