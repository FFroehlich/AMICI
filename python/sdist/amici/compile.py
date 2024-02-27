"""
Functionality for building the C++ extensions of an amici-created model
package.
"""
import subprocess
import sys
from typing import Optional, Union
from pathlib import Path
import os


def build_model_extension(
    package_dir: Union[str, Path],
    verbose: Optional[Union[bool, int]] = False,
    compiler: Optional[str] = None,
    extra_msg: Optional[str] = None,
) -> None:
    """
    Compile the model extension of an amici-created model package.

    :param package_dir:
        Directory of the model package to be compiled. I.e., the directory
        containing the `setup.py` file.

    :param verbose:
        Make model compilation verbose.

    :param compiler:
        Absolute path to the compiler executable to be used to build the Python
        extension, e.g. ``/usr/bin/clang``.

    :param extra_msg:
        Additional message to be printed in case of a failed build.
    """
    # setup.py assumes it is run from within the model directory
    package_dir = Path(package_dir)
    script_args = [sys.executable, package_dir / "setup.py"]

    if verbose:
        script_args.append("--verbose")
    else:
        script_args.append("--quiet")

    script_args.extend(
        [
            "build_ext",
            f"--build-lib={package_dir}",
            # This is generally not required, but helps to reduce the path
            # length of intermediate build files, that may easily become
            # problematic on Windows, due to its ridiculous 255-character path
            # length limit.
            f'--build-temp={package_dir / "build"}',
        ]
    )

    env = os.environ.copy()
    if compiler is not None:
        # CMake will use the compiler specified in the CXX environment variable
        env["CXX"] = compiler

    # distutils.core.run_setup looks nicer, but does not let us check the
    # result easily
    try:
        result = subprocess.run(
            script_args,
            cwd=str(package_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
            env=env,
        )
    except subprocess.CalledProcessError as e:
        print(e.output.decode("utf-8"))
        print("Failed building the model extension.")
        if extra_msg:
            print(f"Note: {extra_msg}")
        raise

    if verbose:
        print(result.stdout.decode("utf-8"))
