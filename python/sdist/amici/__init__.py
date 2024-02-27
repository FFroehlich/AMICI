"""
AMICI
-----

The AMICI Python module provides functionality for importing SBML or PySB
models and turning them into C++ Python extensions.
"""

import contextlib
import importlib
import os
import re
import sys
from pathlib import Path
from types import ModuleType as ModelModule
from typing import Any, Callable, Union


def _get_amici_path():
    """
    Determine package installation path, or, if used directly from git
    repository, get repository root
    """
    basedir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if os.path.exists(os.path.join(basedir, ".git")):
        return os.path.abspath(basedir)
    return os.path.dirname(__file__)


def _get_commit_hash():
    """Get commit hash from file"""
    basedir = os.path.dirname(os.path.dirname(os.path.dirname(amici_path)))
    commitfile = next(
        (
            file
            for file in [
                os.path.join(basedir, ".git", "FETCH_HEAD"),
                os.path.join(basedir, ".git", "ORIG_HEAD"),
            ]
            if os.path.isfile(file)
        ),
        None,
    )

    if commitfile:
        with open(commitfile) as f:
            return str(re.search(r"^([\w]*)", f.read().strip()).group())
    return "unknown"


def _imported_from_setup() -> bool:
    """Check whether this module is imported from `setup.py`"""

    from inspect import currentframe, getouterframes
    from os import sep

    # in case we are imported from setup.py, this will be the AMICI package
    # root directory (otherwise it is most likely the Python library directory,
    # we are not interested in)
    package_root = os.path.realpath(os.path.dirname(os.path.dirname(__file__)))

    for frame in getouterframes(currentframe(), context=0):
        # Need to compare the full path, in case a user tries to import AMICI
        # from a module `*setup.py`. Will still cause trouble if some package
        # requires the AMICI extension during its installation, but seems
        # unlikely...
        frame_path = os.path.realpath(os.path.expanduser(frame.filename))
        if frame_path == os.path.join(
            package_root, "setup.py"
        ) or frame_path.endswith(f"{sep}setuptools{sep}build_meta.py"):
            return True

    return False


# Initialize AMICI paths
#: absolute root path of the amici repository or Python package
amici_path = _get_amici_path()
#: absolute path of the amici swig directory
amiciSwigPath = os.path.join(amici_path, "swig")
#: absolute path of the amici source directory
amiciSrcPath = os.path.join(amici_path, "src")
#: absolute root path of the amici module
amiciModulePath = os.path.dirname(__file__)
#: boolean indicating if this is the full package with swig interface or
#  the raw package without extension
has_clibs: bool = any(
    os.path.isfile(os.path.join(amici_path, wrapper))
    for wrapper in ["amici.py", "amici_without_hdf5.py"]
)
#: boolean indicating if amici was compiled with hdf5 support
hdf5_enabled: bool = False

# Get version number from file
with open(os.path.join(amici_path, "version.txt")) as f:
    __version__ = f.read().strip()

__commit__ = _get_commit_hash()

# Import SWIG module and swig-dependent submodules if required and available
if not _imported_from_setup():
    if has_clibs:
        from . import amici
        from .amici import *

        # has to be done before importing readSolverSettingsFromHDF5
        #  from .swig_wrappers
        hdf5_enabled = "readSolverSettingsFromHDF5" in dir()
        # These modules require the swig interface and other dependencies
        from .numpy import ExpDataView, ReturnDataView  # noqa: F401
        from .pandas import *
        from .swig_wrappers import *

    # These modules don't require the swig interface
    from typing import Protocol, runtime_checkable

    from .de_export import DEExporter, DEModel  # noqa: F401
    from .sbml_import import (  # noqa: F401
        SbmlImporter,
        assignmentRules2observables,
    )

    @runtime_checkable
    class ModelModule(Protocol):  # noqa: F811
        """Type of AMICI-generated model modules.

        To enable static type checking."""

        def getModel(self) -> amici.Model:
            """Create a model instance."""
            ...

        def get_model(self) -> amici.Model:
            """Create a model instance."""
            ...

    AmiciModel = Union[amici.Model, amici.ModelPtr]


class add_path:
    """Context manager for temporarily changing PYTHONPATH"""

    def __init__(self, path: Union[str, Path]):
        self.path: str = str(path)

    def __enter__(self):
        if self.path:
            sys.path.insert(0, self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        with contextlib.suppress(ValueError):
            sys.path.remove(self.path)


def import_model_module(
    module_name: str, module_path: Union[Path, str]
) -> ModelModule:
    """
    Import Python module of an AMICI model

    :param module_name:
        Name of the python package of the model
    :param module_path:
        Absolute or relative path of the package directory
    :return:
        The model module
    """
    module_path = str(module_path)

    # ensure we will find the newly created module
    importlib.invalidate_caches()

    if not os.path.isdir(module_path):
        raise ValueError(f"module_path '{module_path}' is not a directory.")

    module_path = os.path.abspath(module_path)

    # module already loaded?
    if module_name in sys.modules:
        # if a module with that name is already in sys.modules, we remove it,
        # along with all other modules from that package. otherwise, there
        # will be trouble if two different models with the same name are to
        # be imported.
        del sys.modules[module_name]
        # collect first, don't delete while iterating
        to_unload = {
            loaded_module_name
            for loaded_module_name in sys.modules.keys()
            if loaded_module_name.startswith(f"{module_name}.")
        }
        for m in to_unload:
            del sys.modules[m]

    with add_path(module_path):
        return importlib.import_module(module_name)


class AmiciVersionError(RuntimeError):
    """Error thrown if an AMICI model is loaded that is incompatible with
    the installed AMICI base package"""

    pass


def _get_default_argument(func: Callable, arg: str) -> Any:
    """Get the default value of the given argument in the given function."""
    import inspect

    signature = inspect.signature(func)
    if (
        default := signature.parameters[arg].default
    ) is not inspect.Parameter.empty:
        return default
    raise ValueError(f"No default value for argument {arg} of {func}.")
