"""Convenience wrappers for the swig interface"""
import sys
from contextlib import contextmanager, suppress
from typing import List, Optional, Union, Sequence, Dict, Any
import amici
from . import numpy

__all__ = [
    'runAmiciSimulation', 'runAmiciSimulations', 'ExpData',
    'readSolverSettingsFromHDF5', 'writeSolverSettingsToHDF5',
    'set_model_settings', 'get_model_settings',
    'AmiciModel', 'AmiciSolver', 'AmiciExpData', 'AmiciReturnData',
    'AmiciExpDataVector'
]

AmiciModel = Union['amici.Model', 'amici.ModelPtr']
AmiciSolver = Union['amici.Solver', 'amici.SolverPtr']
AmiciExpData = Union['amici.ExpData', 'amici.ExpDataPtr']
AmiciReturnData = Union['amici.ReturnData', 'amici.ReturnDataPtr']
AmiciExpDataVector = Union['amici.ExpDataPtrVector', Sequence[AmiciExpData]]


try:
    from wurlitzer import sys_pipes
except ModuleNotFoundError:
    sys_pipes = suppress


@contextmanager
def _capture_cstdout():
    """Redirect C/C++ stdout to python stdout if python stdout is redirected,
    e.g. in ipython notebook"""
    if sys.stdout == sys.__stdout__:
        yield
    else:
        with sys_pipes():
            yield


def _get_ptr(
        obj: Union[AmiciModel, AmiciExpData, AmiciSolver, AmiciReturnData]
) -> Union['amici.Model', 'amici.ExpData', 'amici.Solver', 'amici.ReturnData']:
    """
    Convenience wrapper that returns the smart pointer pointee, if applicable

    :param obj:
        Potential smart pointer

    :returns:
        Non-smart pointer
    """
    if isinstance(obj, (amici.ModelPtr, amici.ExpDataPtr, amici.SolverPtr,
                        amici.ReturnDataPtr)):
        return obj.get()
    return obj


def runAmiciSimulation(
        model: AmiciModel,
        solver: AmiciSolver,
        edata: Optional[AmiciExpData] = None
) -> 'numpy.ReturnDataView':
    """
    Convenience wrapper around :py:func:`amici.amici.runAmiciSimulation`
    (generated by swig)

    :param model:
        Model instance

`   :param solver:
        Solver instance, must be generated from
        :py:meth:`amici.amici.Model.getSolver`

    :param edata:
        ExpData instance (optional)

    :returns:
        ReturnData object with simulation results
    """
    with _capture_cstdout():
        rdata = amici.runAmiciSimulation(_get_ptr(solver), _get_ptr(edata),
                                         _get_ptr(model))
    return numpy.ReturnDataView(rdata)


def ExpData(*args) -> 'amici.ExpData':
    """
    Convenience wrapper for :py:class:`amici.amici.ExpData` constructors

    :param args: arguments

    :returns: ExpData Instance
    """
    import amici.amici as ext

    if isinstance(args[0], numpy.ReturnDataView):
        return ext.ExpData(_get_ptr(args[0]['ptr']), *args[1:])
    elif isinstance(args[0], (ext.ExpData, ext.ExpDataPtr)):
        # the *args[:1] should be empty, but by the time you read this,
        # the constructor signature may have changed, and you are glad this
        # wrapper did not break.
        return ext.ExpData(_get_ptr(args[0]), *args[1:])
    elif isinstance(args[0], (ext.Model, ext.ModelPtr)):
        return ext.ExpData(_get_ptr(args[0]))
    else:
        return ext.ExpData(*args)


def runAmiciSimulations(
        model: AmiciModel,
        solver: AmiciSolver,
        edata_list: AmiciExpDataVector,
        failfast: bool = True,
        num_threads: int = 1,
) -> List['numpy.ReturnDataView']:
    """
    Convenience wrapper for loops of amici.runAmiciSimulation

    :param model: Model instance
    :param solver: Solver instance, must be generated from Model.getSolver()
    :param edata_list: list of ExpData instances
    :param failfast: returns as soon as an integration failure is encountered
    :param num_threads: number of threads to use (only used if compiled
        with openmp)

    :returns: list of simulation results
    """
    with _capture_cstdout():
        edata_ptr_vector = amici.ExpDataPtrVector(edata_list)
        rdata_ptr_list = amici.runAmiciSimulations(_get_ptr(solver),
                                                   edata_ptr_vector,
                                                   _get_ptr(model),
                                                   failfast,
                                                   num_threads)
    return [numpy.ReturnDataView(r) for r in rdata_ptr_list]


def readSolverSettingsFromHDF5(
        file: str,
        solver: AmiciSolver,
        location: Optional[str] = 'solverSettings'
) -> None:
    """
    Convenience wrapper for :py:func:`amici.readSolverSettingsFromHDF5`

    :param file: hdf5 filename
    :param solver: Solver instance to which settings will be transferred
    :param location: location of solver settings in hdf5 file
    """
    amici.readSolverSettingsFromHDF5(file, _get_ptr(solver), location)


def writeSolverSettingsToHDF5(
        solver: AmiciSolver,
        file: Union[str, object],
        location: Optional[str] = 'solverSettings'
) -> None:
    """
    Convenience wrapper for :py:func:`amici.amici.writeSolverSettingsToHDF5`

    :param file: hdf5 filename, can also be an object created by
        :py:func:`amici.amici.createOrOpenForWriting`
    :param solver: Solver instance from which settings will be stored
    :param location: location of solver settings in hdf5 file
    """
    amici.writeSolverSettingsToHDF5(_get_ptr(solver), file, location)


# Values are suffixes of `get[...]` and `set[...]` `amici.Model` methods.
# If either the getter or setter is not named with this pattern, then the value
# is a tuple where the first and second elements are the getter and setter
# methods, respectively.
model_instance_settings = [
    'AddSigmaResiduals',
    'AlwaysCheckFinite',
    'FixedParameters',
    'InitialStates',
    'InitialStateSensitivities',
    'MinimumSigmaResiduals',
    ('nMaxEvent', 'setNMaxEvent'),
    'Parameters',
    'ParameterList',
    'ParameterScale',  # getter returns a SWIG object
    'ReinitializationStateIdxs',
    'ReinitializeFixedParameterInitialStates',
    'StateIsNonNegative',
    'SteadyStateSensitivityMode',
    ('t0', 'setT0'),
    'Timepoints',
]


def get_model_settings(
        model: AmiciModel,
) -> Dict[str, Any]:
    """Get model settings that are set independently of the compiled model.

    :param model: The AMICI model instance.

    :returns: Keys are AMICI model attributes, values are attribute values.
    """
    settings = {}
    for setting in model_instance_settings:
        getter = setting[0] if isinstance(setting, tuple) else f'get{setting}'
        settings[setting] = getattr(model, getter)()
        # TODO `amici.Model.getParameterScale` returns a SWIG object instead
        # of a Python list/tuple.
        if setting == 'ParameterScale':
            settings[setting] = tuple(settings[setting])
    return settings


def set_model_settings(
        model: AmiciModel,
        settings: Dict[str, Any],
) -> None:
    """Set model settings.

    :param model: The AMICI model instance.
    :param settings: Keys are callable attributes (setters) of an AMICI model,
        values are provided to the setters.
    """
    for setting, value in settings.items():
        setter = setting[1] if isinstance(setting, tuple) else f'set{setting}'
        getattr(model, setter)(value)
