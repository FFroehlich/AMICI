%define DOCSTRING
"""
Core C++ bindings

This module encompasses the complete public C++ API of AMICI, which was
exposed via swig. All functions listed here are directly accessible in the
main amici package, i.e., :py:class:`amici.amici.ExpData` is available as
``amici.ExpData``.
Usage of functions and classes from the base ``amici`` package is
generally recommended as they often include convenience wrappers that avoid
common pitfalls when accessing C++ types from python and implement some
nonstandard type conversions.
"""
%enddef
%module (docstring=DOCSTRING) amici

// typemaps for docstrings
%typemap(doctype) std::unique_ptr< amici::ExpData >::pointer "ExpData";
%typemap(doctype) std::unique_ptr< amici::Model > "ModelPtr";
%typemap(doctype) std::unique_ptr< amici::Solver > "SolverPtr";
%typemap(doctype) std::vector< amici::realtype,std::allocator< amici::realtype > > "DoubleVector";
%typemap(doctype) std::vector< double,std::allocator< double > > "DoubleVector";
%typemap(doctype) std::vector< int,std::allocator< int > > "IntVector";
%typemap(doctype) std::vector< amici::ParameterScaling,std::allocator< amici::ParameterScaling > > "ParameterScalingVector";
%typemap(doctype) std::vector< std::string,std::allocator< std::string > > "StringVector";
%typemap(doctype) std::vector< bool,std::allocator< bool > > "BoolVector";
%typemap(doctype) std::map< std::string,amici::realtype,std::less< std::string >, std::allocator< std::pair< std::string const,amici::realtype > > > "StringDoubleMap";
%typemap(doctype) std::vector< amici::ExpData *,std::allocator< amici::ExpData * > > "ExpDataPtrVector";
%typemap(doctype) std::vector< std::unique_ptr< amici::ReturnData >,std::allocator< std::unique_ptr< amici::ReturnData > > > "Iterable[ReturnData]";
%typemap(doctype) void "None";
%typemap(doctype) std::unique_ptr< amici::Solver > "Solver";
%typemap(doctype) amici::InternalSensitivityMethod "InternalSensitivityMethod";
%typemap(doctype) amici::InterpolationType "InterpolationType";
%typemap(doctype) amici::LinearMultistepMethod "LinearMultistepMethod";
%typemap(doctype) amici::LinearSolver "LinearSolver";
%typemap(doctype) amici::Model * "Model";
%typemap(doctype) amici::Model const * "Model";
%typemap(doctype) amici::NewtonDampingFactorMode "NewtonDampingFactorMode";
%typemap(doctype) amici::NonlinearSolverIteration "NonlinearSolverIteration";
%typemap(doctype) amici::RDataReporting "RDataReporting";
%typemap(doctype) amici::SensitivityMethod "SensitivityMethod";
%typemap(doctype) amici::SensitivityOrder "SensitivityOrder";
%typemap(doctype) amici::Solver * "Solver";
%typemap(doctype) amici::SteadyStateSensitivityMode "SteadyStateSensitivityMode";
%typemap(doctype) amici::realtype "float";
%typemap(doctype) DoubleVector "numpy.ndarray";
%typemap(doctype) IntVector "list[int]";
%typemap(doctype) std::pair< size_t,size_t > "tuple[int, int]";
%typemap(doctype) std::string "str";
%typemap(doctype) std::string const & "str";
%typemap(doctype) std::unique_ptr< amici::ExpData >   "ExpData";
%typemap(doctype) std::unique_ptr< amici::ReturnData > "ReturnData";
%typemap(doctype) size_t "int";


%include <exception.i>
%exception {
    try {
        $action
    } catch(std::exception const& ex) {
        SWIG_exception_fail(SWIG_RuntimeError, ex.what());
    } catch(...) {
        SWIG_exception_fail(SWIG_RuntimeError, "Unknown exception occurred");
    }
}

// Warning 503: Can't wrap 'operator ==' unless renamed to a valid identifier.
%rename("__eq__") operator ==;

%{
#define SWIG_FILE_WITH_INIT
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <vector>
#include <stdexcept>
%}

// numpy
%init %{
import_array();
%}

// Expose vectors
%include <stl.i>
%template(DoubleVector) std::vector<double>;
%extend std::vector<double> {
%pythoncode %{
def __repr__(self):
    return self.this.__repr__()[:-1] + '; ' + repr(np.asarray(self, dtype=np.float64)) + ' >'

%}
};
%template(IntVector) std::vector<int>;
%extend std::vector<int> {
%pythoncode %{
def __repr__(self):
    return self.this.__repr__()[:-1] + '; ' + repr(np.asarray(self, dtype=np.int64)) + ' >'

%}
};
%template(BoolVector) std::vector<bool>;
%extend std::vector<bool> {
%pythoncode %{
def __repr__(self):
    return self.this.__repr__()[:-1] + '; ' + repr(np.asarray(self, dtype=np.bool_)) + ' >'

%}
};
%template(StringVector) std::vector<std::string>;
%extend std::vector<string> {
%pythoncode %{
def __repr__(self):
    return self.this.__repr__()[:-1] + '; ' + repr(list(self)) + ' >'

%}
};
%feature("docstring") std::map<std::string, double>
"Swig-Generated class templating :class:`Dict`
[:class:`str`, :class:`float`] to facilitate interfacing with C++ bindings.";
%template(StringDoubleMap) std::map<std::string, double>;
%extend std::map<std::string, double> {
%pythoncode %{
def __repr__(self):
    return self.this.__repr__()[:-1] + '; ' + repr(dict(self)) + ' >'

%}
};

// Let numpy access std::vector
%{
static_assert (sizeof(double) == sizeof (npy_double), "Numpy double size mismatch");
static_assert (sizeof(int) == sizeof (npy_int), "Numpy integer size mismatch");

#include "../swig/stdvec2numpy.h"
using namespace amici;
%}
%include "stdvec2numpy.h"

typedef double realtype;

%include std_unique_ptr.i
wrap_unique_ptr(SolverPtr, amici::Solver)
wrap_unique_ptr(ReturnDataPtr, amici::ReturnData)
wrap_unique_ptr(ModelPtr, amici::Model)
wrap_unique_ptr(ExpDataPtr, amici::ExpData)

%naturalvar amici::SimulationParameters::x0;
%naturalvar amici::SimulationParameters::sx0;
%naturalvar amici::SimulationParameters::parameters;
%naturalvar amici::SimulationParameters::pscale;
%naturalvar amici::SimulationParameters::plist;
%naturalvar amici::SimulationParameters::fixedParameters;
%naturalvar amici::SimulationParameters::fixedParametersPreequilibration;
%naturalvar amici::SimulationParameters::fixedParametersPresimulation;
%naturalvar amici::SimulationParameters::reinitialization_state_idxs_sim;
%naturalvar amici::SimulationParameters::reinitialization_state_idxs_presim;

// DO NOT IGNORE amici::SimulationParameters, amici::ModelDimensions, amici::CpuTimer
%ignore amici::ModelContext;
%ignore amici::ContextManager;
%ignore amici::ModelState;
%ignore amici::ModelStateDerived;
%ignore amici::unravel_index;
%ignore amici::backtraceString;
%ignore amici::Logger;
%ignore amici::SimulationState;

// Include before any other header which uses enums defined there
%include "amici/defines.h"

%include "amici/model_dimensions.h"
%include "amici/model_state.h"
%include "amici/simulation_parameters.h"

%include abstract_model.i
%include misc.i
%include edata.i
%include solver.i
%include solver_idas.i
%include solver_cvodes.i
%include model.i
%include model_ode.i
%include model_dae.i
%include logging.i
%include rdata.i

#ifndef AMICI_SWIG_WITHOUT_HDF5
%include hdf5.i
#endif

// Return Python list of raw pointers instead of std::vector<std::unique_ptr> which is a huge pain
%typemap(out) std::vector<std::unique_ptr<amici::ReturnData>> %{
    $result = PyList_New($1.size());
    for (int i = 0; i < (int) $1.size(); i++) {
        PyObject *o = SWIG_NewPointerObj($1.at(i).release(), $descriptor(amici::ReturnData*), SWIG_POINTER_OWN);
        PyList_SetItem($result, i, o);
    }
%}

// Add necessary symbols to generated header
%{
#include "amici/amici.h"
using namespace amici;
%}

// Prevent using ValueWrapper, but don't expose unique_ptr vector
%ignore std::vector<std::unique_ptr<amici::ReturnData>>;
%template(ReturnDataPtrVector) std::vector<std::unique_ptr<amici::ReturnData>>;
%extend std::vector<std::unique_ptr<amici::ReturnData>> {
%pythoncode %{
def __repr__(self):
    return self.this.__repr__()[:-1] + '; ' + repr(list(self)) + ' >'

%}
};

// Process symbols in header
%include "amici/amici.h"

// Expose vectors
%template(ExpDataPtrVector) std::vector<amici::ExpData*>;
%extend std::vector<amici::ExpData*> {
%pythoncode %{
def __repr__(self):
    return self.this.__repr__()[:-1] + '; ' + repr(list(self)) + ' >'

%}
};
%template(LogItemVector) std::vector<amici::LogItem>;
%extend std::vector<amici::LogItem> {
%pythoncode %{
def __repr__(self):
    return self.this.__repr__()[:-1] + '; ' + repr(list(self)) + ' >'

%}
};


// Convert integer values to enum class
// defeats the purpose of enum class, but didn't find a better way to allow for
// vectors of enum class types in python
%feature("docstring") parameterScalingFromIntVector
"Swig-Generated class, which, in contrast to other Vector
classes, does not allow for simple interoperability with common
Python types, but must be created using
:func:`amici.amici.parameterScalingFromIntVector`";
%{
namespace amici {
std::vector<amici::ParameterScaling> parameterScalingFromIntVector(std::vector<int> const& intVec) {
    std::vector<amici::ParameterScaling> result(intVec.size());
    for (int i = 0; i < (int) result.size(); ++i) {
        result[i] = static_cast<amici::ParameterScaling>(intVec[i]);
    }
    return result;
}
}; // namespace amici
%}
%extend amici::Model {
    void setParameterScale(std::vector<int> const& intVec) {
        std::vector<amici::ParameterScaling> result(intVec.size());
        for (int i = 0; i < (int) result.size(); ++i) {
            result[i] = static_cast<amici::ParameterScaling>(intVec[i]);
        }
        $self->setParameterScale(result);
    }
}
namespace amici {
    std::vector<amici::ParameterScaling> parameterScalingFromIntVector(std::vector<int> const& intVec);
    void Model::setParameterScale(std::vector<int> const& intVec);
}
%template(ParameterScalingVector) std::vector<amici::ParameterScaling>;
%extend std::vector<amici::ParameterScaling> {
%pythoncode %{
def __repr__(self):
    return self.this.__repr__()[:-1] + '; ' + repr(list(self)) + ' >'

%}
};


// Add function to check if amici was compiled with OpenMP
%feature("docstring") compiledWithOpenMP
    "AMICI extension was compiled with OpenMP?";
%{
namespace amici {
/** AMICI extension was compiled with OpenMP? */
bool compiledWithOpenMP() {
#if defined(_OPENMP)
    return true;
#else
    return false;
#endif
}
};
%}
namespace amici {
bool compiledWithOpenMP();
}

%pythoncode %{
from enum import IntEnum
def enum(prefix):
    values = {k:v for k,v in globals().items() if k.startswith(prefix + '_')}
    values = {k[len(prefix)+1:]:v for k,v in values.items()}
    return IntEnum(prefix, values)
ParameterScaling = enum('ParameterScaling')
ObservableScaling = enum('ObservableScaling')
SecondOrderMode = enum('SecondOrderMode')
SensitivityOrder = enum('SensitivityOrder')
SensitivityMethod = enum('SensitivityMethod')
LinearSolver = enum('LinearSolver')
InternalSensitivityMethod = enum('InternalSensitivityMethod')
InterpolationType = enum('InterpolationType')
LinearMultistepMethod = enum('LinearMultistepMethod')
NonlinearSolverIteration = enum('NonlinearSolverIteration')
SteadyStateComputationMode = enum('SteadyStateComputationMode')
SteadyStateSensitivityMode = enum('SteadyStateSensitivityMode')
SteadyStateStatus = enum('SteadyStateStatus')
NewtonDampingFactorMode = enum('NewtonDampingFactorMode')
FixedParameterContext = enum('FixedParameterContext')
RDataReporting = enum('RDataReporting')
%}

%template(SteadyStateStatusVector) std::vector<amici::SteadyStateStatus>;
%extend std::vector<amici::SteadyStateStatus> {
%pythoncode %{
def __repr__(self):
    return self.this.__repr__()[:-1] + '; ' + repr(list(self)) + ' >'

%}
};

// Handle AMICI_DLL_DIRS environment variable
%pythonbegin %{
import sys
import os

if sys.platform == 'win32' and (dll_dirs := os.environ.get('AMICI_DLL_DIRS')):
    for dll_dir in dll_dirs.split(os.pathsep):
        os.add_dll_directory(dll_dir)

%}

// import additional types for typehints
// also import np for use in __repr__ functions
%pythonbegin %{
from typing import TYPE_CHECKING, Iterable, Sequence
import numpy as np
if TYPE_CHECKING:
    import numpy
%}

%pythoncode %{


__all__ = [
    x
    for x in dir(sys.modules[__name__])
    if not x.startswith('_')
    and x not in {"np", "sys", "os", "numpy", "IntEnum", "enum", "pi", "TYPE_CHECKING", "Iterable", "Sequence"}
]
%}
