%module solver_idas

// Add necessary symbols to generated header
%{
#include "amici/solver_idas.h"
using namespace amici;
%}

%newobject amici::IDASolver::clone;
%feature("notabstract") amici::IDASolver;

// Required for SWIG 4.2.0 https://github.com/AMICI-dev/AMICI/issues/2275
%extend amici::IDASolver {
    IDASolver() {
        return new IDASolver();
    }

    IDASolver(Solver const& solver) {
        return new IDASolver(dynamic_cast<IDASolver const&>(solver));
    }
}

// Process symbols in header
%include "amici/solver_idas.h"
