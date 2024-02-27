%module solver_cvodes

// Add necessary symbols to generated header
%{
#include "amici/solver_cvodes.h"
using namespace amici;
%}

%newobject amici::CVodeSolver::clone;
%feature("notabstract") amici::CVodeSolver;

// Required with SWIG 4.2.0 https://github.com/AMICI-dev/AMICI/issues/2275
%extend amici::CVodeSolver {
    CVodeSolver() {
        return new CVodeSolver();
    }

    CVodeSolver(Solver const& solver) {
        return new CVodeSolver(dynamic_cast<CVodeSolver const&>(solver));
    }
}

// Process symbols in header
%include "amici/solver_cvodes.h"
