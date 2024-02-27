%module solver

// Add necessary symbols to generated header
%{
#include "amici/solver.h"
using namespace amici;
%}

%rename(equals) operator==;

// remove functions that use AmiVector(Array) since that class anyways cannot
// be exposed in swig
%ignore getAdjointDerivativeState;
%ignore getAdjointQuadrature;
%ignore getQuadrature;
%ignore getAdjointState;
%ignore getDerivativeState;
%ignore getState;
%ignore getStateSensitivity;
%ignore quadReInitB;
%ignore reInit;
%ignore reInitB;
%ignore sensReInit;
%ignore setup;
%ignore setupB;
%ignore setupSteadystate;
%ignore writeSolution;
%ignore writeSolutionB;
%ignore calcIC;
%ignore calcICB;
%ignore sensToggleOff;
%ignore solveB;
%ignore step;
%ignore run;
%ignore runB;
%ignore resetDiagnosis;
%ignore storeDiagnosis;
%ignore storeDiagnosisB;
%ignore turnOffRootFinding;
%ignore getRootInfo;
%ignore updateAndReinitStatesAndSensitivities;
%ignore getCpuTime;
%ignore getCpuTimeB;
%ignore getLastOrder;
%ignore getNumErrTestFails;
%ignore getNumErrTestFailsB;
%ignore getNumNonlinSolvConvFails;
%ignore getNumNonlinSolvConvFailsB;
%ignore getNumRhsEvals;
%ignore getNumRhsEvalsB;
%ignore getNumSteps;
%ignore getNumStepsB;
%ignore gett;
%ignore startTimer;
%ignore switchForwardSensisOff;
%ignore timeExceeded;

// Solver.__repr__
%pythoncode %{
def _solver_repr(self: "Solver"):
    return "\n".join([
        self.this.__repr__()[:-1],
        "  reporting_mode: "
        f"{RDataReporting(self.getReturnDataReportingMode())!r}",
        f"  sens_meth: {SensitivityMethod(self.getSensitivityMethod())!r}",
        f"  sens_order: {SensitivityOrder(self.getSensitivityOrder())!r}",
        f"  sens_meth_preeq: {SensitivityMethod(self.getSensitivityMethodPreequilibration())!r}",
        f"  maxsteps: {self.getMaxSteps()}",
        f"  maxtime: {self.getMaxTime()}s",
        f"  abs_tol: {self.getAbsoluteTolerance()}",
        f"  rel_tol: {self.getRelativeTolerance()}",
        f"  abs_tol_b: {self.getAbsoluteToleranceB()}",
        f"  rel_tol_b: {self.getRelativeToleranceB()}",
        f"  abs_tol_fsa: {self.getAbsoluteToleranceFSA()}",
        f"  rel_tol_fsa: {self.getRelativeToleranceFSA()}",
        f"  abs_tol_quad: {self.getAbsoluteToleranceQuadratures()}",
        f"  rel_tol_quad: {self.getRelativeToleranceQuadratures()}",
        f"  abs_tol_ss: {self.getAbsoluteToleranceSteadyState()}",
        f"  rel_tol_ss: {self.getRelativeToleranceSteadyState()}",
        f"  abs_tol_sss: {self.getAbsoluteToleranceSteadyStateSensi()}",
        f"  rel_tol_sss: {self.getRelativeToleranceSteadyStateSensi()}",
        f"  int_sens_meth: {InternalSensitivityMethod(self.getInternalSensitivityMethod())!r}",
        f"  int_type: {InterpolationType(self.getInterpolationType())!r}",
        f"  linsol: {LinearSolver(self.getLinearSolver())!r}",
        f"  lmm: {LinearMultistepMethod(self.getLinearMultistepMethod())!r}",
        f"  newton_damp_mode: {NewtonDampingFactorMode(self.getNewtonDampingFactorMode())!r}",
        f"  newton_damp_lb: {self.getNewtonDampingFactorLowerBound()}",
        f"  newton_maxsteps: {self.getNewtonMaxSteps()}",
        f"  newton_ss_check: {self.getNewtonStepSteadyStateCheck()}",
        f"  sens_ss_check: {self.getSensiSteadyStateCheck()}",
        f"  interpolation_type: {InterpolationType(self.getInterpolationType())!r}",
        f"  ism: {InternalSensitivityMethod(self.getInternalSensitivityMethod())!r}",
        f"  nlsol_iter: {NonlinearSolverIteration(self.getNonlinearSolverIteration())!r}",
        f"  stability_limit: {self.getStabilityLimitFlag()}",
        f"  state_ordering: {self.getStateOrdering()}",
        ">"
    ])
%}
%extend amici::CVodeSolver {
%pythoncode %{
def __repr__(self):
    return _solver_repr(self)
%}
};
%extend amici::IDASolver {
%pythoncode %{
def __repr__(self):
    return _solver_repr(self)
%}
};

%extend std::unique_ptr<amici::Solver> {
%pythoncode %{
def __repr__(self):
    return _solver_repr(self)

def __deepcopy__(self, memo):
    return self.clone()
%}
};

%extend amici::Solver {
%pythoncode %{
def __deepcopy__(self, memo):
    return self.clone()
%}
};

%newobject amici::Solver::clone;
// Process symbols in header
%include "amici/solver.h"
