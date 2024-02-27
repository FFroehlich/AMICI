#ifndef AMICI_DEFINES_H
#define AMICI_DEFINES_H

#if !defined(_USE_MATH_DEFINES)
#define _USE_MATH_DEFINES
#endif

#include <cmath>

/* Math constants in case _USE_MATH_DEFINES is not supported */
#if defined(_USE_MATH_DEFINES)
#if !defined(M_E)
#define M_E 2.71828182845904523536
#endif
#if !defined(M_LOG2E)
#define M_LOG2E 1.44269504088896340736
#endif
#if !defined(M_LOG10E)
#define M_LOG10E 0.434294481903251827651
#endif
#if !defined(M_LN2)
#define M_LN2 0.693147180559945309417
#endif
#if !defined(M_LN10)
#define M_LN10 2.30258509299404568402
#endif
#if !defined(M_PI)
#define M_PI 3.14159265358979323846
#endif
#if !defined(M_PI_2)
#define M_PI_2 1.57079632679489661923
#endif
#if !defined(M_PI_4)
#define M_PI_4 0.785398163397448309616
#endif
#if !defined(M_1_PI)
#define M_1_PI 0.318309886183790671538
#endif
#if !defined(M_2_PI)
#define M_2_PI 0.636619772367581343076
#endif
#if !defined(M_2_SQRTPI)
#define M_2_SQRTPI 1.12837916709551257390
#endif
#if !defined(M_SQRT2)
#define M_SQRT2 1.41421356237309504880
#endif
#if !defined(M_SQRT1_2)
#define M_SQRT1_2 0.707106781186547524401
#endif
#endif

namespace amici {

constexpr double pi = M_PI;

// clang-format off

constexpr int AMICI_ONEOUTPUT=                 5;

// Return codes
//
// NOTE: When adding / removing / renaming return codes,
//       please update simulation_status_to_str_map in amici.h
constexpr int AMICI_RECOVERABLE_ERROR=         1;
constexpr int AMICI_UNRECOVERABLE_ERROR=     -10;
constexpr int AMICI_TOO_MUCH_WORK=            -1;
constexpr int AMICI_TOO_MUCH_ACC=             -2;
constexpr int AMICI_ERR_FAILURE=              -3;
constexpr int AMICI_CONV_FAILURE=             -4;
constexpr int AMICI_RHSFUNC_FAIL=             -8;
constexpr int AMICI_FIRST_RHSFUNC_ERR=        -9;
constexpr int AMICI_ILL_INPUT=               -22;
constexpr int AMICI_ERROR=                   -99;
constexpr int AMICI_NO_STEADY_STATE=         -81;
constexpr int AMICI_DAMPING_FACTOR_ERROR=    -86;
constexpr int AMICI_SINGULAR_JACOBIAN=      -809;
constexpr int AMICI_NOT_IMPLEMENTED=        -999;
constexpr int AMICI_MAX_TIME_EXCEEDED  =   -1000;
constexpr int AMICI_NOT_RUN=               -1001;
constexpr int AMICI_SUCCESS=                   0;
constexpr int AMICI_DATA_RETURN=               1;
constexpr int AMICI_ROOT_RETURN=               2;

constexpr int AMICI_NORMAL=                    1;
constexpr int AMICI_ONE_STEP=                  2;

constexpr int AMICI_PREEQUILIBRATE=           -1;

/** defines variable type for simulation variables
 * (determines numerical accuracy) */
using realtype = double;

/** BLAS Matrix Layout, affects dgemm and gemv calls */
enum class BLASLayout{
    rowMajor = 101,
    colMajor = 102
};

/** BLAS Matrix Transposition, affects dgemm and gemv calls */
enum class BLASTranspose {
    noTrans = 111,
    trans = 112,
    conjTrans = 113
};

/** modes for parameter transformations */
enum class ParameterScaling {
    none,
    ln,
    log10
};

/** modes for observable scaling */
enum class ObservableScaling {
    lin,
    log,
    log10
};

/** modes for second order sensitivity analysis */
enum class SecondOrderMode {
    none,
    full,
    directional
};

/** orders of sensitivity analysis */
enum class SensitivityOrder {
    /** Don't compute sensitivities. */
    none,
    /** First-order sensitivities. */
    first,
    /** Second-order sensitivities. */
    second
};

/** methods for sensitivity computation */
enum class SensitivityMethod {
    /** Don't compute sensitivities. */
    none,
    /** Forward sensitivity analysis. */
    forward,
    /** Adjoint sensitivity analysis. */
    adjoint
};

/** linear solvers for CVODES/IDAS */
enum class LinearSolver {
    dense       = 1,
    band        = 2,
    LAPACKDense = 3,
    LAPACKBand  = 4,
    diag        = 5,
    SPGMR       = 6,
    SPBCG       = 7,
    SPTFQMR     = 8,
    KLU         = 9,
    SuperLUMT   = 10,
};

/** CVODES/IDAS forward sensitivity computation method */
enum class InternalSensitivityMethod {
    simultaneous = 1,
    staggered = 2,
    staggered1 = 3
};

/** CVODES/IDAS state interpolation for adjoint sensitivity analysis */
enum class InterpolationType {
    hermite = 1,
    polynomial = 2
};

/** CVODES/IDAS linear multistep method */
enum class LinearMultistepMethod {
    adams = 1,
    BDF = 2
};

/** CVODES/IDAS Nonlinear Iteration method */
enum class NonlinearSolverIteration {
    functional = 1, /** deprecated */
    fixedpoint = 1,
    newton = 2
};

/** Steady-state computation mode in steadyStateProblem */
enum class SteadyStateComputationMode {
    newtonOnly,
    integrationOnly,
    integrateIfNewtonFails
};

/** Sensitivity computation mode in steadyStateProblem */
enum class SteadyStateSensitivityMode {
    newtonOnly,
    integrationOnly,
    integrateIfNewtonFails
};

/** State in which the steady state computation finished */
enum class SteadyStateStatus {
    failed_too_long_simulation = -5,
    failed_damping = -4,
    failed_factorization = -3,
    failed_convergence = -2,
    failed = -1,
    not_run = 0,
    success = 1
};

/** Context for which the sensitivity flag should be computed */
enum class SteadyStateContext {
    newtonSensi = 0,
    sensiStorage = 1,
    solverCreation = 2
};

/** Damping factor flag for the Newton method */
enum class NewtonDampingFactorMode {
    off = 0,
    on = 1
};

/** fixedParameter to be used in condition context */
enum class FixedParameterContext {
    simulation = 0,
    preequilibration = 1,
    presimulation = 2,
};

enum class RDataReporting {
    full,
    residuals,
    likelihood,
};

/** boundary conditions for splines */
enum class SplineBoundaryCondition {
    given                 = -1,
    zeroDerivative        =  0,
    natural               =  1,
    naturalZeroDerivative =  2,
    periodic              =  3,
};

/** extrapolation methods for splines */
enum class SplineExtrapolation {
    noExtrapolation = -1,
    constant        =  0,
    linear          =  1,
    polynomial      =  2,
    periodic        =  3,
};

// clang-format on

} // namespace amici

#endif
