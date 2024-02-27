#include "amici/steadystateproblem.h"
#include "amici/backwardproblem.h"
#include "amici/defines.h"
#include "amici/edata.h"
#include "amici/misc.h"
#include "amici/model.h"
#include "amici/newton_solver.h"
#include "amici/solver.h"

#include <cmath>
#include <cvodes/cvodes.h>
#include <memory>
#include <sundials/sundials_dense.h>

constexpr realtype conv_thresh = 1.0;

namespace amici {

SteadystateProblem::SteadystateProblem(Solver const& solver, Model const& model)
    : delta_(model.nx_solver)
    , delta_old_(model.nx_solver)
    , ewt_(model.nx_solver)
    , ewtQB_(model.nplist())
    , x_old_(model.nx_solver)
    , xdot_(model.nx_solver)
    , sdx_(model.nx_solver, model.nplist())
    , xB_(model.nJ * model.nx_solver)
    , xQ_(model.nJ * model.nx_solver)
    , xQB_(model.nplist())
    , xQBdot_(model.nplist())
    , max_steps_(solver.getNewtonMaxSteps())
    , dJydx_(model.nJ * model.nx_solver * model.nt(), 0.0)
    , state_(
          {INFINITY,                                        // t
           AmiVector(model.nx_solver),                      // x
           AmiVector(model.nx_solver),                      // dx
           AmiVectorArray(model.nx_solver, model.nplist()), // sx
           model.getModelState()}
      )
    , // state
    atol_(solver.getAbsoluteToleranceSteadyState())
    , rtol_(solver.getRelativeToleranceSteadyState())
    , atol_sensi_(solver.getAbsoluteToleranceSteadyStateSensi())
    , rtol_sensi_(solver.getRelativeToleranceSteadyStateSensi())
    , atol_quad_(solver.getAbsoluteToleranceQuadratures())
    , rtol_quad_(solver.getRelativeToleranceQuadratures())
    , newton_solver_(NewtonSolver::getSolver(solver, model))
    , damping_factor_mode_(solver.getNewtonDampingFactorMode())
    , damping_factor_lower_bound_(solver.getNewtonDampingFactorLowerBound())
    , newton_step_conv_(solver.getNewtonStepSteadyStateCheck())
    , check_sensi_conv_(solver.getSensiSteadyStateCheck()) {
    /* Check for compatibility of options */
    if (solver.getSensitivityMethod() == SensitivityMethod::forward
        && solver.getSensitivityMethodPreequilibration()
               == SensitivityMethod::adjoint
        && solver.getSensitivityOrder() > SensitivityOrder::none)
        throw AmiException("Preequilibration using adjoint sensitivities "
                           "is not compatible with using forward "
                           "sensitivities during simulation");
    if (solver.getSensitivityMethod() == SensitivityMethod::forward
        && model.getSteadyStateComputationMode()
               == SteadyStateComputationMode::newtonOnly
        && model.getSteadyStateSensitivityMode()
               == SteadyStateSensitivityMode::integrationOnly)
        throw AmiException("For forward sensitivity analysis steady-state "
                           "computation mode 'newtonOnly' and steady-state "
                           "sensitivity mode 'integrationOnly' are not "
                           "compatible as numerical integration of the model "
                           "ODEs and corresponding forward sensitivities ODEs "
                           "is coupled");
}

void SteadystateProblem::workSteadyStateProblem(
    Solver const& solver, Model& model, int it
) {
    initializeForwardProblem(it, solver, model);

    /* Compute steady state, track computation time */
    CpuTimer cpu_timer;
    findSteadyState(solver, model, it);

    /* Check whether state sensis still need to be computed */
    if (getSensitivityFlag(
            model, solver, it, SteadyStateContext::newtonSensi
        )) {
        try {
            /* this might still fail, if the Jacobian is singular and
             simulation did not find a steady state */
            newton_solver_->computeNewtonSensis(state_.sx, model, state_);
        } catch (NewtonFailure const&) {
            throw AmiException(
                "Steady state sensitivity computation failed due "
                "to unsuccessful factorization of RHS Jacobian"
            );
        }
    }
    cpu_time_ = cpu_timer.elapsed_milliseconds();
}

void SteadystateProblem::workSteadyStateBackwardProblem(
    Solver const& solver, Model& model, BackwardProblem const* bwd
) {

    if (!initializeBackwardProblem(solver, model, bwd))
        return;

    /* compute quadratures, track computation time */
    CpuTimer cpu_timer;
    computeSteadyStateQuadrature(solver, model);
    cpu_timeB_ = cpu_timer.elapsed_milliseconds();
}

void SteadystateProblem::findSteadyState(
    Solver const& solver, Model& model, int it
) {
    steady_state_status_.resize(3, SteadyStateStatus::not_run);
    /* Turn off Newton's method if 'integrationOnly' approach is chosen for
    steady-state computation or newton_maxsteps is set to 0 or
    if 'integrationOnly' approach is chosen for sensitivity computation
    in combination with forward sensitivities approach. The latter is necessary
    as numerical integration of the model ODEs and corresponding
    forward sensitivities ODEs is coupled. If 'integrationOnly' approach is
    chosen for sensitivity computation it is enforced that steady state is
    computed only by numerical integration as well. */
    bool turnOffNewton
        = model.getSteadyStateComputationMode()
              == SteadyStateComputationMode::integrationOnly
          || solver.getNewtonMaxSteps() == 0
          || (model.getSteadyStateSensitivityMode()
                  == SteadyStateSensitivityMode::integrationOnly
              && ((it == -1
                   && solver.getSensitivityMethodPreequilibration()
                          == SensitivityMethod::forward)
                  || solver.getSensitivityMethod() == SensitivityMethod::forward
              ));

    bool turnOffSimulation = model.getSteadyStateComputationMode()
                             == SteadyStateComputationMode::newtonOnly;

    /* First, try to run the Newton solver */
    if (!turnOffNewton)
        findSteadyStateByNewtonsMethod(model, false);

    /* Newton solver didn't work, so try to simulate to steady state */
    if (!turnOffSimulation && !checkSteadyStateSuccess())
        findSteadyStateBySimulation(solver, model, it);

    /* Simulation didn't work, retry the Newton solver from last sim state. */
    if (!turnOffNewton && !turnOffSimulation && !checkSteadyStateSuccess())
        findSteadyStateByNewtonsMethod(model, true);

    /* Nothing worked, throw an as informative error as possible */
    if (!checkSteadyStateSuccess())
        handleSteadyStateFailure(
            !turnOffNewton, !turnOffSimulation,
            !turnOffNewton && !turnOffSimulation
        );
}

void SteadystateProblem::findSteadyStateByNewtonsMethod(
    Model& model, bool newton_retry
) {
    int ind = newton_retry ? 2 : 0;
    try {
        applyNewtonsMethod(model, newton_retry);
        steady_state_status_[ind] = SteadyStateStatus::success;
    } catch (NewtonFailure const& ex) {
        /* nothing to be done */
        switch (ex.error_code) {
        case AMICI_TOO_MUCH_WORK:
            steady_state_status_[ind] = SteadyStateStatus::failed_convergence;
            break;
        case AMICI_NO_STEADY_STATE:
            steady_state_status_[ind]
                = SteadyStateStatus::failed_too_long_simulation;
            break;
        case AMICI_SINGULAR_JACOBIAN:
            steady_state_status_[ind] = SteadyStateStatus::failed_factorization;
            break;
        case AMICI_DAMPING_FACTOR_ERROR:
            steady_state_status_[ind] = SteadyStateStatus::failed_damping;
            break;
        default:
            steady_state_status_[ind] = SteadyStateStatus::failed;
            break;
        }
    }
}

void SteadystateProblem::findSteadyStateBySimulation(
    Solver const& solver, Model& model, int it
) {
    try {
        if (it < 0) {
            /* Preequilibration? -> Create a new solver instance for sim */
            bool integrateSensis = getSensitivityFlag(
                model, solver, it, SteadyStateContext::solverCreation
            );
            auto newtonSimSolver = createSteadystateSimSolver(
                solver, model, integrateSensis, false
            );
            runSteadystateSimulation(*newtonSimSolver, model, false);
        } else {
            /* Solver was already created, use this one */
            runSteadystateSimulation(solver, model, false);
        }
        steady_state_status_[1] = SteadyStateStatus::success;
    } catch (IntegrationFailure const& ex) {
        switch (ex.error_code) {
        case AMICI_TOO_MUCH_WORK:
            steady_state_status_[1] = SteadyStateStatus::failed_convergence;
            if (model.logger)
                model.logger->log(
                    LogSeverity::debug, "EQUILIBRATION_FAILURE",
                    "AMICI equilibration exceeded maximum number of"
                    " integration steps at t=%g.",
                    ex.time
                );
            break;
        case AMICI_RHSFUNC_FAIL:
            steady_state_status_[1]
                = SteadyStateStatus::failed_too_long_simulation;
            if (model.logger)
                model.logger->log(
                    LogSeverity::debug, "EQUILIBRATION_FAILURE",
                    "AMICI equilibration was stopped after exceedingly"
                    " long simulation time at t=%g.",
                    ex.time
                );
            break;
        default:
            steady_state_status_[1] = SteadyStateStatus::failed;
            if (model.logger)
                model.logger->log(
                    LogSeverity::debug, "OTHER",
                    "AMICI equilibration failed at t=%g.", ex.time
                );
        }
    } catch (AmiException const& ex) {
        if (model.logger)
            model.logger->log(
                LogSeverity::debug, "OTHER", "AMICI equilibration failed: %s",
                ex.what()
            );
        steady_state_status_[1] = SteadyStateStatus::failed;
    }
}

void SteadystateProblem::initializeForwardProblem(
    int it, Solver const& solver, Model& model
) {
    newton_solver_->reinitialize();
    /* process solver handling for pre- or postequilibration */
    if (it == -1) {
        /* solver was not run before, set up everything */
        auto roots_found = std::vector<int>(model.ne, 0);
        model.initialize(
            state_.x, state_.dx, state_.sx, sdx_,
            solver.getSensitivityOrder() >= SensitivityOrder::first, roots_found
        );
        state_.t = model.t0();
        solver.setup(state_.t, &model, state_.x, state_.dx, state_.sx, sdx_);
    } else {
        /* solver was run before, extract current state from solver */
        solver.writeSolution(&state_.t, state_.x, state_.dx, state_.sx, xQ_);
    }

    /* overwrite starting timepoint */
    if (it < 1) /* No previous time point computed, set t = t0 */
        state_.t = model.t0();
    else /* Carry on simulating from last point */
        state_.t = model.getTimepoint(it - 1);

    state_.state = model.getModelState();
    flagUpdatedState();
}

bool SteadystateProblem::initializeBackwardProblem(
    Solver const& solver, Model& model, BackwardProblem const* bwd
) {
    newton_solver_->reinitialize();
    /* note that state_ is still set from forward run */
    if (bwd) {
        /* preequilibration */
        if (solver.getSensitivityMethodPreequilibration()
            != SensitivityMethod::adjoint)
            return false; /* if not adjoint mode, there's nothing to do */

        /* If we need to reinitialize solver states, this won't work yet. */
        if (model.nx_reinit() > 0)
            throw NewtonFailure(
                AMICI_NOT_IMPLEMENTED,
                "Adjoint preequilibration with reinitialization of "
                "non-constant states is not yet implemented. Stopping."
            );

        solver.reInit(state_.t, state_.x, state_.dx);
        solver.updateAndReinitStatesAndSensitivities(&model);
        xB_.copy(bwd->getAdjointState());
    }
    /* postequilibration does not need a reInit */

    /* initialize quadratures */
    xQ_.zero();
    xQB_.zero();
    xQBdot_.zero();

    return true;
}

void SteadystateProblem::computeSteadyStateQuadrature(
    Solver const& solver, Model& model
) {
    /* This routine computes the quadratures:
         xQB = Integral[ xB(x(t), t, p) * dxdot/dp(x(t), t, p) | dt ]
     As we're in steady state, we have x(t) = x_ss (x_steadystate), hence
         xQB = Integral[ xB(x_ss, t, p) | dt ] * dxdot/dp(x_ss, t, p)
     We therefore compute the integral over xB first and then do a
     matrix-vector multiplication */

    auto sensitivityMode = model.getSteadyStateSensitivityMode();

    /* Try to compute the analytical solution for quadrature algebraically */
    if (sensitivityMode == SteadyStateSensitivityMode::newtonOnly
        || sensitivityMode
               == SteadyStateSensitivityMode::integrateIfNewtonFails)
        getQuadratureByLinSolve(model);

    /* Perform simulation */
    if (sensitivityMode == SteadyStateSensitivityMode::integrationOnly
        || (sensitivityMode
                == SteadyStateSensitivityMode::integrateIfNewtonFails
            && !hasQuadrature()))
        getQuadratureBySimulation(solver, model);

    /* If analytic solution and integration did not work, throw an Exception */
    if (!hasQuadrature())
        throw AmiException(
            "Steady state backward computation failed: Linear "
            "system could not be solved (possibly due to singular Jacobian), "
            "and numerical integration did not equilibrate within maxsteps"
        );
}

void SteadystateProblem::getQuadratureByLinSolve(Model& model) {
    /* Computes the integral over the adjoint state xB:
     If the Jacobian has full rank, this has an analytical solution, since
     d/dt[ xB(t) ] = JB^T(x(t), p) xB(t) = JB^T(x_ss, p) xB(t)
     This linear ODE system with time-constant matrix has the solution
     xB(t) = exp( t * JB^T(x_ss, p) ) * xB(0)
     This integral xQ over xB is given as the solution of
     JB^T(x_ss, p) * xQ = xB(0)
     So we first try to solve the linear system, if possible. */

    /* copy content of xB into vector with integral */
    xQ_.copy(xB_);

    /* try to solve the linear system */
    try {
        /* compute integral over xB and write to xQ */
        newton_solver_->prepareLinearSystemB(model, state_);
        newton_solver_->solveLinearSystem(xQ_);
        /* Compute the quadrature as the inner product xQ * dxdotdp */
        computeQBfromQ(model, xQ_, xQB_);
        /* set flag that quadratures is available (for processing in rdata) */
        hasQuadrature_ = true;

        /* Finalize by setting adjoint state to zero (its steady state) */
        xB_.zero();
    } catch (NewtonFailure const&) {
        hasQuadrature_ = false;
    }
}

void SteadystateProblem::getQuadratureBySimulation(
    Solver const& solver, Model& model
) {
    /* If the Jacobian is singular, the integral over xB must be computed
       by usual integration over time, but  simplifications can be applied:
       x is not time dependent, no forward trajectory is needed. */

    /* set starting timepoint for the simulation solver */
    state_.t = model.t0();
    /* xQ was written in getQuadratureByLinSolve() -> set to zero */
    xQ_.zero();

    /* create a new solver object */
    auto simSolver = createSteadystateSimSolver(solver, model, false, true);

    /* perform integration and quadrature */
    try {
        runSteadystateSimulation(*simSolver, model, true);
        hasQuadrature_ = true;
    } catch (NewtonFailure const&) {
        hasQuadrature_ = false;
    }
}

[[noreturn]] void SteadystateProblem::handleSteadyStateFailure(
    bool tried_newton_1, bool tried_simulation, bool tried_newton_2
) {
    /* Throw error message according to error codes */
    std::string errorString = "Steady state computation failed.";
    if (tried_newton_1) {
        errorString.append(" First run of Newton solver failed");
        writeErrorString(&errorString, steady_state_status_[0]);
    }
    if (tried_simulation) {
        errorString.append(" Simulation to steady state failed");
        writeErrorString(&errorString, steady_state_status_[1]);
    }
    if (tried_newton_2) {
        errorString.append(" Second run of Newton solver failed");
        writeErrorString(&errorString, steady_state_status_[2]);
    }
    throw AmiException(errorString.c_str());
}

void SteadystateProblem::writeErrorString(
    std::string* errorString, SteadyStateStatus status
) const {
    /* write error message according to steady state status */
    switch (status) {
    case SteadyStateStatus::failed_too_long_simulation:
        (*errorString).append(": System could not be equilibrated.");
        break;
    case SteadyStateStatus::failed_damping:
        (*errorString).append(": Damping factor reached lower bound.");
        break;
    case SteadyStateStatus::failed_factorization:
        (*errorString).append(": RHS could not be factorized.");
        break;
    case SteadyStateStatus::failed_convergence:
        (*errorString).append(": No convergence was achieved.");
        break;
    default:
        (*errorString).append(".");
        break;
    }
}

bool SteadystateProblem::getSensitivityFlag(
    Model const& model, Solver const& solver, int it, SteadyStateContext context
) {
    /* We need to check whether we need to compute forward sensitivities.
       Depending on the situation (pre-/postequilibration) and the solver
       settings, the logic may be involved and is handled here.
       Most of these boolean operation could be simplified. However,
       clarity is more important than brevity. */

    /* Are we running in preequilibration (and hence create)? */
    bool preequilibration = (it == -1);

    /* Have we maybe already computed forward sensitivities? */
    bool forwardSensisAlreadyComputed
        = solver.getSensitivityOrder() >= SensitivityOrder::first
          && steady_state_status_[1] == SteadyStateStatus::success
          && (model.getSteadyStateSensitivityMode()
                  == SteadyStateSensitivityMode::integrationOnly
              || model.getSteadyStateSensitivityMode()
                     == SteadyStateSensitivityMode::integrateIfNewtonFails);

    bool simulationStartedInSteadystate
        = steady_state_status_[0] == SteadyStateStatus::success
          && numsteps_[0] == 0;

    /* Do we need forward sensis for postequilibration? */
    bool needForwardSensisPosteq
        = !preequilibration && !forwardSensisAlreadyComputed
          && solver.getSensitivityOrder() >= SensitivityOrder::first
          && solver.getSensitivityMethod() == SensitivityMethod::forward;

    /* Do we need forward sensis for preequilibration? */
    bool needForwardSensisPreeq
        = preequilibration && !forwardSensisAlreadyComputed
          && solver.getSensitivityMethodPreequilibration()
                 == SensitivityMethod::forward
          && solver.getSensitivityOrder() >= SensitivityOrder::first;

    /* Do we need to do the linear system solve to get forward sensitivities? */
    bool needForwardSensisNewton
        = (needForwardSensisPreeq || needForwardSensisPosteq)
          && !simulationStartedInSteadystate;

    /* When we're creating a new solver object */
    bool needForwardSensiAtCreation
        = needForwardSensisPreeq
          && (model.getSteadyStateSensitivityMode()
                  == SteadyStateSensitivityMode::integrationOnly
              || model.getSteadyStateSensitivityMode()
                     == SteadyStateSensitivityMode::integrateIfNewtonFails);

    /* Check if we need to store sensis */
    switch (context) {
    case SteadyStateContext::newtonSensi:
        return needForwardSensisNewton;

    case SteadyStateContext::sensiStorage:
        return needForwardSensisNewton || forwardSensisAlreadyComputed
               || simulationStartedInSteadystate;

    case SteadyStateContext::solverCreation:
        return needForwardSensiAtCreation;

    default:
        throw AmiException("Requested invalid context in sensitivity "
                           "processing during steady state computation");
    }
}

realtype SteadystateProblem::getWrmsNorm(
    AmiVector const& x, AmiVector const& xdot, realtype atol, realtype rtol,
    AmiVector& ewt
) const {
    /* Depending on what convergence we want to check (xdot, sxdot, xQBdot)
       we need to pass ewt[QB], as xdot and xQBdot have different sizes */
    /* ewt = x */
    N_VAbs(const_cast<N_Vector>(x.getNVector()), ewt.getNVector());
    /* ewt *= rtol */
    N_VScale(rtol, ewt.getNVector(), ewt.getNVector());
    /* ewt += atol */
    N_VAddConst(ewt.getNVector(), atol, ewt.getNVector());
    /* ewt = 1/ewt (ewt = 1/(rtol*x+atol)) */
    N_VInv(ewt.getNVector(), ewt.getNVector());
    /* wrms = sqrt(sum((xdot/ewt)**2)/n) where n = size of state vector */
    return N_VWrmsNorm(
        const_cast<N_Vector>(xdot.getNVector()), ewt.getNVector()
    );
}

realtype
SteadystateProblem::getWrms(Model& model, SensitivityMethod sensi_method) {
    realtype wrms = INFINITY;
    if (sensi_method == SensitivityMethod::adjoint) {
        /* In the adjoint case, only xQB contributes to the gradient, the exact
           steadystate is less important, as xB = xQdot may even not converge
           to zero at all. So we need xQBdot, hence compute xQB first. */
        computeQBfromQ(model, xQ_, xQB_);
        computeQBfromQ(model, xB_, xQBdot_);
        if (newton_step_conv_)
            throw NewtonFailure(
                AMICI_NOT_IMPLEMENTED,
                "Newton type convergence check is not implemented for adjoint "
                "steady state computations. Stopping."
            );
        wrms = getWrmsNorm(xQB_, xQBdot_, atol_quad_, rtol_quad_, ewtQB_);
    } else {
        /* If we're doing a forward simulation (with or without sensitivities:
           Get RHS and compute weighted error norm */
        if (newton_step_conv_)
            getNewtonStep(model);
        else
            updateRightHandSide(model);
        wrms = getWrmsNorm(
            state_.x, newton_step_conv_ ? delta_ : xdot_, atol_, rtol_, ewt_
        );
    }
    return wrms;
}

realtype SteadystateProblem::getWrmsFSA(Model& model) {
    /* Forward sensitivities: Compute weighted error norm for their RHS */
    realtype wrms = 0.0;

    /* we don't need to call prepareLinearSystem in this function, since it was
     already computed in the preceding getWrms call and both equations have the
     same jacobian */

    xdot_updated_ = false;
    for (int ip = 0; ip < model.nplist(); ++ip) {
        model.fsxdot(
            state_.t, state_.x, state_.dx, ip, state_.sx[ip], state_.dx, xdot_
        );
        if (newton_step_conv_)
            newton_solver_->solveLinearSystem(xdot_);
        wrms
            = getWrmsNorm(state_.sx[ip], xdot_, atol_sensi_, rtol_sensi_, ewt_);
        /* ideally this function would report the maximum of all wrms over
         all ip, but for practical purposes we can just report the wrms for
         the first ip where we know that the convergence threshold is not
         satisfied. */
        if (wrms > conv_thresh)
            break;
    }
    /* just report the parameter for the last ip, value doesn't matter it's
     only important that all of them satisfy the convergence threshold */
    return wrms;
}

bool SteadystateProblem::checkSteadyStateSuccess() const {
    /* Did one of the attempts yield s steady state? */
    return std::any_of(
        steady_state_status_.begin(), steady_state_status_.end(),
        [](SteadyStateStatus status) {
            return status == SteadyStateStatus::success;
        }
    );
}

void SteadystateProblem::applyNewtonsMethod(Model& model, bool newton_retry) {
    int& i_newtonstep = numsteps_.at(newton_retry ? 2 : 0);
    i_newtonstep = 0;
    gamma_ = 1.0;

    if (model.nx_solver == 0)
        return;

    /* initialize output of linear solver for Newton step */
    delta_.zero();
    x_old_.copy(state_.x);
    bool converged = false;
    wrms_ = getWrms(model, SensitivityMethod::none);
    converged = newton_retry ? false : wrms_ < conv_thresh;
    bool update_direction = true;

    while (!converged && i_newtonstep < max_steps_) {

        /* If Newton steps are necessary, compute the initial search
         direction */
        if (update_direction) {
            getNewtonStep(model);
            /* we store delta_ here as later convergence checks may
             update it */
            delta_old_.copy(delta_);
        }

        /* Try step with new gamma_/delta_ */
        linearSum(
            1.0, x_old_, gamma_, update_direction ? delta_ : delta_old_,
            state_.x
        );
        flagUpdatedState();

        /* Compute new xdot and residuals */
        realtype wrms_tmp = getWrms(model, SensitivityMethod::none);

        bool step_successful = wrms_tmp < wrms_;
        if (step_successful) {
            /* If new residuals are smaller than old ones, update state */
            wrms_ = wrms_tmp;
            /* precheck convergence */
            converged = wrms_ < conv_thresh;
            if (converged) {
                converged = makePositiveAndCheckConvergence(model);
            }
            /* update x_old_ _after_ positivity was enforced */
            x_old_.copy(state_.x);
        }

        update_direction = updateDampingFactor(step_successful);
        /* increase step counter */
        i_newtonstep++;
    }

    if (!converged)
        throw NewtonFailure(AMICI_TOO_MUCH_WORK, "applyNewtonsMethod");
}

bool SteadystateProblem::makePositiveAndCheckConvergence(Model& model) {
    /* Ensure positivity of the found state and recheck if
       the convergence still holds */
    auto nonnegative = model.getStateIsNonNegative();
    for (int ix = 0; ix < model.nx_solver; ix++) {
        if (state_.x[ix] < 0.0 && nonnegative[ix]) {
            state_.x[ix] = 0.0;
            flagUpdatedState();
        }
    }
    wrms_ = getWrms(model, SensitivityMethod::none);
    return wrms_ < conv_thresh;
}

bool SteadystateProblem::updateDampingFactor(bool step_successful) {
    if (damping_factor_mode_ != NewtonDampingFactorMode::on)
        return true;

    if (step_successful)
        gamma_ = fmin(1.0, 2.0 * gamma_);
    else
        gamma_ = gamma_ / 4.0;

    if (gamma_ < damping_factor_lower_bound_)
        throw NewtonFailure(
            AMICI_DAMPING_FACTOR_ERROR,
            "Newton solver failed: the damping factor "
            "reached its lower bound"
        );
    return step_successful;
}

void SteadystateProblem::runSteadystateSimulation(
    Solver const& solver, Model& model, bool backward
) {
    if (model.nx_solver == 0)
        return;
    /* Loop over steps and check for convergence
       NB: This function is used for forward and backward simulation, and may
       be called by workSteadyStateProblem and workSteadyStateBackwardProblem.
       Whether we simulate forward or backward in time is reflected by the
       flag "backward". */

    /* Do we also have to check for convergence of sensitivities? */
    SensitivityMethod sensitivityFlag = SensitivityMethod::none;
    if (solver.getSensitivityOrder() > SensitivityOrder::none
        && solver.getSensitivityMethod() == SensitivityMethod::forward)
        sensitivityFlag = SensitivityMethod::forward;
    /* If flag for forward sensitivity computation by simulation is not set,
     disable forward sensitivity integration. Sensitivities will be computed
     by newtonsolver.computeNewtonSensis then */
    if (model.getSteadyStateSensitivityMode()
        == SteadyStateSensitivityMode::newtonOnly) {
        solver.switchForwardSensisOff();
        sensitivityFlag = SensitivityMethod::none;
    }
    if (backward)
        sensitivityFlag = SensitivityMethod::adjoint;

    int& sim_steps = backward ? numstepsB_ : numsteps_.at(1);

    int convergence_check_frequency = 1;

    if (newton_step_conv_)
        convergence_check_frequency = 25;

    while (true) {
        if (sim_steps % convergence_check_frequency == 0) {
            // Check for convergence (already before simulation, since we might
            // start in steady state)
            wrms_ = getWrms(model, sensitivityFlag);
            if (wrms_ < conv_thresh) {
                if (check_sensi_conv_
                    && sensitivityFlag == SensitivityMethod::forward) {
                    updateSensiSimulation(solver);
                    // getWrms needs to be called before getWrmsFSA
                    // such that the linear system is prepared for newton-type
                    // convergence check
                    if (getWrmsFSA(model) < conv_thresh)
                        break; // converged
                } else {
                    break; // converged
                }
            }
        }

        /* check for maxsteps  */
        if (sim_steps >= solver.getMaxSteps()) {
            throw IntegrationFailure(AMICI_TOO_MUCH_WORK, state_.t);
        }

        /* increase counter */
        sim_steps++;
        /* One step of ODE integration
         reason for tout specification:
         max with 1 ensures correct direction (any positive value would do)
         multiplication with 10 ensures nonzero difference and should ensure
         stable computation value is not important for AMICI_ONE_STEP mode,
         only direction w.r.t. current t
         */
        solver.step(std::max(state_.t, 1.0) * 10);

        if (backward) {
            solver.writeSolution(&state_.t, xB_, state_.dx, state_.sx, xQ_);
        } else {
            solver.writeSolution(
                &state_.t, state_.x, state_.dx, state_.sx, xQ_
            );
            flagUpdatedState();
        }
    }

    // if check_sensi_conv_ is deactivated, we still have to update sensis
    if (sensitivityFlag == SensitivityMethod::forward)
        updateSensiSimulation(solver);
}

std::unique_ptr<Solver> SteadystateProblem::createSteadystateSimSolver(
    Solver const& solver, Model& model, bool forwardSensis, bool backward
) const {
    /* Create new CVode solver object */
    auto sim_solver = std::unique_ptr<Solver>(solver.clone());

    sim_solver->logger = solver.logger;

    switch (solver.getLinearSolver()) {
    case LinearSolver::dense:
        break;
    case LinearSolver::KLU:
        break;
    default:
        throw AmiException("invalid solver for steadystate simulation");
    }
    /* do we need sensitivities? */
    if (forwardSensis) {
        /* need forward to compute sx0 */
        sim_solver->setSensitivityMethod(SensitivityMethod::forward);
    } else {
        sim_solver->setSensitivityMethod(SensitivityMethod::none);
        sim_solver->setSensitivityOrder(SensitivityOrder::none);
    }
    /* use x and sx as dummies for dx and sdx
     (they wont get touched in a CVodeSolver) */
    sim_solver->setup(model.t0(), &model, state_.x, state_.dx, state_.sx, sdx_);
    if (backward) {
        sim_solver->setup(model.t0(), &model, xB_, xB_, state_.sx, sdx_);
        sim_solver->setupSteadystate(
            model.t0(), &model, state_.x, state_.dx, xB_, xB_, xQ_
        );
    } else {
        sim_solver->setup(
            model.t0(), &model, state_.x, state_.dx, state_.sx, sdx_
        );
    }

    return sim_solver;
}

void SteadystateProblem::computeQBfromQ(
    Model& model, AmiVector const& yQ, AmiVector& yQB
) const {
    /* Compute the quadrature as the inner product: yQB = dxdotdp * yQ */

    /* set to zero first, as multiplication adds to existing value */
    yQB.zero();
    /* multiply */
    if (model.pythonGenerated) {
        /* fill dxdotdp with current values */
        auto const& plist = model.getParameterList();
        model.fdxdotdp(state_.t, state_.x, state_.dx);
        model.get_dxdotdp_full().multiply(
            yQB.getNVector(), yQ.getNVector(), plist, true
        );
    } else {
        for (int ip = 0; ip < model.nplist(); ++ip)
            yQB[ip] = dotProd(yQ, model.get_dxdotdp()[ip]);
    }
}

void SteadystateProblem::getAdjointUpdates(Model& model, ExpData const& edata) {
    xB_.zero();
    for (int it = 0; it < model.nt(); it++) {
        if (std::isinf(model.getTimepoint(it))) {
            model.getAdjointStateObservableUpdate(
                slice(dJydx_, it, model.nx_solver * model.nJ), it, state_.x,
                edata
            );
            for (int ix = 0; ix < model.nxtrue_solver; ix++)
                xB_[ix] += dJydx_[ix + it * model.nx_solver];
        }
    }
}

void SteadystateProblem::flagUpdatedState() {
    xdot_updated_ = false;
    delta_updated_ = false;
    sensis_updated_ = false;
}

void SteadystateProblem::updateSensiSimulation(Solver const& solver) {
    if (sensis_updated_)
        return;
    state_.sx = solver.getStateSensitivity(state_.t);
    sensis_updated_ = true;
}

void SteadystateProblem::updateRightHandSide(Model& model) {
    if (xdot_updated_)
        return;
    model.fxdot(state_.t, state_.x, state_.dx, xdot_);
    xdot_updated_ = true;
}

void SteadystateProblem::getNewtonStep(Model& model) {
    if (delta_updated_)
        return;
    updateRightHandSide(model);
    delta_.copy(xdot_);
    newton_solver_->getStep(delta_, model, state_);
    delta_updated_ = true;
}
} // namespace amici
