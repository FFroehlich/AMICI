/**
 * @file   amici.cpp
 * @brief  core routines for integration
 */

#include "amici/amici.h"

#include "amici/backwardproblem.h"
#include "amici/forwardproblem.h"
#include "amici/logging.h"
#include "amici/steadystateproblem.h"

#include <sundials/sundials_types.h> //sunrealtype

#include <map>
#include <memory>
#include <type_traits>

static_assert(
    amici::AMICI_SINGULAR_JACOBIAN == SUN_ERR_EXT_FAIL,
    "AMICI_SINGULAR_JACOBIAN != SUN_ERR_EXT_FAIL"
);
static_assert(
    std::is_same<amici::realtype, sunrealtype>::value,
    "Definition of realtype does not match"
);

namespace amici {

std::map<int, std::string> simulation_status_to_str_map = {
    {AMICI_RECOVERABLE_ERROR, "AMICI_RECOVERABLE_ERROR"},
    {AMICI_UNRECOVERABLE_ERROR, "AMICI_UNRECOVERABLE_ERROR"},
    {AMICI_TOO_MUCH_WORK, "AMICI_TOO_MUCH_WORK"},
    {AMICI_TOO_MUCH_ACC, "AMICI_TOO_MUCH_ACC"},
    {AMICI_ERR_FAILURE, "AMICI_ERR_FAILURE"},
    {AMICI_CONV_FAILURE, "AMICI_CONV_FAILURE"},
    {AMICI_FIRST_RHSFUNC_ERR, "AMICI_FIRST_RHSFUNC_ERR"},
    {AMICI_CONSTR_FAIL, "AMICI_CONSTR_FAIL"},
    {AMICI_CVODES_CONSTR_FAIL, "AMICI_CVODES_CONSTR_FAIL"},
    {AMICI_IDAS_CONSTR_FAIL, "AMICI_IDAS_CONSTR_FAIL"},
    {AMICI_RHSFUNC_FAIL, "AMICI_RHSFUNC_FAIL"},
    {AMICI_ILL_INPUT, "AMICI_ILL_INPUT"},
    {AMICI_ERROR, "AMICI_ERROR"},
    {AMICI_NO_STEADY_STATE, "AMICI_NO_STEADY_STATE"},
    {AMICI_DAMPING_FACTOR_ERROR, "AMICI_DAMPING_FACTOR_ERROR"},
    {AMICI_SINGULAR_JACOBIAN, "AMICI_SINGULAR_JACOBIAN"},
    {AMICI_NOT_IMPLEMENTED, "AMICI_NOT_IMPLEMENTED"},
    {AMICI_MAX_TIME_EXCEEDED, "AMICI_MAX_TIME_EXCEEDED"},
    {AMICI_SUCCESS, "AMICI_SUCCESS"},
    {AMICI_NOT_RUN, "AMICI_NOT_RUN"},
    {AMICI_LSETUP_FAIL, "AMICI_LSETUP_FAIL"},
    {AMICI_FIRST_QRHSFUNC_ERR, "AMICI_FIRST_QRHSFUNC_ERR"},
    {AMICI_WARNING, "AMICI_WARNING"},
    {AMICI_BAD_T, "AMICI_BAD_T"},
    {AMICI_BAD_DKY, "AMICI_BAD_DKY"},
    {AMICI_FIRST_SRHSFUNC_ERR, "AMICI_FIRST_SRHSFUNC_ERR"},
    {AMICI_SRHSFUNC_FAIL, "AMICI_SRHSFUNC_FAIL"},
    {AMICI_REPTD_SRHSFUNC_ERR, "AMICI_REPTD_SRHSFUNC_ERR"},
    {AMICI_UNREC_SRHSFUNC_ERR, "AMICI_UNREC_SRHSFUNC_ERR"},
};

std::unique_ptr<ReturnData> runAmiciSimulation(
    Solver& solver, ExpData const* edata, Model& model, bool rethrow
) {
    // create a temporary logger instance for Solver and Model to capture
    // messages from only this simulation
    Logger logger;
    solver.logger = &logger;
    model.logger = &logger;
    // prevent dangling pointer
    auto _ = gsl::finally([&solver, &model] {
        solver.logger = model.logger = nullptr;
    });

    CpuTimer cpu_timer;
    solver.startTimer();

    /* Applies condition-specific model settings and restores them when going
     * out of scope */
    ConditionContext cc1(&model, edata, FixedParameterContext::simulation);

    std::unique_ptr<ReturnData> rdata
        = std::make_unique<ReturnData>(solver, model);
    if (edata) {
        rdata->id = edata->id;
    }

    std::unique_ptr<SteadystateProblem> preeq{};
    std::unique_ptr<ForwardProblem> fwd{};
    std::unique_ptr<BackwardProblem> bwd{};
    std::unique_ptr<SteadystateProblem> posteq{};
    // tracks whether backwards integration finished without exceptions
    bool bwd_success = true;

    try {
        if (edata && !edata->fixedParametersPreequilibration.empty()) {
            ConditionContext cc2(
                &model, edata, FixedParameterContext::preequilibration
            );

            preeq = std::make_unique<SteadystateProblem>(solver, model);
            preeq->workSteadyStateProblem(solver, model, -1);
        }

        fwd = std::make_unique<ForwardProblem>(
            edata, &model, &solver, preeq.get()
        );
        fwd->workForwardProblem();

        if (fwd->getCurrentTimeIteration() < model.nt()) {
            posteq = std::make_unique<SteadystateProblem>(solver, model);
            posteq->workSteadyStateProblem(
                solver, model, fwd->getCurrentTimeIteration()
            );
        }

        if (edata && solver.computingASA()) {
            fwd->getAdjointUpdates(model, *edata);
            if (posteq) {
                posteq->getAdjointUpdates(model, *edata);
                posteq->workSteadyStateBackwardProblem(
                    solver, model, bwd.get()
                );
            }

            bwd_success = false;

            bwd = std::make_unique<BackwardProblem>(*fwd, posteq.get());
            bwd->workBackwardProblem();

            bwd_success = true;

            if (preeq) {
                ConditionContext cc2(
                    &model, edata, FixedParameterContext::preequilibration
                );
                preeq->workSteadyStateBackwardProblem(solver, model, bwd.get());
            }
        }

        rdata->status = AMICI_SUCCESS;

    } catch (amici::IntegrationFailure const& ex) {
        if (ex.error_code == AMICI_RHSFUNC_FAIL && solver.timeExceeded()) {
            rdata->status = AMICI_MAX_TIME_EXCEEDED;
            if (rethrow)
                throw;
            logger.log(
                LogSeverity::error, "MAXTIME_EXCEEDED",
                "AMICI forward simulation failed at t = %g: "
                "Maximum time exceeded in forward solve.",
                ex.time
            );
        } else {
            rdata->status = ex.error_code;
            if (rethrow)
                throw;
            logger.log(
                LogSeverity::error, "FORWARD_FAILURE",
                "AMICI forward simulation failed at t = %g: %s", ex.time,
                ex.what()
            );
        }
    } catch (amici::IntegrationFailureB const& ex) {
        if (ex.error_code == AMICI_RHSFUNC_FAIL && solver.timeExceeded()) {
            rdata->status = AMICI_MAX_TIME_EXCEEDED;
            if (rethrow)
                throw;
            logger.log(
                LogSeverity::error, "MAXTIME_EXCEEDED",
                "AMICI backward simulation failed when trying to solve until "
                "t = %g: Maximum time exceeded in backward solve.",
                ex.time
            );

        } else {
            rdata->status = ex.error_code;
            if (rethrow)
                throw;
            logger.log(
                LogSeverity::error, "BACKWARD_FAILURE",
                "AMICI backward simulation failed when trying to solve until t "
                "= %g"
                " (check debug logs for details): %s",
                ex.time, ex.what()
            );
        }
    } catch (amici::AmiException const& ex) {
        rdata->status = AMICI_ERROR;
        if (rethrow)
            throw;
        logger.log(
            LogSeverity::error, "OTHER", "AMICI simulation failed: %s",
            ex.what()
        );
#ifndef NDEBUG
        logger.log(
            LogSeverity::debug, "BACKTRACE",
            "The previous error occurred at:\n%s", ex.getBacktrace()
        );
#endif
    } catch (std::exception const& ex) {
        rdata->status = AMICI_ERROR;
        if (rethrow)
            throw;
        logger.log(
            LogSeverity::error, "OTHER", "AMICI simulation failed: %s",
            ex.what()
        );
    }

    try {
        rdata->processSimulationObjects(
            preeq.get(), fwd.get(), bwd_success ? bwd.get() : nullptr,
            posteq.get(), model, solver, edata
        );
    } catch (std::exception const& ex) {
        rdata->status = AMICI_ERROR;
        if (rethrow)
            throw;
        logger.log(
            LogSeverity::error, "OTHER", "AMICI simulation failed: %s",
            ex.what()
        );
    }

    rdata->t_last = solver.gett();

    rdata->cpu_time_total = cpu_timer.elapsed_milliseconds();

    // verify that reported CPU times are plausible
    gsl_EnsuresDebug(rdata->cpu_time <= rdata->cpu_time_total);
    gsl_EnsuresDebug(rdata->cpu_timeB <= rdata->cpu_time_total);
    gsl_EnsuresDebug(rdata->preeq_cpu_time <= rdata->cpu_time_total);
    gsl_EnsuresDebug(rdata->preeq_cpu_timeB <= rdata->cpu_time_total);
    gsl_EnsuresDebug(rdata->posteq_cpu_time <= rdata->cpu_time_total);
    gsl_EnsuresDebug(rdata->posteq_cpu_timeB <= rdata->cpu_time_total);
    if (!posteq)
        gsl_EnsuresDebug(
            std::is_sorted(rdata->numsteps.begin(), rdata->numsteps.end())
            || rdata->status != AMICI_SUCCESS
        );
    if (!preeq)
        gsl_EnsuresDebug(
            std::is_sorted(rdata->numstepsB.begin(), rdata->numstepsB.end())
            || rdata->status != AMICI_SUCCESS
        );
    rdata->messages = logger.items;
    return rdata;
}

std::vector<std::unique_ptr<ReturnData>> runAmiciSimulations(
    Solver const& solver, std::vector<ExpData*> const& edatas,
    Model const& model, bool failfast,
#if defined(_OPENMP)
    int num_threads
#else
    int /* num_threads */
#endif

) {
    std::vector<std::unique_ptr<ReturnData>> results(edatas.size());
    // is set to true if one simulation fails and we should skip the rest.
    // shared across threads.
    bool skipThrough = false;

#if defined(_OPENMP)
#pragma omp parallel for num_threads(num_threads)
#endif
    for (int i = 0; i < (int)edatas.size(); ++i) {
        // must catch exceptions in parallel section to avoid termination
        try {
            auto mySolver = std::unique_ptr<Solver>(solver.clone());
            auto myModel = std::unique_ptr<Model>(model.clone());

            /* if we fail we need to write empty return datas for the python
             interface */
            if (skipThrough) {
                ConditionContext conditionContext(myModel.get(), edatas[i]);
                results[i]
                    = std::unique_ptr<ReturnData>(new ReturnData(solver, model)
                    );
            } else {
                results[i] = runAmiciSimulation(*mySolver, edatas[i], *myModel);
            }
        } catch (std::exception const& ex) {
            results[i]
                = std::unique_ptr<ReturnData>(new ReturnData(solver, model));
            results[i]->status = AMICI_ERROR;
            results[i]->messages.push_back(
                LogItem(LogSeverity::error, "OTHER", ex.what())
            );
        }

        skipThrough |= failfast && results[i]->status < 0;
    }

    return results;
}


std::unique_ptr<ReturnData> runAmiciSimulationPeriod(
    Solver& solver, ExpData& const edata, Model& model,
    &std::map(std::string, ForwardProblem) fwds,
    &std::map(std::string, SteadystateProblem) peqs,
    bool rethrow
) {

    // create a temporary logger instance for Solver and Model to capture
    // messages from only this simulation
    Logger logger;
    solver.logger = &logger;
    model.logger = &logger;
    // prevent dangling pointer
    auto _ = gsl::finally([&solver, &model] {
        solver.logger = model.logger = nullptr;
    });

    CpuTimer cpu_timer;
    solver.startTimer();

    /* Applies condition-specific model settings and restores them when going
     * out of scope */
    ConditionContext cc(&model, edata, FixedParameterContext::simulation);

    std::unique_ptr<ReturnData> rdata
        = std::make_unique<ReturnData>(solver, model);
    rdata->id = edata->id;

    try {
        auto peq = peqs.find(edata.id);
        auto fwd = fwds.find(edata.id);
        // simulate
        if (peq) {
            // steadystate
            peq.workSteadyStateProblem(solver, model, -1);
        } else {
            // forward
            fwd.workForwardProblem();
        }

        rdata->status = AMICI_SUCCESS;
    } catch (amici::IntegrationFailure const& ex) {
        if (ex.error_code == AMICI_RHSFUNC_FAIL && solver.timeExceeded()) {
            rdata->status = AMICI_MAX_TIME_EXCEEDED;
            if (rethrow)
                throw;
            logger.log(
                LogSeverity::error, "MAXTIME_EXCEEDED",
                "AMICI forward simulation failed at t = %g: "
                "Maximum time exceeded in forward solve.",
                ex.time
            );
        } else {
            rdata->status = ex.error_code;
            if (rethrow)
                throw;
            logger.log(
                LogSeverity::error, "FORWARD_FAILURE",
                "AMICI forward simulation failed at t = %g: %s", ex.time,
                ex.what()
            );
        }
    } catch (amici::AmiException const& ex) {
        rdata->status = AMICI_ERROR;
        if (rethrow)
            throw;
        logger.log(
            LogSeverity::error, "OTHER", "AMICI simulation failed: %s",
            ex.what()
        );
#ifndef NDEBUG
        logger.log(
            LogSeverity::debug, "BACKTRACE",
            "The previous error occurred at:\n%s", ex.getBacktrace()
        );
#endif
    } catch (std::exception const& ex) {
        rdata->status = AMICI_ERROR;
        if (rethrow)
            throw;
        logger.log(
            LogSeverity::error, "OTHER", "AMICI simulation failed: %s",
            ex.what()
        );
    }

    rdata->cpu_time = cpu_timer.elapsed_milliseconds();
    rdata->cpu_time_total = rdata->cpu_time;

    rdata->messages = logger.items;
    return rdata;
}

std::unique_ptr<ReturnData> runAmiciSimulationPeriodB(
    Solver& solver, ExpData& const edata, Model& model,
    const &std::map(std::string, ForwardProblem) fwds,
    const &std::map(std::string, SteadystateProblem) peqs,
    &std::map(std::string, BackwardProblem) bwds,
    bool rethrow
) {
    Logger logger;
    solver.logger = &logger;
    model.logger = &logger;
    // prevent dangling pointer
    auto _ = gsl::finally([&solver, &model] {
        solver.logger = model.logger = nullptr;
    });

    CpuTimer cpu_timerB;
    solver.startTimer();

    /* Applies condition-specific model settings and restores them when going
     * out of scope */
    ConditionContext cc(&model, edata, FixedParameterContext::simulation);

    bool bwd_success = false;

    try {
        if (solver.computingASA()) {
            bwd = std::make_unique<BackwardProblem>(*fwd, posteq.get());
            auto peq = peqs.find(edata.id);
            auto fwd = fwds.find(edata.id);

            if (peq != peqs.end()) {
                // initialize from preequilibration
                peq->getAdjointUpdates(model, *edata);
                peq->workSteadyStateBackwardProblem(
                    solver, model, bwd.get()
                );

            } else (fwd != fwds.end()) {
                // initialize from simulation
                fwd->getAdjointUpdates(model, *edata);
                bwd->workBackwardProblem();
            }
        }
        bwd_success = true;
    } catch (amici::IntegrationFailureB const& ex) {
        if (ex.error_code == AMICI_RHSFUNC_FAIL && solver.timeExceeded()) {
            rdata->status = AMICI_MAX_TIME_EXCEEDED;
            if (rethrow)
                throw;
            logger.log(
                LogSeverity::error, "MAXTIME_EXCEEDED",
                "AMICI backward simulation failed when trying to solve until "
                "t = %g: Maximum time exceeded in backward solve.",
                ex.time
            );

        } else {
            rdata->status = ex.error_code;
            if (rethrow)
                throw;
            logger.log(
                LogSeverity::error, "BACKWARD_FAILURE",
                "AMICI backward simulation failed when trying to solve until t "
                "= %g"
                " (check debug logs for details): %s",
                ex.time, ex.what()
            );
        }
    } catch (amici::AmiException const& ex) {
        rdata->status = AMICI_ERROR;
        if (rethrow)
            throw;
        logger.log(
            LogSeverity::error, "OTHER", "AMICI simulation failed: %s",
            ex.what()
        );
#ifndef NDEBUG
        logger.log(
            LogSeverity::debug, "BACKTRACE",
            "The previous error occurred at:\n%s", ex.getBacktrace()
        );
#endif
    } catch (std::exception const& ex) {
        rdata->status = AMICI_ERROR;
        if (rethrow)
            throw;
        logger.log(
            LogSeverity::error, "OTHER", "AMICI simulation failed: %s",
            ex.what()
        );
    }

    try {
        rdata->processSimulationObjects(
            nullptr, fwd.get(), bwd_success ? bwd.get() : nullptr,
            nullptr, model, solver, edata
        );
    } catch (std::exception const& ex) {
        rdata->status = AMICI_ERROR;
        if (rethrow)
            throw;
        logger.log(
            LogSeverity::error, "OTHER", "AMICI simulation failed: %s",
            ex.what()
        );
    }

    rdata->cpu_timeB = cpu_timerB.elapsed_milliseconds();
    rdata->cpu_time_total += rdata->cpu_timeB;

    gsl_EnsuresDebug(rdata->cpu_timeB <= rdata->cpu_time_total);

    gsl_EnsuresDebug(
        std::is_sorted(rdata->numstepsB.begin(), rdata->numstepsB.end())
        || rdata->status != AMICI_SUCCESS
    );

    // TODO(ff): append rather than set
    rdata->messages = logger.items;
    return rdata;
}

void initializePeriodFromPreviousPeriod(
    Model& model, std::string initialization_id,
    std::map(std::string, ForwardProblem) fwds,
    std::map(std::string, SteadystateProblem) peqs,
){
    realtype t0;
    amiVector x0;
    amiVector dx0;
    amiVector sx0;
    amiVector sdx0;
    // initialize from initialization_id
    if (edata.initialization_id == "")
        return

    auto peq = peqs.find(edata.initialization_id);
    auto fwd = fwds.find(edata.initialization_id);
    if (peq != peqs.end()) {
        // initialize from preequilibration
        t0 = edata.t_start_;
        x0 = peq.getState();
        dx0 = peq.getStateDerivative();
    } else if (fwd != fwds.end()) {
        // initialize from simulation
        t0 = fwd.getTime();
        x0 = fwd.getState();
        dx0 = fwd.getStateDerivative();
        if solver.computingFSA() {
            sx0 = fwd.getStateSensitivity();
            sdx0 = fwd.getStateDerivativeSensitivity();
        }
    } else {
        throw amici::AmiException("Invalid initialization condition %s in simulation condition %s", edata.initialization_id, edata.id);
    }
    // TODO(ff) do we need a context manager here?
    if (edata.isEquilibration()) {
        // steadystate
        peqs[edata.id] = std::make_unique<SteadystateProblem>(solver, model);
        peqs.initialize(x0, dx0, sx0, sdx0);
    } else {
        // forward
        fwds[edata.id] = std::make_unique<ForwardProblem>(
            edata, &model, &solver, nullptr
        );
        fwd.initialize(x0, dx0, sx0, sdx0);
    }
    // TODO(ff) necessary?
    model.setT0(t0);
}

void initializePeriodFromNextPeriods(
    Model& model, std::string initialization_id,
    const std::vector<ExpData> edatas,
    const std::map(std::string, ForwardProblem) fwds,
    const std::map(std::string, SteadystateProblem) peqs,
) {
    realtype t0;
    amiVector xB0;
    amiVector dxB0;
    amiVector xQB0;
    // initialize from initialization_id
    for (int i = 0; i < (int)edatas.size(); ++i) {
        auto edata = edatas[i];
        if (edata.initialization_id != initialization_id)
            continue;

        auto bwd = bwds.find(edata.initialization_id);
        if (peq != peqs.end()) {
            // initialize from preequilibration
            xB0 += peq.getAdjointState();
            dxB0 += peq.getAdjointStateDerivative();
            xQB0 += peq.getAdjointStateQuadrature();
        } else if (fwd != fwds.end()) {
            // initialize from simulation
            t0 = fwd.getTime();
            x0 = fwd.getState();
            dx0 = fwd.getStateDerivative();
            if solver
                .computingFSA() {
                    sx0 = fwd.getStateSensitivity();
                    sdx0 = fwd.getStateDerivativeSensitivity();
                }
        } else {
            throw amici::AmiException(
                "Invalid initialization condition %s in simulation condition %s",
                edata.initialization_id, edata.id
            );
        }
    }
    // TODO(ff) do we need a context manager here?
    if (edata.isEquilibration()) {
        // steadystate
        peqs[edata.id] = std::make_unique<SteadystateProblem>(solver, model);
        peqs.initialize(x0, dx0, sx0, sdx0);
    } else {
        // forward
        fwds[edata.id] = std::make_unique<ForwardProblem>(
            edata, &model, &solver, nullptr
        );
        fwd.initialize(x0, dx0, sx0, sdx0);
    }
    // TODO(ff) necessary?
    model.setT0(t0);
}

std::vector<std::unique_ptr<ReturnData>> runAmiciSimulationsTimeseries(
    Solver const& solver, std::vector<ExpData*> const& edatas,
    Model const& model, bool failfast
) {
    std::vector<std::unique_ptr<ReturnData>> results(edatas.size());
    // is set to true if one simulation fails and we should skip the rest.
    // shared across threads.

    std::map(std::string, ForwardProblem) fwds(edatas.size());
    std::map(std::string, SteadystateProblem) peqs(edatas.size());
    std::map(std::string, BackwardProblem) bwds(edatas.size());

    bool skipThrough = false;

    // forward pass
    for (int i = 0; i < (int)edatas.size(); ++i) {
        auto edata = edatas[i];
        auto mySolver = std::unique_ptr<Solver>(solver.clone());
        auto myModel = std::unique_ptr<Model>(model.clone());

        if (skipThrough) {
            ConditionContext conditionContext(myModel.get(), edata);
            results[i]
                = std::unique_ptr<ReturnData>(new ReturnData(solver, model));
            continue
        }

        initializeModelFromPreviousPeriod(myModel, edata, fwds, peqs);
        results[i] = runAmiciSimulationPeriod(mySolver, edata, myModel);

        skipThrough |= failfast && results[i]->status < 0;
    }

    // backward pass
    for (int i = ((int)edatas.size()) -1; i >= 0; --i) {
        auto edata = edatas[i];
        auto mySolver = std::unique_ptr<Solver>(solver.clone());
        auto myModel = std::unique_ptr<Model>(model.clone());

        if (skipThrough) {
            ConditionContext conditionContext(myModel.get(), edata);
            results[i]
                = std::unique_ptr<ReturnData>(new ReturnData(solver, model));
            continue
        }

        initializeModelFromPreviousPeriod(myModel, edata, fwds, peqs);
        results[i] = runAmiciSimulationPeriod(mySolver, edata, myModel);

        skipThrough |= failfast && results[i]->status < 0;

    }

    return results;
}

std::string simulation_status_to_str(int status) {
    try {
        return simulation_status_to_str_map.at(status);
    } catch (std::out_of_range const&) {
        // Missing mapping - terminate if this is a debug build,
        // but show the number if non-debug.
        fprintf(stderr, "Unknown simulation status: %d\n", status);
        gsl_ExpectsDebug(false);
        return std::to_string(status);
    }
}

} // namespace amici
