#include "amici/rdata.h"

#include "amici/backwardproblem.h"
#include "amici/edata.h"
#include "amici/forwardproblem.h"
#include "amici/misc.h"
#include "amici/model.h"
#include "amici/solver.h"
#include "amici/steadystateproblem.h"
#include "amici/symbolic_functions.h"

#include <cmath>

namespace amici {

ReturnData::ReturnData(Solver const& solver, Model const& model)
    : ReturnData(
        model.getTimepoints(),
        ModelDimensions(static_cast<ModelDimensions const&>(model)),
        model.nplist(), model.nMaxEvent(), model.nt(),
        solver.getNewtonMaxSteps(), model.getParameterScale(), model.o2mode,
        solver.getSensitivityOrder(), solver.getSensitivityMethod(),
        solver.getReturnDataReportingMode(), model.hasQuadraticLLH(),
        model.getAddSigmaResiduals(), model.getMinimumSigmaResiduals()
    ) {}

ReturnData::ReturnData(
    std::vector<realtype> ts, ModelDimensions const& model_dimensions,
    int nplist, int nmaxevent, int nt, int newton_maxsteps,
    std::vector<ParameterScaling> pscale, SecondOrderMode o2mode,
    SensitivityOrder sensi, SensitivityMethod sensi_meth, RDataReporting rdrm,
    bool quadratic_llh, bool sigma_res, realtype sigma_offset
)
    : ModelDimensions(model_dimensions)
    , ts(std::move(ts))
    , nx(nx_rdata)
    , nxtrue(nxtrue_rdata)
    , nplist(nplist)
    , nmaxevent(nmaxevent)
    , nt(nt)
    , newton_maxsteps(newton_maxsteps)
    , pscale(std::move(pscale))
    , o2mode(o2mode)
    , sensi(sensi)
    , sensi_meth(sensi_meth)
    , rdata_reporting(rdrm)
    , sigma_res(sigma_res)
    , sigma_offset(sigma_offset)
    , x_solver_(nx_solver)
    , sx_solver_(nx_solver, nplist)
    , x_rdata_(nx)
    , sx_rdata_(nx, nplist)
    , nroots_(ne) {
    switch (rdata_reporting) {
    case RDataReporting::full:
        initializeFullReporting(quadratic_llh);
        break;

    case RDataReporting::residuals:
        initializeResidualReporting(quadratic_llh);
        break;

    case RDataReporting::likelihood:
        initializeLikelihoodReporting(quadratic_llh);
        break;
    }
}

void ReturnData::initializeLikelihoodReporting(bool enable_fim) {
    llh = getNaN();
    chi2 = getNaN();
    if (sensi >= SensitivityOrder::first) {
        sllh.resize(nplist, getNaN());
        if (sensi >= SensitivityOrder::second)
            s2llh.resize(nplist * (nJ - 1), getNaN());

        if ((sensi_meth == SensitivityMethod::forward
             || sensi >= SensitivityOrder::second)
            && enable_fim)
            FIM.resize(nplist * nplist, 0.0);
    }
}

void ReturnData::initializeResidualReporting(bool enable_res) {
    y.resize(nt * ny, 0.0);
    sigmay.resize(nt * ny, 0.0);
    if (enable_res)
        res.resize((sigma_res ? 2 : 1) * nt * nytrue, 0.0);

    if ((sensi_meth == SensitivityMethod::forward
         && sensi >= SensitivityOrder::first)
        || sensi >= SensitivityOrder::second) {

        sy.resize(nt * ny * nplist, 0.0);
        ssigmay.resize(nt * ny * nplist, 0.0);
        if (enable_res)
            sres.resize((sigma_res ? 2 : 1) * nt * nytrue * nplist, 0.0);
    }
}

void ReturnData::initializeFullReporting(bool quadratic_llh) {

    initializeLikelihoodReporting(quadratic_llh);
    initializeResidualReporting(quadratic_llh);

    xdot.resize(nx_solver, getNaN());

    J.resize(nx_solver * nx_solver, getNaN());

    // initialize with 0.0, so we only need to write non-zero values
    z.resize(nmaxevent * nz, 0.0);
    sigmaz.resize(nmaxevent * nz, 0.0);

    rz.resize(nmaxevent * nz, 0.0);
    x.resize(nt * nx, 0.0);
    w.resize(nt * nw, 0.0);

    preeq_numsteps.resize(3, 0);
    preeq_status.resize(3, SteadyStateStatus::not_run);
    posteq_numsteps.resize(3, 0);
    posteq_status.resize(3, SteadyStateStatus::not_run);

    if (nt > 0) {
        numsteps.resize(nt, 0);
        numrhsevals.resize(nt, 0);
        numerrtestfails.resize(nt, 0);
        numnonlinsolvconvfails.resize(nt, 0);
        order.resize(nt, 0);

        if (sensi_meth == SensitivityMethod::adjoint
            && sensi >= SensitivityOrder::first) {
            numstepsB.resize(nt, 0);
            numrhsevalsB.resize(nt, 0);
            numerrtestfailsB.resize(nt, 0);
            numnonlinsolvconvfailsB.resize(nt, 0);
        }
    }

    x0.resize(nx, getNaN());
    x_ss.resize(nx, getNaN());

    if (sensi >= SensitivityOrder::first) {
        sx0.resize(nx * nplist, getNaN());
        sx_ss.resize(nx * nplist, getNaN());

        if (sensi_meth == SensitivityMethod::forward
            || sensi >= SensitivityOrder::second) {
            // for second order we can fill in from the augmented states
            sx.resize(nt * nx * nplist, 0.0);
            sz.resize(nmaxevent * nz * nplist, 0.0);
            srz.resize(nmaxevent * nz * nplist, 0.0);
        }

        ssigmaz.resize(nmaxevent * nz * nplist, 0.0);
        if (sensi >= SensitivityOrder::second
            && sensi_meth == SensitivityMethod::forward)
            s2rz.resize(nmaxevent * nztrue * nplist * nplist, 0.0);
    }
}

void ReturnData::processSimulationObjects(
    SteadystateProblem const* preeq, ForwardProblem const* fwd,
    BackwardProblem const* bwd, SteadystateProblem const* posteq, Model& model,
    Solver const& solver, ExpData const* edata
) {
    ModelContext mc(&model);

    processSolver(solver);

    if (preeq)
        processPreEquilibration(*preeq, model);

    if (fwd)
        processForwardProblem(*fwd, model, edata);
    else
        invalidate(0);

    if (posteq)
        processPostEquilibration(*posteq, model, edata);

    if (fwd && !posteq)
        storeJacobianAndDerivativeInReturnData(*fwd, model);
    else if (posteq)
        storeJacobianAndDerivativeInReturnData(*posteq, model);

    if (fwd && bwd)
        processBackwardProblem(*fwd, *bwd, preeq, model);
    else if (solver.computingASA())
        invalidateSLLH();

    applyChainRuleFactorToSimulationResults(model);
}

void ReturnData::processPreEquilibration(
    SteadystateProblem const& preeq, Model& model
) {
    readSimulationState(preeq.getFinalSimulationState(), model);

    if (!x_ss.empty()) {
        model.fx_rdata(x_rdata_, x_solver_);
        writeSlice(x_rdata_, x_ss);
    }
    if (!sx_ss.empty() && sensi >= SensitivityOrder::first) {
        model.fsx_rdata(sx_rdata_, sx_solver_, x_solver_);
        for (int ip = 0; ip < nplist; ip++)
            writeSlice(sx_rdata_[ip], slice(sx_ss, ip, nx));
    }
    /* Get cpu time for pre-equilibration in milliseconds */
    preeq_cpu_time = preeq.getCPUTime();
    preeq_cpu_timeB = preeq.getCPUTimeB();
    preeq_numstepsB = preeq.getNumStepsB();
    preeq_wrms = preeq.getResidualNorm();
    preeq_status = preeq.getSteadyStateStatus();
    if (preeq_status[1] == SteadyStateStatus::success)
        preeq_t = preeq.getSteadyStateTime();
    if (!preeq_numsteps.empty())
        writeSlice(preeq.getNumSteps(), preeq_numsteps);
}

void ReturnData::processPostEquilibration(
    SteadystateProblem const& posteq, Model& model, ExpData const* edata
) {
    for (int it = 0; it < nt; it++) {
        auto t = model.getTimepoint(it);
        if (std::isinf(t)) {
            readSimulationState(posteq.getFinalSimulationState(), model);
            getDataOutput(it, model, edata);
        }
    }
    /* Get cpu time for Newton solve in milliseconds */
    posteq_cpu_time = posteq.getCPUTime();
    posteq_cpu_timeB = posteq.getCPUTimeB();
    posteq_numstepsB = posteq.getNumStepsB();
    posteq_wrms = posteq.getResidualNorm();
    posteq_status = posteq.getSteadyStateStatus();
    if (posteq_status[1] == SteadyStateStatus::success)
        posteq_t = posteq.getSteadyStateTime();
    if (!posteq_numsteps.empty())
        writeSlice(posteq.getNumSteps(), posteq_numsteps);
}

void ReturnData::processForwardProblem(
    ForwardProblem const& fwd, Model& model, ExpData const* edata
) {
    if (edata)
        initializeObjectiveFunction(model.hasQuadraticLLH());

    auto initialState = fwd.getInitialSimulationState();
    if (initialState.x.getLength() == 0 && model.nx_solver > 0)
        return; // if x wasn't set forward problem failed during initialization
    readSimulationState(initialState, model);

    if (!x0.empty()) {
        model.fx_rdata(x_rdata_, x_solver_);
        writeSlice(x_rdata_, x0);
    }

    if (!sx0.empty()) {
        model.fsx_rdata(sx_rdata_, sx_solver_, x_solver_);
        for (int ip = 0; ip < nplist; ip++)
            writeSlice(sx_rdata_[ip], slice(sx0, ip, nx));
    }

    // process timepoint data
    realtype tf = fwd.getFinalTime();
    for (int it = 0; it < model.nt(); it++) {
        if (model.getTimepoint(it) <= tf) {
            readSimulationState(fwd.getSimulationStateTimepoint(it), model);
            getDataOutput(it, model, edata);
        } else {
            // check for integration failure but consider postequilibration
            if (!std::isinf(model.getTimepoint(it)))
                invalidate(it);
        }
    }

    // process event data
    if (nz > 0) {
        auto rootidx = fwd.getRootIndexes();
        for (int iroot = 0; iroot <= fwd.getEventCounter(); iroot++) {
            readSimulationState(fwd.getSimulationStateEvent(iroot), model);
            getEventOutput(t_, rootidx.at(iroot), model, edata);
        }
    }
}

void ReturnData::getDataOutput(int it, Model& model, ExpData const* edata) {
    if (!x.empty()) {
        model.fx_rdata(x_rdata_, x_solver_);
        writeSlice(x_rdata_, slice(x, it, nx));
    }
    if (!w.empty())
        model.getExpression(slice(w, it, nw), ts[it], x_solver_);
    if (!y.empty())
        model.getObservable(slice(y, it, ny), ts[it], x_solver_);
    if (!sigmay.empty())
        model.getObservableSigma(slice(sigmay, it, ny), it, edata);

    if (edata) {
        if (!isNaN(llh))
            model.addObservableObjective(llh, it, x_solver_, *edata);
        fres(it, model, *edata);
        fchi2(it, *edata);
    }

    if (sensi >= SensitivityOrder::first && nplist > 0) {

        if (sensi_meth == SensitivityMethod::forward) {
            getDataSensisFSA(it, model, edata);
        } else if (edata && !sllh.empty()) {
            model.addPartialObservableObjectiveSensitivity(
                sllh, s2llh, it, x_solver_, *edata
            );
        }

        if (!ssigmay.empty())
            model.getObservableSigmaSensitivity(
                slice(ssigmay, it, nplist * ny), slice(sy, it, nplist * ny), it,
                edata
            );
    }
}

void ReturnData::getDataSensisFSA(int it, Model& model, ExpData const* edata) {
    if (!sx.empty()) {
        model.fsx_rdata(sx_rdata_, sx_solver_, x_solver_);
        for (int ip = 0; ip < nplist; ip++) {
            writeSlice(sx_rdata_[ip], slice(sx, it * nplist + ip, nx));
        }
    }

    if (!sy.empty()) {
        model.getObservableSensitivity(
            slice(sy, it, nplist * ny), ts[it], x_solver_, sx_solver_
        );
    }

    if (edata) {
        if (!sllh.empty())
            model.addObservableObjectiveSensitivity(
                sllh, s2llh, it, x_solver_, sx_solver_, *edata
            );
        fsres(it, model, *edata);
        fFIM(it, model, *edata);
    }
}

void ReturnData::getEventOutput(
    realtype t, std::vector<int> rootidx, Model& model, ExpData const* edata
) {

    for (int ie = 0; ie < ne; ie++) {
        if (rootidx.at(ie) != 1 || nroots_.at(ie) >= nmaxevent)
            continue;

        /* get event output */
        if (!z.empty())
            model.getEvent(slice(z, nroots_.at(ie), nz), ie, t, x_solver_);
        /* if called from fillEvent at last timepoint,
         then also get the root function value */
        if (t == model.getTimepoint(nt - 1))
            if (!rz.empty())
                model.getEventRegularization(
                    slice(rz, nroots_.at(ie), nz), ie, t, x_solver_
                );

        if (edata) {
            if (!sigmaz.empty())
                model.getEventSigma(
                    slice(sigmaz, nroots_.at(ie), nz), ie, nroots_.at(ie), t,
                    edata
                );
            if (!isNaN(llh))
                model.addEventObjective(
                    llh, ie, nroots_.at(ie), t, x_solver_, *edata
                );

            /* if called from fillEvent at last timepoint,
               add regularization based on rz */
            if (t == model.getTimepoint(nt - 1) && !isNaN(llh)) {
                model.addEventObjectiveRegularization(
                    llh, ie, nroots_.at(ie), t, x_solver_, *edata
                );
            }
        }

        if (sensi >= SensitivityOrder::first) {
            if (sensi_meth == SensitivityMethod::forward) {
                getEventSensisFSA(ie, t, model, edata);
            } else if (edata && !sllh.empty()) {
                model.addPartialEventObjectiveSensitivity(
                    sllh, s2llh, ie, nroots_.at(ie), t, x_solver_, *edata
                );
            }
        }
        nroots_.at(ie)++;
    }
}

void ReturnData::getEventSensisFSA(
    int ie, realtype t, Model& model, ExpData const* edata
) {
    if (t == model.getTimepoint(nt - 1)) {
        // call from fillEvent at last timepoint
        if (!sz.empty())
            model.getUnobservedEventSensitivity(
                slice(sz, nroots_.at(ie), nz * nplist), ie
            );
        if (!srz.empty())
            model.getEventRegularizationSensitivity(
                slice(srz, nroots_.at(ie), nz * nplist), ie, t, x_solver_,
                sx_solver_
            );
    } else if (!sz.empty()) {
        model.getEventSensitivity(
            slice(sz, nroots_.at(ie), nz * nplist), ie, t, x_solver_, sx_solver_
        );
    }

    if (edata && !sllh.empty()) {
        model.addEventObjectiveSensitivity(
            sllh, s2llh, ie, nroots_.at(ie), t, x_solver_, sx_solver_, *edata
        );
    }
}

void ReturnData::processBackwardProblem(
    ForwardProblem const& fwd, BackwardProblem const& bwd,
    SteadystateProblem const* preeq, Model& model
) {
    if (sllh.empty())
        return;
    readSimulationState(fwd.getInitialSimulationState(), model);

    std::vector<realtype> llhS0(model.nJ * model.nplist(), 0.0);
    auto xB = bwd.getAdjointState();
    auto xQB = bwd.getAdjointQuadrature();

    if (preeq && preeq->hasQuadrature()) {
        handleSx0Backward(model, *preeq, llhS0, xQB);
    } else {
        handleSx0Forward(model, llhS0, xB);
    }

    for (int iJ = 0; iJ < model.nJ; iJ++) {
        for (int ip = 0; ip < model.nplist(); ip++) {
            if (iJ == 0) {
                sllh.at(ip) -= llhS0[ip] + xQB[ip * model.nJ];
            } else {
                s2llh.at(iJ - 1 + ip * (model.nJ - 1))
                    -= llhS0[ip + iJ * model.nplist()]
                       + xQB[iJ + ip * model.nJ];
            }
        }
    }
}

void ReturnData::handleSx0Backward(
    Model const& model, SteadystateProblem const& preeq,
    std::vector<realtype>& llhS0, AmiVector& xQB
) const {
    /* If preequilibration is run in adjoint mode, the scalar product of sx0
       with its adjoint counterpart (see handleSx0Forward()) is not necessary:
       the actual simulation is "extended" by the preequilibration time.
       At initialization (at t=-inf), the adjoint state is in steady state (= 0)
       and so is the scalar product. Instead of the scalar product, the
       quadratures xQB from preequilibration contribute to the gradient
       (see example notebook on equilibration for further documentation). */
    auto const& xQBpreeq = preeq.getAdjointQuadrature();
    for (int ip = 0; ip < model.nplist(); ++ip)
        xQB[ip] += xQBpreeq.at(ip);

    /* We really need references here, as sx0 can be large... */
    auto const& sx0preeq = preeq.getStateSensitivity();
    auto const& xBpreeq = preeq.getAdjointState();

    /* Add the contribution for sx0 from preequilibration. If backward
     * preequilibration was done by simulation due to a singular Jacobian,
     * xB is not necessarily 0 and we may get a non-zero contribution here. */
    for (int ip = 0; ip < model.nplist(); ++ip) {
        llhS0[ip] = 0.0;
        for (int ix = 0; ix < model.nxtrue_solver; ++ix) {
            llhS0[ip] += xBpreeq.at(ix) * sx0preeq.at(ix, ip);
        }
    }
}

void ReturnData::handleSx0Forward(
    Model const& model, std::vector<realtype>& llhS0, AmiVector& xB
) const {
    /* If preequilibration is run in forward mode or is not needed, then adjoint
       sensitivity analysis still needs the state sensitivities at t=0 (sx0),
       to compute the gradient. For each parameter, the scalar product of sx0
       with its adjoint counterpart contributes to the gradient. */
    for (int iJ = 0; iJ < model.nJ; iJ++) {
        if (iJ == 0) {
            for (int ip = 0; ip < model.nplist(); ++ip) {
                llhS0[ip] = 0.0;
                for (int ix = 0; ix < model.nxtrue_solver; ++ix) {
                    llhS0[ip] += xB[ix] * sx_solver_.at(ix, ip);
                }
            }
        } else {
            for (int ip = 0; ip < model.nplist(); ++ip) {
                llhS0[ip + iJ * model.nplist()] = 0.0;
                for (int ix = 0; ix < model.nxtrue_solver; ++ix) {
                    llhS0[ip + iJ * model.nplist()]
                        += xB[ix + iJ * model.nxtrue_solver]
                               * sx_solver_.at(ix, ip)
                           + xB[ix]
                                 * sx_solver_.at(
                                     ix + iJ * model.nxtrue_solver, ip
                                 );
                }
            }
        }
    }
}

void ReturnData::processSolver(Solver const& solver) {

    cpu_time = solver.getCpuTime();

    std::vector<int> const* tmp;

    if (!numsteps.empty()) {
        tmp = &solver.getNumSteps();
        // copy_n instead of assignment to ensure length `nt`
        // (vector from solver may be shorter in case of integration errors)
        std::copy_n(tmp->cbegin(), tmp->size(), numsteps.begin());
    }

    if (!numsteps.empty()) {
        tmp = &solver.getNumRhsEvals();
        std::copy_n(tmp->cbegin(), tmp->size(), numrhsevals.begin());
    }

    if (!numerrtestfails.empty()) {
        tmp = &solver.getNumErrTestFails();
        std::copy_n(tmp->cbegin(), tmp->size(), numerrtestfails.begin());
    }

    if (!numnonlinsolvconvfails.empty()) {
        tmp = &solver.getNumNonlinSolvConvFails();
        std::copy_n(tmp->cbegin(), tmp->size(), numnonlinsolvconvfails.begin());
    }

    if (!order.empty()) {
        tmp = &solver.getLastOrder();
        std::copy_n(tmp->cbegin(), tmp->size(), order.begin());
    }

    cpu_timeB = solver.getCpuTimeB();

    if (!numstepsB.empty()) {
        tmp = &solver.getNumStepsB();
        std::copy_n(tmp->cbegin(), tmp->size(), numstepsB.begin());
    }

    if (!numrhsevalsB.empty()) {
        tmp = &solver.getNumRhsEvalsB();
        std::copy_n(tmp->cbegin(), tmp->size(), numrhsevalsB.begin());
    }

    if (!numerrtestfailsB.empty()) {
        tmp = &solver.getNumErrTestFailsB();
        std::copy_n(tmp->cbegin(), tmp->size(), numerrtestfailsB.begin());
    }

    if (!numnonlinsolvconvfailsB.empty()) {
        tmp = &solver.getNumNonlinSolvConvFailsB();
        std::copy_n(
            tmp->cbegin(), tmp->size(), numnonlinsolvconvfailsB.begin()
        );
    }
}

void ReturnData::readSimulationState(
    SimulationState const& state, Model& model
) {
    x_solver_ = state.x;
    dx_solver_ = state.dx;
    if (computingFSA() || state.t == model.t0())
        sx_solver_ = state.sx;
    t_ = state.t;
    model.setModelState(state.state);
}

void ReturnData::invalidate(int const it_start) {
    if (it_start >= nt)
        return;

    invalidateLLH();
    invalidateSLLH();

    if (!x.empty())
        std::fill(x.begin() + nx * it_start, x.end(), getNaN());
    if (!y.empty())
        std::fill(y.begin() + ny * it_start, y.end(), getNaN());
    if (!w.empty())
        std::fill(w.begin() + nw * it_start, w.end(), getNaN());

    if (!sx.empty())
        std::fill(sx.begin() + nx * nplist * it_start, sx.end(), getNaN());
    if (!sy.empty())
        std::fill(sy.begin() + ny * nplist * it_start, sy.end(), getNaN());
}

void ReturnData::invalidateLLH() {
    llh = getNaN();
    chi2 = getNaN();
}

void ReturnData::invalidateSLLH() {
    if (!sllh.empty()) {
        std::fill(sllh.begin(), sllh.end(), getNaN());
        std::fill(s2llh.begin(), s2llh.end(), getNaN());
    }
}

void ReturnData::applyChainRuleFactorToSimulationResults(Model const& model) {
    // chain-rule factor: multiplier for am_p
    std::vector<realtype> coefficient(nplist, 1.0);
    std::vector<realtype> pcoefficient(nplist, 1.0);

    std::vector<realtype> unscaledParameters = model.getParameters();
    unscaleParameters(
        unscaledParameters, model.getParameterScale(), unscaledParameters
    );

    std::vector<realtype> augcoefficient(np, 1.0);

    if (sensi == SensitivityOrder::second && o2mode == SecondOrderMode::full) {
        for (int ip = 0; ip < np; ++ip) {
            switch (pscale[ip]) {
            case ParameterScaling::log10:
                augcoefficient.at(ip) = unscaledParameters.at(ip) * log(10);
                break;
            case ParameterScaling::ln:
                augcoefficient.at(ip) = unscaledParameters.at(ip);
                break;
            case ParameterScaling::none:
                break;
            }
        }
    }

    for (int ip = 0; ip < nplist; ++ip) {
        switch (pscale[model.plist(ip)]) {
        case ParameterScaling::log10:
            coefficient.at(ip) = log(10.0);
            pcoefficient.at(ip)
                = unscaledParameters.at(model.plist(ip)) * log(10);
            break;
        case ParameterScaling::ln:
            pcoefficient.at(ip) = unscaledParameters.at(model.plist(ip));
            break;
        case ParameterScaling::none:
            break;
        }
    }

    if (sensi >= SensitivityOrder::first) {
        // recover first order sensitivities from states for adjoint sensitivity
        // analysis
        if (sensi == SensitivityOrder::second && o2mode == SecondOrderMode::full
            && sensi_meth == SensitivityMethod::adjoint) {
            if (!sx.empty() && !x.empty())
                for (int ip = 0; ip < nplist; ++ip)
                    for (int ix = 0; ix < nxtrue; ++ix)
                        for (int it = 0; it < nt; ++it)
                            sx.at(ix + nxtrue * (ip + it * nplist))
                                = x.at(it * nx + nxtrue + ip * nxtrue + ix);

            if (!sy.empty() && !y.empty())
                for (int ip = 0; ip < nplist; ++ip)
                    for (int iy = 0; iy < nytrue; ++iy)
                        for (int it = 0; it < nt; ++it)
                            sy.at(iy + nytrue * (ip + it * nplist))
                                = y.at(it * ny + nytrue + ip * nytrue + iy);

            if (!sz.empty() && !z.empty())
                for (int ip = 0; ip < nplist; ++ip)
                    for (int iz = 0; iz < nztrue; ++iz)
                        for (int it = 0; it < nt; ++it)
                            sz.at(iz + nztrue * (ip + it * nplist))
                                = z.at(it * nz + nztrue + ip * nztrue + iz);
        }

        if (!sllh.empty())
            for (int ip = 0; ip < nplist; ++ip)
                sllh.at(ip) *= pcoefficient.at(ip);

        if (!sres.empty())
            for (int ires = 0; ires < gsl::narrow<int>(res.size()); ++ires)
                for (int ip = 0; ip < nplist; ++ip)
                    sres.at((ires * nplist + ip)) *= pcoefficient.at(ip);

        if (!FIM.empty())
            for (int ip = 0; ip < nplist; ++ip)
                for (int jp = 0; jp < nplist; ++jp)
                    FIM.at(jp + ip * nplist)
                        *= pcoefficient.at(ip) * pcoefficient.at(jp);

#define chainRule(QUANT, IND1, N1T, N1, IND2, N2)                              \
    if (!s##QUANT.empty())                                                     \
        for (int IND1 = 0; (IND1) < (N1T); ++(IND1))                           \
            for (int ip = 0; ip < nplist; ++ip)                                \
                for (int IND2 = 0; (IND2) < (N2); ++(IND2)) {                  \
                    s##QUANT.at(((IND2) * nplist + ip) * (N1) + (IND1))        \
                        *= pcoefficient.at(ip);                                \
                }

        chainRule(x, ix, nxtrue, nx, it, nt);
        chainRule(y, iy, nytrue, ny, it, nt);
        chainRule(sigmay, iy, nytrue, ny, it, nt);
        chainRule(z, iz, nztrue, nz, ie, nmaxevent);
        chainRule(sigmaz, iz, nztrue, nz, ie, nmaxevent);
        chainRule(rz, iz, nztrue, nz, ie, nmaxevent);
        chainRule(x0, ix, nxtrue, nx, it, 1);
    }

    if (o2mode == SecondOrderMode::full) { // full
        if (!s2llh.empty() && !sllh.empty()) {
            for (int ip = 0; ip < nplist; ++ip) {
                for (int iJ = 1; iJ < nJ; ++iJ) {
                    s2llh[ip * nplist + (iJ - 1)]
                        *= pcoefficient.at(ip) * augcoefficient[iJ - 1];
                    if (model.plist(ip) == iJ - 1)
                        s2llh[ip * nplist + (iJ - 1)]
                            += sllh.at(ip) * coefficient.at(ip);
                }
            }
        }

#define s2ChainRule(QUANT, IND1, N1T, N1, IND2, N2)                            \
    if (!s##QUANT.empty())                                                     \
        for (int ip = 0; ip < nplist; ++ip)                                    \
            for (int iJ = 1; iJ < nJ; ++iJ)                                    \
                for (int IND1 = 0; IND1 < N1T; ++IND1)                         \
                    for (int IND2 = 0; IND2 < N2; ++IND2) {                    \
                        s##QUANT.at(                                           \
                            (IND2 * nplist + ip) * N1 + IND1 + iJ * N1T        \
                        ) *= pcoefficient.at(ip) * augcoefficient[iJ - 1];     \
                        if (model.plist(ip) == iJ - 1)                         \
                            s##QUANT.at(                                       \
                                (IND2 * nplist + ip) * N1 + IND1 + iJ * N1T    \
                            ) += s##QUANT.at((IND2 * nplist + ip) * N1 + IND1) \
                                 * coefficient[ip];                            \
                    }

        s2ChainRule(x, ix, nxtrue, nx, it, nt);
        s2ChainRule(y, iy, nytrue, ny, it, nt);
        s2ChainRule(sigmay, iy, nytrue, ny, it, nt);
        s2ChainRule(z, iz, nztrue, nz, ie, nmaxevent);
        s2ChainRule(sigmaz, iz, nztrue, nz, ie, nmaxevent);
        s2ChainRule(rz, iz, nztrue, nz, ie, nmaxevent);
    }

    if (o2mode == SecondOrderMode::directional) { // directional
        for (int ip = 0; ip < nplist; ++ip) {
            s2llh.at(ip) *= pcoefficient.at(ip);
            s2llh.at(ip) += model.k()[nk - nplist + ip] * sllh.at(ip)
                            / unscaledParameters[model.plist(ip)];
        }

#define s2vecChainRule(QUANT, IND1, N1T, N1, IND2, N2)                         \
    if (!s##QUANT.empty())                                                     \
        for (int ip = 0; ip < nplist; ++ip)                                    \
            for (int IND1 = 0; IND1 < N1T; ++IND1)                             \
                for (int IND2 = 0; IND2 < N2; ++IND2) {                        \
                    s##QUANT.at((IND2 * nplist + ip) * N1 + IND1 + N1T)        \
                        *= pcoefficient.at(ip);                                \
                    s##QUANT.at((IND2 * nplist + ip) * N1 + IND1 + N1T)        \
                        += model.k()[nk - nplist + ip]                         \
                           * s##QUANT.at((IND2 * nplist + ip) * N1 + IND1)     \
                           / unscaledParameters[model.plist(ip)];              \
                }

        s2vecChainRule(x, ix, nxtrue, nx, it, nt);
        s2vecChainRule(y, iy, nytrue, ny, it, nt);
        s2vecChainRule(sigmay, iy, nytrue, ny, it, nt);
        s2vecChainRule(z, iz, nztrue, nz, ie, nmaxevent);
        s2vecChainRule(sigmaz, iz, nztrue, nz, ie, nmaxevent);
        s2vecChainRule(rz, iz, nztrue, nz, ie, nmaxevent);
    }
}

void ReturnData::initializeObjectiveFunction(bool enable_chi2) {
    if (rdata_reporting == RDataReporting::likelihood
        || rdata_reporting == RDataReporting::full) {
        llh = 0.0;
        std::fill(sllh.begin(), sllh.end(), 0.0);
        std::fill(s2llh.begin(), s2llh.end(), 0.0);
    }
    if ((rdata_reporting == RDataReporting::residuals
         || rdata_reporting == RDataReporting::full)
        && enable_chi2)
        chi2 = 0.0;
}

static realtype
fres(realtype y, realtype my, realtype sigma_y, ObservableScaling scale) {
    switch (scale) {
    case amici::ObservableScaling::lin:
        return (y - my) / sigma_y;
    case amici::ObservableScaling::log:
        return (std::log(y) - std::log(my)) / sigma_y;
    case amici::ObservableScaling::log10:
        return (std::log10(y) - std::log10(my)) / sigma_y;
    default:
        throw std::invalid_argument("only lin, log, log10 allowed.");
    }
}

static realtype fres_error(realtype sigma_y, realtype sigma_offset) {
    return sqrt(2 * log(sigma_y) + sigma_offset);
}

void ReturnData::fres(int const it, Model& model, ExpData const& edata) {
    if (res.empty())
        return;

    std::vector<realtype> y_it(ny, 0.0);
    model.getObservable(y_it, ts[it], x_solver_);

    std::vector<realtype> sigmay_it(ny, 0.0);
    model.getObservableSigma(sigmay_it, it, &edata);

    auto observedData = edata.getObservedDataPtr(it);
    for (int iy = 0; iy < nytrue; ++iy) {
        int iyt = iy + it * edata.nytrue();
        if (!edata.isSetObservedData(it, iy))
            continue;

        res.at(iyt) = amici::fres(
            y_it.at(iy), observedData[iy], sigmay_it.at(iy),
            model.getObservableScaling(iy)
        );

        if (sigma_res)
            res.at(iyt + nt * nytrue)
                = fres_error(sigmay_it.at(iy), sigma_offset);
    }
}

void ReturnData::fchi2(int const it, ExpData const& edata) {
    if (res.empty() || isNaN(chi2))
        return;

    for (int iy = 0; iy < nytrue; ++iy) {
        int iyt_true = iy + it * nytrue;
        chi2 += pow(res.at(iyt_true), 2);
        if (sigma_res && edata.isSetObservedData(it, iy))
            chi2 += pow(res.at(iyt_true + nt * nytrue), 2) - sigma_offset;
    }
}

static realtype fsres(
    realtype y, realtype sy, realtype my, realtype sigma_y, realtype ssigma_y,
    ObservableScaling scale
) {
    auto res = fres(y, my, sigma_y, scale);
    switch (scale) {
    case amici::ObservableScaling::lin:
        return (sy - ssigma_y * res) / sigma_y;
    case amici::ObservableScaling::log:
        return (sy / y - ssigma_y * res) / sigma_y;
    case amici::ObservableScaling::log10:
        return (sy / (y * std::log(10)) - ssigma_y * res) / sigma_y;
    default:
        throw std::invalid_argument("only lin, log, log10 allowed.");
    }
}
static realtype
fsres_error(realtype sigma_y, realtype ssigma_y, realtype sigma_offset) {
    return ssigma_y / (fres_error(sigma_y, sigma_offset) * sigma_y);
}

void ReturnData::fsres(int const it, Model& model, ExpData const& edata) {
    if (sres.empty())
        return;

    std::vector<realtype> y_it(ny, 0.0);
    model.getObservable(y_it, ts[it], x_solver_);
    std::vector<realtype> sy_it(ny * nplist, 0.0);
    model.getObservableSensitivity(sy_it, ts[it], x_solver_, sx_solver_);

    std::vector<realtype> sigmay_it(ny, 0.0);
    model.getObservableSigma(sigmay_it, it, &edata);
    std::vector<realtype> ssigmay_it(ny * nplist, 0.0);
    model.getObservableSigmaSensitivity(ssigmay_it, sy_it, it, &edata);

    auto observedData = edata.getObservedDataPtr(it);
    for (int iy = 0; iy < nytrue; ++iy) {
        if (!edata.isSetObservedData(it, iy))
            continue;
        for (int ip = 0; ip < nplist; ++ip) {
            int idx = (iy + it * edata.nytrue()) * nplist + ip;

            sres.at(idx) = amici::fsres(
                y_it.at(iy), sy_it.at(iy + ny * ip), observedData[iy],
                sigmay_it.at(iy), ssigmay_it.at(iy + ny * ip),
                model.getObservableScaling(iy)
            );

            if (sigma_res) {
                int idx_res
                    = (iy + it * edata.nytrue() + edata.nytrue() * edata.nt())
                          * nplist
                      + ip;
                sres.at(idx_res) = amici::fsres_error(
                    sigmay_it.at(iy), ssigmay_it.at(iy + ny * ip), sigma_offset
                );
            }
        }
    }
}

void ReturnData::fFIM(int it, Model& model, ExpData const& edata) {
    if (FIM.empty())
        return;

    std::vector<realtype> y_it(ny, 0.0);
    model.getObservable(y_it, ts[it], x_solver_);
    std::vector<realtype> sy_it(ny * nplist, 0.0);
    model.getObservableSensitivity(sy_it, ts[it], x_solver_, sx_solver_);

    std::vector<realtype> sigmay_it(ny, 0.0);
    model.getObservableSigma(sigmay_it, it, &edata);
    std::vector<realtype> ssigmay_it(ny * nplist, 0.0);
    model.getObservableSigmaSensitivity(ssigmay_it, sy_it, it, &edata);

    /*
     * https://www.wolframalpha.com/input/?i=d%2Fdu+d%2Fdv+log%28s%28u%2Cv%29%29+%2B+0.5+*+%28r%28u%2Cv%29%2Fs%28u%2Cv%29%29%5E2
     * r = (y - m)
     * r_du = y_du
     * d/du(d/(dv)(log(s) + 0.5 (r/s)^2)) =
     * -(2*y_du*s_dv)*r/s^3 - (2*y_dv*s_du)*r/s^3 + y_du_dv*r/s^2
     * 22222222222222222222   2222222222222222222   #############
     *
     * + (y_dv*y_du)/s^2 + (3*s_dv*s_du)*r^2/s^4 - s_du_dv*r^2/s^3
     *   111111111111111   333333333333333333333   ***************
     *
     * - (s_dv*s_du)/s^2 + (s_du_dv)/s
     *                     +++++++++++
     *
     *
     * we compute this using fsres:
     * sres_u * sres_v = (y_du*s - s_du * r) * (y_dv*s - s_dv * r) / s^4
     * = y_du*y_dv/s^2 - (y_du*s_dv + y_dv*s_du)*r/s^3 + s_du*s_dv*r^2/s^4
     *   1111111111111   22222222222222222222222222222   33333333333333333
     *
     * r should be on the same order as s. We keep 1/s^2 and drop 1/s terms.
     * drop:
     * ********: r^2/s^3 term, typically zero anyways
     * ########: r/s^2 term, requires second order sensitivities
     * ++++++++: 1/s term, typically zero anyways
     *
     * keep:
     * 123123123: accounted for by sres*sres,
     * but
     * - (s_dv*s_du)/s^2 is unaccounted for and
     * - (s_dv*y_du + s_du*y_dv)*r/s^3 is missing from 2222 and
     * + 2*(s_du*s_dv)*r^2/s^4 is missing from 3333
     *
     * s_dv*y_du and s_du*y_dv are usually zero since we do not have parameters
     * that affect both observables and sigmas. Accordingly, it is hard to know
     * emprically whether these terms are important or not.
     *
     * This leaves us with
     * + (s_du*s_dv)(2*r^2-s^2)/s^4
     * which may be problematic, since this expression does not factorise and
     * may thus introduce directions of negative curvature.
     *
     * For the least squares trick, where error residuals
     * er = sqrt(log(s) + c), with sensitivity er_du = s_du/(2*s*er). This
     * would yield terms (s_du*s_dv)*(s^2/(4*er^2))/s^4.
     * These terms are guaranteed to yield positive curvature, but go to zero
     * in the limit c -> Infty.
     *
     * Empirically, simply taking this limit and dropping all missing terms,
     * works substantially better. This was evaluated using the fides optimizer
     * on the Boehm2014 Benchmark example.
     */

    auto observedData = edata.getObservedDataPtr(it);

    for (int iy = 0; iy < nytrue; ++iy) {
        if (!edata.isSetObservedData(it, iy))
            continue;
        auto y = y_it.at(iy);
        auto m = observedData[iy];
        auto s = sigmay_it.at(iy);
        auto os = model.getObservableScaling(iy);
        // auto r = amici::fres(y, m, s);
        for (int ip = 0; ip < nplist; ++ip) {
            auto dy_i = sy_it.at(iy + ny * ip);
            auto ds_i = ssigmay_it.at(iy + ny * ip);
            auto sr_i = amici::fsres(y, dy_i, m, s, ds_i, os);
            realtype sre_i = 0.0;
            if (sigma_res)
                sre_i = amici::fsres_error(s, ds_i, sigma_offset);
            for (int jp = 0; jp < nplist; ++jp) {
                auto dy_j = sy_it.at(iy + ny * jp);
                auto ds_j = ssigmay_it.at(iy + ny * jp);
                auto sr_j = amici::fsres(y, dy_j, m, s, ds_j, os);
                FIM.at(ip + nplist * jp) += sr_i * sr_j;
                if (sigma_res) {
                    auto sre_j = amici::fsres_error(s, ds_j, sigma_offset);
                    FIM.at(ip + nplist * jp) += sre_i * sre_j;
                }
                /*+ ds_i*ds_j*(2*pow(r/pow(s,2.0), 2.0) - 1/pow(s,2.0));*/
            }
        }
    }
}

ModelContext::ModelContext(Model* model)
    : model_(model)
    , original_state_(model->getModelState()) {}

ModelContext::~ModelContext() { restore(); }

void ModelContext::restore() { model_->setModelState(original_state_); }

} // namespace amici
