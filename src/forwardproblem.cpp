#include "amici/forwardproblem.h"

#include "amici/edata.h"
#include "amici/exception.h"
#include "amici/misc.h"
#include "amici/model.h"
#include "amici/solver.h"
#include "amici/steadystateproblem.h"

#include <algorithm>
#include <cmath>

namespace amici {

/**
 * @brief Check if the next timepoint is too close to the current timepoint.
 *
 * Based on CVODES' `cvHin`.
 * @param cur_t Current time.
 * @param t_next Next stop time.
 * @return True if too close, false otherwise.
 */
bool is_next_t_too_close(realtype cur_t, realtype t_next) {
    auto tdiff = t_next - cur_t;
    if (tdiff == 0.0)
        return true;

    auto tdist = std::fabs(tdiff);
    auto tround = std::numeric_limits<realtype>::epsilon()
                  * std::max(std::fabs(cur_t), std::fabs(t_next));
    if (tdist < 2.0 * tround)
        return true;

    return false;
}

ForwardProblem::ForwardProblem(
    ExpData const* edata, Model* model, Solver* solver,
    SteadystateProblem const* preeq
)
    : model(model)
    , solver(solver)
    , edata(edata)
    , nroots_(gsl::narrow<decltype(nroots_)::size_type>(model->ne), 0)
    , rootvals_(gsl::narrow<decltype(rootvals_)::size_type>(model->ne), 0.0)
    , rval_tmp_(gsl::narrow<decltype(rval_tmp_)::size_type>(model->ne), 0.0)
    , dJydx_(model->nJ * model->nx_solver * model->nt(), 0.0)
    , dJzdx_(model->nJ * model->nx_solver * model->nMaxEvent(), 0.0)
    , t_(model->t0())
    , roots_found_(model->ne, 0)
    , x_(model->nx_solver)
    , x_old_(model->nx_solver)
    , dx_(model->nx_solver)
    , dx_old_(model->nx_solver)
    , xdot_(model->nx_solver)
    , xdot_old_(model->nx_solver)
    , sx_(model->nx_solver, model->nplist())
    , sdx_(model->nx_solver, model->nplist())
    , stau_(model->nplist()) {
    if (preeq) {
        x_ = preeq->getState();
        sx_ = preeq->getStateSensitivity();
        preequilibrated_ = true;
    }
}

void ForwardProblem::workForwardProblem() {
    FinalStateStorer fss(this);

    auto presimulate = edata && edata->t_presim > 0;

    /* if preequilibration was done, model was already initialized */
    if (!preequilibrated_)
        model->initialize(
            x_, dx_, sx_, sdx_,
            solver->getSensitivityOrder() >= SensitivityOrder::first,
            roots_found_
        );
    else if (model->ne) {
        model->initEvents(x_, dx_, roots_found_);
    }

    /* compute initial time and setup solver for (pre-)simulation */
    auto t0 = model->t0();
    if (presimulate)
        t0 -= edata->t_presim;
    solver->setup(t0, model, x_, dx_, sx_, sdx_);

    if (model->ne
        && std::any_of(roots_found_.begin(), roots_found_.end(), [](int rf) {
               return rf == 1;
           }))
        handleEvent(&t0, false, true);

    /* perform presimulation if necessary */
    if (presimulate) {
        if (solver->computingASA())
            throw AmiException("Presimulation with adjoint sensitivities"
                               " is currently not implemented.");
        handlePresimulation();
        t_ = model->t0();
        if (model->ne) {
            model->initEvents(x_, dx_, roots_found_);
            if (std::any_of(
                    roots_found_.begin(), roots_found_.end(),
                    [](int rf) { return rf == 1; }
                ))
                handleEvent(&t0, false, true);
        }
    }

    /* when computing adjoint sensitivity analysis with presimulation,
     we need to store sx after the reinitialization after preequilibration
     but before reinitialization after presimulation. As presimulation with ASA
     will not update sx, we can simply extract the values here.*/
    if (solver->computingASA() && presimulate)
        sx_ = solver->getStateSensitivity(model->t0());

    if (presimulate || preequilibrated_)
        solver->updateAndReinitStatesAndSensitivities(model);

    // update x0 after computing consistence IC/reinitialization
    x_ = solver->getState(model->t0());
    /* when computing forward sensitivities, we generally want to update sx
     after presimulation/preequilibration, and if we didn't do either this also
     wont harm. when computing ASA, we only want to update here, if we didn't
     update before presimulation (if applicable).
    */
    if (solver->computingFSA() || (solver->computingASA() && !presimulate))
        sx_ = solver->getStateSensitivity(model->t0());

    /* store initial state and sensitivity*/
    initial_state_ = getSimulationState();
    // store root information at t0
    model->froot(t_, x_, dx_, rootvals_);

    // get list of trigger timepoints for fixed-time triggered events
    auto trigger_timepoints = model->get_trigger_timepoints();
    auto it_trigger_timepoints = std::find_if(
        trigger_timepoints.begin(), trigger_timepoints.end(),
        [this](auto t) { return t > this->t_; }
    );

    /* loop over timepoints */
    for (it_ = 0; it_ < model->nt(); it_++) {
        // next output time-point
        auto next_t_out = model->getTimepoint(it_);

        if (std::isinf(next_t_out))
            break;

        if (next_t_out > model->t0()) {
            // Solve for next output timepoint
            while (t_ < next_t_out) {
                if (is_next_t_too_close(t_, next_t_out)) {
                    // next timepoint is too close to current timepoint.
                    // we use the state of the current timepoint.
                    break;
                }

                // next stop time is next output timepoint or next
                // time-triggered event
                auto next_t_event
                    = it_trigger_timepoints != trigger_timepoints.end()
                          ? *it_trigger_timepoints
                          : std::numeric_limits<realtype>::infinity();
                auto next_t_stop = std::min(next_t_out, next_t_event);

                int status = solver->run(next_t_stop);
                /* sx will be copied from solver on demand if sensitivities
                 are computed */
                solver->writeSolution(&t_, x_, dx_, sx_, dx_);

                if (status == AMICI_ILL_INPUT) {
                    /* clustering of roots => turn off root-finding */
                    solver->turnOffRootFinding();
                } else if (status == AMICI_ROOT_RETURN || t_ == next_t_event) {
                    // solver-tracked or time-triggered event
                    solver->getRootInfo(roots_found_.data());

                    // check if we are at a trigger timepoint.
                    // if so, set the root-found flag
                    if (t_ == next_t_event) {
                        for (auto ie : model->state_independent_events_[t_]) {
                            // determine direction of root crossing from
                            // root function value at the previous event
                            roots_found_[ie] = std::copysign(1, -rootvals_[ie]);
                        }
                        ++it_trigger_timepoints;
                    }

                    handleEvent(&tlastroot_, false, false);
                }
            }
        }
        handleDataPoint(next_t_out);
    }

    /* fill events */
    if (model->nz > 0 && model->nt() > 0) {
        fillEvents(model->nMaxEvent());
    }
}

void ForwardProblem::handlePresimulation() {
    // Are there dedicated condition preequilibration parameters provided?
    ConditionContext cond(model, edata, FixedParameterContext::presimulation);
    solver->updateAndReinitStatesAndSensitivities(model);

    solver->run(model->t0());
    solver->writeSolution(&t_, x_, dx_, sx_, dx_);
}

void ForwardProblem::handleEvent(
    realtype* tlastroot, bool const seflag, bool const initial_event
) {
    /* store Heaviside information at event occurrence */
    model->froot(t_, x_, dx_, rootvals_);

    /* store timepoint at which the event occurred */
    discs_.push_back(t_);

    root_idx_.push_back(roots_found_);

    rval_tmp_ = rootvals_;

    if (!seflag && !initial_event) {
        /* only check this in the first event fired, otherwise this will always
         * be true */
        if (t_ == *tlastroot) {
            throw AmiException("AMICI is stuck in an event, as the initial "
                               "step-size after the event is too small. "
                               "To fix this, increase absolute and relative "
                               "tolerances!");
        }
        *tlastroot = t_;
    }

    if (model->nz > 0)
        storeEvent();

    store_pre_event_state(seflag, initial_event);

    if (!initial_event)
        model->updateHeaviside(roots_found_);

    applyEventBolus();

    if (solver->computingFSA()) {
        /* compute the new xdot  */
        model->fxdot(t_, x_, dx_, xdot_);
        applyEventSensiBolusFSA();
    }

    handle_secondary_event(tlastroot);

    /* only reinitialise in the first event fired */
    if (!seflag) {
        solver->reInit(t_, x_, dx_);
        if (solver->computingFSA()) {
            solver->sensReInit(sx_, sdx_);
        }
    }
}

void ForwardProblem::storeEvent() {
    if (t_ == model->getTimepoint(model->nt() - 1)) {
        // call from fillEvent at last timepoint
        model->froot(t_, x_, dx_, rootvals_);
        for (int ie = 0; ie < model->ne; ie++) {
            roots_found_.at(ie) = (nroots_.at(ie) < model->nMaxEvent()) ? 1 : 0;
        }
        root_idx_.push_back(roots_found_);
    }

    if (getRootCounter() < getEventCounter()) {
        /* update stored state (sensi) */
        event_states_.at(getRootCounter()) = getSimulationState();
    } else {
        /* add stored state (sensi) */
        event_states_.push_back(getSimulationState());
    }

    /* EVENT OUTPUT */
    for (int ie = 0; ie < model->ne; ie++) {
        /* only look for roots of the rootfunction not discontinuities */
        if (nroots_.at(ie) >= model->nMaxEvent())
            continue;

        /* only consider transitions false -> true or event filling */
        if (roots_found_.at(ie) != 1
            && t_ != model->getTimepoint(model->nt() - 1)) {
            continue;
        }

        if (edata && solver->computingASA())
            model->getAdjointStateEventUpdate(
                slice(dJzdx_, nroots_.at(ie), model->nx_solver * model->nJ), ie,
                nroots_.at(ie), t_, x_, *edata
            );

        nroots_.at(ie)++;
    }

    if (t_ == model->getTimepoint(model->nt() - 1)) {
        // call from fillEvent at last timepoint
        // loop until all events are filled
        fillEvents(model->nMaxEvent());
    }
}

void ForwardProblem::store_pre_event_state(bool seflag, bool initial_event) {
    /* if we need to do forward sensitivities later on we need to store the old
     * x and the old xdot */
    if (solver->getSensitivityOrder() >= SensitivityOrder::first) {
        /* store x and xdot to compute jump in sensitivities */
        x_old_.copy(x_);
    }
    if (solver->computingFSA()) {
        model->fxdot(t_, x_, dx_, xdot_);
        xdot_old_.copy(xdot_);
        dx_old_.copy(dx_);
        /* compute event-time derivative only for primary events, we get
         * into trouble with multiple simultaneously firing events here (but
         * is this really well defined then?), in that case just use the
         * last ie and hope for the best. */
        if (!seflag && !initial_event) {
            for (int ie = 0; ie < model->ne; ie++) {
                if (roots_found_.at(ie) == 1) {
                    /* only consider transitions false -> true */
                    model->getEventTimeSensitivity(stau_, t_, ie, x_, sx_);
                }
            }
        }
        if (initial_event) // t0 has no parameter dependency
            std::fill(stau_.begin(), stau_.end(), 0.0);
    } else if (solver->computingASA()) {
        /* store x to compute jump in discontinuity */
        x_disc_.push_back(x_);
        xdot_disc_.push_back(xdot_);
        xdot_old_disc_.push_back(xdot_old_);
    }
}

void ForwardProblem::handle_secondary_event(realtype* tlastroot) {
    int secondevent = 0;

    /* check whether we need to fire a secondary event */
    model->froot(t_, x_, dx_, rootvals_);
    for (int ie = 0; ie < model->ne; ie++) {
        /* the same event should not trigger itself */
        if (roots_found_.at(ie) == 0) {
            /* check whether there was a zero-crossing */
            if (0 > rval_tmp_.at(ie) * rootvals_.at(ie)) {
                if (rval_tmp_.at(ie) < rootvals_.at(ie)) {
                    roots_found_.at(ie) = 1;
                } else {
                    roots_found_.at(ie) = -1;
                }
                secondevent++;
            } else {
                roots_found_.at(ie) = 0;
            }
        } else {
            /* don't fire the same event again */
            roots_found_.at(ie) = 0;
        }
    }
    /* fire the secondary event */
    if (secondevent > 0) {
        /* Secondary events may result in wrong forward sensitivities,
         * if the secondary event has a bolus... */
        if (solver->computingFSA() && solver->logger)
            solver->logger->log(
                LogSeverity::warning, "SECONDARY_EVENT",
                "Secondary event was triggered. Depending on "
                "the bolus of the secondary event, forward "
                "sensitivities can be incorrect."
            );
        handleEvent(tlastroot, true, false);
    }
}

void ForwardProblem::handleDataPoint(realtype t) {
    /* We only store the simulation state if it's not the initial state, as the
       initial state is stored anyway and we want to avoid storing it twice */
    if (t != model->t0() && timepoint_states_.count(t) == 0)
        timepoint_states_[t] = getSimulationState();
    /* store diagnosis information for debugging */
    solver->storeDiagnosis();
}

void ForwardProblem::applyEventBolus() {
    for (int ie = 0; ie < model->ne; ie++)
        if (roots_found_.at(ie) == 1) // only consider transitions false -> true
            model->addStateEventUpdate(x_, ie, t_, xdot_, xdot_old_);
}

void ForwardProblem::applyEventSensiBolusFSA() {
    for (int ie = 0; ie < model->ne; ie++)
        if (roots_found_.at(ie) == 1) // only consider transitions false -> true
            /*  */
            model->addStateSensitivityEventUpdate(
                sx_, ie, t_, x_old_, xdot_, xdot_old_, stau_
            );
}

void ForwardProblem::getAdjointUpdates(Model& model, ExpData const& edata) {
    for (int it = 0; it < model.nt(); it++) {
        if (std::isinf(model.getTimepoint(it)))
            return;
        model.getAdjointStateObservableUpdate(
            slice(dJydx_, it, model.nx_solver * model.nJ), it,
            getSimulationStateTimepoint(it).x, edata
        );
    }
}

SimulationState ForwardProblem::getSimulationState() const {
    auto state = SimulationState();
    state.t = t_;
    state.x = x_;
    state.dx = dx_;
    if (solver->computingFSA() || t_ == model->t0())
        state.sx = sx_;
    state.state = model->getModelState();
    return state;
}

} // namespace amici
