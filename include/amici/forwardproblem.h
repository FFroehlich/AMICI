#ifndef AMICI_FORWARDPROBLEM_H
#define AMICI_FORWARDPROBLEM_H

#include "amici/defines.h"
#include "amici/misc.h"
#include "amici/model.h"
#include "amici/vector.h"
#include <amici/amici.h>

#include <sundials/sundials_direct.h>
#include <vector>

namespace amici {

class ExpData;
class Solver;
class SteadystateProblem;
class FinalStateStorer;

/**
 * @brief The ForwardProblem class groups all functions for solving the
 * forward problem.
 */
class ForwardProblem {
  public:
    /**
     * @brief Constructor
     * @param edata pointer to ExpData instance
     * @param model pointer to Model instance
     * @param solver pointer to Solver instance
     * @param preeq preequilibration with which to initialize the forward
     * problem, pass nullptr for no initialization
     */
    ForwardProblem(
        ExpData const* edata, Model* model, Solver* solver,
        SteadystateProblem const* preeq
    );

    ~ForwardProblem() = default;

    /** allow FinalStateStorer to access private members and functions */
    friend ::amici::FinalStateStorer;

    /**
     * @brief Solve the forward problem.
     *
     * If forward sensitivities are enabled this will also compute
     * sensitivities.
     */
    void workForwardProblem();

    /**
     * @brief computes adjoint updates dJydx according to provided model and
     * expdata
     * @param model Model instance
     * @param edata experimental data
     */
    void getAdjointUpdates(Model& model, ExpData const& edata);

    /**
     * @brief Accessor for t
     * @return t
     */
    realtype getTime() const { return t_; }

    /**
     * @brief Accessor for x
     * @return x
     */
    AmiVector const& getState() const { return x_; }

    /**
     * @brief Accessor for dx
     * @return dx
     */
    AmiVector const& getStateDerivative() const { return dx_; }

    /**
     * @brief Accessor for sx
     * @return sx
     */
    AmiVectorArray const& getStateSensitivity() const { return sx_; }

    /**
     * @brief Accessor for x_disc
     * @return x_disc
     */
    std::vector<AmiVector> const& getStatesAtDiscontinuities() const {
        return x_disc_;
    }

    /**
     * @brief Accessor for xdot_disc
     * @return xdot_disc
     */
    std::vector<AmiVector> const& getRHSAtDiscontinuities() const {
        return xdot_disc_;
    }

    /**
     * @brief Accessor for xdot_old_disc
     * @return xdot_old_disc
     */
    std::vector<AmiVector> const& getRHSBeforeDiscontinuities() const {
        return xdot_old_disc_;
    }

    /**
     * @brief Accessor for nroots
     * @return nroots
     */
    std::vector<int> const& getNumberOfRoots() const { return nroots_; }

    /**
     * @brief Accessor for discs
     * @return discs
     */
    std::vector<realtype> const& getDiscontinuities() const { return discs_; }

    /**
     * @brief Accessor for rootidx
     * @return rootidx
     */
    std::vector<std::vector<int>> const& getRootIndexes() const {
        return root_idx_;
    }

    /**
     * @brief Accessor for dJydx
     * @return dJydx
     */
    std::vector<realtype> const& getDJydx() const { return dJydx_; }

    /**
     * @brief Accessor for dJzdx
     * @return dJzdx
     */
    std::vector<realtype> const& getDJzdx() const { return dJzdx_; }

    /**
     * @brief Accessor for pointer to x
     * @return &x
     */
    AmiVector* getStatePointer() { return &x_; }

    /**
     * @brief Accessor for pointer to dx
     * @return &dx
     */
    AmiVector* getStateDerivativePointer() { return &dx_; }

    /**
     * @brief accessor for pointer to sx
     * @return &sx
     */
    AmiVectorArray* getStateSensitivityPointer() { return &sx_; }

    /**
     * @brief Accessor for pointer to sdx
     * @return &sdx
     */
    AmiVectorArray* getStateDerivativeSensitivityPointer() { return &sdx_; }

    /**
     * @brief Accessor for it
     * @return it
     */
    int getCurrentTimeIteration() const { return it_; }

    /**
     * @brief Returns final time point for which simulations are available
     * @return time point
     */
    realtype getFinalTime() const { return final_state_.t; }

    /**
     * @brief Returns maximal event index for which simulations are available
     * @return index
     */
    int getEventCounter() const {
        return gsl::narrow<int>(event_states_.size()) - 1;
    }

    /**
     * @brief Returns maximal event index for which the timepoint is available
     * @return index
     */
    int getRootCounter() const { return gsl::narrow<int>(discs_.size()) - 1; }

    /**
     * @brief Retrieves the carbon copy of the simulation state variables at
     * the specified timepoint index
     * @param it timepoint index
     * @return state
     */
    SimulationState const& getSimulationStateTimepoint(int it) const {
        if (model->getTimepoint(it) == initial_state_.t)
            return getInitialSimulationState();
        auto map_iter = timepoint_states_.find(model->getTimepoint(it));
        assert(map_iter != timepoint_states_.end());
        return map_iter->second;
    };

    /**
     * @brief Retrieves the carbon copy of the simulation state variables at
     * the specified event index
     * @param iroot event index
     * @return SimulationState
     */
    SimulationState const& getSimulationStateEvent(int iroot) const {
        return event_states_.at(iroot);
    };

    /**
     * @brief Retrieves the carbon copy of the simulation state variables at the
     * initial timepoint
     * @return SimulationState
     */
    SimulationState const& getInitialSimulationState() const {
        return initial_state_;
    };

    /**
     * @brief Retrieves the carbon copy of the simulation state variables at the
     * final timepoint (or when simulation failed)
     * @return SimulationState
     */
    SimulationState const& getFinalSimulationState() const {
        return final_state_;
    };

    /** pointer to model instance */
    Model* model;

    /** pointer to solver instance */
    Solver* solver;

    /** pointer to experimental data instance */
    ExpData const* edata;

  private:
    void handlePresimulation();

    /**
     * @brief Execute everything necessary for the handling of events
     *
     * @param tlastroot pointer to the timepoint of the last event
     * @param seflag Secondary event flag
     * @param initial_event initial event flag
     */

    void handleEvent(realtype* tlastroot, bool seflag, bool initial_event);

    /**
     * @brief Store pre-event model state
     *
     * @param seflag Secondary event flag
     * @param initial_event initial event flag
     */
    void store_pre_event_state(bool seflag, bool initial_event);

    /**
     * @brief Check for, and if applicable, handle any secondary events
     *
     * @param tlastroot pointer to the timepoint of the last event
     */
    void handle_secondary_event(realtype* tlastroot);

    /**
     * @brief Extract output information for events
     */
    void storeEvent();

    /**
     * @brief Execute everything necessary for the handling of data points
     *
     * @param t measurement timepoint
     */
    void handleDataPoint(realtype t);

    /**
     * @brief Applies the event bolus to the current state
     */
    void applyEventBolus();

    /**
     * @brief Applies the event bolus to the current sensitivities
     */
    void applyEventSensiBolusFSA();

    /**
     * @brief checks whether there are any events to fill
     *
     * @param nmaxevent maximal number of events
     */
    bool checkEventsToFill(int nmaxevent) const {
        return std::any_of(
            nroots_.cbegin(), nroots_.cend(),
            [nmaxevent](int curNRoots) { return curNRoots < nmaxevent; }
        );
    };

    /**
     * @brief fills events at final timepoint if necessary
     *
     * @param nmaxevent maximal number of events
     */
    void fillEvents(int nmaxevent) {
        if (checkEventsToFill(nmaxevent)) {
            discs_.push_back(t_);
            storeEvent();
        }
    }

    /**
     * @brief Creates a carbon copy of the current simulation state variables
     * @return state
     */
    SimulationState getSimulationState() const;

    /** array of index vectors (dimension ne) indicating whether the respective
     * root has been detected for all so far encountered discontinuities,
     * extended as needed (dimension: dynamic) */
    std::vector<std::vector<int>> root_idx_;

    /** array of number of found roots for a certain event type
     * (dimension: ne) */
    std::vector<int> nroots_;

    /** array of values of the root function (dimension: ne) */
    std::vector<realtype> rootvals_;

    /** temporary rootval storage to check crossing in secondary event
     * (dimension: ne) */
    std::vector<realtype> rval_tmp_;

    /** array containing the time-points of discontinuities
     * (dimension: nmaxevent x ne, ordering = ?) */
    std::vector<realtype> discs_;

    /** array containing the index of discontinuities
     * (dimension: nmaxevent x ne, ordering = ?) */
    std::vector<realtype> irdiscs_;

    /** array of state vectors (dimension nx) for all so far encountered
     * discontinuities, extended as needed (dimension dynamic) */
    std::vector<AmiVector> x_disc_;

    /** array of differential state vectors (dimension nx) for all so far
     * encountered discontinuities, extended as needed (dimension dynamic) */
    std::vector<AmiVector> xdot_disc_;

    /** array of old differential state vectors (dimension nx) for all so far
     * encountered discontinuities, extended as needed (dimension dynamic) */
    std::vector<AmiVector> xdot_old_disc_;

    /** state derivative of data likelihood
     * (dimension nJ x nx x nt, ordering =?) */
    std::vector<realtype> dJydx_;

    /** state derivative of event likelihood
     * (dimension nJ x nx x nMaxEvent, ordering =?) */
    std::vector<realtype> dJzdx_;

    /** current time */
    realtype t_;

    /**
     * @brief Array of flags indicating which root has been found.
     *
     * Array of length nr (ne) with the indices of the user functions gi found
     * to have a root. For i = 0, . . . ,nr 1 or -1 if gi has a root, and = 0
     * if not. See CVodeGetRootInfo for details.
     */
    std::vector<int> roots_found_;

    /** simulation states history at timepoints  */
    std::map<realtype, SimulationState> timepoint_states_;

    /** simulation state history at events*/
    std::vector<SimulationState> event_states_;

    /** simulation state after initialization*/
    SimulationState initial_state_;

    /** simulation state after simulation*/
    SimulationState final_state_;

    /** state vector (dimension: nx_solver) */
    AmiVector x_;

    /** old state vector (dimension: nx_solver) */
    AmiVector x_old_;

    /** differential state vector (dimension: nx_solver) */
    AmiVector dx_;

    /** old differential state vector (dimension: nx_solver) */
    AmiVector dx_old_;

    /** time derivative state vector (dimension: nx_solver) */
    AmiVector xdot_;

    /** old time derivative state vector (dimension: nx_solver) */
    AmiVector xdot_old_;

    /** sensitivity state vector array (dimension: nx_cl x nplist, row-major) */
    AmiVectorArray sx_;

    /** differential sensitivity state vector array
     * (dimension: nx_cl x nplist, row-major) */
    AmiVectorArray sdx_;

    /** sensitivity of the event timepoint (dimension: nplist) */
    std::vector<realtype> stau_;

    /** storage for last found root */
    realtype tlastroot_{0.0};

    /** flag to indicate whether solver was preeinitialized via preequilibration
     */
    bool preequilibrated_{false};

    /** current iteration number for time index */
    int it_;
};

/**
 * @brief stores the stimulation state when it goes out of scope
 */
class FinalStateStorer : public ContextManager {
  public:
    /**
     * @brief constructor, attaches problem pointer
     * @param fwd problem from which the simulation state is to be stored
     */
    explicit FinalStateStorer(ForwardProblem* fwd)
        : fwd_(fwd) {}

    FinalStateStorer& operator=(FinalStateStorer const& other) = delete;

    /**
     * @brief destructor, stores simulation state
     */
    ~FinalStateStorer() {
        if (fwd_)
            fwd_->final_state_ = fwd_->getSimulationState();
    }

  private:
    ForwardProblem* fwd_;
};

} // namespace amici

#endif // FORWARDPROBLEM_H
