#ifndef AMICI_EDATA_H
#define AMICI_EDATA_H

#include "amici/defines.h"
#include "amici/misc.h"
#include "amici/simulation_parameters.h"

#include <vector>

namespace amici {

class Model;
class ReturnData;

/**
 * @brief ExpData carries all information about experimental or
 * condition-specific data.
 */
class ExpData : public SimulationParameters {

  public:
    /**
     * @brief Default constructor.
     */
    ExpData() = default;

    /**
     * @brief Copy constructor.
     */
    // needs to be declared to be wrapped by SWIG
    ExpData(ExpData const&) = default;

    /**
     * @brief Constructor that only initializes dimensions.
     *
     * @param nytrue Number of observables
     * @param nztrue Number of event outputs
     * @param nmaxevent Maximal number of events to track
     */
    ExpData(int nytrue, int nztrue, int nmaxevent);

    /**
     * @brief constructor that initializes timepoints from vectors
     *
     * @param nytrue Number of observables
     * @param nztrue Number of event outputs
     * @param nmaxevent Maximal number of events to track
     * @param ts Timepoints (dimension: nt)
     */
    ExpData(int nytrue, int nztrue, int nmaxevent, std::vector<realtype> ts);

    /**
     * @brief constructor that initializes timepoints and fixed parameters from
     * vectors
     *
     * @param nytrue Number of observables
     * @param nztrue Number of event outputs
     * @param nmaxevent Maximal number of events to track
     * @param ts Timepoints (dimension: nt)
     * @param fixedParameters Model constants (dimension: nk)
     */
    ExpData(
        int nytrue, int nztrue, int nmaxevent, std::vector<realtype> ts,
        std::vector<realtype> fixedParameters
    );

    /**
     * @brief constructor that initializes timepoints and data from vectors
     *
     * @param nytrue Number of observables
     * @param nztrue Number of event outputs
     * @param nmaxevent Maximal number of events to track
     * @param ts Timepoints (dimension: nt)
     * @param observedData observed data (dimension: nt x nytrue, row-major)
     * @param observedDataStdDev standard deviation of observed data
     * (dimension: nt x nytrue, row-major)
     * @param observedEvents observed events
     * (dimension: nmaxevents x nztrue, row-major)
     * @param observedEventsStdDev standard deviation of observed events/roots
     * (dimension: nmaxevents x nztrue, row-major)
     */
    ExpData(
        int nytrue, int nztrue, int nmaxevent, std::vector<realtype> ts,
        std::vector<realtype> const& observedData,
        std::vector<realtype> const& observedDataStdDev,
        std::vector<realtype> const& observedEvents,
        std::vector<realtype> const& observedEventsStdDev
    );

    /**
     * @brief constructor that initializes with Model
     *
     * @param model pointer to model specification object
     */
    explicit ExpData(Model const& model);

    /**
     * @brief constructor that initializes with returnData, adds noise according
     * to specified sigmas
     *
     * @param rdata return data pointer with stored simulation results
     * @param sigma_y scalar standard deviations for all observables
     * @param sigma_z scalar standard deviations for all event observables
     */
    ExpData(ReturnData const& rdata, realtype sigma_y, realtype sigma_z);

    /**
     * @brief constructor that initializes with returnData, adds noise according
     * to specified sigmas
     *
     * @param rdata return data pointer with stored simulation results
     * @param sigma_y vector of standard deviations for observables
     * (dimension: nytrue or nt x nytrue, row-major)
     * @param sigma_z vector of standard deviations for event observables
     * (dimension: nztrue or nmaxevent x nztrue, row-major)
     */
    ExpData(
        ReturnData const& rdata, std::vector<realtype> sigma_y,
        std::vector<realtype> sigma_z
    );

    ~ExpData() = default;

    friend inline bool operator==(ExpData const& lhs, ExpData const& rhs);

    /**
     * @brief number of observables of the non-augmented model
     *
     * @return number of observables of the non-augmented model
     */
    int nytrue() const;

    /**
     * @brief number of event observables of the non-augmented model
     *
     * @return number of event observables of the non-augmented model
     */
    int nztrue() const;

    /**
     * @brief maximal number of events to track
     *
     * @return maximal number of events to track
     */
    int nmaxevent() const;

    /**
     * @brief number of timepoints
     *
     * @return number of timepoints
     */
    int nt() const;

    /**
     * @brief Set output timepoints.
     *
     * If the number of timepoint increases, this will grow the
     * observation/sigma matrices and fill new entries with NaN.
     * If the number of timepoints decreases, this will shrink the
     * observation/sigma matrices.
     *
     * Note that the mapping from timepoints to measurements will not be
     * preserved. E.g., say there are measurements at t = 2, and this
     * function is called with [1, 2], then the old measurements will belong to
     * t = 1.
     *
     * @param ts timepoints
     */
    void setTimepoints(std::vector<realtype> const& ts);

    /**
     * @brief Get output timepoints.
     *
     * @return ExpData::ts
     */
    std::vector<realtype> const& getTimepoints() const;

    /**
     * @brief Get timepoint for the given index
     *
     * @param it timepoint index
     *
     * @return timepoint timepoint at index
     */
    realtype getTimepoint(int it) const;

    /**
     * @brief Set all measurements.
     *
     * @param observedData observed data (dimension: nt x nytrue, row-major)
     */
    void setObservedData(std::vector<realtype> const& observedData);

    /**
     * @brief Set measurements for a given observable index
     *
     * @param observedData observed data (dimension: nt)
     * @param iy observed data index
     */
    void setObservedData(std::vector<realtype> const& observedData, int iy);

    /**
     * @brief Whether there is a measurement for the given time- and observable-
     * index.
     *
     * @param it time index
     * @param iy observable index
     *
     * @return boolean specifying if data was set
     */
    bool isSetObservedData(int it, int iy) const;

    /**
     * @brief Get all measurements.
     *
     * @return observed data (dimension: nt x nytrue, row-major)
     */
    std::vector<realtype> const& getObservedData() const;

    /**
     * @brief Get measurements for a given timepoint index.
     *
     * @param it timepoint index
     *
     * @return pointer to observed data at index (dimension: nytrue)
     */
    realtype const* getObservedDataPtr(int it) const;

    /**
     * @brief Set standard deviations for measurements.
     *
     * @param observedDataStdDev standard deviation of observed data (dimension:
     * nt x nytrue, row-major)
     */
    void setObservedDataStdDev(std::vector<realtype> const& observedDataStdDev);

    /**
     * @brief Set identical standard deviation for all measurements.
     *
     * @param stdDev standard deviation (dimension: scalar)
     */
    void setObservedDataStdDev(realtype stdDev);

    /**
     * @brief Set standard deviations of observed data for a
     * specific observable index.
     *
     * @param observedDataStdDev standard deviation of observed data (dimension:
     * nt)
     * @param iy observed data index
     */
    void setObservedDataStdDev(
        std::vector<realtype> const& observedDataStdDev, int iy
    );

    /**
     * @brief Set all standard deviation for a given observable index to the
     * input value.
     *
     * @param stdDev standard deviation (dimension: scalar)
     * @param iy observed data index
     */
    void setObservedDataStdDev(realtype stdDev, int iy);

    /**
     * @brief Whether standard deviation for a measurement at
     * specified timepoint- and observable index has been set.
     *
     * @param it time index
     * @param iy observable index
     * @return boolean specifying if standard deviation of data was set
     */
    bool isSetObservedDataStdDev(int it, int iy) const;

    /**
     * @brief Get measurement standard deviations.
     *
     * @return standard deviation of observed data
     */
    std::vector<realtype> const& getObservedDataStdDev() const;

    /**
     * @brief Get pointer to measurement standard deviations.
     *
     * @param it timepoint index
     * @return pointer to standard deviation of observed data at index
     */
    realtype const* getObservedDataStdDevPtr(int it) const;

    /**
     * @brief Set observed event data.
     *
     * @param observedEvents observed data (dimension: nmaxevent x nztrue,
     * row-major)
     */
    void setObservedEvents(std::vector<realtype> const& observedEvents);

    /**
     * @brief Set observed event data for specific event observable.
     *
     * @param observedEvents observed data (dimension: nmaxevent)
     * @param iz observed event data index
     */
    void setObservedEvents(std::vector<realtype> const& observedEvents, int iz);

    /**
     * @brief Check whether event data at specified indices has been set.
     *
     * @param ie event index
     * @param iz event observable index
     * @return boolean specifying if data was set
     */
    bool isSetObservedEvents(int ie, int iz) const;

    /**
     * @brief Get observed event data.
     *
     * @return observed event data
     */
    std::vector<realtype> const& getObservedEvents() const;

    /**
     * @brief Get pointer to observed data at ie-th occurrence.
     *
     * @param ie event occurrence
     *
     * @return pointer to observed event data at ie-th occurrence
     */
    realtype const* getObservedEventsPtr(int ie) const;

    /**
     * @brief Set standard deviation of observed event data.
     *
     * @param observedEventsStdDev standard deviation of observed event data
     */
    void
    setObservedEventsStdDev(std::vector<realtype> const& observedEventsStdDev);

    /**
     * @brief Set standard deviation of observed event data.
     *
     * @param stdDev standard deviation (dimension: scalar)
     */
    void setObservedEventsStdDev(realtype stdDev);

    /**
     * @brief Set standard deviation of observed data for a specific observable.
     *
     * @param observedEventsStdDev standard deviation of observed data
     * (dimension: nmaxevent)
     * @param iz observed data index
     */
    void setObservedEventsStdDev(
        std::vector<realtype> const& observedEventsStdDev, int iz
    );

    /**
     * @brief Set all standard deviations of a specific event-observable.
     *
     * @param stdDev standard deviation (dimension: scalar)
     * @param iz observed data index
     */
    void setObservedEventsStdDev(realtype stdDev, int iz);

    /**
     * @brief Check whether standard deviation of event data
     * at specified indices has been set.
     *
     * @param ie event index
     * @param iz event observable index
     * @return boolean specifying if standard deviation of event data was set
     */
    bool isSetObservedEventsStdDev(int ie, int iz) const;

    /**
     * @brief Get standard deviation of observed event data.
     *
     * @return standard deviation of observed event data
     */
    std::vector<realtype> const& getObservedEventsStdDev() const;

    /**
     * @brief Get pointer to standard deviation of
     * observed event data at ie-th occurrence.
     *
     * @param ie event occurrence
     *
     * @return pointer to standard deviation of observed event data at ie-th
     * occurrence
     */
    realtype const* getObservedEventsStdDevPtr(int ie) const;

    /**
     * @brief Set all observations and their standard deviations to NaN.
     *
     * Useful, e.g., after calling ExpData::setTimepoints.
     */
    void clear_observations();

    /**
     * @brief Arbitrary (not necessarily unique) identifier.
     */
    std::string id;

  protected:
    /**
     * @brief resizes observedData, observedDataStdDev, observedEvents and
     * observedEventsStdDev
     */
    void applyDimensions();

    /**
     * @brief resizes observedData and observedDataStdDev
     */
    void applyDataDimension();

    /**
     * @brief resizes observedEvents and observedEventsStdDev
     */
    void applyEventDimension();

    /**
     * @brief checker for dimensions of input observedData or observedDataStdDev
     *
     * @param input vector input to be checked
     * @param fieldname name of the input
     */
    void checkDataDimension(
        std::vector<realtype> const& input, char const* fieldname
    ) const;

    /**
     * @brief checker for dimensions of input observedEvents or
     * observedEventsStdDev
     *
     * @param input vector input to be checked
     * @param fieldname name of the input
     */
    void checkEventsDimension(
        std::vector<realtype> const& input, char const* fieldname
    ) const;

    /** @brief number of observables */
    int nytrue_{0};

    /** @brief number of event observables */
    int nztrue_{0};

    /** @brief maximal number of event occurrences */
    int nmaxevent_{0};

    /** @brief observed data (dimension: nt x nytrue, row-major) */
    std::vector<realtype> observed_data_;

    /**
     * @brief standard deviation of observed data (dimension: nt x nytrue,
     * row-major)
     */
    std::vector<realtype> observed_data_std_dev_;

    /**
     * @brief observed events (dimension: nmaxevents x nztrue, row-major)
     */
    std::vector<realtype> observed_events_;

    /**
     * @brief standard deviation of observed events/roots
     * (dimension: nmaxevents x nztrue, row-major)
     */
    std::vector<realtype> observed_events_std_dev_;
};

/**
 * @brief Equality operator
 * @param lhs some object
 * @param rhs another object
 * @return `true`, if both arguments are equal; `false` otherwise.
 */
inline bool operator==(ExpData const& lhs, ExpData const& rhs) {
    return *dynamic_cast<SimulationParameters const*>(&lhs)
               == *dynamic_cast<SimulationParameters const*>(&rhs)
           && lhs.id == rhs.id && lhs.nytrue_ == rhs.nytrue_
           && lhs.nztrue_ == rhs.nztrue_ && lhs.nmaxevent_ == rhs.nmaxevent_
           && is_equal(lhs.observed_data_, rhs.observed_data_)
           && is_equal(lhs.observed_data_std_dev_, rhs.observed_data_std_dev_)
           && is_equal(lhs.observed_events_, rhs.observed_events_)
           && is_equal(
               lhs.observed_events_std_dev_, rhs.observed_events_std_dev_
           );
};

/**
 * @brief checks input vector of sigmas for not strictly positive values
 *
 * @param sigmaVector vector input to be checked
 * @param vectorName name of the input
 */
void checkSigmaPositivity(
    std::vector<realtype> const& sigmaVector, char const* vectorName
);

/**
 * @brief checks input scalar sigma for not strictly positive value
 *
 * @param sigma input to be checked
 * @param sigmaName name of the input
 */
void checkSigmaPositivity(realtype sigma, char const* sigmaName);

/**
 * @brief The ConditionContext class applies condition-specific amici::Model
 * settings and restores them when going out of scope
 */
class ConditionContext : public ContextManager {
  public:
    /**
     * @brief Apply condition-specific settings from edata to model while
     * keeping a backup of the original values.
     *
     * @param model
     * @param edata
     * @param fpc flag indicating which fixedParameter from edata to apply
     */
    explicit ConditionContext(
        Model* model, ExpData const* edata = nullptr,
        FixedParameterContext fpc = FixedParameterContext::simulation
    );

    ConditionContext& operator=(ConditionContext const& other) = delete;

    ~ConditionContext();

    /**
     * @brief Apply condition-specific settings from edata to the
     * constructor-supplied model, not changing the settings which were
     * backed-up in the constructor call.
     *
     * @param edata
     * @param fpc flag indicating which fixedParameter from edata to apply
     */
    void applyCondition(ExpData const* edata, FixedParameterContext fpc);

    /**
     * @brief Restore original settings on constructor-supplied amici::Model.
     * Will be called during destruction. Explicit call is generally not
     * necessary.
     */
    void restore();

  private:
    Model* model_ = nullptr;
    std::vector<realtype> original_x0_;
    std::vector<realtype> original_sx0_;
    std::vector<realtype> original_parameters_;
    std::vector<realtype> original_fixed_parameters_;
    realtype original_tstart_;
    std::vector<realtype> original_timepoints_;
    std::vector<int> original_parameter_list_;
    std::vector<amici::ParameterScaling> original_scaling_;
    bool original_reinitialize_fixed_parameter_initial_states_;
    std::vector<int> original_reinitialization_state_idxs;
};

} // namespace amici

#endif /* AMICI_EDATA_H */
