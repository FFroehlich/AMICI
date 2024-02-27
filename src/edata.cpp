#include "amici/edata.h"
#include "amici/defines.h"
#include "amici/model.h"
#include "amici/rdata.h"
#include "amici/symbolic_functions.h" // getNaN

#include <algorithm>
#include <random>
#include <utility>

namespace amici {

ExpData::ExpData(int nytrue, int nztrue, int nmaxevent)
    : nytrue_(nytrue)
    , nztrue_(nztrue)
    , nmaxevent_(nmaxevent) {
    applyDimensions();
}

ExpData::ExpData(
    int nytrue, int nztrue, int nmaxevent, std::vector<realtype> ts
)
    : SimulationParameters(ts)
    , nytrue_(nytrue)
    , nztrue_(nztrue)
    , nmaxevent_(nmaxevent) {
    applyDimensions();
}

ExpData::ExpData(
    int nytrue, int nztrue, int nmaxevent, std::vector<realtype> ts,
    std::vector<realtype> fixedParameters
)
    : SimulationParameters(ts)
    , nytrue_(nytrue)
    , nztrue_(nztrue)
    , nmaxevent_(nmaxevent) {
    this->fixedParameters = std::move(fixedParameters);
    applyDimensions();
}

ExpData::ExpData(
    int nytrue, int nztrue, int nmaxevent, std::vector<realtype> ts,
    std::vector<realtype> const& observedData,
    std::vector<realtype> const& observedDataStdDev,
    std::vector<realtype> const& observedEvents,
    std::vector<realtype> const& observedEventsStdDev
)
    : SimulationParameters(ts)
    , nytrue_(nytrue)
    , nztrue_(nztrue)
    , nmaxevent_(nmaxevent) {
    applyDimensions();
    setObservedData(observedData);
    setObservedDataStdDev(observedDataStdDev);
    setObservedEvents(observedEvents);
    setObservedEventsStdDev(observedEventsStdDev);
}

ExpData::ExpData(Model const& model)
    : ExpData(
        model.nytrue, model.nztrue, model.nMaxEvent(), model.getTimepoints(),
        model.getFixedParameters()
    ) {
    reinitializeFixedParameterInitialStates
        = model.getReinitializeFixedParameterInitialStates()
          && model.getReinitializationStateIdxs().empty();
    reinitialization_state_idxs_sim = model.getReinitializationStateIdxs();
}

ExpData::ExpData(ReturnData const& rdata, realtype sigma_y, realtype sigma_z)
    : ExpData(
        rdata, std::vector<realtype>(rdata.nytrue * rdata.nt, sigma_y),
        std::vector<realtype>(rdata.nztrue * rdata.nmaxevent, sigma_z)
    ) {}

ExpData::ExpData(
    ReturnData const& rdata, std::vector<realtype> sigma_y,
    std::vector<realtype> sigma_z
)
    : ExpData(rdata.nytrue, rdata.nztrue, rdata.nmaxevent, rdata.ts) {
    if (sigma_y.size() != (unsigned)nytrue_
        && sigma_y.size() != (unsigned)nytrue_ * nt())
        throw AmiException(
            "Dimension of sigma_y must be %d or %d, was %d", nytrue_,
            nytrue_ * nt(), sigma_y.size()
        );

    if (sigma_z.size() != (unsigned)nztrue_
        && sigma_z.size() != (unsigned)nztrue_ * nmaxevent_)
        throw AmiException(
            "Dimension of sigma_z must be %d or %d, was %d", nztrue_,
            nztrue_ * nmaxevent_, sigma_z.size()
        );

    std::random_device rd{};
    std::mt19937 gen{rd()};

    realtype sigma;

    checkSigmaPositivity(sigma_y, "sigma_y");
    checkSigmaPositivity(sigma_z, "sigma_z");

    for (int iy = 0; iy < nytrue_; ++iy) {
        for (int it = 0; it < nt(); ++it) {
            sigma = sigma_y.size() == (unsigned)nytrue_
                        ? sigma_y.at(iy)
                        : sigma_y.at(iy + nytrue_ * it);
            std::normal_distribution<> e{0, sigma};
            observed_data_.at(iy + nytrue_ * it)
                = rdata.y.at(iy + rdata.ny * it) + e(gen);
            observed_data_std_dev_.at(iy + nytrue_ * it) = sigma;
        }
    }

    for (int iz = 0; iz < nztrue_; ++iz) {
        for (int ie = 0; ie < nmaxevent_; ++ie) {
            sigma = sigma_z.size() == (unsigned)nztrue_
                        ? sigma_z.at(iz)
                        : sigma_z.at(iz + nztrue_ * ie);
            std::normal_distribution<> e{0, sigma};
            observed_events_.at(iz + rdata.nztrue * ie)
                = rdata.z.at(iz + rdata.nz * ie) + e(gen);
            observed_data_std_dev_.at(iz + rdata.nztrue * ie) = sigma;
        }
    }

    id = rdata.id;
}

void ExpData::setTimepoints(std::vector<realtype> const& ts) {
    if (!std::is_sorted(ts.begin(), ts.end()))
        throw AmiException(
            "Encountered non-monotonic timepoints, please order timepoints "
            "such that they are monotonically increasing!"
        );
    ts_ = ts;
    applyDataDimension();
}

std::vector<realtype> const& ExpData::getTimepoints() const { return ts_; }

int ExpData::nt() const { return gsl::narrow<int>(ts_.size()); }

realtype ExpData::getTimepoint(int it) const { return ts_.at(it); }

void ExpData::setObservedData(std::vector<realtype> const& observedData) {
    checkDataDimension(observedData, "observedData");

    if (observedData.size() == (unsigned)nt() * nytrue_)
        observed_data_ = observedData;
    else if (observedData.empty())
        observed_data_.clear();
}

void ExpData::setObservedData(
    std::vector<realtype> const& observedData, int iy
) {
    if (observedData.size() != (unsigned)nt())
        throw AmiException(
            "Input observedData did not match dimensions nt (%i), was %i", nt(),
            observedData.size()
        );

    for (int it = 0; it < nt(); ++it)
        observed_data_.at(iy + it * nytrue_) = observedData.at(it);
}

bool ExpData::isSetObservedData(int it, int iy) const {
    return !observed_data_.empty()
           && !isNaN(observed_data_.at(it * nytrue_ + iy));
}

std::vector<realtype> const& ExpData::getObservedData() const {
    return observed_data_;
}

realtype const* ExpData::getObservedDataPtr(int it) const {
    if (!observed_data_.empty())
        return &observed_data_.at(it * nytrue_);

    return nullptr;
}

void ExpData::setObservedDataStdDev(
    std::vector<realtype> const& observedDataStdDev
) {
    checkDataDimension(observedDataStdDev, "observedDataStdDev");
    checkSigmaPositivity(observedDataStdDev, "observedDataStdDev");

    if (observedDataStdDev.size() == (unsigned)nt() * nytrue_)
        observed_data_std_dev_ = observedDataStdDev;
    else if (observedDataStdDev.empty())
        observed_data_std_dev_.clear();
}

void ExpData::setObservedDataStdDev(realtype const stdDev) {
    checkSigmaPositivity(stdDev, "stdDev");
    std::fill(
        observed_data_std_dev_.begin(), observed_data_std_dev_.end(), stdDev
    );
}

void ExpData::setObservedDataStdDev(
    std::vector<realtype> const& observedDataStdDev, int iy
) {
    if (observedDataStdDev.size() != (unsigned)nt())
        throw AmiException(
            "Input observedDataStdDev did not match dimensions nt (%i), was %i",
            nt(), observedDataStdDev.size()
        );
    checkSigmaPositivity(observedDataStdDev, "observedDataStdDev");

    for (int it = 0; it < nt(); ++it)
        observed_data_std_dev_.at(iy + it * nytrue_)
            = observedDataStdDev.at(it);
}

void ExpData::setObservedDataStdDev(realtype const stdDev, int iy) {
    checkSigmaPositivity(stdDev, "stdDev");
    for (int it = 0; it < nt(); ++it)
        observed_data_std_dev_.at(iy + it * nytrue_) = stdDev;
}

bool ExpData::isSetObservedDataStdDev(int it, int iy) const {
    return !observed_data_std_dev_.empty()
           && !isNaN(observed_data_std_dev_.at(it * nytrue_ + iy));
}

std::vector<realtype> const& ExpData::getObservedDataStdDev() const {
    return observed_data_std_dev_;
}

realtype const* ExpData::getObservedDataStdDevPtr(int it) const {
    if (!observed_data_std_dev_.empty())
        return &observed_data_std_dev_.at(it * nytrue_);

    return nullptr;
}

void ExpData::setObservedEvents(std::vector<realtype> const& observedEvents) {
    checkEventsDimension(observedEvents, "observedEvents");

    if (observedEvents.size() == (unsigned)nmaxevent_ * nztrue_)
        observed_events_ = observedEvents;
    else if (observedEvents.empty())
        observed_events_.clear();
}

void ExpData::setObservedEvents(
    std::vector<realtype> const& observedEvents, int iz
) {
    if (observedEvents.size() != (unsigned)nmaxevent_) {
        throw AmiException(
            "Input observedEvents did not match dimensions nmaxevent (%i), was "
            "%i",
            nmaxevent_, observedEvents.size()
        );
    }

    for (int ie = 0; ie < nmaxevent_; ++ie)
        observed_events_.at(iz + ie * nztrue_) = observedEvents.at(ie);
}

bool ExpData::isSetObservedEvents(int ie, int iz) const {
    return !observed_events_.empty()
           && !isNaN(observed_events_.at(ie * nztrue_ + iz));
}

std::vector<realtype> const& ExpData::getObservedEvents() const {
    return observed_events_;
}

realtype const* ExpData::getObservedEventsPtr(int ie) const {
    if (!observed_events_.empty())
        return &observed_events_.at(ie * nztrue_);

    return nullptr;
}

void ExpData::setObservedEventsStdDev(
    std::vector<realtype> const& observedEventsStdDev
) {
    checkEventsDimension(observedEventsStdDev, "observedEventsStdDev");
    checkSigmaPositivity(observedEventsStdDev, "observedEventsStdDev");

    if (observedEventsStdDev.size() == (unsigned)nmaxevent_ * nztrue_)
        observed_events_std_dev_ = observedEventsStdDev;
    else if (observedEventsStdDev.empty())
        observed_events_std_dev_.clear();
}

void ExpData::setObservedEventsStdDev(realtype const stdDev) {
    checkSigmaPositivity(stdDev, "stdDev");
    std::fill(
        observed_events_std_dev_.begin(), observed_events_std_dev_.end(), stdDev
    );
}

void ExpData::setObservedEventsStdDev(
    std::vector<realtype> const& observedEventsStdDev, int iz
) {
    if (observedEventsStdDev.size() != (unsigned)nmaxevent_)
        throw AmiException(
            "Input observedEventsStdDev did not match dimensions nmaxevent "
            "(%i), was %i",
            nmaxevent_, observedEventsStdDev.size()
        );
    checkSigmaPositivity(observedEventsStdDev, "observedEventsStdDev");

    for (int ie = 0; ie < nmaxevent_; ++ie)
        observed_events_std_dev_.at(iz + ie * nztrue_)
            = observedEventsStdDev.at(ie);
}

void ExpData::setObservedEventsStdDev(realtype const stdDev, int iz) {
    checkSigmaPositivity(stdDev, "stdDev");

    for (int ie = 0; ie < nmaxevent_; ++ie)
        observed_events_std_dev_.at(iz + ie * nztrue_) = stdDev;
}

bool ExpData::isSetObservedEventsStdDev(int ie, int iz) const {
    if (!observed_events_std_dev_.empty()) // avoid out of bound memory access
        return !isNaN(observed_events_std_dev_.at(ie * nztrue_ + iz));

    return false;
}

std::vector<realtype> const& ExpData::getObservedEventsStdDev() const {
    return observed_events_std_dev_;
}

realtype const* ExpData::getObservedEventsStdDevPtr(int ie) const {
    if (!observed_events_std_dev_.empty())
        return &observed_events_std_dev_.at(ie * nztrue_);

    return nullptr;
}

void ExpData::clear_observations() {
    std::fill(observed_data_.begin(), observed_data_.end(), getNaN());
    std::fill(
        observed_data_std_dev_.begin(), observed_data_std_dev_.end(), getNaN()
    );
    std::fill(observed_events_.begin(), observed_events_.end(), getNaN());
    std::fill(
        observed_events_std_dev_.begin(), observed_events_std_dev_.end(),
        getNaN()
    );
}

void ExpData::applyDimensions() {
    applyDataDimension();
    applyEventDimension();
}

void ExpData::applyDataDimension() {
    observed_data_.resize(nt() * nytrue_, getNaN());
    observed_data_std_dev_.resize(nt() * nytrue_, getNaN());
}

void ExpData::applyEventDimension() {
    observed_events_.resize(nmaxevent_ * nztrue_, getNaN());
    observed_events_std_dev_.resize(nmaxevent_ * nztrue_, getNaN());
}

void ExpData::checkDataDimension(
    std::vector<realtype> const& input, char const* fieldname
) const {
    if (input.size() != (unsigned)nt() * nytrue_ && !input.empty())
        throw AmiException(
            "Input %s did not match dimensions nt (%i) x nytrue (%i), was %i",
            fieldname, nt(), nytrue_, input.size()
        );
}

void ExpData::checkEventsDimension(
    std::vector<realtype> const& input, char const* fieldname
) const {
    if (input.size() != (unsigned)nmaxevent_ * nztrue_ && !input.empty())
        throw AmiException(
            "Input %s did not match dimensions nt (%i) x nytrue (%i), was %i",
            fieldname, nmaxevent_, nztrue_, input.size()
        );
}

void checkSigmaPositivity(
    std::vector<realtype> const& sigmaVector, char const* vectorName
) {
    for (auto&& sigma : sigmaVector)
        checkSigmaPositivity(sigma, vectorName);
}

void checkSigmaPositivity(realtype const sigma, char const* sigmaName) {
    if (sigma <= 0.0)
        throw AmiException(
            "Encountered sigma <= 0 in %s! value: %f", sigmaName, sigma
        );
}

int ExpData::nytrue() const { return nytrue_; }

int ExpData::nztrue() const { return nztrue_; }

int ExpData::nmaxevent() const { return nmaxevent_; }

ConditionContext::ConditionContext(
    Model* model, ExpData const* edata, FixedParameterContext fpc
)
    : model_(model)
    , original_parameters_(model->getParameters())
    , original_fixed_parameters_(model->getFixedParameters())
    , original_tstart_(model->t0())
    , original_timepoints_(model->getTimepoints())
    , original_parameter_list_(model->getParameterList())
    , original_scaling_(model->getParameterScale())
    , original_reinitialize_fixed_parameter_initial_states_(
          model->getReinitializeFixedParameterInitialStates()
          && model->getReinitializationStateIdxs().empty()
      )
    , original_reinitialization_state_idxs(model->getReinitializationStateIdxs()
      ) {
    if (model->hasCustomInitialStates())
        original_x0_ = model->getInitialStates();

    if (model->hasCustomInitialStateSensitivities())
        original_sx0_ = model->getInitialStateSensitivities();

    applyCondition(edata, fpc);
}

ConditionContext::~ConditionContext() { restore(); }

void ConditionContext::applyCondition(
    ExpData const* edata, FixedParameterContext fpc
) {
    if (!edata)
        return;

    // this needs to go first, otherwise nplist will not have the right
    // dimension for all other fields that depend on Model::nplist
    if (!edata->plist.empty())
        model_->setParameterList(edata->plist);

    // this needs to go second as setParameterScale will reset sx0
    if (!edata->pscale.empty()) {
        if (edata->pscale.size() != (unsigned)model_->np())
            throw AmiException(
                "Number of parameters (%d) in model does not"
                " match ExpData (%zd).",
                model_->np(), edata->pscale.size()
            );
        model_->setParameterScale(edata->pscale);
    }

    // this needs to be set in the model before handling initial state
    // sensitivities, which may be unscaled using model parameter values
    if (!edata->parameters.empty()) {
        if (edata->parameters.size() != (unsigned)model_->np())
            throw AmiException(
                "Number of parameters (%d) in model does not"
                " match ExpData (%zd).",
                model_->np(), edata->parameters.size()
            );
        model_->setParameters(edata->parameters);
    }

    if (!edata->x0.empty()) {
        if (edata->x0.size() != (unsigned)model_->nx_rdata)
            throw AmiException(
                "Number of initial conditions (%d) in model does"
                " not match ExpData (%zd).",
                model_->nx_rdata, edata->x0.size()
            );
        model_->setInitialStates(edata->x0);
    }

    if (!edata->sx0.empty()) {
        if (edata->sx0.size() != (unsigned)model_->nx_rdata * model_->nplist())
            throw AmiException(
                "Number of initial conditions sensitivities (%d)"
                " in model does not match ExpData (%zd).",
                model_->nx_rdata * model_->nplist(), edata->sx0.size()
            );
        model_->setInitialStateSensitivities(edata->sx0);
    }

    model_->setReinitializeFixedParameterInitialStates(
        edata->reinitializeFixedParameterInitialStates
    );

    switch (fpc) {
    case FixedParameterContext::simulation:
        if (!edata->fixedParameters.empty()) {
            // fixed parameter in model are superseded by those provided in
            // edata
            if (edata->fixedParameters.size() != (unsigned)model_->nk())
                throw AmiException(
                    "Number of fixed parameters (%d) in model does"
                    "not match ExpData (%zd).",
                    model_->nk(), edata->fixedParameters.size()
                );
            model_->setFixedParameters(edata->fixedParameters);
            if (!edata->reinitializeFixedParameterInitialStates)
                model_->setReinitializationStateIdxs(
                    edata->reinitialization_state_idxs_sim
                );
        }
        break;
    case FixedParameterContext::preequilibration:
        if (!edata->fixedParametersPreequilibration.empty()) {
            // fixed parameter in model are superseded by those provided in
            // edata
            if (edata->fixedParametersPreequilibration.size()
                != (unsigned)model_->nk())
                throw AmiException(
                    "Number of fixed parameters (%d) in model does"
                    "not match ExpData (preequilibration) (%zd).",
                    model_->nk(), edata->fixedParametersPreequilibration.size()
                );
            model_->setFixedParameters(edata->fixedParametersPreequilibration);
        }
        break;
    case FixedParameterContext::presimulation:
        if (!edata->fixedParametersPresimulation.empty()) {
            // fixed parameter in model are superseded by those provided in
            // edata
            if (edata->fixedParametersPresimulation.size()
                != (unsigned)model_->nk())
                throw AmiException(
                    "Number of fixed parameters (%d) in model does"
                    " not match ExpData (presimulation) (%zd).",
                    model_->nk(), edata->fixedParametersPresimulation.size()
                );
            model_->setFixedParameters(edata->fixedParametersPresimulation);
            if (!edata->reinitializeFixedParameterInitialStates)
                model_->setReinitializationStateIdxs(
                    edata->reinitialization_state_idxs_presim
                );
        }
        break;
    }

    model_->setT0(edata->tstart_);
    if (edata->nt()) {
        // fixed parameter in model are superseded by those provided in edata
        model_->setTimepoints(edata->getTimepoints());
    }
}

void ConditionContext::restore() {
    // parameter list has to be set before initial state sensitivities
    model_->setParameterList(original_parameter_list_);
    // parameter scale has to be set before initial state sensitivities
    model_->setParameterScale(original_scaling_);

    if (!original_x0_.empty())
        model_->setInitialStates(original_x0_);

    if (!original_sx0_.empty())
        model_->setUnscaledInitialStateSensitivities(original_sx0_);

    model_->setParameters(original_parameters_);
    model_->setFixedParameters(original_fixed_parameters_);
    model_->setT0(original_tstart_);
    model_->setTimepoints(original_timepoints_);
    model_->setReinitializeFixedParameterInitialStates(
        original_reinitialize_fixed_parameter_initial_states_
    );
    model_->setReinitializationStateIdxs(original_reinitialization_state_idxs);
}

} // namespace amici
