#ifndef AMICI_SIMULATION_PARAMETERS_H
#define AMICI_SIMULATION_PARAMETERS_H

#include "amici/defines.h"

#include <vector>

namespace amici {

/**
 * @brief Container for various simulation parameters.
 */
class SimulationParameters {
  public:
    SimulationParameters() = default;

    /**
     * @brief Constructor
     * @param timepoints Timepoints for which simulation results are requested
     */
    explicit SimulationParameters(std::vector<realtype> timepoints)
        : ts_(std::move(timepoints)) {}

    /**
     * @brief Constructor
     * @param fixedParameters Model constants
     * @param parameters Model parameters
     */
    SimulationParameters(
        std::vector<realtype> fixedParameters, std::vector<realtype> parameters
    )
        : fixedParameters(std::move(fixedParameters))
        , parameters(std::move(parameters))
        , pscale(std::vector<ParameterScaling>(
              this->parameters.size(), ParameterScaling::none
          )) {}

#ifndef SWIGPYTHON
    /*
     * include/amici/simulation_parameters.h:71: Warning 509: Overloaded method
     * amici::SimulationParameters::SimulationParameters(std::vector<
     * amici::realtype,std::allocator< amici::realtype > >,std::vector<
     * amici::realtype,std::allocator< amici::realtype > >,std::vector<
     * amici::realtype,std::allocator< amici::realtype > >) effectively ignored,
     * include/amici/simulation_parameters.h:54: Warning 509: as it is shadowed
     * by amici::SimulationParameters::SimulationParameters(std::vector<
     * amici::realtype,std::allocator< amici::realtype > >,std::vector<
     * amici::realtype,std::allocator< amici::realtype > >,std::vector<
     * int,std::allocator< int > >).
     */
    /**
     * @brief Constructor
     * @param fixedParameters Model constants
     * @param parameters Model parameters
     * @param plist Model parameter indices w.r.t. which sensitivities are to be
     * computed
     */
    SimulationParameters(
        std::vector<realtype> fixedParameters, std::vector<realtype> parameters,
        std::vector<int> plist
    )
        : fixedParameters(std::move(fixedParameters))
        , parameters(std::move(parameters))
        , pscale(std::vector<ParameterScaling>(
              this->parameters.size(), ParameterScaling::none
          ))
        , plist(std::move(plist)) {}

    /**
     * @brief Constructor
     * @param timepoints Timepoints for which simulation results are requested
     * @param fixedParameters Model constants
     * @param parameters Model parameters
     */
    SimulationParameters(
        std::vector<realtype> timepoints, std::vector<realtype> fixedParameters,
        std::vector<realtype> parameters
    )
        : fixedParameters(std::move(fixedParameters))
        , parameters(std::move(parameters))
        , pscale(std::vector<ParameterScaling>(
              this->parameters.size(), ParameterScaling::none
          ))
        , ts_(std::move(timepoints)) {}
#endif

    /**
     * @brief Set reinitialization of all states based on model constants for
     * presimulation (only meaningful if preequilibration is performed).
     *
     * Convenience function to populate
     * `reinitialization_state_idxs_presim` and
     * `reinitialization_state_idxs_sim`
     *
     * @param nx_rdata Number of states (Model::nx_rdata)
     */
    void reinitializeAllFixedParameterDependentInitialStatesForPresimulation(
        int nx_rdata
    );

    /**
     * @brief Set reinitialization of all states based on model constants for
     * the 'main' simulation (only meaningful if presimulation or
     * preequilibration is performed).
     *
     * Convenience function to populate
     * `reinitialization_state_idxs_presim` and
     * `reinitialization_state_idxs_sim`
     *
     * @param nx_rdata Number of states (Model::nx_rdata)
     */
    void reinitializeAllFixedParameterDependentInitialStatesForSimulation(
        int nx_rdata
    );

    /**
     * @brief Set reinitialization of all states based on model constants for
     * all simulation phases.
     *
     * Convenience function to populate
     * `reinitialization_state_idxs_presim` and
     * `reinitialization_state_idxs_sim`
     *
     * @param nx_rdata Number of states (Model::nx_rdata)
     */
    void reinitializeAllFixedParameterDependentInitialStates(int nx_rdata);

    /**
     * @brief Model constants
     *
     * Vector of size Model::nk() or empty
     */
    std::vector<realtype> fixedParameters;

    /**
     * @brief Model constants for pre-equilibration
     *
     * Vector of size Model::nk() or empty.
     */
    std::vector<realtype> fixedParametersPreequilibration;

    /**
     * @brief Model constants for pre-simulation
     *
     * Vector of size Model::nk() or empty.
     */
    std::vector<realtype> fixedParametersPresimulation;

    /**
     * @brief Model parameters
     *
     * Vector of size Model::np() or empty with parameter scaled according to
     * SimulationParameter::pscale.
     */
    std::vector<realtype> parameters;

    /**
     * @brief Initial state
     *
     * Vector of size Model::nx() or empty
     */
    std::vector<realtype> x0;

    /**
     * @brief Initial state sensitivities
     *
     * Dimensions:
     * Model::nx() * Model::nplist(), Model::nx() * ExpData::plist.size(), if
     * ExpData::plist is not empty, or empty
     */
    std::vector<realtype> sx0;

    /**
     * @brief Parameter scales
     *
     * Vector of parameter scale of size Model::np(), indicating how/if each
     * parameter is to be scaled.
     */
    std::vector<ParameterScaling> pscale;

    /**
     * @brief Parameter indices w.r.t. which to compute sensitivities
     */
    std::vector<int> plist;

    /**
     * @brief Starting time of the simulation.
     *
     * Output timepoints are absolute timepoints, independent of
     * \f$ t_{start} \f$.
     * For output timepoints \f$ t <  t_{start} \f$, the initial state will be
     * returned.
     */
    realtype tstart_{0.0};

    /**
     * @brief Duration of pre-simulation.
     *
     * If this is > 0, presimulation will be performed from
     * (model->t0 - t_presim) to model->t0 using the fixedParameters in
     * fixedParametersPresimulation
     */
    realtype t_presim{0.0};

    /**
     * @brief Timepoints for which model state/outputs/... are requested
     *
     * Vector of timepoints.
     */
    std::vector<realtype> ts_;

    /**
     * @brief Flag indicating whether reinitialization of states depending on
     * fixed parameters is activated
     */
    bool reinitializeFixedParameterInitialStates{false};

    /**
     * @brief Indices of states to be reinitialized based on provided
     * presimulation constants / fixed parameters.
     */
    std::vector<int> reinitialization_state_idxs_presim;

    /**
     * @brief Indices of states to be reinitialized based on provided
     * constants / fixed parameters.
     */
    std::vector<int> reinitialization_state_idxs_sim;
};

bool operator==(SimulationParameters const& a, SimulationParameters const& b);

} // namespace amici

#endif // AMICI_SIMULATION_PARAMETERS_H
