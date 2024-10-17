#ifndef AMICI_MODEL_H
#define AMICI_MODEL_H

#include "amici/abstract_model.h"
#include "amici/defines.h"
#include "amici/logging.h"
#include "amici/model_dimensions.h"
#include "amici/model_state.h"
#include "amici/simulation_parameters.h"
#include "amici/splinefunctions.h"
#include "amici/sundials_matrix_wrapper.h"
#include "amici/vector.h"

#include <map>
#include <vector>

namespace amici {

class ExpData;
class Model;
class Solver;

} // namespace amici

// for serialization friend in amici::Model
namespace boost {
namespace serialization {
template <class Archive>
void serialize(Archive& ar, amici::Model& m, unsigned int version);
}
} // namespace boost

namespace amici {

/**
 * @brief Describes the various model quantities
 */
enum class ModelQuantity {
    J,
    JB,
    Jv,
    JvB,
    JDiag,
    sx,
    sy,
    sz,
    srz,
    ssigmay,
    ssigmaz,
    xdot,
    sxdot,
    xBdot,
    x0_rdata,
    x0,
    x_rdata,
    x,
    dwdw,
    dwdx,
    dwdp,
    y,
    dydp,
    dydx,
    w,
    root,
    qBdot,
    qBdot_ss,
    xBdot_ss,
    JSparseB_ss,
    deltax,
    deltasx,
    deltaxB,
    k,
    p,
    ts,
    dJydy,
    dJydy_matlab,
    deltaqB,
    dsigmaydp,
    dsigmaydy,
    dsigmazdp,
    dJydsigma,
    dJydx,
    dzdx,
    dzdp,
    dJrzdsigma,
    dJrzdz,
    dJrzdx,
    dJzdsigma,
    dJzdz,
    dJzdx,
    drzdp,
    drzdx,
};

extern std::map<ModelQuantity, std::string> const model_quantity_to_str;

/**
 * @brief The Model class represents an AMICI ODE/DAE model.
 *
 * The model can compute various model related quantities based on symbolically
 * generated code.
 */
class Model : public AbstractModel, public ModelDimensions {
  public:
    /** Default constructor */
    Model() = default;

    /**
     * @brief Constructor with model dimensions.
     * @param model_dimensions Model dimensions
     * @param simulation_parameters Simulation parameters
     * @param o2mode Second order sensitivity mode
     * @param idlist Indexes indicating algebraic components (DAE only)
     * @param z2event Mapping of event outputs to events
     * @param state_independent_events Map of events with state-independent
     * triggers functions, mapping trigger timepoints to event indices.
     */
    Model(
        ModelDimensions const& model_dimensions,
        SimulationParameters simulation_parameters,
        amici::SecondOrderMode o2mode, std::vector<amici::realtype> idlist,
        std::vector<int> z2event,
        std::map<realtype, std::vector<int>> state_independent_events = {}
    );

    /** Destructor. */
    ~Model() override = default;

    /**
     * @brief Copy assignment is disabled until const members are removed.
     * @param other Object to copy from
     * @return
     */
    Model& operator=(Model const& other) = delete;

    /**
     * @brief Clone this instance.
     * @return The clone
     */
    virtual Model* clone() const = 0;

    /**
     * @brief Serialize Model (see `boost::serialization::serialize`).
     * @param ar Archive to serialize to
     * @param m Data to serialize
     * @param version Version number
     */
    template <class Archive>
    friend void boost::serialization::serialize(
        Archive& ar, Model& m, unsigned int version
    );

    /**
     * @brief Check equality of data members.
     * @param a First model instance
     * @param b Second model instance
     * @return Equality
     */
    friend bool operator==(Model const& a, Model const& b);

    // Overloaded base class methods
    using AbstractModel::fdeltaqB;
    using AbstractModel::fdeltasx;
    using AbstractModel::fdeltax;
    using AbstractModel::fdeltaxB;
    using AbstractModel::fdJrzdsigma;
    using AbstractModel::fdJrzdz;
    using AbstractModel::fdJydsigma;
    using AbstractModel::fdJydy;
    using AbstractModel::fdJydy_colptrs;
    using AbstractModel::fdJydy_rowvals;
    using AbstractModel::fdJzdsigma;
    using AbstractModel::fdJzdz;
    using AbstractModel::fdrzdp;
    using AbstractModel::fdrzdx;
    using AbstractModel::fdsigmaydp;
    using AbstractModel::fdsigmaydy;
    using AbstractModel::fdsigmazdp;
    using AbstractModel::fdtotal_cldp;
    using AbstractModel::fdtotal_cldx_rdata;
    using AbstractModel::fdtotal_cldx_rdata_colptrs;
    using AbstractModel::fdtotal_cldx_rdata_rowvals;
    using AbstractModel::fdwdp;
    using AbstractModel::fdwdp_colptrs;
    using AbstractModel::fdwdp_rowvals;
    using AbstractModel::fdwdw;
    using AbstractModel::fdwdw_colptrs;
    using AbstractModel::fdwdw_rowvals;
    using AbstractModel::fdwdx;
    using AbstractModel::fdwdx_colptrs;
    using AbstractModel::fdwdx_rowvals;
    using AbstractModel::fdx_rdatadp;
    using AbstractModel::fdx_rdatadtcl;
    using AbstractModel::fdx_rdatadtcl_colptrs;
    using AbstractModel::fdx_rdatadtcl_rowvals;
    using AbstractModel::fdx_rdatadx_solver;
    using AbstractModel::fdx_rdatadx_solver_colptrs;
    using AbstractModel::fdx_rdatadx_solver_rowvals;
    using AbstractModel::fdydp;
    using AbstractModel::fdydx;
    using AbstractModel::fdzdp;
    using AbstractModel::fdzdx;
    using AbstractModel::fJrz;
    using AbstractModel::fJy;
    using AbstractModel::fJz;
    using AbstractModel::frz;
    using AbstractModel::fsigmay;
    using AbstractModel::fsigmaz;
    using AbstractModel::fsrz;
    using AbstractModel::fstau;
    using AbstractModel::fsx0;
    using AbstractModel::fsx0_fixedParameters;
    using AbstractModel::fsz;
    using AbstractModel::fw;
    using AbstractModel::fx0;
    using AbstractModel::fx0_fixedParameters;
    using AbstractModel::fy;
    using AbstractModel::fz;

    /**
     * @brief Initialize model properties.
     * @param x Reference to state variables
     * @param dx Reference to time derivative of states (DAE only)
     * @param sx Reference to state variable sensitivities
     * @param sdx Reference to time derivative of state sensitivities (DAE only)
     * @param computeSensitivities Flag indicating whether sensitivities are to
     * be computed
     * @param roots_found boolean indicators indicating whether roots were found
     * at t0 by this fun
     */
    void initialize(
        AmiVector& x, AmiVector& dx, AmiVectorArray& sx, AmiVectorArray& sdx,
        bool computeSensitivities, std::vector<int>& roots_found
    );

    /**
     * @brief Re-initialize model properties after changing simulation context.
     * @param t Timepoint
     * @param x Reference to state variables
     * @param sx Reference to state variable sensitivities
     * @param computeSensitivities Flag indicating whether sensitivities are to
     * be computed
     */
    void reinitialize(
        realtype t, AmiVector& x, AmiVectorArray& sx, bool computeSensitivities
    );

    /**
     * @brief Initialize model properties.
     * @param xB Adjoint state variables
     * @param dxB Time derivative of adjoint states (DAE only)
     * @param xQB Adjoint quadratures
     * @param posteq Flag indicating whether postequilibration was performed
     */
    void initializeB(AmiVector& xB, AmiVector& dxB, AmiVector& xQB, bool posteq)
        const;

    /**
     * @brief Initialize initial states.
     * @param x State vector to be initialized
     */
    void initializeStates(AmiVector& x);

    /**
     * @brief Initialize initial state sensitivities.
     * @param sx Reference to state variable sensitivities
     * @param x Reference to state variables
     */
    void initializeStateSensitivities(AmiVectorArray& sx, AmiVector const& x);

    /**
     * @brief Initialization of spline functions
     */
    void initializeSplines();

    /**
     * @brief Initialization of spline sensitivity functions
     */
    void initializeSplineSensitivities();

    /**
     * @brief Initialize the Heaviside variables `h` at the initial time `t0`.
     *
     * Heaviside variables activate/deactivate on event occurrences.
     *
     * @param x Reference to state variables
     * @param dx Reference to time derivative of states (DAE only)
     * @param roots_found boolean indicators indicating whether roots were found
     * at t0 by this fun
     */
    void initEvents(
        AmiVector const& x, AmiVector const& dx, std::vector<int>& roots_found
    );

    /**
     * @brief Get number of parameters wrt to which sensitivities are computed.
     * @return Length of sensitivity index vector
     */
    int nplist() const;

    /**
     * @brief Get total number of model parameters.
     * @return Length of parameter vector
     */
    int np() const;

    /**
     * @brief Get number of constants
     * @return Length of constant vector
     */
    int nk() const;

    /**
     * @brief Get number of conservation laws.
     * @return Number of conservation laws (i.e., difference between `nx_rdata`
     * and `nx_solver`).
     */
    int ncl() const;

    /**
     * @brief Get number of solver states subject to reinitialization.
     * @return Model member `nx_solver_reinit`
     */
    int nx_reinit() const;

    /**
     * @brief Get fixed parameters.
     * @return Pointer to constants array
     */
    double const* k() const;

    /**
     * @brief Get maximum number of events that may occur for each type.
     * @return Maximum number of events that may occur for each type
     */
    int nMaxEvent() const;

    /**
     * @brief Set maximum number of events that may occur for each type.
     * @param nmaxevent Maximum number of events that may occur for each type
     */
    void setNMaxEvent(int nmaxevent);

    /**
     * @brief Get number of timepoints.
     * @return Number of timepoints
     */
    int nt() const;

    /**
     * @brief Get parameter scale for each parameter.
     * @return Vector of parameter scales
     */
    std::vector<ParameterScaling> const& getParameterScale() const;

    /**
     * @brief Set parameter scale for each parameter.
     *
     * NOTE: Resets initial state sensitivities.
     *
     * @param pscale Scalar parameter scale to be set for all parameters
     */
    void setParameterScale(ParameterScaling pscale);

    /**
     * @brief Set parameter scale for each parameter.
     *
     * NOTE: Resets initial state sensitivities.
     *
     * @param pscaleVec Vector of parameter scales
     */
    void setParameterScale(std::vector<ParameterScaling> const& pscaleVec);

    /**
     * @brief Get parameters with transformation according to parameter scale
     * applied.
     * @return Unscaled parameters
     */
    std::vector<realtype> const& getUnscaledParameters() const;

    /**
     * @brief Get parameter vector.
     * @return The user-set parameters (see also `Model::getUnscaledParameters`)
     */
    std::vector<realtype> const& getParameters() const;

    /**
     * @brief Get value of first model parameter with the specified ID.
     * @param par_id Parameter ID
     * @return Parameter value
     */
    realtype getParameterById(std::string const& par_id) const;

    /**
     * @brief Get value of first model parameter with the specified name.
     * @param par_name Parameter name
     * @return Parameter value
     */
    realtype getParameterByName(std::string const& par_name) const;

    /**
     * @brief Set the parameter vector.
     * @param p Vector of parameters
     */
    void setParameters(std::vector<realtype> const& p);

    /**
     * @brief Set model parameters according to the parameter IDs and mapped
     * values.
     * @param p Map of parameters IDs and values
     * @param ignoreErrors Ignore errors such as parameter IDs in p which are
     * not model parameters
     */
    void setParameterById(
        std::map<std::string, realtype> const& p, bool ignoreErrors = false
    );

    /**
     * @brief Set value of first model parameter with the specified ID.
     * @param par_id Parameter ID
     * @param value Parameter value
     */
    void setParameterById(std::string const& par_id, realtype value);

    /**
     * @brief Set all values of model parameters with IDs matching the specified
     * regular expression.
     * @param par_id_regex Parameter ID regex
     * @param value Parameter value
     * @return Number of parameter IDs that matched the regex
     */
    int setParametersByIdRegex(std::string const& par_id_regex, realtype value);

    /**
     * @brief Set value of first model parameter with the specified name.
     * @param par_name Parameter name
     * @param value Parameter value
     */
    void setParameterByName(std::string const& par_name, realtype value);

    /**
     * @brief Set model parameters according to the parameter name and mapped
     * values.
     * @param p Map of parameters names and values
     * @param ignoreErrors Ignore errors such as parameter names in p which are
     * not model parameters
     */
    void setParameterByName(
        std::map<std::string, realtype> const& p, bool ignoreErrors = false
    );

    /**
     * @brief Set all values of all model parameters with names matching the
     * specified regex.
     * @param par_name_regex Parameter name regex
     * @param value Parameter value
     * @return Number of fixed parameter names that matched the regex
     */
    int
    setParametersByNameRegex(std::string const& par_name_regex, realtype value);

    /**
     * @brief Get values of fixed parameters.
     * @return Vector of fixed parameters with same ordering as in
     * Model::getFixedParameterIds
     */
    std::vector<realtype> const& getFixedParameters() const;

    /**
     * @brief Get value of fixed parameter with the specified ID.
     * @param par_id Parameter ID
     * @return Parameter value
     */
    realtype getFixedParameterById(std::string const& par_id) const;

    /**
     * @brief Get value of fixed parameter with the specified name.
     *
     * If multiple parameters have the same name, the first parameter with
     * matching name is returned.
     *
     * @param par_name Parameter name
     * @return Parameter value
     */
    realtype getFixedParameterByName(std::string const& par_name) const;

    /**
     * @brief Set values for constants.
     * @param k Vector of fixed parameters
     */
    void setFixedParameters(std::vector<realtype> const& k);

    /**
     * @brief Set value of first fixed parameter with the specified ID.
     * @param par_id Fixed parameter id
     * @param value Fixed parameter value
     */
    void setFixedParameterById(std::string const& par_id, realtype value);

    /**
     * @brief Set values of all fixed parameters with the ID matching the
     * specified regex.
     * @param par_id_regex Fixed parameter name regex
     * @param value Fixed parameter value
     * @return Number of fixed parameter IDs that matched the regex
     */
    int setFixedParametersByIdRegex(
        std::string const& par_id_regex, realtype value
    );

    /**
     * @brief Set value of first fixed parameter with the specified name.
     * @param par_name Fixed parameter ID
     * @param value Fixed parameter value
     */
    void setFixedParameterByName(std::string const& par_name, realtype value);

    /**
     * @brief Set value of all fixed parameters with name matching the specified
     * regex.
     * @param par_name_regex Fixed parameter name regex
     * @param value Fixed parameter value
     * @return Number of fixed parameter names that matched the regex
     */
    int setFixedParametersByNameRegex(
        std::string const& par_name_regex, realtype value
    );

    /**
     * @brief Get the model name.
     * @return Model name
     */
    virtual std::string getName() const;

    /**
     * @brief Report whether the model has parameter names set.
     *
     * @return Boolean indicating whether parameter names were set. Also returns
     * `true` if the number of corresponding variables is just zero.
     */
    virtual bool hasParameterNames() const;

    /**
     * @brief Get names of the model parameters.
     * @return The parameter names
     */
    virtual std::vector<std::string> getParameterNames() const;

    /**
     * @brief Report whether the model has state names set.
     *
     * @return Boolean indicating whether state names were set. Also returns
     * `true` if the number of corresponding variables is just zero.
     */
    virtual bool hasStateNames() const;

    /**
     * @brief Get names of the model states.
     * @return State names
     */
    virtual std::vector<std::string> getStateNames() const;

    /**
     * @brief Get names of the solver states.
     * @return State names
     */
    virtual std::vector<std::string> getStateNamesSolver() const;

    /**
     * @brief Report whether the model has fixed parameter names set.
     * @return Boolean indicating whether fixed parameter names were set. Also
     * returns `true` if the number of corresponding variables is just zero.
     */
    virtual bool hasFixedParameterNames() const;

    /**
     * @brief Get names of the fixed model parameters.
     * @return Fixed parameter names
     */
    virtual std::vector<std::string> getFixedParameterNames() const;

    /**
     * @brief Report whether the model has observable names set.
     * @return Boolean indicating whether observable names were set. Also
     * returns `true` if the number of corresponding variables is just zero.
     */
    virtual bool hasObservableNames() const;

    /**
     * @brief Get names of the observables.
     * @return Observable names
     */
    virtual std::vector<std::string> getObservableNames() const;

    /**
     * @brief Report whether the model has expression names set.
     * @return Boolean indicating whether expression names were set. Also
     * returns `true` if the number of corresponding variables is just zero.
     */
    virtual bool hasExpressionNames() const;

    /**
     * @brief Get names of the expressions.
     * @return Expression names
     */
    virtual std::vector<std::string> getExpressionNames() const;

    /**
     * @brief Report whether the model has parameter IDs set.
     * @return Boolean indicating whether parameter IDs were set. Also returns
     * `true` if the number of corresponding variables is just zero.
     */
    virtual bool hasParameterIds() const;

    /**
     * @brief Get IDs of the model parameters.
     * @return Parameter IDs
     */
    virtual std::vector<std::string> getParameterIds() const;

    /**
     * @brief Report whether the model has state IDs set.
     * @return Boolean indicating whether state IDs were set. Also returns
     * `true` if the number of corresponding variables is just zero.
     */
    virtual bool hasStateIds() const;

    /**
     * @brief Get IDs of the model states.
     * @return State IDs
     */
    virtual std::vector<std::string> getStateIds() const;

    /**
     * @brief Get IDs of the solver states.
     * @return State IDs
     */
    virtual std::vector<std::string> getStateIdsSolver() const;

    /**
     * @brief Report whether the model has fixed parameter IDs set.
     * @return Boolean indicating whether fixed parameter IDs were set. Also
     * returns `true` if the number of corresponding variables is just zero.
     */
    virtual bool hasFixedParameterIds() const;

    /**
     * @brief Get IDs of the fixed model parameters.
     * @return Fixed parameter IDs
     */
    virtual std::vector<std::string> getFixedParameterIds() const;

    /**
     * @brief Report whether the model has observable IDs set.
     * @return Boolean indicating whether observable ids were set. Also returns
     * `true` if the number of corresponding variables is just zero.
     */
    virtual bool hasObservableIds() const;

    /**
     * @brief Get IDs of the observables.
     * @return Observable IDs
     */
    virtual std::vector<std::string> getObservableIds() const;

    /**
     * @brief Report whether the model has expression IDs set.
     * @return Boolean indicating whether expression ids were set. Also returns
     * `true` if the number of corresponding variables is just zero.
     */
    virtual bool hasExpressionIds() const;

    /**
     * @brief Get IDs of the expression.
     * @return Expression IDs
     */
    virtual std::vector<std::string> getExpressionIds() const;

    /**
     * @brief Checks whether the defined noise model is gaussian, i.e., the nllh
     * is quadratic
     * @return boolean flag
     */
    virtual bool hasQuadraticLLH() const;

    /**
     * @brief Get the timepoint vector.
     * @return Timepoint vector
     */
    std::vector<realtype> const& getTimepoints() const;

    /**
     * @brief Get simulation timepoint for time index `it`.
     * @param it Time index
     * @return Timepoint
     */
    realtype getTimepoint(int it) const;

    /**
     * @brief Set the timepoint vector.
     * @param ts New timepoint vector
     */
    void setTimepoints(std::vector<realtype> const& ts);

    /**
     * @brief Get simulation start time.
     * @return Simulation start time
     */
    double t0() const;

    /**
     * @brief Set simulation start time.
     *
     * Output timepoints are absolute timepoints, independent of
     * \f$ t_{0} \f$.
     * For output timepoints \f$ t <  t_{0} \f$, the initial state will be
     * returned.

     * @param t0 Simulation start time
     */
    void setT0(double t0);

    /**
     * @brief Get flags indicating whether states should be treated as
     * non-negative.
     * @return Vector of flags
     */
    std::vector<bool> const& getStateIsNonNegative() const;

    /**
     * @brief Set flags indicating whether states should be treated as
     * non-negative.
     * @param stateIsNonNegative Vector of flags
     */
    void setStateIsNonNegative(std::vector<bool> const& stateIsNonNegative);

    /**
     * @brief Set flags indicating that all states should be treated as
     * non-negative.
     */
    void setAllStatesNonNegative();

    /**
     * @brief Get the current model state.
     * @return Current model state
     */
    ModelState const& getModelState() const { return state_; };

    /**
     * @brief Set the current model state.
     * @param state Model state
     */
    void setModelState(ModelState const& state) {
        if (gsl::narrow<int>(state.unscaledParameters.size()) != np())
            throw AmiException("Mismatch in parameter size");
        if (gsl::narrow<int>(state.fixedParameters.size()) != nk())
            throw AmiException("Mismatch in fixed parameter size");
        if (gsl::narrow<int>(state.h.size()) != ne)
            throw AmiException("Mismatch in Heaviside size");
        if (gsl::narrow<int>(state.total_cl.size()) != ncl())
            throw AmiException("Mismatch in conservation law size");
        if (gsl::narrow<int>(state.stotal_cl.size()) != ncl() * np())
            throw AmiException("Mismatch in conservation law sensitivity size");
        state_ = state;
    };

    /**
     * @brief Sets the estimated lower boundary for sigma_y. When
     * :meth:`setAddSigmaResiduals` is activated, this lower boundary must
     * ensure that log(sigma) + min_sigma > 0.
     * @param min_sigma lower boundary
     */
    void setMinimumSigmaResiduals(double min_sigma) { min_sigma_ = min_sigma; }

    /**
     * @brief Gets the specified estimated lower boundary for sigma_y.
     * @return lower boundary
     */
    realtype getMinimumSigmaResiduals() const { return min_sigma_; }

    /**
     * @brief Specifies whether residuals should be added to account for
     * parameter dependent sigma.
     *
     * If set to true, additional residuals of the form \f$ \sqrt{\log(\sigma) +
     * C} \f$ will be added. This enables least-squares optimization for
     * variables with Gaussian noise assumption and parameter dependent standard
     * deviation sigma. The constant \f$ C \f$ can be set via
     * :meth:`setMinimumSigmaResiduals`.
     *
     * @param sigma_res if true, additional residuals are added
     */
    void setAddSigmaResiduals(bool sigma_res) { sigma_res_ = sigma_res; }

    /**
     * @brief Checks whether residuals should be added to account for parameter
     * dependent sigma.
     * @return sigma_res
     */
    bool getAddSigmaResiduals() const { return sigma_res_; }

    /**
     * @brief Get the list of parameters for which sensitivities are computed.
     * @return List of parameter indices
     */
    std::vector<int> const& getParameterList() const;

    /**
     * @brief Get entry in parameter list by index.
     * @param pos Index in sensitivity parameter list
     * @return Index in parameter list
     */
    int plist(int pos) const;

    /**
     * @brief Set the list of parameters for which sensitivities are to be
     * computed.
     *
     * NOTE: Resets initial state sensitivities.
     *
     * @param plist List of parameter indices
     */
    void setParameterList(std::vector<int> const& plist);

    /**
     * @brief Get the initial states.
     * @return Initial state vector
     */
    std::vector<realtype> getInitialStates();

    /**
     * @brief Set the initial states.
     * @param x0 Initial state vector
     */
    void setInitialStates(std::vector<realtype> const& x0);

    /**
     * @brief Return whether custom initial states have been set.
     * @return `true` if has custom initial states, otherwise `false`
     */
    bool hasCustomInitialStates() const;

    /**
     * @brief Get the initial states sensitivities.
     * @return vector of initial state sensitivities
     */
    std::vector<realtype> getInitialStateSensitivities();

    /**
     * @brief Set the initial state sensitivities.
     * @param sx0 vector of initial state sensitivities with chainrule applied.
     * This could be a slice of ReturnData::sx or ReturnData::sx0
     */
    void setInitialStateSensitivities(std::vector<realtype> const& sx0);

    /**
     * @brief Return whether custom initial state sensitivities have been set.
     * @return `true` if has custom initial state sensitivities, otherwise
     * `false`.
     */
    bool hasCustomInitialStateSensitivities() const;

    /**
     * @brief Set the initial state sensitivities.
     * @param sx0 Vector of initial state sensitivities without chainrule
     * applied. This could be the readin from a `model.sx0data` saved to HDF5.
     */
    void setUnscaledInitialStateSensitivities(std::vector<realtype> const& sx0);

    /**
     * @brief Set the mode how steady state is computed in the steadystate
     * simulation.
     * @param mode Steadystate computation mode
     */
    void setSteadyStateComputationMode(SteadyStateComputationMode mode);

    /**
     * @brief Gets the mode how steady state is computed in the steadystate
     * simulation.
     * @return Mode
     */
    SteadyStateComputationMode getSteadyStateComputationMode() const;

    /**
     * @brief Set the mode how sensitivities are computed in the steadystate
     * simulation.
     * @param mode Steadystate sensitivity mode
     */
    void setSteadyStateSensitivityMode(SteadyStateSensitivityMode mode);

    /**
     * @brief Gets the mode how sensitivities are computed in the steadystate
     * simulation.
     * @return Mode
     */
    SteadyStateSensitivityMode getSteadyStateSensitivityMode() const;

    /**
     * @brief Set whether initial states depending on fixed parameters are to be
     * reinitialized after preequilibration and presimulation.
     * @param flag Fixed parameters reinitialized?
     */
    void setReinitializeFixedParameterInitialStates(bool flag);

    /**
     * @brief Get whether initial states depending on fixedParameters are to be
     * reinitialized after preequilibration and presimulation.
     * @return flag `true` / `false`
     */
    bool getReinitializeFixedParameterInitialStates() const;

    /**
     * @brief Require computation of sensitivities for all parameters p [0..np[
     * in natural order.
     *
     * NOTE: Resets initial state sensitivities.
     */
    void requireSensitivitiesForAllParameters();

    /**
     * @brief Get time-resolved `w`.
     * @param w Buffer (shape `nw`)
     * @param t Current timepoint
     * @param x Current state
     */
    void
    getExpression(gsl::span<realtype> w, realtype const t, AmiVector const& x);

    /**
     * @brief Get time-resolved observables.
     * @param y Buffer (shape `ny`)
     * @param t Current timepoint
     * @param x Current state
     */
    void
    getObservable(gsl::span<realtype> y, realtype const t, AmiVector const& x);

    /**
     * @brief Get scaling type for observable
     * @param iy observable index
     * @return scaling type
     */
    virtual ObservableScaling getObservableScaling(int iy) const;

    /**
     * @brief Get sensitivity of time-resolved observables.
     *
     * Total derivative \f$ sy = dydx * sx + dydp\f$
     * (only for forward sensitivities).
     * @param sy buffer (shape `ny` x `nplist`, row-major)
     * @param t Timpoint
     * @param x State variables
     * @param sx State sensitivities
     */
    void getObservableSensitivity(
        gsl::span<realtype> sy, realtype const t, AmiVector const& x,
        AmiVectorArray const& sx
    );

    /**
     * @brief Get time-resolved observable standard deviations
     * @param sigmay Buffer (shape `ny`)
     * @param it Timepoint index
     * @param edata Pointer to experimental data instance (optional, pass
     * `nullptr` to ignore)
     */
    void getObservableSigma(
        gsl::span<realtype> sigmay, int const it, ExpData const* edata
    );

    /**
     * @brief Sensitivity of time-resolved observable standard deviation.
     *
     * Total derivative (can be used with both adjoint and forward sensitivity).
     *
     * @param ssigmay Buffer (shape `ny` x `nplist`, row-major)
     * @param sy Sensitivity of time-resolved observables for current timepoint
     * @param it Timepoint index
     * @param edata Pointer to experimental data instance (optional, pass
     * `nullptr` to ignore)
     */
    void getObservableSigmaSensitivity(
        gsl::span<realtype> ssigmay, gsl::span<realtype const> sy, int const it,
        ExpData const* edata
    );

    /**
     * @brief Add time-resolved measurement negative log-likelihood \f$ Jy \f$.
     * @param Jy Buffer (shape 1)
     * @param it Timepoint index
     * @param x State variables
     * @param edata Experimental data
     */
    void addObservableObjective(
        realtype& Jy, int const it, AmiVector const& x, ExpData const& edata
    );

    /**
     * @brief Add sensitivity of time-resolved measurement negative
     * log-likelihood \f$ Jy \f$.
     *
     * @param sllh First-order buffer (shape `nplist`)
     * @param s2llh Second-order buffer (shape `nJ - 1` x `nplist`, row-major)
     * @param it Timepoint index
     * @param x State variables
     * @param sx State sensitivities
     * @param edata Experimental data
     */
    void addObservableObjectiveSensitivity(
        std::vector<realtype>& sllh, std::vector<realtype>& s2llh, int const it,
        AmiVector const& x, AmiVectorArray const& sx, ExpData const& edata
    );

    /**
     * @brief Add sensitivity of time-resolved measurement negative
     * log-likelihood \f$ Jy \f$.
     *
     * Partial derivative (to be used with adjoint sensitivities).
     *
     * @param sllh First order output buffer (shape `nplist`)
     * @param s2llh Second order output buffer
     * (shape `nJ - 1` x `nplist`, row-major)
     * @param it Timepoint index
     * @param x State variables
     * @param edata Experimental data
     */
    void addPartialObservableObjectiveSensitivity(
        std::vector<realtype>& sllh, std::vector<realtype>& s2llh, int const it,
        AmiVector const& x, ExpData const& edata
    );

    /**
     * @brief Get state sensitivity of the negative loglikelihood \f$ Jy \f$,
     * partial derivative (to be used with adjoint sensitivities).
     *
     * @param dJydx Output buffer (shape `nJ` x `nx_solver`, row-major)
     * @param it Timepoint index
     * @param x State variables
     * @param edata Experimental data instance
     */
    void getAdjointStateObservableUpdate(
        gsl::span<realtype> dJydx, int const it, AmiVector const& x,
        ExpData const& edata
    );

    /**
     * @brief Get event-resolved observables.
     * @param z Output buffer (shape `nz`)
     * @param ie Event index
     * @param t Timepoint
     * @param x State variables
     */
    void getEvent(
        gsl::span<realtype> z, int const ie, realtype const t,
        AmiVector const& x
    );
    /**
     * @brief Get sensitivities of event-resolved observables.
     *
     * Total derivative (only forward sensitivities).
     *
     * @param sz Output buffer (shape `nz x nplist`, row-major)
     * @param ie Event index
     * @param t Timepoint
     * @param x State variables
     * @param sx State sensitivities
     */
    void getEventSensitivity(
        gsl::span<realtype> sz, int const ie, realtype const t,
        AmiVector const& x, AmiVectorArray const& sx
    );

    /**
     * @brief Get sensitivity of `z` at final timepoint.
     *
     * Ignores sensitivity of timepoint. Total derivative.
     *
     * @param sz Output buffer (shape `nz x nplist`, row-major)
     * @param ie Event index
     */
    void getUnobservedEventSensitivity(gsl::span<realtype> sz, int const ie);

    /**
     * @brief Get regularization for event-resolved observables.
     * @param rz Output buffer (shape `nz`)
     * @param ie Event index
     * @param t Timepoint
     * @param x State variables
     */
    void getEventRegularization(
        gsl::span<realtype> rz, int const ie, realtype const t,
        AmiVector const& x
    );

    /**
     * @brief Get sensitivities of regularization for event-resolved
     * observables.
     *
     * Total derivative. Only forward sensitivities.
     *
     * @param srz Output buffer (shape `nz x nplist`, row-major)
     * @param ie Event index
     * @param t Timepoint
     * @param x State variables
     * @param sx State sensitivities
     */
    void getEventRegularizationSensitivity(
        gsl::span<realtype> srz, int const ie, realtype const t,
        AmiVector const& x, AmiVectorArray const& sx
    );
    /**
     * @brief Get event-resolved observable standard deviations.
     * @param sigmaz Output buffer (shape `nz`)
     * @param ie Event index
     * @param nroots Event occurrence
     * @param t Timepoint
     * @param edata Pointer to experimental data (optional, pass
     * `nullptr` to ignore)
     */
    void getEventSigma(
        gsl::span<realtype> sigmaz, int const ie, int const nroots,
        realtype const t, ExpData const* edata
    );

    /**
     * @brief Get sensitivities of event-resolved observable standard
     * deviations.
     *
     * Total derivative (only forward sensitivities).
     *
     * @param ssigmaz Output buffer (shape `nz x nplist`, row-major)
     * @param ie Event index
     * @param nroots Event occurrence
     * @param t Timepoint
     * @param edata Pointer to experimental data (optional, pass
     * `nullptr` to ignore)
     */
    void getEventSigmaSensitivity(
        gsl::span<realtype> ssigmaz, int const ie, int const nroots,
        realtype const t, ExpData const* edata
    );

    /**
     * @brief Add event-resolved observable negative log-likelihood.
     * @param Jz Output buffer (shape 1)
     * @param ie Event index
     * @param nroots Event occurrence
     * @param t Timepoint
     * @param x State variables
     * @param edata Experimental data
     */
    void addEventObjective(
        realtype& Jz, int const ie, int const nroots, realtype const t,
        AmiVector const& x, ExpData const& edata
    );

    /**
     * @brief Add event-resolved observable negative log-likelihood.
     * @param Jrz Output buffer (shape 1)
     * @param ie Event index
     * @param nroots Event occurrence
     * @param t Timepoint
     * @param x State variables
     * @param edata Experimental data
     */
    void addEventObjectiveRegularization(
        realtype& Jrz, int const ie, int const nroots, realtype const t,
        AmiVector const& x, ExpData const& edata
    );

    /**
     * @brief Add sensitivity of time-resolved measurement negative
     * log-likelihood \f$ Jy \f$.
     *
     * Total derivative (to be used with forward sensitivities).
     *
     * @param sllh First order buffer (shape `nplist`)
     * @param s2llh Second order buffer
     * (shape `nJ-1` x `nplist`, row-major)
     * @param ie Event index
     * @param nroots Event occurrence
     * @param t Timepoint
     * @param x State variables
     * @param sx State sensitivities
     * @param edata Experimental data
     */
    void addEventObjectiveSensitivity(
        std::vector<realtype>& sllh, std::vector<realtype>& s2llh, int const ie,
        int const nroots, realtype const t, AmiVector const& x,
        AmiVectorArray const& sx, ExpData const& edata
    );

    /**
     * @brief Add sensitivity of time-resolved measurement negative
     * log-likelihood \f$ Jy \f$.
     *
     * Partial derivative (to be used with adjoint sensitivities).
     *
     * @param sllh First order buffer (shape `nplist`)
     * @param s2llh Second order buffer
     * (shape `(nJ-1)` x `nplist`, row-major)
     * @param ie Event index
     * @param nroots Event occurrence
     * @param t Timepoint
     * @param x State variables
     * @param edata Experimental data
     */
    void addPartialEventObjectiveSensitivity(
        std::vector<realtype>& sllh, std::vector<realtype>& s2llh, int const ie,
        int const nroots, realtype const t, AmiVector const& x,
        ExpData const& edata
    );

    /**
     * @brief State sensitivity of the negative loglikelihood \f$ Jz \f$.
     *
     * Partial derivative (to be used with adjoint sensitivities).
     *
     * @param dJzdx Output buffer (shape `nJ` x `nx_solver`, row-major)
     * @param ie Event index
     * @param nroots Event occurrence
     * @param t Timepoint
     * @param x State variables
     * @param edata Experimental data
     */
    void getAdjointStateEventUpdate(
        gsl::span<realtype> dJzdx, int const ie, int const nroots,
        realtype const t, AmiVector const& x, ExpData const& edata
    );

    /**
     * @brief Sensitivity of event timepoint, total derivative.
     *
     * Only forward sensitivities.
     *
     * @param stau Timepoint sensitivity (shape `nplist`)
     * @param t Timepoint
     * @param ie Event index
     * @param x State variables
     * @param sx State sensitivities
     */
    void getEventTimeSensitivity(
        std::vector<realtype>& stau, realtype const t, int const ie,
        AmiVector const& x, AmiVectorArray const& sx
    );

    /**
     * @brief Update state variables after event.
     * @param x Current state (will be overwritten)
     * @param ie Event index
     * @param t Current timepoint
     * @param xdot Current residual function values
     * @param xdot_old Value of residual function before event
     */
    void addStateEventUpdate(
        AmiVector& x, int const ie, realtype const t, AmiVector const& xdot,
        AmiVector const& xdot_old
    );

    /**
     * @brief Update state sensitivity after event.
     * @param sx Current state sensitivity (will be overwritten)
     * @param ie Event index
     * @param t Current timepoint
     * @param x_old Current state
     * @param xdot Current residual function values
     * @param xdot_old Value of residual function before event
     * @param stau Timepoint sensitivity, to be computed with
     * `Model::getEventTimeSensitivity`
     */
    void addStateSensitivityEventUpdate(
        AmiVectorArray& sx, int const ie, realtype const t,
        AmiVector const& x_old, AmiVector const& xdot,
        AmiVector const& xdot_old, std::vector<realtype> const& stau
    );

    /**
     * @brief Update adjoint state after event.
     * @param xB Current adjoint state (will be overwritten)
     * @param ie Event index
     * @param t Current timepoint
     * @param x Current state
     * @param xdot Current residual function values
     * @param xdot_old Value of residual function before event
     * @param xBdot right hand side of adjoint state
     */
    void addAdjointStateEventUpdate(
        AmiVector& xB, int const ie, realtype const t, AmiVector const& x,
        AmiVector const& xdot, AmiVector const& xdot_old, AmiVector const& xBdot
    );

    /**
     * @brief Update adjoint quadratures after event.
     * @param xQB Current quadrature state (will be overwritten)
     * @param ie Event index
     * @param t Current timepoint
     * @param x Current state
     * @param xB Current adjoint state
     * @param xdot Current residual function values
     * @param xdot_old Value of residual function before event
     * @param xBdot right hand side of adjoint state
     */
    void addAdjointQuadratureEventUpdate(
        AmiVector& xQB, int const ie, realtype const t, AmiVector const& x,
        AmiVector const& xB, AmiVector const& xdot, AmiVector const& xdot_old,
        AmiVector const& xBdot
    );

    /**
     * @brief Update the Heaviside variables `h` on event occurrences.
     *
     * @param rootsfound Provides the direction of the zero-crossing, so adding
     * it will give the right update to the Heaviside variables (zero if no root
     * was found)
     */
    void updateHeaviside(std::vector<int> const& rootsfound);

    /**
     * @brief Updates the Heaviside variables `h` on event occurrences in the
     * backward problem.
     * @param rootsfound Provides the direction of the zero-crossing, so adding
     * it will give the right update to the Heaviside variables (zero if no root
     * was found)
     */
    void updateHeavisideB(int const* rootsfound);

    /**
     * @brief Check if the given array has only finite elements.
     *
     * For (1D) spans.
     *
     * @param array
     * @param model_quantity The model quantity `array` corresponds to
     * @param t Current timepoint
     * @return
     */
    int checkFinite(
        gsl::span<realtype const> array, ModelQuantity model_quantity,
        realtype t
    ) const;
    /**
     * @brief Check if the given array has only finite elements.
     *
     * For flattened 2D arrays.
     *
     * @param array Flattened matrix
     * @param model_quantity The model quantity `array` corresponds to
     * @param num_cols Number of columns of the non-flattened matrix
     * @param t Current timepoint
     * @return
     */
    int checkFinite(
        gsl::span<realtype const> array, ModelQuantity model_quantity,
        size_t num_cols, realtype t
    ) const;

    /**
     * @brief Check if the given array has only finite elements.
     *
     * For SUNMatrix.
     *
     * @param m Matrix to check
     * @param model_quantity The model quantity `m` corresponds to
     * @param t current timepoint
     * @return
     */
    int
    checkFinite(SUNMatrix m, ModelQuantity model_quantity, realtype t) const;

    /**
     * @brief Set whether the result of every call to `Model::f*` should be
     * checked for finiteness.
     * @param alwaysCheck
     */
    void setAlwaysCheckFinite(bool alwaysCheck);

    /**
     * @brief Get setting of whether the result of every call to `Model::f*`
     * should be checked for finiteness.
     * @return that
     */
    bool getAlwaysCheckFinite() const;

    /**
     * @brief Compute/get initial states.
     * @param x Output buffer.
     */
    void fx0(AmiVector& x);

    /**
     * @brief Set only those initial states that are specified via
     * fixed parameters.
     * @param x Output buffer.
     */
    void fx0_fixedParameters(AmiVector& x);

    /**
     * @brief Compute/get initial value for initial state sensitivities.
     * @param sx Output buffer for state sensitivities
     * @param x State variables
     */
    void fsx0(AmiVectorArray& sx, AmiVector const& x);

    /**
     * @brief Get only those initial states sensitivities that are affected
     * from `amici::Model::fx0_fixedParameters`.
     * @param sx Output buffer for state sensitivities
     * @param x State variables
     */
    void fsx0_fixedParameters(AmiVectorArray& sx, AmiVector const& x);

    /**
     * @brief Compute sensitivity of derivative initial states sensitivities
     * `sdx0`.
     *
     * Only necessary for DAEs.
     */
    virtual void fsdx0();

    /**
     * @brief Expand conservation law for states.
     * @param x_rdata Output buffer for state variables with conservation laws
     * expanded (stored in `amici::ReturnData`).
     * @param x_solver State variables with conservation laws applied
     * (solver returns this)
     */
    void fx_rdata(gsl::span<realtype> x_rdata, AmiVector const& x_solver);

    /**
     * @brief Expand conservation law for state sensitivities.
     * @param sx_rdata Output buffer for state variables sensitivities with
     * conservation laws expanded
     * (stored in `amici::ReturnData` shape `nplist` x `nx`, row-major).
     * @param sx_solver State variables sensitivities with conservation laws
     * applied (solver returns this)
     * @param x_solver State variables with conservation laws
     * applied (solver returns this)
     */
    void fsx_rdata(
        gsl::span<realtype> sx_rdata, AmiVectorArray const& sx_solver,
        AmiVector const& x_solver
    );

    /**
     * @brief Set indices of states to be reinitialized based on provided
     * constants / fixed parameters
     * @param idxs Array of state indices
     */
    void setReinitializationStateIdxs(std::vector<int> const& idxs);

    /**
     * @brief Return indices of states to be reinitialized based on provided
     * constants / fixed parameters
     * @return Those indices.
     */
    std::vector<int> const& getReinitializationStateIdxs() const;

    /**
     * @brief getter for dxdotdp (matlab generated)
     * @return dxdotdp
     */
    AmiVectorArray const& get_dxdotdp() const;

    /**
     * @brief getter for dxdotdp (python generated)
     * @return dxdotdp
     */
    SUNMatrixWrapper const& get_dxdotdp_full() const;

    /**
     * @brief Get trigger times for events that don't require root-finding.
     *
     * @return List of unique trigger points for events that don't require
     * root-finding (i.e. that trigger at predetermined timepoints),
     * in ascending order.
     */
    virtual std::vector<double> get_trigger_timepoints() const;

    /**
     * @brief Get steady-state mask as std::vector.
     *
     * See `set_steadystate_mask` for details.
     *
     * @return Steady-state mask
     */
    std::vector<realtype> get_steadystate_mask() const {
        return steadystate_mask_;
    };

    /**
     * @brief Set steady-state mask.
     *
     * The mask is used to exclude certain state variables from the steady-state
     * convergence check. Positive values indicate that the corresponding state
     * variable should be included in the convergence check, while non-positive
     * values indicate that the corresponding state variable should be excluded.
     * An empty mask is interpreted as including all state variables.
     *
     * @param mask Mask of length `nx_solver`.
     */
    void set_steadystate_mask(std::vector<realtype> const& mask);

    /**
     * Flag indicating whether for
     * `amici::Solver::sensi_` == `amici::SensitivityOrder::second`
     * directional or full second order derivative will be computed
     */
    SecondOrderMode o2mode{SecondOrderMode::none};

    /** Flag array for DAE equations */
    std::vector<realtype> idlist;

    /** Logger */
    Logger* logger = nullptr;

    /**
     * @brief Map of trigger timepoints to event indices for events that don't
     * require root-finding.
     */
    std::map<realtype, std::vector<int>> state_independent_events_ = {};

  protected:
    /**
     * @brief Write part of a slice to a buffer according to indices specified
     * in z2event.
     * @param slice Input data slice
     * @param buffer Output data slice
     * @param ie Event index
     */
    void writeSliceEvent(
        gsl::span<realtype const> slice, gsl::span<realtype> buffer,
        int const ie
    );

    /**
     * @brief Write part of a sensitivity slice to a buffer according to
     * indices specified in z2event.
     * @param slice source data slice
     * @param buffer output data slice
     * @param ie event index
     */
    void writeSensitivitySliceEvent(
        gsl::span<realtype const> slice, gsl::span<realtype> buffer,
        int const ie
    );

    /**
     * @brief Separate first and second order objective sensitivity information
     * and write them into the respective buffers.
     * @param dLLhdp Data with mangled first- and second-order information
     * @param sllh First order buffer
     * @param s2llh Second order buffer
     */
    void writeLLHSensitivitySlice(
        std::vector<realtype> const& dLLhdp, std::vector<realtype>& sllh,
        std::vector<realtype>& s2llh
    );

    /**
     * @brief Verify that the provided buffers have the expected size.
     * @param sllh first order buffer
     * @param s2llh second order buffer
     */
    void checkLLHBufferSize(
        std::vector<realtype> const& sllh, std::vector<realtype> const& s2llh
    ) const;

    /**
     * @brief Set the nplist-dependent vectors to their proper sizes.
     */
    void initializeVectors();

    /**
     * @brief Compute observables / measurements.
     * @param t Current timepoint
     * @param x Current state
     */
    void fy(realtype t, AmiVector const& x);

    /**
     * @brief Compute partial derivative of observables \f$ y \f$ w.r.t. model
     * parameters `p`.
     * @param t Current timepoint
     * @param x Current state
     */
    void fdydp(realtype t, AmiVector const& x);

    /**
     * @brief Compute partial derivative of observables \f$ y \f$ w.r.t. state
     * variables `x`.
     * @param t Current timepoint
     * @param x Current state
     */
    void fdydx(realtype t, AmiVector const& x);

    /**
     * @brief Compute standard deviation of measurements.
     * @param it Timepoint index
     * @param edata Experimental data
     */
    void fsigmay(int it, ExpData const* edata);

    /**
     * @brief Compute partial derivative of standard deviation of measurements
     * w.r.t. model parameters.
     * @param it Timepoint index
     * @param edata pointer to `amici::ExpData` data instance holding sigma
     * values
     */
    void fdsigmaydp(int it, ExpData const* edata);

    /**
     * @brief Compute partial derivative of standard deviation of measurements
     * w.r.t. model outputs.
     * @param it Timepoint index
     * @param edata pointer to `amici::ExpData` data instance holding sigma
     * values
     */
    void fdsigmaydy(int it, ExpData const* edata);

    /**
     * @brief Compute negative log-likelihood of measurements \f$ y \f$.
     *
     * @param Jy Variable to which llh will be added
     * @param it Timepoint index
     * @param y Simulated observable
     * @param edata Pointer to experimental data instance
     */
    void fJy(realtype& Jy, int it, AmiVector const& y, ExpData const& edata);

    /**
     * @brief Compute partial derivative of time-resolved measurement negative
     * log-likelihood \f$ Jy \f$.
     * @param it timepoint index
     * @param x state variables
     * @param edata Pointer to experimental data
     */
    void fdJydy(int it, AmiVector const& x, ExpData const& edata);

    /**
     * @brief Sensitivity of time-resolved measurement negative log-likelihood
     * Jy w.r.t. standard deviation sigma.
     * @param it timepoint index
     * @param x state variables
     * @param edata pointer to experimental data instance
     */
    void fdJydsigma(int it, AmiVector const& x, ExpData const& edata);

    /**
     * @brief Compute sensitivity of time-resolved measurement negative
     * log-likelihood \f$ Jy \f$ w.r.t. parameters for the given timepoint.
     * @param it timepoint index
     * @param x state variables
     * @param edata pointer to experimental data instance
     */
    void fdJydp(int const it, AmiVector const& x, ExpData const& edata);

    /**
     * @brief Sensitivity of time-resolved measurement negative log-likelihood
     * \f$ Jy \f$ w.r.t. state variables.
     * @param it Timepoint index
     * @param x State variables
     * @param edata Pointer to experimental data instance
     */
    void fdJydx(int const it, AmiVector const& x, ExpData const& edata);

    /**
     * @brief Compute event-resolved output.
     * @param ie Event index
     * @param t Current timepoint
     * @param x Current state
     */
    void fz(int ie, realtype t, AmiVector const& x);

    /**
     * @brief Compute partial derivative of event-resolved output `z` w.r.t.
     * model parameters `p`
     * @param ie event index
     * @param t current timepoint
     * @param x current state
     */
    void fdzdp(int ie, realtype t, AmiVector const& x);

    /**
     * @brief Compute partial derivative of event-resolved output `z` w.r.t.
     * model states `x`.
     * @param ie Event index
     * @param t Current timepoint
     * @param x Current state
     */
    void fdzdx(int ie, realtype t, AmiVector const& x);

    /**
     * @brief Compute event root function of events.
     *
     * Equal to `Model::froot` but does not include non-output events.
     *
     * @param ie Event index
     * @param t Current timepoint
     * @param x Current state
     */
    void frz(int ie, realtype t, AmiVector const& x);

    /**
     * @brief Compute sensitivity of event-resolved root output w.r.t. model
     * parameters `p`.
     * @param ie Event index
     * @param t Current timepoint
     * @param x Current state
     */
    void fdrzdp(int ie, realtype t, AmiVector const& x);

    /**
     * @brief Compute sensitivity of event-resolved measurements \f$ rz \f$
     * w.r.t. model states `x`.
     * @param ie Event index
     * @param t Current timepoint
     * @param x Current state
     */
    void fdrzdx(int ie, realtype t, AmiVector const& x);

    /**
     * @brief Compute standard deviation of events.
     * @param ie Event index
     * @param nroots Event index
     * @param t Current timepoint
     * @param edata Experimental data
     */
    void fsigmaz(
        int const ie, int const nroots, realtype const t, ExpData const* edata
    );

    /**
     * @brief Compute sensitivity of standard deviation of events measurements
     * w.r.t. model parameters `p`.
     * @param ie Event index
     * @param nroots Event occurrence
     * @param t Current timepoint
     * @param edata Pointer to experimental data instance
     */
    void fdsigmazdp(int ie, int nroots, realtype t, ExpData const* edata);

    /**
     * @brief Compute negative log-likelihood of event-resolved measurements
     * `z`.
     * @param Jz Variable to which llh will be added
     * @param nroots Event index
     * @param z Simulated event
     * @param edata Experimental data
     */
    void
    fJz(realtype& Jz, int nroots, AmiVector const& z, ExpData const& edata);

    /**
     * @brief Compute partial derivative of event measurement negative
     * log-likelihood \f$ Jz \f$.
     * @param ie Event index
     * @param nroots Event index
     * @param t Current timepoint
     * @param x State variables
     * @param edata Experimental data
     */
    void fdJzdz(
        int const ie, int const nroots, realtype const t, AmiVector const& x,
        ExpData const& edata
    );

    /**
     * @brief Compute sensitivity of event measurement negative log-likelihood
     * \f$ Jz \f$ w.r.t. standard deviation sigmaz.
     * @param ie Event index
     * @param nroots Event index
     * @param t Current timepoint
     * @param x State variables
     * @param edata Pointer to experimental data instance
     */
    void fdJzdsigma(
        int const ie, int const nroots, realtype const t, AmiVector const& x,
        ExpData const& edata
    );

    /**
     * @brief Compute sensitivity of event-resolved measurement negative
     * log-likelihood Jz w.r.t. parameters.
     * @param ie Event index
     * @param nroots Event index
     * @param t Current timepoint
     * @param x State variables
     * @param edata Pointer to experimental data instance
     */
    void fdJzdp(
        int const ie, int const nroots, realtype t, AmiVector const& x,
        ExpData const& edata
    );

    /**
     * @brief Compute sensitivity of event-resolved measurement negative
     * log-likelihood Jz w.r.t. state variables.
     * @param ie Event index
     * @param nroots Event index
     * @param t Current timepoint
     * @param x State variables
     * @param edata Experimental data
     */
    void fdJzdx(
        int const ie, int const nroots, realtype t, AmiVector const& x,
        ExpData const& edata
    );

    /**
     * @brief Compute regularization of negative log-likelihood with roots of
     * event-resolved measurements rz.
     * @param Jrz Variable to which regularization will be added
     * @param nroots Event index
     * @param rz Regularization variable
     * @param edata Experimental data
     */
    void
    fJrz(realtype& Jrz, int nroots, AmiVector const& rz, ExpData const& edata);

    /**
     * @brief Compute partial derivative of event measurement negative
     * log-likelihood J.
     * @param ie Event index
     * @param nroots Event index
     * @param t Current timepoint
     * @param x State variables
     * @param edata Experimental data
     */
    void fdJrzdz(
        int const ie, int const nroots, realtype const t, AmiVector const& x,
        ExpData const& edata
    );

    /**
     * @brief Compute sensitivity of event measurement negative log-likelihood
     * Jz w.r.t. standard deviation sigmaz
     * @param ie event index
     * @param nroots event index
     * @param t current timepoint
     * @param x state variables
     * @param edata pointer to experimental data instance
     */
    void fdJrzdsigma(
        int const ie, int const nroots, realtype const t, AmiVector const& x,
        ExpData const& edata
    );

    /**
     * @brief Spline functions
     * @param t timepoint
     */
    void fspl(realtype t);

    /**
     * @brief Parametric derivatives of splines functions
     * @param t timepoint
     */
    void fsspl(realtype t);

    /**
     * @brief Compute recurring terms in xdot.
     * @param t Timepoint
     * @param x Array with the states
     * @param include_static Whether to (re-)evaluate only dynamic expressions
     * (false) or also static expressions (true).
     * Dynamic expressions are those that depend directly or indirectly on time,
     * static expressions are those that don't.
     */
    void fw(realtype t, realtype const* x, bool include_static = true);

    /**
     * @brief Compute parameter derivative for recurring terms in xdot.
     * @param t Timepoint
     * @param x Array with the states
     * @param include_static Whether to (re-)evaluate only dynamic expressions
     * (false) or also static expressions (true).
     * Dynamic expressions are those that depend directly or indirectly on time,
     * static expressions are those that don't.
     */
    void fdwdp(realtype t, realtype const* x, bool include_static = true);

    /**
     * @brief Compute state derivative for recurring terms in xdot.
     * @param t Timepoint
     * @param x Array with the states
     * @param include_static Whether to (re-)evaluate only dynamic expressions
     * (false) or also static expressions (true).
     * Dynamic expressions are those that depend directly or indirectly on time,
     * static expressions are those that don't.
     */
    void fdwdx(realtype t, realtype const* x, bool include_static = true);

    /**
     * @brief Compute self derivative for recurring terms in xdot.
     * @param t Timepoint
     * @param x Array with the states
     * @param include_static Whether to (re-)evaluate only dynamic expressions
     * (false) or also static expressions (true).
     * Dynamic expressions are those that depend directly or indirectly on time,
     * static expressions are those that don't.
     */
    void fdwdw(realtype t, realtype const* x, bool include_static = true);

    /**
     * @brief Compute fx_rdata.
     *
     * To be implemented by derived class if applicable.
     *
     * @param x_rdata State variables with conservation laws expanded
     * @param x_solver State variables with conservation laws applied
     * @param tcl Total abundances for conservation laws
     * @param p parameter vector
     * @param k constant vector
     */
    virtual void fx_rdata(
        realtype* x_rdata, realtype const* x_solver, realtype const* tcl,
        realtype const* p, realtype const* k
    );

    /**
     * @brief Compute fsx_solver.
     *
     * To be implemented by derived class if applicable.
     *
     * @param sx_rdata State sensitivity variables with conservation laws
     * expanded
     * @param sx_solver State sensitivity variables with conservation laws
     * applied
     * @param stcl Sensitivities of total abundances for conservation laws
     * @param p parameter vector
     * @param k constant vector
     * @param x_solver State variables with conservation laws applied
     * @param tcl Total abundances for conservation laws
     * @param ip Sensitivity index
     */
    virtual void fsx_rdata(
        realtype* sx_rdata, realtype const* sx_solver, realtype const* stcl,
        realtype const* p, realtype const* k, realtype const* x_solver,
        realtype const* tcl, int const ip
    );

    /**
     * @brief Compute fx_solver.
     *
     * To be implemented by derived class if applicable.
     *
     * @param x_solver State variables with conservation laws applied
     * @param x_rdata State variables with conservation laws expanded
     */
    virtual void fx_solver(realtype* x_solver, realtype const* x_rdata);

    /**
     * @brief Compute fsx_solver.
     *
     * To be implemented by derived class if applicable.
     *
     * @param sx_rdata State sensitivity variables with conservation laws
     * expanded
     * @param sx_solver State sensitivity variables with conservation laws
     * applied
     */
    virtual void fsx_solver(realtype* sx_solver, realtype const* sx_rdata);

    /**
     * @brief Compute ftotal_cl.
     *
     * To be implemented by derived class if applicable.
     *
     * @param total_cl Total abundances of conservation laws
     * @param x_rdata State variables with conservation laws expanded
     * @param p parameter vector
     * @param k constant vector
     */
    virtual void ftotal_cl(
        realtype* total_cl, realtype const* x_rdata, realtype const* p,
        realtype const* k
    );

    /**
     * @brief Compute fstotal_cl
     *
     * To be implemented by derived class if applicable.
     *
     * @param stotal_cl Sensitivities for the total abundances of conservation
     * laws
     * @param sx_rdata State sensitivity variables with conservation laws
     * expanded
     * @param ip Sensitivity index
     * @param x_rdata State variables with conservation laws expanded
     * @param p parameter vector
     * @param k constant vector
     * @param tcl Total abundances for conservation laws
     */
    virtual void fstotal_cl(
        realtype* stotal_cl, realtype const* sx_rdata, int const ip,
        realtype const* x_rdata, realtype const* p, realtype const* k,
        realtype const* tcl
    );

    /**
     * @brief Compute non-negative state vector.
     *
     * Compute non-negative state vector according to stateIsNonNegative.
     * If anyStateNonNegative is set to `false`, i.e., all entries in
     * stateIsNonNegative are `false`, this function directly returns `x`,
     * otherwise all entries of x are copied in to `amici::Model::x_pos_tmp_`
     * and negative values are replaced by `0` where applicable.
     *
     * @param x State vector possibly containing negative values
     * @return State vector with negative values replaced by `0` according to
     * stateIsNonNegative
     */
    const_N_Vector computeX_pos(const_N_Vector x);

    /**
     * @brief Compute non-negative state vector.
     *
     * Compute non-negative state vector according to stateIsNonNegative.
     * If anyStateNonNegative is set to `false`, i.e., all entries in
     * stateIsNonNegative are `false`, this function directly returns `x`,
     * otherwise all entries of x are copied in to `amici::Model::x_pos_tmp_`
     * and negative values are replaced by `0` where applicable.
     *
     * @param x State vector possibly containing negative values
     * @return State vector with negative values replaced by `0` according to
     * stateIsNonNegative
     */
    realtype const* computeX_pos(AmiVector const& x);

    /** All variables necessary for function evaluation */
    ModelState state_;

    /**
     * Storage for model quantities beyond ModelState for the current timepoint
     */
    ModelStateDerived derived_state_;

    /** Storage for splines of the model */
    std::vector<HermiteSpline> splines_;

    /** index indicating to which event an event output belongs */
    std::vector<int> z2event_;

    /** state initialization (size nx_solver) */
    std::vector<realtype> x0data_;

    /** sensitivity initialization (size nx_rdata x nplist, row-major) */
    std::vector<realtype> sx0data_;

    /** vector of bools indicating whether state variables are to be assumed to
     * be positive */
    std::vector<bool> state_is_non_negative_;

    /** Vector of booleans indicating the initial boolean value for every event
     * trigger function. Events at t0 can only trigger if the initial value is
     * set to `false`. Must be specified during model compilation by setting the
     * `initialValue` attribute of an event trigger. */
    std::vector<bool> root_initial_values_;

    /** boolean indicating whether any entry in stateIsNonNegative is `true` */
    bool any_state_non_negative_{false};

    /** maximal number of events to track */
    int nmaxevent_{10};

    /** method for steady-state computation */
    SteadyStateComputationMode steadystate_computation_mode_{
        SteadyStateComputationMode::integrateIfNewtonFails
    };

    /** method for steadystate sensitivities computation */
    SteadyStateSensitivityMode steadystate_sensitivity_mode_{
        SteadyStateSensitivityMode::integrateIfNewtonFails
    };

    /**
     * Indicates whether the result of every call to `Model::f*` should be
     * checked for finiteness
     */
#ifdef NDEBUG
    bool always_check_finite_{false};
#else
    bool always_check_finite_{true};
#endif

    /** indicates whether sigma residuals are to be added for every datapoint */
    bool sigma_res_{false};

    /** offset to ensure positivity of sigma residuals, only has an effect when
     * `sigma_res_` is `true`  */
    realtype min_sigma_{50.0};

  private:
    /** Simulation parameters, initial state, etc. */
    SimulationParameters simulation_parameters_;

    /**
     * Mask for state variables that should be checked for steady state
     * during pre-/post-equilibration. Positive values indicate that the
     * corresponding state variable should be checked for steady state.
     * Negative values indicate that the corresponding state variable should
     * be ignored.
     */
    std::vector<realtype> steadystate_mask_;
};

bool operator==(Model const& a, Model const& b);
bool operator==(ModelDimensions const& a, ModelDimensions const& b);

} // namespace amici

#endif // AMICI_MODEL_H
