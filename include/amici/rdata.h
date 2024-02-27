#ifndef AMICI_RDATA_H
#define AMICI_RDATA_H

#include "amici/defines.h"
#include "amici/logging.h"
#include "amici/misc.h"
#include "amici/model.h"
#include "amici/vector.h"

#include <vector>

namespace amici {
class ReturnData;
class Solver;
class ExpData;
class ForwardProblem;
class BackwardProblem;
class SteadystateProblem;
} // namespace amici

namespace boost {
namespace serialization {
template <class Archive>
void serialize(Archive& ar, amici::ReturnData& r, unsigned int version);
}
} // namespace boost

namespace amici {

/**
 * @brief Stores all data to be returned by amici::runAmiciSimulation.
 *
 * NOTE: multi-dimensional arrays are stored in row-major order (C-style)
 */
class ReturnData : public ModelDimensions {
  public:
    /**
     * @brief Default constructor
     */
    ReturnData() = default;

    /**
     * @brief Constructor
     * @param ts see amici::SimulationParameters::ts
     * @param model_dimensions Model dimensions
     * @param nplist see amici::ModelDimensions::nplist
     * @param nmaxevent see amici::ModelDimensions::nmaxevent
     * @param nt see amici::ModelDimensions::nt
     * @param newton_maxsteps see amici::Solver::newton_maxsteps
     * @param pscale see amici::SimulationParameters::pscale
     * @param o2mode see amici::SimulationParameters::o2mode
     * @param sensi see amici::Solver::sensi
     * @param sensi_meth see amici::Solver::sensi_meth
     * @param rdrm see amici::Solver::rdata_reporting
     * @param quadratic_llh whether model defines a quadratic nllh and
     * computing res, sres and FIM makes sense
     * @param sigma_res indicates whether additional residuals are to be added
     * for each sigma
     * @param sigma_offset offset to ensure real-valuedness of sigma residuals
     */
    ReturnData(
        std::vector<realtype> ts, ModelDimensions const& model_dimensions,
        int nplist, int nmaxevent, int nt, int newton_maxsteps,
        std::vector<ParameterScaling> pscale, SecondOrderMode o2mode,
        SensitivityOrder sensi, SensitivityMethod sensi_meth,
        RDataReporting rdrm, bool quadratic_llh, bool sigma_res,
        realtype sigma_offset
    );

    /**
     * @brief constructor that uses information from model and solver to
     * appropriately initialize fields
     * @param solver solver instance
     * @param model model instance
     */
    ReturnData(Solver const& solver, Model const& model);

    ~ReturnData() = default;

    /**
     * @brief constructor that uses information from model and solver to
     * appropriately initialize fields
     * @param preeq simulated preequilibration problem, pass `nullptr` to ignore
     * @param fwd simulated forward problem, pass `nullptr` to ignore
     * @param bwd simulated backward problem, pass `nullptr` to ignore
     * @param posteq simulated postequilibration problem, pass `nullptr` to
     * ignore
     * @param model matching model instance
     * @param solver matching solver instance
     * @param edata matching experimental data
     */
    void processSimulationObjects(
        SteadystateProblem const* preeq, ForwardProblem const* fwd,
        BackwardProblem const* bwd, SteadystateProblem const* posteq,
        Model& model, Solver const& solver, ExpData const* edata
    );
    /**
     * @brief Arbitrary (not necessarily unique) identifier.
     */
    std::string id;

    /**
     * timepoints (shape `nt`)
     */
    std::vector<realtype> ts;

    /** time derivative (shape `nx`) */
    std::vector<realtype> xdot;

    /**
     * Jacobian of differential equation right hand side (shape `nx` x `nx`,
     * row-major)
     */
    std::vector<realtype> J;

    /**
     * w data from the model (recurring terms in xdot, for imported SBML models
     * from python, this contains the flux vector) (shape `nt` x `nw`, row
     * major)
     */
    std::vector<realtype> w;

    /** event output (shape `nmaxevent` x `nz`, row-major) */
    std::vector<realtype> z;

    /**
     * event output sigma standard deviation (shape `nmaxevent` x `nz`,
     * row-major)
     */
    std::vector<realtype> sigmaz;

    /**
     * parameter derivative of event output
     * (shape `nmaxevent` x `nplist` x `nz`, row-major)
     */
    std::vector<realtype> sz;

    /**
     * parameter derivative of event output standard deviation
     * (shape `nmaxevent` x `nplist` x `nz`, row-major)
     */
    std::vector<realtype> ssigmaz;

    /** event trigger output (shape `nmaxevent` x `nz`, row-major)*/
    std::vector<realtype> rz;

    /**
     * parameter derivative of event trigger output
     * (shape `nmaxevent` x `nplist` x `nz`, row-major)
     */
    std::vector<realtype> srz;

    /**
     * second-order parameter derivative of event trigger output (shape
     * `nmaxevent` x `nztrue` x `nplist` x `nplist`, row-major)
     */
    std::vector<realtype> s2rz;

    /** state (shape `nt` x `nx`, row-major) */
    std::vector<realtype> x;

    /**
     * parameter derivative of state (shape `nt` x `nplist` x `nx`, row-major)
     */
    std::vector<realtype> sx;

    /** observable (shape `nt` x `ny`, row-major) */
    std::vector<realtype> y;

    /** observable standard deviation (shape `nt` x `ny`, row-major) */
    std::vector<realtype> sigmay;

    /**
     * parameter derivative of observable (shape `nt` x `nplist` x `ny`,
     * row-major)
     */
    std::vector<realtype> sy;

    /**
     * parameter derivative of observable standard deviation
     * (shape `nt` x `nplist` x `ny`, row-major)
     */
    std::vector<realtype> ssigmay;

    /** observable (shape `nt*ny`, row-major) */
    std::vector<realtype> res;

    /**
     * parameter derivative of residual (shape `nt*ny` x `nplist`, row-major)
     */
    std::vector<realtype> sres;

    /** fisher information matrix (shape `nplist` x `nplist`, row-major) */
    std::vector<realtype> FIM;

    /** number of integration steps forward problem (shape `nt`) */
    std::vector<int> numsteps;

    /** number of integration steps backward problem (shape `nt`) */
    std::vector<int> numstepsB;

    /** number of right hand side evaluations forward problem (shape `nt`) */
    std::vector<int> numrhsevals;

    /** number of right hand side evaluations backward problem (shape `nt`) */
    std::vector<int> numrhsevalsB;

    /** number of error test failures forward problem (shape `nt`) */
    std::vector<int> numerrtestfails;

    /** number of error test failures backward problem (shape `nt`) */
    std::vector<int> numerrtestfailsB;

    /**
     * number of linear solver convergence failures forward problem (shape `nt`)
     */
    std::vector<int> numnonlinsolvconvfails;

    /**
     * number of linear solver convergence failures backward problem (shape
     * `nt`)
     */
    std::vector<int> numnonlinsolvconvfailsB;

    /** employed order forward problem (shape `nt`) */
    std::vector<int> order;

    /**
     * @brief computation time of forward solve [ms]
     *
     * .. warning::
     *      If AMICI was built without boost, this tracks the CPU-time of the
     *      current process. Therefore, in a multi-threaded context, this value
     *      may be incorrect.
     *
     */
    double cpu_time = 0.0;

    /**
     * @brief computation time of backward solve [ms]
     *
     * .. warning::
     *      If AMICI was built without boost, this tracks the CPU-time of the
     *      current process. Therefore, in a multi-threaded context, this value
     *      may be incorrect.
     *
     */
    double cpu_timeB = 0.0;

    /**
     * @brief total CPU time from entering runAmiciSimulation until exiting [ms]
     *
     * .. warning::
     *      If AMICI was built without boost, this tracks the CPU-time of the
     *      current process. Therefore, in a multi-threaded context, this value
     *      may be incorrect.
     *
     */
    double cpu_time_total = 0.0;

    /** flags indicating success of steady state solver (preequilibration) */
    std::vector<SteadyStateStatus> preeq_status;

    /**
     * @brief computation time of the steady state solver [ms]
     * (preequilibration)
     *
     * .. warning::
     *      If AMICI was built without boost, this tracks the CPU-time of the
     *      current process. Therefore, in a multi-threaded context, this value
     *      may be incorrect.
     *
     */
    double preeq_cpu_time = 0.0;

    /**
     * @brief computation time of the steady state solver of the backward
     * problem [ms] (preequilibration)
     *
     * .. warning::
     *      If AMICI was built without boost, this tracks the CPU-time of the
     *      current process. Therefore, in a multi-threaded context, this value
     *      may be incorrect.
     *
     */
    double preeq_cpu_timeB = 0.0;

    /** flags indicating success of steady state solver  (postequilibration) */
    std::vector<SteadyStateStatus> posteq_status;

    /**
     * @brief computation time of the steady state solver [ms]
     * (postequilibration)
     *
     * .. warning::
     *      If AMICI was built without boost, this tracks the CPU-time of the
     *      current process. Therefore, in a multi-threaded context, this value
     *      may be incorrect.
     *
     */
    double posteq_cpu_time = 0.0;

    /**
     * @brief computation time of the steady state solver of the backward
     * problem [ms] (postequilibration)
     *
     * .. warning::
     *      If AMICI was built without boost, this tracks the CPU-time of the
     *      current process. Therefore, in a multi-threaded context, this value
     *      may be incorrect.
     *
     */
    double posteq_cpu_timeB = 0.0;

    /**
     * number of Newton steps for steady state problem (preequilibration)
     * [newton, simulation, newton] (length = 3)
     */
    std::vector<int> preeq_numsteps;

    /**
     * number of simulation steps for adjoint steady state problem
     * (preequilibration) [== 0 if analytical solution worked, > 0 otherwise]
     */
    int preeq_numstepsB = 0;

    /**
     * number of Newton steps for steady state problem (preequilibration)
     * [newton, simulation, newton] (shape `3`) (postequilibration)
     */
    std::vector<int> posteq_numsteps;

    /**
     * number of simulation steps for adjoint steady state problem
     * (postequilibration) [== 0 if analytical solution worked, > 0 otherwise]
     */
    int posteq_numstepsB = 0;

    /**
     * time when steadystate was reached via simulation (preequilibration)
     */
    realtype preeq_t = NAN;

    /**
     * weighted root-mean-square of the rhs when steadystate
     * was reached (preequilibration)
     */
    realtype preeq_wrms = NAN;

    /**
     * time when steadystate was reached via simulation (postequilibration)
     */
    realtype posteq_t = NAN;

    /**
     * weighted root-mean-square of the rhs when steadystate
     * was reached (postequilibration)
     */
    realtype posteq_wrms = NAN;

    /** initial state (shape `nx`) */
    std::vector<realtype> x0;

    /** preequilibration steady state (shape `nx`) */
    std::vector<realtype> x_ss;

    /** initial sensitivities (shape `nplist` x `nx`, row-major) */
    std::vector<realtype> sx0;

    /**
     * preequilibration sensitivities
     * (shape `nplist` x `nx`, row-major)
     */
    std::vector<realtype> sx_ss;

    /** log-likelihood value */
    realtype llh = 0.0;

    /** \f$\chi^2\f$ value */
    realtype chi2 = 0.0;

    /** parameter derivative of log-likelihood (shape `nplist`) */
    std::vector<realtype> sllh;

    /**
     * second-order parameter derivative of log-likelihood
     * (shape `nJ-1` x `nplist`, row-major)
     */
    std::vector<realtype> s2llh;

    /**
     * @brief Simulation status code.
     *
     * One of:
     *
     * * AMICI_SUCCESS, indicating successful simulation
     * * AMICI_MAX_TIME_EXCEEDED, indicating that the simulation did not finish
     *   within the allowed time (see Solver.{set,get}MaxTime)
     * * AMICI_ERROR, indicating that some error occurred during simulation
     *   (a more detailed error message will have been printed).
     * * AMICI_NOT_RUN, if no simulation was started
     */
    int status = AMICI_NOT_RUN;

    /** number of states (alias `nx_rdata`, kept for backward compatibility) */
    int nx{0};

    /**
     * number of states in the unaugmented system
     * (alias nxtrue_rdata, kept for backward compatibility)
     */
    int nxtrue{0};

    /** number of parameter for which sensitivities were requested */
    int nplist{0};

    /** maximal number of occurring events (for every event type) */
    int nmaxevent{0};

    /** number of considered timepoints */
    int nt{0};

    /** maximal number of newton iterations for steady state calculation */
    int newton_maxsteps{0};

    /** scaling of parameterization */
    std::vector<ParameterScaling> pscale;

    /** flag indicating whether second-order sensitivities were requested */
    SecondOrderMode o2mode{SecondOrderMode::none};

    /** sensitivity order */
    SensitivityOrder sensi{SensitivityOrder::none};

    /** sensitivity method */
    SensitivityMethod sensi_meth{SensitivityMethod::none};

    /** reporting mode */
    RDataReporting rdata_reporting{RDataReporting::full};

    /**
     * @brief Serialize ReturnData (see boost::serialization::serialize)
     * @param ar Archive to serialize to
     * @param r Data to serialize
     * @param version Version number
     */
    template <class Archive>
    friend void boost::serialization::serialize(
        Archive& ar, ReturnData& r, unsigned int version
    );

    /** boolean indicating whether residuals for standard deviations have been
     * added */
    bool sigma_res;

    /** log messages */
    std::vector<LogItem> messages;

  protected:
    /** offset for sigma_residuals */
    realtype sigma_offset;

    /** timepoint for model evaluation*/
    realtype t_;

    /** partial state vector, excluding states eliminated from conservation laws
     */
    AmiVector x_solver_;

    /** partial time derivative of state vector, excluding states eliminated
     * from conservation laws */
    AmiVector dx_solver_;

    /** partial sensitivity state vector array, excluding states eliminated from
     * conservation laws */
    AmiVectorArray sx_solver_;

    /** full state vector, including states eliminated from conservation laws */
    AmiVector x_rdata_;

    /** full sensitivity state vector array, including states eliminated from
     * conservation laws */
    AmiVectorArray sx_rdata_;

    /** array of number of found roots for a certain event type
     * (shape `ne`) */
    std::vector<int> nroots_;

    /**
     * @brief initializes storage for likelihood reporting mode
     * @param quadratic_llh whether model defines a quadratic nllh and computing
     * res, sres and FIM makes sense.
     */
    void initializeLikelihoodReporting(bool quadratic_llh);

    /**
     * @brief initializes storage for residual reporting mode
     * @param enable_res whether residuals are to be computed
     */
    void initializeResidualReporting(bool enable_res);

    /**
     * @brief initializes storage for full reporting mode
     * @param enable_fim whether FIM Hessian approximation is to be computed
     */
    void initializeFullReporting(bool enable_fim);

    /**
     * @brief initialize values for chi2 and llh and derivatives
     * @param enable_chi2 whether chi2 values are to be computed
     */
    void initializeObjectiveFunction(bool enable_chi2);

    /**
     * @brief extracts data from a preequilibration SteadystateProblem
     * @param preeq SteadystateProblem for preequilibration
     * @param model Model instance to compute return values
     */
    void processPreEquilibration(SteadystateProblem const& preeq, Model& model);

    /**
     * @brief extracts data from a preequilibration SteadystateProblem
     * @param posteq SteadystateProblem for postequilibration
     * @param model Model instance to compute return values
     * @param edata ExpData instance containing observable data
     */
    void processPostEquilibration(
        SteadystateProblem const& posteq, Model& model, ExpData const* edata
    );

    /**
     * @brief extracts results from forward problem
     * @param fwd forward problem
     * @param model model that was used for forward simulation
     * @param edata ExpData instance containing observable data
     */
    void processForwardProblem(
        ForwardProblem const& fwd, Model& model, ExpData const* edata
    );

    /**
     * @brief extracts results from backward problem
     * @param fwd forward problem
     * @param bwd backward problem
     * @param preeq SteadystateProblem for preequilibration
     * @param model model that was used for forward/backward simulation
     */
    void processBackwardProblem(
        ForwardProblem const& fwd, BackwardProblem const& bwd,
        SteadystateProblem const* preeq, Model& model
    );

    /**
     * @brief extracts results from solver
     * @param solver solver that was used for forward/backward simulation
     */
    void processSolver(Solver const& solver);

    /**
     * @brief Evaluates and stores the Jacobian and right hand side at final
     * timepoint
     * @param problem forward problem or steadystate problem
     * @param model model that was used for forward/backward simulation
     */
    template <class T>
    void
    storeJacobianAndDerivativeInReturnData(T const& problem, Model& model) {
        readSimulationState(problem.getFinalSimulationState(), model);

        AmiVector xdot(nx_solver);
        if (!this->xdot.empty() || !this->J.empty())
            model.fxdot(t_, x_solver_, dx_solver_, xdot);

        if (!this->xdot.empty())
            writeSlice(xdot, this->xdot);

        if (!this->J.empty()) {
            SUNMatrixWrapper J(nx_solver, nx_solver);
            model.fJ(t_, 0.0, x_solver_, dx_solver_, xdot, J);
            // CVODES uses colmajor, so we need to transform to rowmajor
            for (int ix = 0; ix < model.nx_solver; ix++)
                for (int jx = 0; jx < model.nx_solver; jx++)
                    this->J.at(ix * model.nx_solver + jx)
                        = J.data()[ix + model.nx_solver * jx];
        }
    }
    /**
     * @brief sets member variables and model state according to provided
     * simulation state
     * @param state simulation state provided by Problem
     * @param model model that was used for forward/backward simulation
     */
    void readSimulationState(SimulationState const& state, Model& model);

    /**
     * @brief Residual function
     * @param it time index
     * @param model model that was used for forward/backward simulation
     * @param edata ExpData instance containing observable data
     */
    void fres(int it, Model& model, ExpData const& edata);

    /**
     * @brief Chi-squared function
     * @param it time index
     * @param edata ExpData instance containing observable data
     */
    void fchi2(int it, ExpData const& edata);

    /**
     * @brief Residual sensitivity function
     * @param it time index
     * @param model model that was used for forward/backward simulation
     * @param edata ExpData instance containing observable data
     */
    void fsres(int it, Model& model, ExpData const& edata);

    /**
     * @brief Fisher information matrix function
     * @param it time index
     * @param model model that was used for forward/backward simulation
     * @param edata ExpData instance containing observable data
     */
    void fFIM(int it, Model& model, ExpData const& edata);

    /**
     * @brief Set likelihood, state variables, outputs and respective
     * sensitivities to NaN (typically after integration failure)
     * @param it_start time index at which to start invalidating
     */
    void invalidate(int it_start);

    /**
     * @brief Set likelihood and chi2 to NaN
     * (typically after integration failure)
     */
    void invalidateLLH();

    /**
     * @brief Set likelihood sensitivities to NaN
     * (typically after integration failure)
     */
    void invalidateSLLH();

    /**
     * @brief applies the chain rule to account for parameter transformation in
     * the sensitivities of simulation results
     * @param model Model from which the ReturnData was obtained
     */
    void applyChainRuleFactorToSimulationResults(Model const& model);

    /**
     * @brief Checks whether forward sensitivity analysis is performed
     * @return boolean indicator
     */
    bool computingFSA() const {
        return (
            sensi_meth == SensitivityMethod::forward
            && sensi >= SensitivityOrder::first
        );
    }

    /**
     * @brief Extracts output information for data-points, expects that
     * x_solver_ and sx_solver_ were set appropriately
     * @param it timepoint index
     * @param model model that was used in forward solve
     * @param edata ExpData instance carrying experimental data
     */
    void getDataOutput(int it, Model& model, ExpData const* edata);

    /**
     * @brief Extracts data information for forward sensitivity analysis,
     * expects that x_solver_ and sx_solver_ were set appropriately
     * @param it index of current timepoint
     * @param model model that was used in forward solve
     * @param edata ExpData instance carrying experimental data
     */
    void getDataSensisFSA(int it, Model& model, ExpData const* edata);

    /**
     * @brief Extracts output information for events, expects that x_solver_
     * and sx_solver_ were set appropriately
     * @param t event timepoint
     * @param rootidx information about which roots fired
     * (1 indicating fired, 0/-1 for not)
     * @param model model that was used in forward solve
     * @param edata ExpData instance carrying experimental data
     */
    void getEventOutput(
        realtype t, std::vector<int> const rootidx, Model& model,
        ExpData const* edata
    );

    /**
     * @brief Extracts event information for forward sensitivity analysis,
     * expects that x_solver_ and sx_solver_ were set appropriately
     * @param ie index of event type
     * @param t event timepoint
     * @param model model that was used in forward solve
     * @param edata ExpData instance carrying experimental data
     */
    void
    getEventSensisFSA(int ie, realtype t, Model& model, ExpData const* edata);

    /**
     * @brief Updates contribution to likelihood from quadratures (xQB),
     * if preequilibration was run in adjoint mode
     * @param model model that was used for forward/backward simulation
     * @param preeq SteadystateProblem for preequilibration
     * @param llhS0 contribution to likelihood for initial state sensitivities
     * of preequilibration
     * @param xQB vector with quadratures from adjoint computation
     */
    void handleSx0Backward(
        Model const& model, SteadystateProblem const& preeq,
        std::vector<realtype>& llhS0, AmiVector& xQB
    ) const;

    /**
     * @brief Updates contribution to likelihood for initial state sensitivities
     * (llhS0), if no preequilibration was run or if forward sensitivities were
     * used
     * @param model model that was used for forward/backward simulation
     * @param llhS0 contribution to likelihood for initial state sensitivities
     * @param xB vector with final adjoint state
     * (excluding conservation laws)
     */
    void handleSx0Forward(
        Model const& model, std::vector<realtype>& llhS0, AmiVector& xB
    ) const;
};

/**
 * @brief The ModelContext temporarily stores amici::Model::state
 * and restores it when going out of scope
 */
class ModelContext : public ContextManager {
  public:
    /**
     * @brief initialize backup of the original values.
     *
     * @param model
     */
    explicit ModelContext(Model* model);

    ModelContext& operator=(ModelContext const& other) = delete;

    ~ModelContext();

    /**
     * @brief Restore original state on constructor-supplied amici::Model.
     * Will be called during destruction. Explicit call is generally not
     * necessary.
     */
    void restore();

  private:
    Model* model_{nullptr};
    ModelState original_state_;
};

} // namespace amici

#endif /* _MY_RDATA */
