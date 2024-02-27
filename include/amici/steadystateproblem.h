#ifndef AMICI_STEADYSTATEPROBLEM_H
#define AMICI_STEADYSTATEPROBLEM_H

#include <amici/defines.h>
#include <amici/model_state.h>
#include <amici/newton_solver.h>
#include <amici/vector.h>

#include <nvector/nvector_serial.h>

#include <memory>

namespace amici {

class ExpData;
class Solver;
class Model;

/**
 * @brief The SteadystateProblem class solves a steady-state problem using
 * Newton's method and falls back to integration on failure.
 */
class SteadystateProblem {
  public:
    /**
     * @brief constructor
     * @param solver Solver instance
     * @param model Model instance
     */
    explicit SteadystateProblem(Solver const& solver, Model const& model);

    /**
     * @brief Handles steady state computation in the forward case:
     * tries to determine the steady state of the ODE system and computes
     * steady state sensitivities if requested.
     * @param solver pointer to the solver object
     * @param model pointer to the model object
     * @param it integer with the index of the current time step
     */
    void workSteadyStateProblem(Solver const& solver, Model& model, int it);

    /**
     * Integrates over the adjoint state backward in time by solving a linear
     * system of equations, which gives the analytical solution.
     * Computes the gradient via adjoint steady state sensitivities
     * @param solver pointer to the solver object
     * @param model pointer to the model object
     * @param bwd backward problem
     */
    void workSteadyStateBackwardProblem(
        Solver const& solver, Model& model, BackwardProblem const* bwd
    );

    /**
     * @brief Returns the stored SimulationState
     * @return stored SimulationState
     */
    SimulationState const& getFinalSimulationState() const { return state_; };

    /**
     * @brief Returns the quadratures from pre- or postequilibration
     * @return xQB Vector with quadratures
     */
    AmiVector const& getEquilibrationQuadratures() const { return xQB_; }
    /**
     * @brief Returns state at steadystate
     * @return x
     */
    AmiVector const& getState() const { return state_.x; };

    /**
     * @brief Returns state sensitivity at steadystate
     * @return sx
     */
    AmiVectorArray const& getStateSensitivity() const { return state_.sx; };

    /**
     * @brief Accessor for dJydx
     * @return dJydx
     */
    std::vector<realtype> const& getDJydx() const { return dJydx_; }

    /**
     * @brief Accessor for run_time of the forward problem
     * @return run_time
     */
    double getCPUTime() const { return cpu_time_; }

    /**
     * @brief Accessor for run_time of the backward problem
     * @return run_time
     */
    double getCPUTimeB() const { return cpu_timeB_; }

    /**
     * @brief Accessor for steady_state_status
     * @return steady_state_status
     */
    std::vector<SteadyStateStatus> const& getSteadyStateStatus() const {
        return steady_state_status_;
    }

    /**
     * @brief Get model time at which steadystate was found through simulation
     * @return t
     */
    realtype getSteadyStateTime() const { return state_.t; }

    /**
     * @brief Accessor for wrms
     * @return wrms
     */
    realtype getResidualNorm() const { return wrms_; }

    /**
     * @brief Accessor for numsteps
     * @return numsteps
     */
    std::vector<int> const& getNumSteps() const { return numsteps_; }

    /**
     * @brief Accessor for numstepsB
     * @return numstepsB
     */
    int getNumStepsB() const { return numstepsB_; }

    /**
     * @brief computes adjoint updates dJydx according to provided model and
     * expdata
     * @param model Model instance
     * @param edata experimental data
     */
    void getAdjointUpdates(Model& model, ExpData const& edata);

    /**
     * @brief Return the adjoint state
     * @return xB adjoint state
     */
    AmiVector const& getAdjointState() const { return xB_; }

    /**
     * @brief Accessor for xQB
     * @return xQB
     */
    AmiVector const& getAdjointQuadrature() const { return xQB_; }

    /**
     * @brief Accessor for hasQuadrature_
     * @return hasQuadrature_
     */
    bool hasQuadrature() const { return hasQuadrature_; }

    /**
     * @brief computes adjoint updates dJydx according to provided model and
     * expdata
     * @return convergence of steady state solver
     */
    bool checkSteadyStateSuccess() const;

  private:
    /**
     * @brief Handles the computation of the steady state, throws an
     * AmiException, if no steady state was found
     * @param solver pointer to the solver object
     * @param model pointer to the model object
     * @param it integer with the index of the current time step
     */
    void findSteadyState(Solver const& solver, Model& model, int it);

    /**
     * @brief Tries to determine the steady state by using Newton's method
     * @param model pointer to the model object
     * @param newton_retry bool flag indicating whether being relaunched
     */
    void findSteadyStateByNewtonsMethod(Model& model, bool newton_retry);

    /**
     * @brief Tries to determine the steady state by using forward simulation
     * @param solver pointer to the solver object
     * @param model pointer to the model object
     * @param it integer with the index of the current time step
     */
    void
    findSteadyStateBySimulation(Solver const& solver, Model& model, int it);

    /**
     * @brief Handles the computation of quadratures in adjoint mode
     * @param solver pointer to the solver object
     * @param model pointer to the model object
     */
    void computeSteadyStateQuadrature(Solver const& solver, Model& model);

    /**
     * @brief Computes the quadrature in steady state backward mode by
     * solving the linear system defined by the backward Jacobian
     * @param model pointer to the model object
     */
    void getQuadratureByLinSolve(Model& model);

    /**
     * @brief Computes the quadrature in steady state backward mode by
     * numerical integration of xB forward in time
     * @param solver pointer to the solver object
     * @param model pointer to the model object
     */
    void getQuadratureBySimulation(Solver const& solver, Model& model);

    /**
     * @brief Stores state and throws an exception if equilibration failed
     * @param tried_newton_1 Whether any Newton step was attempted before
     * simulation
     * @param tried_simulation Whether simulation was attempted
     * @param tried_newton_2 Whether any Newton step was attempted after
     * simulation
     */
    [[noreturn]] void handleSteadyStateFailure(
        bool tried_newton_1, bool tried_simulation, bool tried_newton_2
    );

    /**
     * @brief Assembles the error message to be thrown.
     * @param errorString const pointer to string with error message
     * @param status Entry of steady_state_status to be processed
     */
    void
    writeErrorString(std::string* errorString, SteadyStateStatus status) const;

    /**
     * @brief Checks depending on the status of the Newton solver,
     * solver settings, and the model, whether state sensitivities
     * still need to be computed via a linear system solve or stored
     * @param model pointer to the model object
     * @param solver pointer to the solver object
     * @param it integer with the index of the current time step
     * @param context SteadyStateContext giving the situation for the flag
     * @return flag telling how to process state sensitivities
     */
    bool getSensitivityFlag(
        Model const& model, Solver const& solver, int it,
        SteadyStateContext context
    );

    /**
     * @brief Computes the weighted root mean square of xdot
     * the weights are computed according to x:
     * w_i = 1 / ( rtol * x_i + atol )
     * @param x current state (sx[ip] for sensitivities)
     * @param xdot current rhs (sxdot[ip] for sensitivities)
     * @param atol absolute tolerance
     * @param rtol relative tolerance
     * @param ewt error weight vector
     * @return root-mean-square norm
     */
    realtype getWrmsNorm(
        AmiVector const& x, AmiVector const& xdot, realtype atol, realtype rtol,
        AmiVector& ewt
    ) const;

    /**
     * @brief Checks convergence for state or adjoint quadratures, depending on
     * sensi method
     * @param model Model instance
     * @param sensi_method sensitivity method
     * @return weighted root mean squared residuals of the RHS
     */
    realtype getWrms(Model& model, SensitivityMethod sensi_method);

    /**
     * @brief Checks convergence for state sensitivities
     * @param model Model instance
     * @return weighted root mean squared residuals of the RHS
     */
    realtype getWrmsFSA(Model& model);

    /**
     * @brief Runs the Newton solver iterations and checks for convergence
     * to steady state
     * @param model pointer to the model object
     * @param newton_retry flag indicating if Newton solver is rerun
     */
    void applyNewtonsMethod(Model& model, bool newton_retry);

    /**
     * @brief Simulation is launched, if Newton solver or linear system solve
     * fails
     * @param solver pointer to the solver object
     * @param model pointer to the model object
     * @param backward flag indicating adjoint mode (including quadrature)
     */
    void
    runSteadystateSimulation(Solver const& solver, Model& model, bool backward);

    /**
     * @brief Initialize CVodeSolver instance for preequilibration simulation
     * @param solver pointer to the solver object
     * @param model pointer to the model object
     * @param forwardSensis flag switching on integration with FSA
     * @param backward flag switching on quadratures computation
     * @return solver instance
     */
    std::unique_ptr<Solver> createSteadystateSimSolver(
        Solver const& solver, Model& model, bool forwardSensis, bool backward
    ) const;

    /**
     * @brief Initialize forward computation
     * @param it integer with the index of the current time step
     * @param solver pointer to the solver object
     * @param model pointer to the model object
     */
    void initializeForwardProblem(int it, Solver const& solver, Model& model);

    /**
     * @brief Initialize backward computation
     * @param solver pointer to the solver object
     * @param model pointer to the model object
     * @param bwd pointer to backward problem
     * @return flag indicating whether backward computation to be carried out
     */
    bool initializeBackwardProblem(
        Solver const& solver, Model& model, BackwardProblem const* bwd
    );

    /**
     * @brief Compute the backward quadratures, which contribute to the
     * gradient (xQB) from the quadrature over the backward state itself (xQ)
     * @param model pointer to the model object
     * @param yQ vector to be multiplied with dxdotdp
     * @param yQB resulting vector after multiplication
     */
    void
    computeQBfromQ(Model& model, AmiVector const& yQ, AmiVector& yQB) const;

    /**
     * @brief Ensures state positivity, if requested and repeats convergence
     * check, if necessary
     * @param model pointer to the model object
     */
    bool makePositiveAndCheckConvergence(Model& model);

    /**
     * @brief Updates the damping factor gamma that determines step size
     *
     * @param step_successful flag indicating whether the previous step was
     * successful
     * @return boolean flag indicating whether search direction should be
     * updated (true) or the same direction should be retried with the updated
     * dampening (false)
     */
    bool updateDampingFactor(bool step_successful);

    /**
     * @brief Updates member variables to indicate that state_.x has been
     * updated and xdot_, delta_, etc. need to be recomputed.
     */
    void flagUpdatedState();

    /**
     * @brief Retrieves simulation sensitivities from the provided solver and
     * sets the corresponding flag to indicate they are up to date
     * @param solver simulation solver instance
     */
    void updateSensiSimulation(Solver const& solver);

    /**
     * @brief Computes the right hand side for the current state_.x and sets the
     * corresponding flag to indicate xdot_ is up to date.
     * @param model model instance
     */
    void updateRightHandSide(Model& model);

    /**
     * @brief Computes the newton step for the current state_.x and sets the
     * corresponding flag to indicate delta_ is up to date.
     * @param model model instance
     */
    void getNewtonStep(Model& model);

    /** newton step */
    AmiVector delta_;
    /** previous newton step */
    AmiVector delta_old_;
    /** error weights for solver state, dimension nx_solver */
    AmiVector ewt_;
    /** error weights for backward quadratures, dimension nplist() */
    AmiVector ewtQB_;
    /** old state vector */
    AmiVector x_old_;
    /** time derivative state vector */
    AmiVector xdot_;
    /** state differential sensitivities */
    AmiVectorArray sdx_;
    /** adjoint state vector */
    AmiVector xB_;
    /** integral over adjoint state vector */
    AmiVector xQ_;
    /** quadrature state vector */
    AmiVector xQB_;
    /** time-derivative of quadrature state vector */
    AmiVector xQBdot_;

    /** maximum number of steps for Newton solver for allocating numlinsteps */
    int max_steps_{0};

    /** weighted root-mean-square error */
    realtype wrms_{NAN};

    /** state derivative of data likelihood
     * (dimension nJ x nx x nt, ordering =?) */
    std::vector<realtype> dJydx_;

    SimulationState state_;

    /** stores diagnostic information about employed number of steps */
    std::vector<int> numsteps_{std::vector<int>(3, 0)};

    /** stores information about employed number of backward steps */
    int numstepsB_{0};

    /** stores diagnostic information about runtime */
    double cpu_time_{0.0};

    /** stores diagnostic information about runtime backward */
    double cpu_timeB_{0.0};

    /** flag indicating whether backward mode was run */
    bool hasQuadrature_{false};

    /** stepsize for newton step */
    double gamma_{1.0};

    /** stores diagnostic information about execution success of the different
     * approaches [newton, simulation, newton] (length = 3)
     */
    std::vector<SteadyStateStatus> steady_state_status_;

    /** absolute tolerance for convergence check (state)*/
    realtype atol_{NAN};
    /** relative tolerance for convergence check (state)*/
    realtype rtol_{NAN};
    /** absolute tolerance for convergence check (state sensi)*/
    realtype atol_sensi_{NAN};
    /** relative tolerance for convergence check (state sensi)*/
    realtype rtol_sensi_{NAN};
    /** absolute tolerance for convergence check (quadratures)*/
    realtype atol_quad_{NAN};
    /** relative tolerance for convergence check (quadratures)*/
    realtype rtol_quad_{NAN};

    /** newton solver */
    std::unique_ptr<NewtonSolver> newton_solver_{nullptr};

    /** damping factor flag */
    NewtonDampingFactorMode damping_factor_mode_{NewtonDampingFactorMode::on};
    /** damping factor lower bound */
    realtype damping_factor_lower_bound_{1e-8};
    /** whether newton step should be used for convergence steps */
    bool newton_step_conv_{false};
    /** whether sensitivities should be checked for convergence to steadystate
     */
    bool check_sensi_conv_{true};

    /** flag indicating whether xdot_ has been computed for the current state */
    bool xdot_updated_{false};
    /** flag indicating whether delta_ has been computed for the current state
     */
    bool delta_updated_{false};
    /** flag indicating whether simulation sensitivities have been retrieved for
     * the current state */
    bool sensis_updated_{false};
};

} // namespace amici
#endif // STEADYSTATEPROBLEM_H
