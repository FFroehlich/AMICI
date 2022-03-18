#ifndef amici_newton_solver_h
#define amici_newton_solver_h

#include "amici/vector.h"
#include "amici/defines.h"
#include "amici/solver.h"
#include "amici/sundials_matrix_wrapper.h"
#include "amici/sundials_linsol_wrapper.h"

#include <memory>

namespace amici {

class Model;
class Solver;
class AmiVector;

/**
 * @brief The NewtonSolver class sets up the linear solver for the Newton
 * method.
 */

class NewtonSolver {

  public:
    /**
     * @brief Initializes all members with the provided objects
     *
     * @param model pointer to the model object
     */
    NewtonSolver(Model *model);

    /**
     * @brief Factory method to create a NewtonSolver based on linsolType
     *
     * @param simulationSolver solver with settings
     * @param model pointer to the model instance
     * @return solver NewtonSolver according to the specified linsolType
     */
    static std::unique_ptr<NewtonSolver> getSolver(
    const Solver &simulationSolver, Model *model);

    /**
     * @brief Computes the solution of one Newton iteration
     *
     * @param ntry integer newton_try integer start number of Newton solver
     * (1 or 2)
     * @param nnewt integer number of current Newton step
     * @param delta containing the RHS of the linear system, will be
     * overwritten by solution to the linear system
     * @param model pointer to the model instance
     */
    void getStep(int ntry, int nnewt, AmiVector &delta, Model *model,
                 const SimulationState &state);

    /**
     * @brief Computes steady state sensitivities
     *
     * @param sx pointer to state variable sensitivities
     */
    void computeNewtonSensis(AmiVectorArray &sx, Model *model,
                             const SimulationState &state);

    /**
     * @brief Writes the Jacobian for the Newton iteration and passes it to the
     * linear solver
     *
     * @param ntry integer newton_try integer start number of Newton solver
     * (1 or 2)
     * @param nnewt integer number of current Newton step
     * @param model pointer to the model instance
     */
    virtual void prepareLinearSystem(int ntry, int nnewt, Model *model,
                                     const SimulationState &state) = 0;

    /**
     * Writes the Jacobian (JB) for the Newton iteration and passes it to the linear
     * solver
     *
     * @param ntry integer newton_try integer start number of Newton solver
     * (1 or 2)
     * @param nnewt integer number of current Newton step
     * @param model pointer to the model instance
     */
    virtual void prepareLinearSystemB(int ntry, int nnewt, Model *model,
                                      const SimulationState &state) = 0;

    /**
     * @brief Solves the linear system for the Newton step
     *
     * @param rhs containing the RHS of the linear system, will be
     * overwritten by solution to the linear system
     */
    virtual void solveLinearSystem(AmiVector &rhs) = 0;
    
    /**
     * @brief Reinitialize the linear solver
     *
     */
    virtual void reinitialize() = 0;

    virtual ~NewtonSolver() = default;

    /** maximum number of allowed linear steps per Newton step for steady state
     * computation */
    int max_lin_steps_ {0};
    /** maximum number of allowed Newton steps for steady state computation */
    int max_steps {0};
    /** damping factor flag */
    NewtonDampingFactorMode damping_factor_mode_ {NewtonDampingFactorMode::on};
    /** damping factor lower bound */
    realtype damping_factor_lower_bound {1e-8};

  protected:
    /** dummy rhs, used as dummy argument when computing J and JB */
    AmiVector xdot_;
    /** dummy state, attached to linear solver */
    AmiVector x_;
    /** dummy adjoint state, used as dummy argument when computing JB */
    AmiVector xB_;
    /** dummy differential adjoint state, used as dummy argument when computing JB */
    AmiVector dxB_;
};

/**
 * @brief The NewtonSolverDense provides access to the dense linear solver for
 * the Newton method.
 */

class NewtonSolverDense : public NewtonSolver {

  public:
    NewtonSolverDense(Model *model);

    NewtonSolverDense(const NewtonSolverDense&) = delete;

    NewtonSolverDense& operator=(const NewtonSolverDense& other) = delete;

    ~NewtonSolverDense() override;

    void solveLinearSystem(AmiVector &rhs) override;

    void prepareLinearSystem(int ntry, int nnewt, Model *model,
                             const SimulationState &state) override;

    void prepareLinearSystemB(int ntry, int nnewt, Model *model,
                              const SimulationState &state) override;
    
    void reinitialize() override;

  private:
    /** temporary storage of Jacobian */
    SUNMatrixWrapper Jtmp_;

    /** dense linear solver */
    SUNLinearSolver linsol_ {nullptr};
};

/**
 * @brief The NewtonSolverSparse provides access to the sparse linear solver for
 * the Newton method.
 */

class NewtonSolverSparse : public NewtonSolver {

  public:
    NewtonSolverSparse(Model *model);

    NewtonSolverSparse(const NewtonSolverSparse&) = delete;

    NewtonSolverSparse& operator=(const NewtonSolverSparse& other) = delete;

    ~NewtonSolverSparse() override;

    void solveLinearSystem(AmiVector &rhs) override;

    void prepareLinearSystem(int ntry, int nnewt, Model *model,
                             const SimulationState &state) override;

    void prepareLinearSystemB(int ntry, int nnewt, Model *model,
                              const SimulationState &state) override;

    void reinitialize() override;
    
  private:
    /** temporary storage of Jacobian */
    SUNMatrixWrapper Jtmp_;

    /** sparse linear solver */
    SUNLinearSolver linsol_ {nullptr};
};


} // namespace amici

#endif // NEWTON_SOLVER
