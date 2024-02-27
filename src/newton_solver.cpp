#include "amici/newton_solver.h"

#include <amici/amici.h>
#include <amici/model.h>
#include <amici/solver.h>

#include <sundials/sundials_config.h>  // roundoffs
#include <sunlinsol/sunlinsol_dense.h> // dense solver
#include <sunlinsol/sunlinsol_klu.h>   // sparse solver

namespace amici {

NewtonSolver::NewtonSolver(Model const& model)
    : xdot_(model.nx_solver)
    , x_(model.nx_solver)
    , xB_(model.nJ * model.nx_solver)
    , dxB_(model.nJ * model.nx_solver) {}

std::unique_ptr<NewtonSolver>
NewtonSolver::getSolver(Solver const& simulationSolver, Model const& model) {

    std::unique_ptr<NewtonSolver> solver;

    switch (simulationSolver.getLinearSolver()) {

    /* DIRECT SOLVERS */
    case LinearSolver::dense:
        solver.reset(new NewtonSolverDense(model));
        break;

    case LinearSolver::band:
        throw NewtonFailure(AMICI_NOT_IMPLEMENTED, "getSolver");

    case LinearSolver::LAPACKDense:
        throw NewtonFailure(AMICI_NOT_IMPLEMENTED, "getSolver");

    case LinearSolver::LAPACKBand:
        throw NewtonFailure(AMICI_NOT_IMPLEMENTED, "getSolver");

    case LinearSolver::diag:
        throw NewtonFailure(AMICI_NOT_IMPLEMENTED, "getSolver");

    /* ITERATIVE SOLVERS */
    case LinearSolver::SPGMR:
        throw NewtonFailure(AMICI_NOT_IMPLEMENTED, "getSolver");

    case LinearSolver::SPBCG:
        throw NewtonFailure(AMICI_NOT_IMPLEMENTED, "getSolver");

    case LinearSolver::SPTFQMR:
        throw NewtonFailure(AMICI_NOT_IMPLEMENTED, "getSolver");

    /* SPARSE SOLVERS */
    case LinearSolver::SuperLUMT:
        throw NewtonFailure(AMICI_NOT_IMPLEMENTED, "getSolver");
    case LinearSolver::KLU:
        solver.reset(new NewtonSolverSparse(model));
        break;
    default:
        throw NewtonFailure(AMICI_NOT_IMPLEMENTED, "getSolver");
    }
    return solver;
}

void NewtonSolver::getStep(
    AmiVector& delta, Model& model, SimulationState const& state
) {
    prepareLinearSystem(model, state);

    delta.minus();
    solveLinearSystem(delta);
}

void NewtonSolver::computeNewtonSensis(
    AmiVectorArray& sx, Model& model, SimulationState const& state
) {
    prepareLinearSystem(model, state);
    model.fdxdotdp(state.t, state.x, state.dx);

    if (model.logger && is_singular(model, state)) {
        model.logger->log(
            LogSeverity::warning, "NEWTON_JAC_SINGULAR",
            "Jacobian is singular at steadystate, "
            "sensitivities may be inaccurate."
        );
    }

    if (model.pythonGenerated) {
        for (int ip = 0; ip < model.nplist(); ip++) {
            N_VConst(0.0, sx.getNVector(ip));
            model.get_dxdotdp_full().scatter(
                model.plist(ip), -1.0, nullptr,
                gsl::make_span(sx.getNVector(ip)), 0, nullptr, 0
            );

            solveLinearSystem(sx[ip]);
        }
    } else {
        for (int ip = 0; ip < model.nplist(); ip++) {
            for (int ix = 0; ix < model.nx_solver; ix++)
                sx.at(ix, ip) = -model.get_dxdotdp().at(ix, ip);

            solveLinearSystem(sx[ip]);
        }
    }
}

NewtonSolverDense::NewtonSolverDense(Model const& model)
    : NewtonSolver(model)
    , Jtmp_(model.nx_solver, model.nx_solver)
    , linsol_(SUNLinSol_Dense(x_.getNVector(), Jtmp_)) {
    auto status = SUNLinSolInitialize_Dense(linsol_);
    if (status != SUNLS_SUCCESS)
        throw NewtonFailure(status, "SUNLinSolInitialize_Dense");
}

void NewtonSolverDense::prepareLinearSystem(
    Model& model, SimulationState const& state
) {
    model.fJ(state.t, 0.0, state.x, state.dx, xdot_, Jtmp_);
    Jtmp_.refresh();
    auto status = SUNLinSolSetup_Dense(linsol_, Jtmp_);
    if (status != SUNLS_SUCCESS)
        throw NewtonFailure(status, "SUNLinSolSetup_Dense");
}

void NewtonSolverDense::prepareLinearSystemB(
    Model& model, SimulationState const& state
) {
    model.fJB(state.t, 0.0, state.x, state.dx, xB_, dxB_, xdot_, Jtmp_);
    Jtmp_.refresh();
    auto status = SUNLinSolSetup_Dense(linsol_, Jtmp_);
    if (status != SUNLS_SUCCESS)
        throw NewtonFailure(status, "SUNLinSolSetup_Dense");
}

void NewtonSolverDense::solveLinearSystem(AmiVector& rhs) {
    auto status = SUNLinSolSolve_Dense(
        linsol_, Jtmp_, rhs.getNVector(), rhs.getNVector(), 0.0
    );
    Jtmp_.refresh();
    // last argument is tolerance and does not have any influence on result

    if (status != SUNLS_SUCCESS)
        throw NewtonFailure(status, "SUNLinSolSolve_Dense");
}

void NewtonSolverDense::reinitialize(){
    /* dense solver does not need reinitialization */
};

bool NewtonSolverDense::is_singular(Model& model, SimulationState const& state)
    const {
    // dense solver doesn't have any implementation for rcond/condest, so use
    // sparse solver interface, not the most efficient solution, but who is
    // concerned about speed and used the dense solver anyways ¯\_(ツ)_/¯
    NewtonSolverSparse sparse_solver(model);
    sparse_solver.prepareLinearSystem(model, state);
    return sparse_solver.is_singular(model, state);
}

NewtonSolverDense::~NewtonSolverDense() {
    if (linsol_)
        SUNLinSolFree_Dense(linsol_);
}

NewtonSolverSparse::NewtonSolverSparse(Model const& model)
    : NewtonSolver(model)
    , Jtmp_(model.nx_solver, model.nx_solver, model.nnz, CSC_MAT)
    , linsol_(SUNKLU(x_.getNVector(), Jtmp_)) {
    auto status = SUNLinSolInitialize_KLU(linsol_);
    if (status != SUNLS_SUCCESS)
        throw NewtonFailure(status, "SUNLinSolInitialize_KLU");
}

void NewtonSolverSparse::prepareLinearSystem(
    Model& model, SimulationState const& state
) {
    /* Get sparse Jacobian */
    model.fJSparse(state.t, 0.0, state.x, state.dx, xdot_, Jtmp_);
    Jtmp_.refresh();
    auto status = SUNLinSolSetup_KLU(linsol_, Jtmp_);
    if (status != SUNLS_SUCCESS)
        throw NewtonFailure(status, "SUNLinSolSetup_KLU");
}

void NewtonSolverSparse::prepareLinearSystemB(
    Model& model, SimulationState const& state
) {
    /* Get sparse Jacobian */
    model.fJSparseB(
        state.t, 0.0, state.x, state.dx, xB_, dxB_, xdot_, Jtmp_
    );
    Jtmp_.refresh();
    auto status = SUNLinSolSetup_KLU(linsol_, Jtmp_);
    if (status != SUNLS_SUCCESS)
        throw NewtonFailure(status, "SUNLinSolSetup_KLU");
}

void NewtonSolverSparse::solveLinearSystem(AmiVector& rhs) {
    /* Pass pointer to the linear solver */
    auto status = SUNLinSolSolve_KLU(
        linsol_, Jtmp_, rhs.getNVector(), rhs.getNVector(), 0.0
    );
    // last argument is tolerance and does not have any influence on result

    if (status != SUNLS_SUCCESS)
        throw NewtonFailure(status, "SUNLinSolSolve_KLU");
}

void NewtonSolverSparse::reinitialize() {
    /* partial reinitialization, don't need to reallocate Jtmp_ */
    auto status = SUNLinSol_KLUReInit(
        linsol_, Jtmp_, Jtmp_.capacity(), SUNKLU_REINIT_PARTIAL
    );
    if (status != SUNLS_SUCCESS)
        throw NewtonFailure(status, "SUNLinSol_KLUReInit");
}

bool NewtonSolverSparse::
    is_singular(Model& /*model*/, SimulationState const& /*state*/) const {
    // adapted from SUNLinSolSetup_KLU in sunlinsol/klu/sunlinsol_klu.c
    auto content = (SUNLinearSolverContent_KLU)(linsol_->content);
    // first cheap check via rcond
    auto status
        = sun_klu_rcond(content->symbolic, content->numeric, &content->common);
    if (status == 0)
        throw NewtonFailure(content->last_flag, "sun_klu_rcond");

    auto precision = std::numeric_limits<realtype>::epsilon();

    if (content->common.rcond < precision) {
        // cheap check indicates singular, expensive check via condest
        status = sun_klu_condest(
            SM_INDEXPTRS_S(Jtmp_.get()), SM_DATA_S(Jtmp_.get()),
            content->symbolic, content->numeric, &content->common
        );
        if (status == 0)
            throw NewtonFailure(content->last_flag, "sun_klu_rcond");
        return content->common.condest > 1.0 / precision;
    }
    return false;
}

NewtonSolverSparse::~NewtonSolverSparse() {
    if (linsol_)
        SUNLinSolFree_KLU(linsol_);
}

} // namespace amici
