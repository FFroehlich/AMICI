#include "amici/newton_solver.h"

#include "amici/model.h"
#include "amici/solver.h"

#include "sunlinsol/sunlinsol_klu.h" // sparse solver
#include "sunlinsol/sunlinsol_dense.h" // dense solver
#include <sundials/sundials_math.h> // roundoffs

#include <cstring>
#include <ctime>
#include <cmath>

// taken from sundials/src/sunlinsol/klu/sunlinsol_klu.c
#if defined(SUNDIALS_INT64_T)
#define KLU_INDEXTYPE long int
#else
#define KLU_INDEXTYPE int
#endif

namespace amici {

NewtonSolver::NewtonSolver(realtype *t, AmiVector *x, Model *model)
    : t_(t), model_(model), xdot_(model->nx_solver), x_(x),
      dx_(model->nx_solver), xB_(model->nx_solver), dxB_(model->nx_solver) {
}

/* ------------------------------------------------------------------------- */

std::unique_ptr<NewtonSolver> NewtonSolver::getSolver(realtype *t, AmiVector *x,
                                                      Solver &simulationSolver,
                                                      Model *model) {

    std::unique_ptr<NewtonSolver> solver;

    switch (simulationSolver.getLinearSolver()) {

    /* DIRECT SOLVERS */
    case LinearSolver::dense:
        solver.reset(new NewtonSolverDense(t, x, model));
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
        solver.reset(new NewtonSolverSparse(t, x, model));
        break;
    default:
        throw NewtonFailure(AMICI_NOT_IMPLEMENTED, "getSolver");
    }
    solver->atol_ = simulationSolver.getAbsoluteToleranceSteadyState();
    solver->rtol_ = simulationSolver.getRelativeToleranceSteadyState();
    solver->max_lin_steps_ = simulationSolver.getNewtonMaxLinearSteps();
    solver->max_steps = simulationSolver.getNewtonMaxSteps();
    solver->damping_factor_mode_ = simulationSolver.getNewtonDampingFactorMode();
    solver->damping_factor_lower_bound =
        simulationSolver.getNewtonDampingFactorLowerBound();

    return solver;
}

/* ------------------------------------------------------------------------- */

void NewtonSolver::getStep(int ntry, int nnewt, AmiVector &delta) {
    prepareLinearSystem(ntry, nnewt);

    delta.minus();
    solveLinearSystem(delta);
}

/* ------------------------------------------------------------------------- */

void NewtonSolver::computeNewtonSensis(AmiVectorArray &sx) {
    prepareLinearSystem(0, -1);
    model_->fdxdotdp(*t_, *x_, dx_);
    
    if (is_singular())
        model_->app->warningF("AMICI:newton",
                              "Jacobian is singular at steadystate, sensitivities may be inaccurate");

    if (model_->pythonGenerated) {
        for (int ip = 0; ip < model_->nplist(); ip++) {
            N_VConst(0.0, sx.getNVector(ip));
            model_->get_dxdotdp_full().scatter(model_->plist(ip), -1.0, nullptr,
                                               gsl::make_span(sx.getNVector(ip)),
                                               0, nullptr, 0);

            solveLinearSystem(sx[ip]);
        }
    } else {
        for (int ip = 0; ip < model_->nplist(); ip++) {
            for (int ix = 0; ix < model_->nx_solver; ix++)
                sx.at(ix,ip) = -model_->get_dxdotdp().at(ix, ip);

            solveLinearSystem(sx[ip]);
        }
    }
}

/* ------------------------------------------------------------------------- */
/* - Dense linear solver --------------------------------------------------- */
/* ------------------------------------------------------------------------- */

/* Derived class for dense linear solver */
NewtonSolverDense::NewtonSolverDense(realtype *t, AmiVector *x, Model *model)
    : NewtonSolver(t, x, model), Jtmp_(model->nx_solver, model->nx_solver),
      linsol_(SUNLinSol_Dense(x->getNVector(), Jtmp_.get())) {
    int status = SUNLinSolInitialize_Dense(linsol_);
    if(status != SUNLS_SUCCESS)
        throw NewtonFailure(status, "SUNLinSolInitialize_Dense");
}

/* ------------------------------------------------------------------------- */

void NewtonSolverDense::prepareLinearSystem(int  /*ntry*/, int  /*nnewt*/) {
    model_->fJ(*t_, 0.0, *x_, dx_, xdot_, Jtmp_.get());
    Jtmp_.refresh();
    int status = SUNLinSolSetup_Dense(linsol_, Jtmp_.get());
    if(status != SUNLS_SUCCESS)
        throw NewtonFailure(status, "SUNLinSolSetup_Dense");
}

/* ------------------------------------------------------------------------- */

void NewtonSolverDense::prepareLinearSystemB(int  /*ntry*/, int  /*nnewt*/) {
    model_->fJB(*t_, 0.0, *x_, dx_, xB_, dxB_, xdot_, Jtmp_.get());
    Jtmp_.refresh();
    int status = SUNLinSolSetup_Dense(linsol_, Jtmp_.get());
    if(status != SUNLS_SUCCESS)
        throw NewtonFailure(status, "SUNLinSolSetup_Dense");
}


/* ------------------------------------------------------------------------- */

void NewtonSolverDense::solveLinearSystem(AmiVector &rhs) {
    int status = SUNLinSolSolve_Dense(linsol_, Jtmp_.get(),
                                      rhs.getNVector(), rhs.getNVector(),
                                      0.0);
    Jtmp_.refresh();
    // last argument is tolerance and does not have any influence on result

    if(status != SUNLS_SUCCESS)
        throw NewtonFailure(status, "SUNLinSolSolve_Dense");
}

/* ------------------------------------------------------------------------- */

bool NewtonSolverDense::is_singular() const {
    // dense solver doesn't have any implementation for rcond/condest, so use
    // sparse solver interface, not the most efficient solution, but who is
    // concerned about speed and used the dense solver anyways ¯\_(ツ)_/¯
    auto sparse_solver = new NewtonSolverSparse(t_, x_, model_);
    sparse_solver->prepareLinearSystem(0,0);
    auto is_singular = sparse_solver->is_singular();
    delete sparse_solver;
    return is_singular;
}

NewtonSolverDense::~NewtonSolverDense() {
    if(linsol_)
        SUNLinSolFree_Dense(linsol_);
}

/* ------------------------------------------------------------------------- */
/* - Sparse linear solver -------------------------------------------------- */
/* ------------------------------------------------------------------------- */

/* Derived class for sparse linear solver */
NewtonSolverSparse::NewtonSolverSparse(realtype *t, AmiVector *x, Model *model)
    : NewtonSolver(t, x, model),
      Jtmp_(model->nx_solver, model->nx_solver, model->nnz, CSC_MAT),
      linsol_(SUNKLU(x->getNVector(), Jtmp_.get())) {
    int status = SUNLinSolInitialize_KLU(linsol_);
    if(status != SUNLS_SUCCESS)
        throw NewtonFailure(status, "SUNLinSolInitialize_KLU");
}

/* ------------------------------------------------------------------------- */

void NewtonSolverSparse::prepareLinearSystem(int  /*ntry*/, int  /*nnewt*/) {
    /* Get sparse Jacobian */
    model_->fJSparse(*t_, 0.0, *x_, dx_, xdot_, Jtmp_.get());
    Jtmp_.refresh();
    int status = SUNLinSolSetup_KLU(linsol_, Jtmp_.get());
    if(status != SUNLS_SUCCESS)
        throw NewtonFailure(status, "SUNLinSolSetup_KLU");
}

/* ------------------------------------------------------------------------- */

void NewtonSolverSparse::prepareLinearSystemB(int  /*ntry*/, int  /*nnewt*/) {
    /* Get sparse Jacobian */
    model_->fJSparseB(*t_, 0.0, *x_, dx_, xB_, dxB_, xdot_, Jtmp_.get());
    Jtmp_.refresh();
    int status = SUNLinSolSetup_KLU(linsol_, Jtmp_.get());
    if(status != SUNLS_SUCCESS)
        throw NewtonFailure(status, "SUNLinSolSetup_KLU");
}

/* ------------------------------------------------------------------------- */

void NewtonSolverSparse::solveLinearSystem(AmiVector &rhs) {
    /* Pass pointer to the linear solver */
    int status = SUNLinSolSolve_KLU(linsol_, Jtmp_.get(),
                                    rhs.getNVector(), rhs.getNVector(), 0.0);
    // last argument is tolerance and does not have any influence on result

    if(status != SUNLS_SUCCESS)
        throw NewtonFailure(status, "SUNLinSolSolve_KLU");
}

/* ------------------------------------------------------------------------- */

bool NewtonSolverSparse::is_singular() const {
    // adapted from SUNLinSolSetup_KLU in sunlinsol/klu/sunlinsol_klu.c
    auto content = (SUNLinearSolverContent_KLU)(linsol_->content);
    // first cheap check via rcond
    int status = sun_klu_rcond(content->symbolic, content->numeric,
                               &content->common);
    if(status != SUNLS_SUCCESS)
        throw NewtonFailure(status, "sun_klu_rcond");
    
    auto precision = SUNRpowerR(UNIT_ROUNDOFF, 2.0/3.0);
    
    if (content->common.rcond < precision) {
        // cheap check indicates singular, expensive check via condest
        status = sun_klu_condest((KLU_INDEXTYPE*) SM_INDEXPTRS_S(Jtmp_.get()),
                                 SM_DATA_S(Jtmp_.get()),
                                 content->symbolic,
                                 content->numeric,
                                 &content->common);
        if(status != SUNLS_SUCCESS)
            throw NewtonFailure(status, "sun_klu_rcond");
        return content->common.condest > 1.0/precision;
    }
    return false;
}

NewtonSolverSparse::~NewtonSolverSparse() {
    if(linsol_)
        SUNLinSolFree_KLU(linsol_);
}


} // namespace amici
