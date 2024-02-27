#include "amici/solver.h"

#include "amici/exception.h"
#include "amici/model.h"
#include "amici/symbolic_functions.h"

#include <cstdio>
#include <memory>

namespace amici {

Solver::Solver(Solver const& other)
    : ism_(other.ism_)
    , lmm_(other.lmm_)
    , iter_(other.iter_)
    , interp_type_(other.interp_type_)
    , maxsteps_(other.maxsteps_)
    , maxtime_(other.maxtime_)
    , simulation_timer_(other.simulation_timer_)
    , sensi_meth_(other.sensi_meth_)
    , sensi_meth_preeq_(other.sensi_meth_preeq_)
    , stldet_(other.stldet_)
    , ordering_(other.ordering_)
    , newton_maxsteps_(other.newton_maxsteps_)
    , newton_damping_factor_mode_(other.newton_damping_factor_mode_)
    , newton_damping_factor_lower_bound_(
          other.newton_damping_factor_lower_bound_
      )
    , linsol_(other.linsol_)
    , atol_(other.atol_)
    , rtol_(other.rtol_)
    , atol_fsa_(other.atol_fsa_)
    , rtol_fsa_(other.rtol_fsa_)
    , atolB_(other.atolB_)
    , rtolB_(other.rtolB_)
    , quad_atol_(other.quad_atol_)
    , quad_rtol_(other.quad_rtol_)
    , ss_tol_factor_(other.ss_tol_factor_)
    , ss_atol_(other.ss_atol_)
    , ss_rtol_(other.ss_rtol_)
    , ss_tol_sensi_factor_(other.ss_tol_sensi_factor_)
    , ss_atol_sensi_(other.ss_atol_sensi_)
    , ss_rtol_sensi_(other.ss_rtol_sensi_)
    , rdata_mode_(other.rdata_mode_)
    , newton_step_steadystate_conv_(other.newton_step_steadystate_conv_)
    , check_sensi_steadystate_conv_(other.check_sensi_steadystate_conv_)
    , maxstepsB_(other.maxstepsB_)
    , sensi_(other.sensi_) {}

void Solver::apply_max_num_steps() const {
    // set remaining steps, setMaxNumSteps only applies to a single call of
    // solve
    long int cursteps;
    getNumSteps(solver_memory_.get(), &cursteps);
    if (maxsteps_ <= cursteps)
        throw AmiException(
            "Reached maximum number of steps %ld before reaching "
            "tout at t=%g.",
            maxsteps_, t_
        );
    setMaxNumSteps(maxsteps_ - cursteps);
}

void Solver::apply_max_num_steps_B() const {
    // set remaining steps, setMaxNumSteps only applies to a single call of
    // solve
    long int curstepsB;
    auto maxstepsB = (maxstepsB_ == 0) ? maxsteps_ * 100 : maxstepsB_;
    for (int i_mem_b = 0; i_mem_b < (int)solver_memory_B_.size(); ++i_mem_b) {
        if (solver_memory_B_.at(i_mem_b)) {
            getNumSteps(solver_memory_B_.at(i_mem_b).get(), &curstepsB);
            if (maxstepsB <= curstepsB)
                throw AmiException(
                    "Reached maximum number of steps %ld before "
                    "reaching tout at t=%g in backward "
                    "problem %i.",
                    maxstepsB_, t_, i_mem_b
                );
            setMaxNumStepsB(i_mem_b, maxstepsB - curstepsB);
        }
    }
}

int Solver::run(realtype const tout) const {
    setStopTime(tout);
    CpuTimer cpu_timer;
    int status = AMICI_SUCCESS;

    apply_max_num_steps();
    if (nx() > 0) {
        if (getAdjInitDone()) {
            status = solveF(tout, AMICI_NORMAL, &ncheckPtr_);
        } else {
            status = solve(tout, AMICI_NORMAL);
        }
    } else {
        t_ = tout;
    }
    cpu_time_ += cpu_timer.elapsed_milliseconds();
    return status;
}

int Solver::step(realtype const tout) const {
    int status = AMICI_SUCCESS;

    apply_max_num_steps();
    if (nx() > 0) {
        if (getAdjInitDone()) {
            status = solveF(tout, AMICI_ONE_STEP, &ncheckPtr_);
        } else {
            status = solve(tout, AMICI_ONE_STEP);
        }
    } else {
        t_ = tout;
    }
    return status;
}

void Solver::runB(realtype const tout) const {
    CpuTimer cpu_timer;

    apply_max_num_steps_B();
    if (nx() > 0) {
        solveB(tout, AMICI_NORMAL);
    }
    cpu_timeB_ += cpu_timer.elapsed_milliseconds();
    t_ = tout;
}

void Solver::setup(
    realtype const t0, Model* model, AmiVector const& x0, AmiVector const& dx0,
    AmiVectorArray const& sx0, AmiVectorArray const& sdx0
) const {
    if (nx() != model->nx_solver || nplist() != model->nplist()
        || nquad() != model->nJ * model->nplist()) {
        resetMutableMemory(
            model->nx_solver, model->nplist(), model->nJ * model->nplist()
        );
    }
    /* Create solver memory object if necessary */
    allocateSolver();
    if (!solver_memory_)
        throw AmiException("Failed to allocated solver memory!");

    /* Initialize CVodes/IDAs solver*/
    init(t0, x0, dx0);

    /* Clear diagnosis storage */
    resetDiagnosis();

    /* Apply stored tolerances to sundials solver */
    applyTolerances();

    /* Set optional inputs */
    setErrHandlerFn();
    /* Attaches userdata */
    user_data = std::make_pair(model, this);
    setUserData();
    /* activates stability limit detection */
    setStabLimDet(stldet_);

    rootInit(model->ne_solver);

    if (nx() == 0)
        return;

    initializeLinearSolver(model);
    initializeNonLinearSolver();

    if (sensi_ >= SensitivityOrder::first
        && sensi_meth_ > SensitivityMethod::none && model->nx_solver > 0) {
        auto plist = model->getParameterList();
        sensInit1(sx0, sdx0);
        if (sensi_meth_ == SensitivityMethod::forward && !plist.empty()) {
            /* Set sensitivity analysis optional inputs */
            auto par = model->getUnscaledParameters();

            /* Activate sensitivity calculations  and apply tolerances */
            initializeNonLinearSolverSens(model);
            setSensParams(par.data(), nullptr, plist.data());
            applyTolerancesFSA();
        } else {
            /* Allocate space for the adjoint computation */
            adjInit();
        }
    }

    setId(model);
    setSuppressAlg(true);
    /* calculate consistent DAE initial conditions (no effect for ODE) */
    if (model->nt() > 1)
        calcIC(model->getTimepoint(1));

    cpu_time_ = 0.0;
    cpu_timeB_ = 0.0;
}

void Solver::setupB(
    int* which, realtype const tf, Model* model, AmiVector const& xB0,
    AmiVector const& dxB0, AmiVector const& xQB0
) const {
    if (!solver_memory_)
        throw AmiException("Solver for the forward problem must be setup first"
        );

    /* allocate memory for the backward problem */
    allocateSolverB(which);

    /* initialise states */
    binit(*which, tf, xB0, dxB0);

    /* Attach user data */
    setUserDataB(*which);

    if (nx() == 0)
        return;

    initializeLinearSolverB(model, *which);
    initializeNonLinearSolverB(*which);

    /* Initialise quadrature calculation */
    qbinit(*which, xQB0);

    applyTolerancesASA(*which);
    applyQuadTolerancesASA(*which);

    setStabLimDetB(*which, stldet_);
}

void Solver::setupSteadystate(
    realtype const t0, Model* model, AmiVector const& x0, AmiVector const& dx0,
    AmiVector const& xB0, AmiVector const& dxB0, AmiVector const& xQ0
) const {
    /* Initialize CVodes/IDAs solver with steadystate RHS function */
    initSteadystate(t0, x0, dx0);

    /* Allocate space for forward quadratures */
    quadInit(xQ0);

    /* Apply tolerances */
    applyQuadTolerances();

    /* Check linear solver (works only with KLU atm) */
    if (linsol_ != LinearSolver::KLU)
        throw AmiException("Backward steady state computation via integration "
                           "is currently only implemented for KLU linear solver"
        );
    /* Set Jacobian function and initialize values */
    setSparseJacFn_ss();
    model->writeSteadystateJB(t0, 0, x0, dx0, xB0, dxB0, xB0);
}

void Solver::updateAndReinitStatesAndSensitivities(Model* model) const {
    model->reinitialize(
        t_, x_, sx_, getSensitivityOrder() >= SensitivityOrder::first
    );

    reInit(t_, x_, dx_);
    if (getSensitivityOrder() >= SensitivityOrder::first
        && getSensitivityMethod() == SensitivityMethod::forward) {
        sensReInit(sx_, sdx_);
    }
}

void Solver::resetDiagnosis() const {
    ns_.clear();
    nrhs_.clear();
    netf_.clear();
    nnlscf_.clear();
    order_.clear();

    nsB_.clear();
    nrhsB_.clear();
    netfB_.clear();
    nnlscfB_.clear();
}

void Solver::storeDiagnosis() const {
    if (!solver_was_called_F_ || !solver_memory_) {
        ns_.push_back(0);
        nrhs_.push_back(0);
        netf_.push_back(0);
        nnlscf_.push_back(0);
        order_.push_back(0);
        return;
    }

    long int lnumber;
    getNumSteps(solver_memory_.get(), &lnumber);
    ns_.push_back(gsl::narrow<int>(lnumber));

    getNumRhsEvals(solver_memory_.get(), &lnumber);
    nrhs_.push_back(gsl::narrow<int>(lnumber));

    getNumErrTestFails(solver_memory_.get(), &lnumber);
    netf_.push_back(gsl::narrow<int>(lnumber));

    getNumNonlinSolvConvFails(solver_memory_.get(), &lnumber);
    nnlscf_.push_back(gsl::narrow<int>(lnumber));

    int number;
    getLastOrder(solver_memory_.get(), &number);
    order_.push_back(number);
}

void Solver::storeDiagnosisB(int const which) const {
    if (!solver_was_called_B_ || !solver_memory_B_.at(which)) {
        nsB_.push_back(0);
        nrhsB_.push_back(0);
        netfB_.push_back(0);
        nnlscfB_.push_back(0);
        return;
    }

    long int number;
    getNumSteps(solver_memory_B_.at(which).get(), &number);
    nsB_.push_back(gsl::narrow<int>(number));

    getNumRhsEvals(solver_memory_B_.at(which).get(), &number);
    nrhsB_.push_back(gsl::narrow<int>(number));

    getNumErrTestFails(solver_memory_B_.at(which).get(), &number);
    netfB_.push_back(gsl::narrow<int>(number));

    getNumNonlinSolvConvFails(solver_memory_B_.at(which).get(), &number);
    nnlscfB_.push_back(gsl::narrow<int>(number));
}

void Solver::initializeLinearSolver(Model const* model) const {
    switch (linsol_) {

        /* DIRECT SOLVERS */

    case LinearSolver::dense:
        linear_solver_ = std::make_unique<SUNLinSolDense>(x_);
        setLinearSolver();
        setDenseJacFn();
        break;

    case LinearSolver::band:
        linear_solver_
            = std::make_unique<SUNLinSolBand>(x_, model->ubw, model->lbw);
        setLinearSolver();
        setBandJacFn();
        break;

    case LinearSolver::LAPACKDense:
        throw AmiException("Solver currently not supported!");

    case LinearSolver::LAPACKBand:
        throw AmiException("Solver currently not supported!");

    case LinearSolver::diag:
        diag();
        setDenseJacFn();
        break;

        /* ITERATIVE SOLVERS */

    case LinearSolver::SPGMR:
        linear_solver_ = std::make_unique<SUNLinSolSPGMR>(x_);
        setLinearSolver();
        setJacTimesVecFn();
        break;

    case LinearSolver::SPBCG:
        linear_solver_ = std::make_unique<SUNLinSolSPBCGS>(x_);
        setLinearSolver();
        setJacTimesVecFn();
        break;

    case LinearSolver::SPTFQMR:
        linear_solver_ = std::make_unique<SUNLinSolSPTFQMR>(x_);
        setLinearSolver();
        setJacTimesVecFn();
        break;

        /* SPARSE SOLVERS */

    case LinearSolver::KLU:
        linear_solver_ = std::make_unique<SUNLinSolKLU>(
            x_, model->nnz, CSC_MAT,
            static_cast<SUNLinSolKLU::StateOrdering>(getStateOrdering())
        );
        setLinearSolver();
        setSparseJacFn();
        break;

#ifdef SUNDIALS_SUPERLUMT
    case LinearSolver::SuperLUMT:
        // TODO state ordering
        linearSolver = std::make_unique<SUNLinSolSuperLUMT>(
            *x, model->nnz, CSC_MAT,
            static_cast<SUNLinSolSuperLUMT::StateOrdering>(getStateOrdering())
        );

        setLinearSolver();
        setSparseJacFn();
        break;
#endif
    default:
        throw AmiException(
            "Invalid choice of solver: %d", static_cast<int>(linsol_)
        );
    }
}

void Solver::initializeNonLinearSolver() const {
    switch (iter_) {
    case NonlinearSolverIteration::newton:
        non_linear_solver_
            = std::make_unique<SUNNonLinSolNewton>(x_.getNVector());
        break;
    case NonlinearSolverIteration::fixedpoint:
        non_linear_solver_
            = std::make_unique<SUNNonLinSolFixedPoint>(x_.getNVector());
        break;
    default:
        throw AmiException(
            "Invalid non-linear solver specified (%d).", static_cast<int>(iter_)
        );
    }

    setNonLinearSolver();
}

void Solver::initializeLinearSolverB(Model const* model, int const which)
    const {
    switch (linsol_) {
    /* DIRECT SOLVERS */
    case LinearSolver::dense:
        linear_solver_B_ = std::make_unique<SUNLinSolDense>(xB_);
        setLinearSolverB(which);
        setDenseJacFnB(which);
        break;

    case LinearSolver::band:
        linear_solver_B_
            = std::make_unique<SUNLinSolBand>(xB_, model->ubw, model->lbw);
        setLinearSolverB(which);
        setBandJacFnB(which);
        break;

    case LinearSolver::LAPACKDense:
        throw AmiException("Solver currently not supported!");

    case LinearSolver::LAPACKBand:
        throw AmiException("Solver currently not supported!");

    case LinearSolver::diag:
        diagB(which);
        setDenseJacFnB(which);
        break;

        /* ITERATIVE SOLVERS */

    case LinearSolver::SPGMR:
        linear_solver_B_ = std::make_unique<SUNLinSolSPGMR>(xB_);
        setLinearSolverB(which);
        setJacTimesVecFnB(which);
        break;

    case LinearSolver::SPBCG:
        linear_solver_B_ = std::make_unique<SUNLinSolSPBCGS>(xB_);
        setLinearSolverB(which);
        setJacTimesVecFnB(which);
        break;

    case LinearSolver::SPTFQMR:
        linear_solver_B_ = std::make_unique<SUNLinSolSPTFQMR>(xB_);
        setLinearSolverB(which);
        setJacTimesVecFnB(which);
        break;

        /* SPARSE SOLVERS */

    case LinearSolver::KLU:
        linear_solver_B_ = std::make_unique<SUNLinSolKLU>(
            xB_, model->nnz, CSC_MAT,
            static_cast<SUNLinSolKLU::StateOrdering>(getStateOrdering())
        );
        setLinearSolverB(which);
        setSparseJacFnB(which);
        break;
#ifdef SUNDIALS_SUPERLUMT
    case LinearSolver::SuperLUMT:
        linearSolverB = std::make_unique<SUNLinSolSuperLUMT>(
            *xB, model->nnz, CSC_MAT,
            static_cast<SUNLinSolSuperLUMT::StateOrdering>(getStateOrdering())
        );
        setLinearSolverB(which);
        setSparseJacFnB(which);
        break;
#endif
    default:
        throw AmiException(
            "Invalid choice of solver: %d", static_cast<int>(linsol_)
        );
    }
}

void Solver::initializeNonLinearSolverB(int const which) const {
    switch (iter_) {
    case NonlinearSolverIteration::newton:
        non_linear_solver_B_
            = std::make_unique<SUNNonLinSolNewton>(xB_.getNVector());
        break;
    case NonlinearSolverIteration::fixedpoint:
        non_linear_solver_B_
            = std::make_unique<SUNNonLinSolFixedPoint>(xB_.getNVector());
        break;
    default:
        throw AmiException(
            "Invalid non-linear solver specified (%d).", static_cast<int>(iter_)
        );
    }

    setNonLinearSolverB(which);
}

bool operator==(Solver const& a, Solver const& b) {
    if (typeid(a) != typeid(b))
        return false;

    return (a.interp_type_ == b.interp_type_) && (a.lmm_ == b.lmm_)
           && (a.iter_ == b.iter_) && (a.stldet_ == b.stldet_)
           && (a.ordering_ == b.ordering_)
           && (a.newton_maxsteps_ == b.newton_maxsteps_)
           && (a.newton_damping_factor_mode_ == b.newton_damping_factor_mode_)
           && (a.newton_damping_factor_lower_bound_
               == b.newton_damping_factor_lower_bound_)
           && (a.ism_ == b.ism_) && (a.linsol_ == b.linsol_)
           && (a.atol_ == b.atol_) && (a.rtol_ == b.rtol_)
           && (a.maxsteps_ == b.maxsteps_) && (a.maxstepsB_ == b.maxstepsB_)
           && (a.quad_atol_ == b.quad_atol_) && (a.quad_rtol_ == b.quad_rtol_)
           && (a.maxtime_ == b.maxtime_)
           && (a.getAbsoluteToleranceSteadyState()
               == b.getAbsoluteToleranceSteadyState())
           && (a.getRelativeToleranceSteadyState()
               == b.getRelativeToleranceSteadyState())
           && (a.getAbsoluteToleranceSteadyStateSensi()
               == b.getAbsoluteToleranceSteadyStateSensi())
           && (a.getRelativeToleranceSteadyStateSensi()
               == b.getRelativeToleranceSteadyStateSensi())
           && (a.rtol_fsa_ == b.rtol_fsa_
               || (isNaN(a.rtol_fsa_) && isNaN(b.rtol_fsa_)))
           && (a.atol_fsa_ == b.atol_fsa_
               || (isNaN(a.atol_fsa_) && isNaN(b.atol_fsa_)))
           && (a.rtolB_ == b.rtolB_ || (isNaN(a.rtolB_) && isNaN(b.rtolB_)))
           && (a.atolB_ == b.atolB_ || (isNaN(a.atolB_) && isNaN(b.atolB_)))
           && (a.sensi_ == b.sensi_) && (a.sensi_meth_ == b.sensi_meth_)
           && (a.newton_step_steadystate_conv_
               == b.newton_step_steadystate_conv_)
           && (a.check_sensi_steadystate_conv_
               == b.check_sensi_steadystate_conv_)
           && (a.rdata_mode_ == b.rdata_mode_);
}

void Solver::applyTolerances() const {
    if (!getInitDone())
        throw AmiException("Solver instance was not yet set up, the "
                           "tolerances cannot be applied yet!");

    setSStolerances(rtol_, atol_);
}

void Solver::applyTolerancesFSA() const {
    if (!getInitDone())
        throw AmiException("Solver instance was not yet set up, the "
                           "tolerances cannot be applied yet!");

    if (sensi_ < SensitivityOrder::first)
        return;

    if (nplist()) {
        std::vector<realtype> atols(nplist(), getAbsoluteToleranceFSA());
        setSensSStolerances(getRelativeToleranceFSA(), atols.data());
        setSensErrCon(true);
    }
}

void Solver::applyTolerancesASA(int const which) const {
    if (!getAdjInitDone())
        throw AmiException("Adjoint solver instance was not yet set up, the "
                           "tolerances cannot be applied yet!");

    if (sensi_ < SensitivityOrder::first)
        return;

    /* specify integration tolerances for backward problem */
    setSStolerancesB(which, getRelativeToleranceB(), getAbsoluteToleranceB());
}

void Solver::applyQuadTolerancesASA(int const which) const {
    if (!getAdjInitDone())
        throw AmiException("Adjoint solver instance was not yet set up, the "
                           "tolerances cannot be applied yet!");

    if (sensi_ < SensitivityOrder::first)
        return;

    realtype quad_rtol = isNaN(quad_rtol_) ? rtol_ : quad_rtol_;
    realtype quad_atol = isNaN(quad_atol_) ? atol_ : quad_atol_;

    /* Enable Quadrature Error Control */
    setQuadErrConB(which, !std::isinf(quad_atol) && !std::isinf(quad_rtol));

    quadSStolerancesB(which, quad_rtol, quad_atol);
}

void Solver::applyQuadTolerances() const {
    if (!getQuadInitDone())
        throw AmiException("Quadratures were not initialized, the "
                           "tolerances cannot be applied yet!");

    if (sensi_ < SensitivityOrder::first)
        return;

    realtype quad_rtolF = isNaN(quad_rtol_) ? rtol_ : quad_rtol_;
    realtype quad_atolF = isNaN(quad_atol_) ? atol_ : quad_atol_;

    /* Enable Quadrature Error Control */
    setQuadErrCon(!std::isinf(quad_atolF) && !std::isinf(quad_rtolF));

    quadSStolerances(quad_rtolF, quad_atolF);
}

void Solver::applySensitivityTolerances() const {
    if (sensi_ < SensitivityOrder::first)
        return;

    if (sensi_meth_ == SensitivityMethod::forward)
        applyTolerancesFSA();
    else if (sensi_meth_ == SensitivityMethod::adjoint && getAdjInitDone()) {
        for (int iMem = 0; iMem < (int)solver_memory_B_.size(); ++iMem)
            applyTolerancesASA(iMem);
    }
}

SensitivityMethod Solver::getSensitivityMethod() const { return sensi_meth_; }

SensitivityMethod Solver::getSensitivityMethodPreequilibration() const {
    return sensi_meth_preeq_;
}

void Solver::setSensitivityMethod(SensitivityMethod const sensi_meth) {
    checkSensitivityMethod(sensi_meth, false);
    this->sensi_meth_ = sensi_meth;
}

void Solver::setSensitivityMethodPreequilibration(
    SensitivityMethod const sensi_meth_preeq
) {
    checkSensitivityMethod(sensi_meth_preeq, true);
    sensi_meth_preeq_ = sensi_meth_preeq;
}

void Solver::checkSensitivityMethod(
    SensitivityMethod const sensi_meth, bool preequilibration
) const {
    if (rdata_mode_ == RDataReporting::residuals
        && sensi_meth == SensitivityMethod::adjoint)
        throw AmiException("Adjoint Sensitivity Analysis is not compatible with"
                           " only reporting residuals!");
    if (!preequilibration && sensi_meth != sensi_meth_)
        resetMutableMemory(nx(), nplist(), nquad());
}

int Solver::getNewtonMaxSteps() const { return newton_maxsteps_; }

void Solver::setNewtonMaxSteps(int const newton_maxsteps) {
    if (newton_maxsteps < 0)
        throw AmiException("newton_maxsteps must be a non-negative number");
    newton_maxsteps_ = newton_maxsteps;
}

NewtonDampingFactorMode Solver::getNewtonDampingFactorMode() const {
    return newton_damping_factor_mode_;
}

void Solver::setNewtonDampingFactorMode(
    NewtonDampingFactorMode dampingFactorMode
) {
    newton_damping_factor_mode_ = dampingFactorMode;
}

double Solver::getNewtonDampingFactorLowerBound() const {
    return newton_damping_factor_lower_bound_;
}

void Solver::setNewtonDampingFactorLowerBound(double dampingFactorLowerBound) {
    newton_damping_factor_lower_bound_ = dampingFactorLowerBound;
}

SensitivityOrder Solver::getSensitivityOrder() const { return sensi_; }

void Solver::setSensitivityOrder(SensitivityOrder const sensi) {
    if (sensi_ != sensi)
        resetMutableMemory(nx(), nplist(), nquad());
    sensi_ = sensi;

    if (getInitDone())
        applySensitivityTolerances();
}

double Solver::getRelativeTolerance() const {
    return static_cast<double>(rtol_);
}

void Solver::setRelativeTolerance(double const rtol) {
    if (rtol < 0)
        throw AmiException("rtol must be a non-negative number");

    rtol_ = static_cast<realtype>(rtol);

    if (getInitDone()) {
        applyTolerances();
        applySensitivityTolerances();
    }
}

double Solver::getAbsoluteTolerance() const {
    return static_cast<double>(atol_);
}

void Solver::setAbsoluteTolerance(double atol) {
    if (atol < 0)
        throw AmiException("atol must be a non-negative number");

    atol_ = static_cast<realtype>(atol);

    if (getInitDone()) {
        applyTolerances();
        applySensitivityTolerances();
    }
}

double Solver::getRelativeToleranceFSA() const {
    return static_cast<double>(isNaN(rtol_fsa_) ? rtol_ : rtol_fsa_);
}

void Solver::setRelativeToleranceFSA(double const rtol) {
    if (rtol < 0)
        throw AmiException("rtol must be a non-negative number");

    rtol_fsa_ = static_cast<realtype>(rtol);

    if (getInitDone()) {
        applySensitivityTolerances();
    }
}

double Solver::getAbsoluteToleranceFSA() const {
    return static_cast<double>(isNaN(atol_fsa_) ? atol_ : atol_fsa_);
}

void Solver::setAbsoluteToleranceFSA(double const atol) {
    if (atol < 0)
        throw AmiException("atol must be a non-negative number");

    atol_fsa_ = static_cast<realtype>(atol);

    if (getInitDone()) {
        applySensitivityTolerances();
    }
}

double Solver::getRelativeToleranceB() const {
    return static_cast<double>(isNaN(rtolB_) ? rtol_ : rtolB_);
}

void Solver::setRelativeToleranceB(double const rtol) {
    if (rtol < 0)
        throw AmiException("rtol must be a non-negative number");

    rtolB_ = static_cast<realtype>(rtol);

    if (getInitDone()) {
        applySensitivityTolerances();
    }
}

double Solver::getAbsoluteToleranceB() const {
    return static_cast<double>(isNaN(atolB_) ? atol_ : atolB_);
}

void Solver::setAbsoluteToleranceB(double const atol) {
    if (atol < 0)
        throw AmiException("atol must be a non-negative number");

    atolB_ = static_cast<realtype>(atol);

    if (getInitDone()) {
        applySensitivityTolerances();
    }
}

double Solver::getRelativeToleranceQuadratures() const {
    return static_cast<double>(quad_rtol_);
}

void Solver::setRelativeToleranceQuadratures(double const rtol) {
    if (rtol < 0)
        throw AmiException("rtol must be a non-negative number");

    quad_rtol_ = static_cast<realtype>(rtol);

    if (sensi_meth_ != SensitivityMethod::adjoint)
        return;

    for (int iMem = 0; iMem < (int)solver_memory_B_.size(); ++iMem)
        if (solver_memory_B_.at(iMem))
            applyQuadTolerancesASA(iMem);
}

double Solver::getAbsoluteToleranceQuadratures() const {
    return static_cast<double>(quad_atol_);
}

void Solver::setAbsoluteToleranceQuadratures(double const atol) {
    if (atol < 0)
        throw AmiException("atol must be a non-negative number");

    quad_atol_ = static_cast<realtype>(atol);

    if (sensi_meth_ != SensitivityMethod::adjoint)
        return;

    for (int iMem = 0; iMem < (int)solver_memory_B_.size(); ++iMem)
        if (solver_memory_B_.at(iMem))
            applyTolerancesASA(iMem);
}

double Solver::getSteadyStateToleranceFactor() const {
    return static_cast<double>(ss_tol_factor_);
}

void Solver::setSteadyStateToleranceFactor(double const ss_tol_factor) {
    if (ss_tol_factor < 0)
        throw AmiException("ss_tol_factor must be a non-negative number");

    ss_tol_factor_ = static_cast<realtype>(ss_tol_factor);
}

double Solver::getRelativeToleranceSteadyState() const {
    return static_cast<double>(
        isNaN(ss_rtol_) ? rtol_ * ss_tol_factor_ : ss_rtol_
    );
}

void Solver::setRelativeToleranceSteadyState(double const rtol) {
    if (rtol < 0)
        throw AmiException("rtol must be a non-negative number");

    ss_rtol_ = static_cast<realtype>(rtol);
}

double Solver::getAbsoluteToleranceSteadyState() const {
    return static_cast<double>(
        isNaN(ss_atol_) ? atol_ * ss_tol_factor_ : ss_atol_
    );
}

void Solver::setAbsoluteToleranceSteadyState(double const atol) {
    if (atol < 0)
        throw AmiException("atol must be a non-negative number");

    ss_atol_ = static_cast<realtype>(atol);
}

double Solver::getSteadyStateSensiToleranceFactor() const {
    return static_cast<double>(ss_tol_sensi_factor_);
}

void Solver::setSteadyStateSensiToleranceFactor(double const ss_tol_sensi_factor
) {
    if (ss_tol_sensi_factor < 0)
        throw AmiException("ss_tol_sensi_factor must be a non-negative number");

    ss_tol_sensi_factor_ = static_cast<realtype>(ss_tol_sensi_factor);
}

double Solver::getRelativeToleranceSteadyStateSensi() const {
    return static_cast<double>(
        isNaN(ss_rtol_sensi_) ? rtol_ * ss_tol_sensi_factor_ : ss_rtol_sensi_
    );
}

void Solver::setRelativeToleranceSteadyStateSensi(double const rtol) {
    if (rtol < 0)
        throw AmiException("rtol must be a non-negative number");

    ss_rtol_sensi_ = static_cast<realtype>(rtol);
}

double Solver::getAbsoluteToleranceSteadyStateSensi() const {
    return static_cast<double>(
        isNaN(ss_atol_sensi_) ? atol_ * ss_tol_sensi_factor_ : ss_atol_sensi_
    );
}

void Solver::setAbsoluteToleranceSteadyStateSensi(double const atol) {
    if (atol < 0)
        throw AmiException("atol must be a non-negative number");

    ss_atol_sensi_ = static_cast<realtype>(atol);
}

long int Solver::getMaxSteps() const { return maxsteps_; }

double Solver::getMaxTime() const { return maxtime_.count(); }

void Solver::setMaxTime(double maxtime) {
    maxtime_ = std::chrono::duration<double>(maxtime);
}

void Solver::startTimer() const { simulation_timer_.reset(); }

bool Solver::timeExceeded(int interval) const {
    static int eval_counter = 0;

    // 0 means infinite time
    if (maxtime_.count() == 0)
        return false;

    if (++eval_counter % interval)
        return false;

    eval_counter = 0;
    auto elapsed_s = simulation_timer_.elapsed_seconds();
    return std::chrono::duration<double>(elapsed_s) > maxtime_;
}

void Solver::setMaxSteps(long int const maxsteps) {
    if (maxsteps <= 0)
        throw AmiException("maxsteps must be a positive number");

    maxsteps_ = maxsteps;
    if (getAdjInitDone())
        resetMutableMemory(nx(), nplist(), nquad());
}

long int Solver::getMaxStepsBackwardProblem() const { return maxstepsB_; }

void Solver::setMaxStepsBackwardProblem(long int const maxsteps) {
    if (maxsteps < 0)
        throw AmiException("maxsteps must be a non-negative number");

    maxstepsB_ = maxsteps;
}

LinearMultistepMethod Solver::getLinearMultistepMethod() const { return lmm_; }

void Solver::setLinearMultistepMethod(LinearMultistepMethod const lmm) {
    if (solver_memory_)
        resetMutableMemory(nx(), nplist(), nquad());
    lmm_ = lmm;
}

NonlinearSolverIteration Solver::getNonlinearSolverIteration() const {
    return iter_;
}

void Solver::setNonlinearSolverIteration(NonlinearSolverIteration const iter) {
    if (solver_memory_)
        resetMutableMemory(nx(), nplist(), nquad());
    iter_ = iter;
}

InterpolationType Solver::getInterpolationType() const { return interp_type_; }

void Solver::setInterpolationType(InterpolationType const interpType) {
    if (!solver_memory_B_.empty())
        resetMutableMemory(nx(), nplist(), nquad());
    interp_type_ = interpType;
}

int Solver::getStateOrdering() const { return ordering_; }

void Solver::setStateOrdering(int ordering) {
    ordering_ = ordering;
    if (solver_memory_ && linsol_ == LinearSolver::KLU) {
        auto klu = dynamic_cast<SUNLinSolKLU*>(linear_solver_.get());
        klu->setOrdering(static_cast<SUNLinSolKLU::StateOrdering>(ordering));
        klu = dynamic_cast<SUNLinSolKLU*>(linear_solver_B_.get());
        klu->setOrdering(static_cast<SUNLinSolKLU::StateOrdering>(ordering));
    }
#ifdef SUNDIALS_SUPERLUMT
    if (solverMemory && linsol == LinearSolver::SuperLUMT) {
        auto klu = dynamic_cast<SUNLinSolSuperLUMT*>(linearSolver.get());
        klu->setOrdering(static_cast<SUNLinSolSuperLUMT::StateOrdering>(ordering
        ));
        klu = dynamic_cast<SUNLinSolSuperLUMT*>(linearSolverB.get());
        klu->setOrdering(static_cast<SUNLinSolSuperLUMT::StateOrdering>(ordering
        ));
    }
#endif
}

bool Solver::getStabilityLimitFlag() const { return stldet_; }

void Solver::setStabilityLimitFlag(bool const stldet) {
    stldet_ = stldet;
    if (solver_memory_) {
        setStabLimDet(stldet);
        for (int iMem = 0; iMem < (int)solver_memory_B_.size(); ++iMem)
            if (solver_memory_B_.at(iMem))
                setStabLimDetB(iMem, stldet);
    }
}

LinearSolver Solver::getLinearSolver() const { return linsol_; }

void Solver::setLinearSolver(LinearSolver linsol) {
    if (solver_memory_)
        resetMutableMemory(nx(), nplist(), nquad());
    linsol_ = linsol;
}

InternalSensitivityMethod Solver::getInternalSensitivityMethod() const {
    return ism_;
}

void Solver::setInternalSensitivityMethod(InternalSensitivityMethod const ism) {
    if (solver_memory_)
        resetMutableMemory(nx(), nplist(), nquad());
    ism_ = ism;
}

RDataReporting Solver::getReturnDataReportingMode() const {
    return rdata_mode_;
};

void Solver::setReturnDataReportingMode(RDataReporting rdrm) {
    if (rdrm == RDataReporting::residuals
        && sensi_meth_ == SensitivityMethod::adjoint)
        throw AmiException("Adjoint Sensitivity Analysis cannot report "
                           "residuals!");
    rdata_mode_ = rdrm;
}

void Solver::initializeNonLinearSolverSens(Model const* model) const {
    switch (iter_) {
    case NonlinearSolverIteration::newton:
        switch (ism_) {
        case InternalSensitivityMethod::staggered:
        case InternalSensitivityMethod::simultaneous:
            non_linear_solver_sens_ = std::make_unique<SUNNonLinSolNewton>(
                1 + model->nplist(), x_.getNVector()
            );
            break;
        case InternalSensitivityMethod::staggered1:
            non_linear_solver_sens_
                = std::make_unique<SUNNonLinSolNewton>(x_.getNVector());
            break;
        default:
            throw AmiException(
                "Unsupported internal sensitivity method selected: %d", ism_
            );
        }
        break;
    case NonlinearSolverIteration::fixedpoint:
        switch (ism_) {
        case InternalSensitivityMethod::staggered:
        case InternalSensitivityMethod::simultaneous:
            non_linear_solver_sens_ = std::make_unique<SUNNonLinSolFixedPoint>(
                1 + model->nplist(), x_.getNVector()
            );
            break;
        case InternalSensitivityMethod::staggered1:
            non_linear_solver_sens_
                = std::make_unique<SUNNonLinSolFixedPoint>(x_.getNVector());
            break;
        default:
            throw AmiException(
                "Unsupported internal sensitivity method selected: %d", ism_
            );
        }
        break;
    default:
        throw AmiException(
            "Invalid non-linear solver specified (%d).", static_cast<int>(iter_)
        );
    }

    setNonLinearSolverSens();
}

int Solver::nplist() const { return sx_.getLength(); }

int Solver::nx() const { return x_.getLength(); }

int Solver::nquad() const { return xQB_.getLength(); }

bool Solver::getInitDone() const { return initialized_; };

bool Solver::getSensInitDone() const { return sens_initialized_; }

bool Solver::getAdjInitDone() const { return adj_initialized_; }

bool Solver::getInitDoneB(int const which) const {
    return static_cast<int>(initializedB_.size()) > which
           && initializedB_.at(which);
}

bool Solver::getQuadInitDoneB(int const which) const {
    return static_cast<int>(initializedQB_.size()) > which
           && initializedQB_.at(which);
}

bool Solver::getQuadInitDone() const { return quad_initialized_; }

void Solver::setInitDone() const { initialized_ = true; };

void Solver::setSensInitDone() const { sens_initialized_ = true; }

void Solver::setAdjInitDone() const { adj_initialized_ = true; }

void Solver::setInitDoneB(int const which) const {
    if (which >= static_cast<int>(initializedB_.size()))
        initializedB_.resize(which + 1, false);
    initializedB_.at(which) = true;
}

void Solver::setQuadInitDoneB(int const which) const {
    if (which >= static_cast<int>(initializedQB_.size()))
        initializedQB_.resize(which + 1, false);
    initializedQB_.at(which) = true;
}

void Solver::setQuadInitDone() const { quad_initialized_ = true; }

void Solver::switchForwardSensisOff() const { sensToggleOff(); }

realtype Solver::getCpuTime() const { return cpu_time_; }

realtype Solver::getCpuTimeB() const { return cpu_timeB_; }

void Solver::resetMutableMemory(int const nx, int const nplist, int const nquad)
    const {
    solver_memory_ = nullptr;
    initialized_ = false;
    adj_initialized_ = false;
    sens_initialized_ = false;
    quad_initialized_ = false;
    solver_was_called_F_ = false;
    solver_was_called_B_ = false;

    x_ = AmiVector(nx);
    dx_ = AmiVector(nx);
    sx_ = AmiVectorArray(nx, nplist);
    sdx_ = AmiVectorArray(nx, nplist);

    xB_ = AmiVector(nx);
    dxB_ = AmiVector(nx);
    xQB_ = AmiVector(nquad);
    xQ_ = AmiVector(nx);

    solver_memory_B_.clear();
    initializedB_.clear();
    initializedQB_.clear();
}

void Solver::writeSolution(
    realtype* t, AmiVector& x, AmiVector& dx, AmiVectorArray& sx, AmiVector& xQ
) const {
    *t = gett();
    if (quad_initialized_)
        xQ.copy(getQuadrature(*t));
    if (sens_initialized_)
        sx.copy(getStateSensitivity(*t));
    x.copy(getState(*t));
    dx.copy(getDerivativeState(*t));
}

void Solver::writeSolutionB(
    realtype* t, AmiVector& xB, AmiVector& dxB, AmiVector& xQB, int const which
) const {
    *t = gett();
    xB.copy(getAdjointState(which, *t));
    dxB.copy(getAdjointDerivativeState(which, *t));
    xQB.copy(getAdjointQuadrature(which, *t));
}

AmiVector const& Solver::getState(realtype const t) const {
    if (t == t_)
        return x_;

    if (solver_was_called_F_)
        getDky(t, 0);

    return dky_;
}

AmiVector const& Solver::getDerivativeState(realtype const t) const {
    if (t == t_)
        return dx_;

    if (solver_was_called_F_)
        getDky(t, 1);

    return dky_;
}

AmiVectorArray const& Solver::getStateSensitivity(realtype const t) const {
    if (sens_initialized_ && solver_was_called_F_) {
        if (t == t_) {
            getSens();
        } else {
            getSensDky(t, 0);
        }
    }
    return sx_;
}

AmiVector const&
Solver::getAdjointState(int const which, realtype const t) const {
    if (adj_initialized_) {
        if (solver_was_called_B_) {
            if (t == t_) {
                getB(which);
                return xB_;
            }
            getDkyB(t, 0, which);
        }
    } else {
        dky_.zero();
    }
    return dky_;
}

AmiVector const&
Solver::getAdjointDerivativeState(int const which, realtype const t) const {
    if (adj_initialized_) {
        if (solver_was_called_B_) {
            if (t == t_) {
                getB(which);
                return dxB_;
            }
            getDkyB(t, 1, which);
        }
    } else {
        dky_.zero();
    }
    return dky_;
}

AmiVector const&
Solver::getAdjointQuadrature(int const which, realtype const t) const {
    if (adj_initialized_) {
        if (solver_was_called_B_) {
            if (t == t_) {
                getQuadB(which);
                return xQB_;
            }
            getQuadDkyB(t, 0, which);
        }
    } else {
        xQB_.zero();
    }
    return xQB_;
}

AmiVector const& Solver::getQuadrature(realtype t) const {
    if (quad_initialized_) {
        if (solver_was_called_F_) {
            if (t == t_) {
                getQuad(t);
                return xQ_;
            }
            getQuadDky(t, 0);
        }
    } else {
        xQ_.zero();
    }
    return xQ_;
}

realtype Solver::gett() const { return t_; }

void wrapErrHandlerFn(
    int error_code, char const* module, char const* function, char* msg,
    void* eh_data
) {
    constexpr int BUF_SIZE = 250;
    char buffer[BUF_SIZE];
    char buffid[BUF_SIZE];
    snprintf(
        buffer, BUF_SIZE, "AMICI ERROR: in module %s in function %s : %s ",
        module, function, msg
    );
    switch (error_code) {
    case 99:
        snprintf(buffid, BUF_SIZE, "%s:%s:WARNING", module, function);
        break;

    case AMICI_TOO_MUCH_WORK:
        snprintf(buffid, BUF_SIZE, "%s:%s:TOO_MUCH_WORK", module, function);
        break;

    case AMICI_TOO_MUCH_ACC:
        snprintf(buffid, BUF_SIZE, "%s:%s:TOO_MUCH_ACC", module, function);
        break;

    case AMICI_ERR_FAILURE:
        snprintf(buffid, BUF_SIZE, "%s:%s:ERR_FAILURE", module, function);
        break;

    case AMICI_CONV_FAILURE:
        snprintf(buffid, BUF_SIZE, "%s:%s:CONV_FAILURE", module, function);
        break;

    case AMICI_RHSFUNC_FAIL:
        snprintf(buffid, BUF_SIZE, "%s:%s:RHSFUNC_FAIL", module, function);
        break;

    case AMICI_FIRST_RHSFUNC_ERR:
        snprintf(buffid, BUF_SIZE, "%s:%s:FIRST_RHSFUNC_ERR", module, function);
        break;

    default:
        snprintf(buffid, BUF_SIZE, "%s:%s:OTHER", module, function);
        break;
    }

    if (!eh_data) {
        throw std::runtime_error("eh_data unset");
    }
    auto solver = static_cast<Solver const*>(eh_data);
    if (solver->logger)
        solver->logger->log(LogSeverity::debug, buffid, buffer);
}

} // namespace amici
