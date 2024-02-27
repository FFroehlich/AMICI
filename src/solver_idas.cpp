#include "amici/solver_idas.h"

#include "amici/exception.h"
#include "amici/model_dae.h"
#include "amici/sundials_linsol_wrapper.h"

#include <idas/idas.h>
#include <idas/idas_impl.h>

#include <amd.h>
#include <btf.h>
#include <colamd.h>
#include <klu.h>

#define ONE RCONST(1.0)

namespace amici {

/*
 * The following static members are callback function to CVODES.
 * Their signatures must not be changes.
 */

static int
fxdot(realtype t, N_Vector x, N_Vector dx, N_Vector xdot, void* user_data);

static int
fJ(realtype t, realtype cj, N_Vector x, N_Vector dx, N_Vector xdot, SUNMatrix J,
   void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

static int fJSparse(
    realtype t, realtype cj, N_Vector x, N_Vector dx, N_Vector xdot,
    SUNMatrix J, void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3
);

static int
fJB(realtype t, realtype cj, N_Vector x, N_Vector dx, N_Vector xB, N_Vector dxB,
    N_Vector xBdot, SUNMatrix JB, void* user_data, N_Vector tmp1B,
    N_Vector tmp2B, N_Vector tmp3B);

static int fJSparseB(
    realtype t, realtype cj, N_Vector x, N_Vector dx, N_Vector xB, N_Vector dxB,
    N_Vector xBdot, SUNMatrix JB, void* user_data, N_Vector tmp1B,
    N_Vector tmp2B, N_Vector tmp3B
);

static int fJBand(
    realtype t, realtype cj, N_Vector x, N_Vector dx, N_Vector xdot,
    SUNMatrix J, void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3
);

static int fJBandB(
    realtype t, realtype cj, N_Vector x, N_Vector dx, N_Vector xB, N_Vector dxB,
    N_Vector xBdot, SUNMatrix JB, void* user_data, N_Vector tmp1B,
    N_Vector tmp2B, N_Vector tmp3B
);

static int
fJv(realtype t, N_Vector x, N_Vector dx, N_Vector xdot, N_Vector v, N_Vector Jv,
    realtype cj, void* user_data, N_Vector tmp1, N_Vector tmp2);

static int fJvB(
    realtype t, N_Vector x, N_Vector dx, N_Vector xB, N_Vector dxB,
    N_Vector xBdot, N_Vector vB, N_Vector JvB, realtype cj, void* user_data,
    N_Vector tmpB1, N_Vector tmpB2
);

static int
froot(realtype t, N_Vector x, N_Vector dx, realtype* root, void* user_data);

static int fxBdot(
    realtype t, N_Vector x, N_Vector dx, N_Vector xB, N_Vector dxB,
    N_Vector xBdot, void* user_data
);

static int fqBdot(
    realtype t, N_Vector x, N_Vector dx, N_Vector xB, N_Vector dxB,
    N_Vector qBdot, void* user_data
);

static int fxBdot_ss(
    realtype t, N_Vector xB, N_Vector dxB, N_Vector xBdot, void* user_data
);

static int fqBdot_ss(
    realtype t, N_Vector xB, N_Vector dxB, N_Vector qBdot, void* user_data
);

static int fJSparseB_ss(
    realtype t, realtype cj, N_Vector x, N_Vector dx, N_Vector xBdot,
    SUNMatrix JB, void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3
);

static int fsxdot(
    int Ns, realtype t, N_Vector x, N_Vector dx, N_Vector xdot, N_Vector* sx,
    N_Vector* sdx, N_Vector* sxdot, void* user_data, N_Vector tmp1,
    N_Vector tmp2, N_Vector tmp3
);

/* Function implementations */

void IDASolver::init(
    realtype const t0, AmiVector const& x0, AmiVector const& dx0
) const {
    int status;
    solver_was_called_F_ = false;
    t_ = t0;
    x_ = x0;
    dx_ = dx0;
    if (getInitDone()) {
        status = IDAReInit(
            solver_memory_.get(), t_, x_.getNVector(), dx_.getNVector()
        );
    } else {
        status = IDAInit(
            solver_memory_.get(), fxdot, t_, x_.getNVector(), dx_.getNVector()
        );
        setInitDone();
    }
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDAInit");
}

void IDASolver::initSteadystate(
    realtype const /*t0*/, AmiVector const& /*x0*/, AmiVector const& /*dx0*/
) const {
    /* We need to set the steadystate rhs function. SUndials doesn't have this
       in its public api, so we have to change it in the solver memory,
       as re-calling init would unset solver settings. */
    auto ida_mem = static_cast<IDAMem>(solver_memory_.get());
    ida_mem->ida_res = fxBdot_ss;
}

void IDASolver::sensInit1(AmiVectorArray const& sx0, AmiVectorArray const& sdx0)
    const {
    int status = IDA_SUCCESS;
    sx_ = sx0;
    sdx_ = sdx0;
    if (getSensitivityMethod() == SensitivityMethod::forward && nplist() > 0) {
        if (getSensInitDone()) {
            status = IDASensReInit(
                solver_memory_.get(),
                static_cast<int>(getInternalSensitivityMethod()),
                sx_.getNVectorArray(), sdx_.getNVectorArray()
            );
        } else {
            status = IDASensInit(
                solver_memory_.get(), nplist(),
                static_cast<int>(getInternalSensitivityMethod()), fsxdot,
                sx_.getNVectorArray(), sdx_.getNVectorArray()
            );
            setSensInitDone();
        }
    }
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDASensInit");
}

void IDASolver::binit(
    int const which, realtype const tf, AmiVector const& xB0,
    AmiVector const& dxB0
) const {
    int status;
    xB_ = xB0;
    dxB_ = dxB0;
    if (getInitDoneB(which))
        status = IDAReInitB(
            solver_memory_.get(), which, tf, xB_.getNVector(), dxB_.getNVector()
        );
    else {

        status = IDAInitB(
            solver_memory_.get(), which, fxBdot, tf, xB_.getNVector(),
            dxB_.getNVector()
        );
        setInitDoneB(which);
    }
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDAInitB");
}

void IDASolver::qbinit(int const which, AmiVector const& xQB0) const {
    int status;
    xQB_.copy(xQB0);
    if (getQuadInitDoneB(which))
        status = IDAQuadReInitB(solver_memory_.get(), which, xQB_.getNVector());
    else {
        status = IDAQuadInitB(
            solver_memory_.get(), which, fqBdot, xQB_.getNVector()
        );
        setQuadInitDoneB(which);
    }
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDAQuadInitB");
}

void IDASolver::rootInit(int ne) const {
    int status = IDARootInit(solver_memory_.get(), ne, froot);
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDARootInit");
}

void IDASolver::setDenseJacFn() const {
    int status = IDASetJacFn(solver_memory_.get(), fJ);
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDADlsSetDenseJacFn");
}

void IDASolver::setSparseJacFn() const {
    int status = IDASetJacFn(solver_memory_.get(), fJSparse);
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDASlsSetSparseJacFn");
}

void IDASolver::setBandJacFn() const {
    int status = IDASetJacFn(solver_memory_.get(), fJBand);
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDADlsSetBandJacFn");
}

void IDASolver::setJacTimesVecFn() const {
    int status = IDASetJacTimes(solver_memory_.get(), nullptr, fJv);
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDASpilsSetJacTimesVecFn");
}

void IDASolver::setDenseJacFnB(int const which) const {
    int status = IDASetJacFnB(solver_memory_.get(), which, fJB);
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDADlsSetDenseJacFnB");
}

void IDASolver::setSparseJacFnB(int const which) const {
    int status = IDASetJacFnB(solver_memory_.get(), which, fJSparseB);
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDASlsSetSparseJacFnB");
}

void IDASolver::setBandJacFnB(int const which) const {
    int status = IDASetJacFnB(solver_memory_.get(), which, fJBandB);
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDADlsSetBandJacFnB");
}

void IDASolver::setJacTimesVecFnB(int const which) const {
    int status = IDASetJacTimesB(solver_memory_.get(), which, nullptr, fJvB);
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDASpilsSetJacTimesVecFnB");
}

void IDASolver::setSparseJacFn_ss() const {
    int status = IDASetJacFn(solver_memory_.get(), fJSparseB_ss);
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDASetJacFn");
}

Solver* IDASolver::clone() const { return new IDASolver(*this); }

void IDASolver::allocateSolver() const {
    if (!solver_memory_)
        solver_memory_ = std::unique_ptr<void, std::function<void(void*)>>(
            IDACreate(), [](void* ptr) { IDAFree(&ptr); }
        );
}

void IDASolver::setSStolerances(realtype const rtol, realtype const atol)
    const {
    int status = IDASStolerances(solver_memory_.get(), rtol, atol);
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDASStolerances");
}
void IDASolver::setSensSStolerances(realtype const rtol, realtype const* atol)
    const {
    int status = IDASensSStolerances(
        solver_memory_.get(), rtol, const_cast<realtype*>(atol)
    );
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDASensEEtolerances");
}

void IDASolver::setSensErrCon(bool const error_corr) const {
    int status = IDASetSensErrCon(solver_memory_.get(), error_corr);
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDASetSensErrCon");
}

void IDASolver::setQuadErrConB(int const which, bool const flag) const {
    int status = IDASetQuadErrConB(solver_memory_.get(), which, flag);
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDASetQuadErrConB");
}

void IDASolver::setQuadErrCon(bool const flag) const {
    int status = IDASetQuadErrCon(solver_memory_.get(), flag);
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDASetQuadErrCon");
}

void IDASolver::getRootInfo(int* rootsfound) const {
    int status = IDAGetRootInfo(solver_memory_.get(), rootsfound);
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDAGetRootInfo");
}

void IDASolver::setErrHandlerFn() const {
    int status = IDASetErrHandlerFn(
        solver_memory_.get(), wrapErrHandlerFn,
        reinterpret_cast<void*>(const_cast<IDASolver*>(this))
    );
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDASetErrHandlerFn");
}

void IDASolver::setUserData() const {
    int status = IDASetUserData(solver_memory_.get(), &user_data);
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDASetUserData");
}

void IDASolver::setUserDataB(int which) const {
    int status = IDASetUserDataB(solver_memory_.get(), which, &user_data);
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDASetUserDataB");
}

void IDASolver::setMaxNumSteps(long int const mxsteps) const {
    int status = IDASetMaxNumSteps(solver_memory_.get(), mxsteps);
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDASetMaxNumSteps");
}

void IDASolver::setStabLimDet(int const /*stldet*/) const {}

void IDASolver::setStabLimDetB(int const /*which*/, int const /*stldet*/)
    const {}

void IDASolver::setId(Model const* model) const {

    N_Vector id = N_VMake_Serial(
        model->nx_solver, const_cast<realtype*>(model->idlist.data())
    );

    int status = IDASetId(solver_memory_.get(), id);
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDASetMaxNumSteps");

    N_VDestroy_Serial(id);
}

void IDASolver::setSuppressAlg(bool const flag) const {
    int status = IDASetSuppressAlg(solver_memory_.get(), flag);
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDASetSuppressAlg");
}

void IDASolver::resetState(
    void* ami_mem, const_N_Vector yy0, const_N_Vector yp0
) const {

    auto ida_mem = static_cast<IDAMem>(ami_mem);
    /* here we force the order in the next step to zero, and update the
     phi arrays, this is largely copied from IDAReInit with
     explanations from idas_impl.h
     */

    /* Initialize the phi array */

    N_VScale(ONE, const_cast<N_Vector>(yy0), ida_mem->ida_phi[0]);
    N_VScale(ONE, const_cast<N_Vector>(yp0), ida_mem->ida_phi[1]);

    /* Set step parameters */

    /* current order */
    ida_mem->ida_kk = 0;
}

void IDASolver::reInitPostProcessF(realtype const tnext) const {
    reInitPostProcess(solver_memory_.get(), &t_, &x_, &dx_, tnext);
}

void IDASolver::reInitPostProcessB(realtype const tnext) const {
    realtype tBret;
    auto ida_mem = static_cast<IDAMem>(solver_memory_.get());
    auto idaadj_mem = ida_mem->ida_adj_mem;
    auto idaB_mem = idaadj_mem->IDAB_mem;
    // loop over all backward problems
    while (idaB_mem != nullptr) {
        // store current backward problem in ca_mem to make it accessible in
        // adjoint rhs wrapper functions
        idaadj_mem->ia_bckpbCrt = idaB_mem;
        reInitPostProcess(
            static_cast<void*>(idaB_mem->IDA_mem), &tBret, &xB_, &dxB_, tnext
        );
        // idaB_mem->ida_tout = tBret;
        idaB_mem = idaB_mem->ida_next;
    }
    force_reinit_postprocess_B_ = false;
}

void IDASolver::reInitPostProcess(
    void* ami_mem, realtype* t, AmiVector* yout, AmiVector* ypout, realtype tout
) const {
    auto ida_mem = static_cast<IDAMem>(ami_mem);
    auto nst_tmp = ida_mem->ida_nst;
    ida_mem->ida_nst = 0;

    auto status = IDASetStopTime(ida_mem, tout);
    if (status != IDA_SUCCESS)
        throw IDAException(status, "CVodeSetStopTime");

    status = IDASolve(
        ami_mem, tout, t, yout->getNVector(), ypout->getNVector(), IDA_ONE_STEP
    );

    if (status != IDA_SUCCESS)
        throw IDAException(status, "reInitPostProcess");

    ida_mem->ida_nst = nst_tmp + 1;
    if (ida_mem->ida_adjMallocDone == SUNTRUE) {
        /* add new step to history array, this is copied from CVodeF */
        auto ia_mem = ida_mem->ida_adj_mem;
        auto dt_mem = ia_mem->dt_mem;

        if (ida_mem->ida_nst % ia_mem->ia_nsteps == 0) {
            /* currently not implemented, we should never get here as we
             limit cv_mem->cv_nst < ca_mem->ca_nsteps, keeping this for
             future regression */
            throw IDAException(AMICI_ERROR, "reInitPostProcess");
        }

        /* Load next point in dt_mem */
        dt_mem[ida_mem->ida_nst % ia_mem->ia_nsteps]->t = *t;
        ia_mem->ia_storePnt(
            ida_mem, dt_mem[ida_mem->ida_nst % ia_mem->ia_nsteps]
        );

        /* Set t1 field of the current ckeck point structure
         for the case in which there will be no future
         check points */
        ia_mem->ck_mem->ck_t1 = *t;

        /* tfinal is now set to *tret */
        ia_mem->ia_tfinal = *t;
    }
}

void IDASolver::reInit(
    realtype const t0, AmiVector const& yy0, AmiVector const& yp0
) const {

    auto ida_mem = static_cast<IDAMem>(solver_memory_.get());
    ida_mem->ida_tn = t0;
    if (solver_was_called_F_)
        force_reinit_postprocess_F_ = true;
    x_.copy(yy0);
    dx_.copy(yp0);
    resetState(ida_mem, x_.getNVector(), xB_.getNVector());
}

void IDASolver::sensReInit(
    AmiVectorArray const& yyS0, AmiVectorArray const& ypS0
) const {
    auto ida_mem = static_cast<IDAMem>(solver_memory_.get());
    /* Initialize znS[0] in the history array */
    for (int is = 0; is < nplist(); is++)
        ida_mem->ida_cvals[is] = ONE;
    if (solver_was_called_F_)
        force_reinit_postprocess_F_ = true;
    sx_.copy(yyS0);
    sdx_.copy(ypS0);
    auto status = N_VScaleVectorArray(
        nplist(), ida_mem->ida_cvals, sx_.getNVectorArray(),
        ida_mem->ida_phiS[0]
    );
    if (status != IDA_SUCCESS)
        throw IDAException(IDA_VECTOROP_ERR, "IDASensReInit");
    status = N_VScaleVectorArray(
        nplist(), ida_mem->ida_cvals, sdx_.getNVectorArray(),
        ida_mem->ida_phiS[1]
    );
    if (status != IDA_SUCCESS)
        throw IDAException(IDA_VECTOROP_ERR, "IDASensReInit");
}

void IDASolver::sensToggleOff() const {
    auto status = IDASensToggleOff(solver_memory_.get());
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDASensToggleOff");
    IDASensFree(solver_memory_.get());
    /* need to deallocate sensi memory, otherwise can't reenable */
    sens_initialized_ = false;
}

void IDASolver::reInitB(
    int const which, realtype const tB0, AmiVector const& yyB0,
    AmiVector const& ypB0
) const {

    auto ida_memB
        = static_cast<IDAMem>(IDAGetAdjIDABmem(solver_memory_.get(), which));
    if (solver_was_called_B_)
        force_reinit_postprocess_B_ = true;
    ida_memB->ida_tn = tB0;
    xB_.copy(yyB0);
    dxB_.copy(ypB0);
    resetState(ida_memB, xB_.getNVector(), dxB_.getNVector());
}

void IDASolver::quadReInitB(int const which, AmiVector const& yQB0) const {
    auto ida_memB
        = static_cast<IDAMem>(IDAGetAdjIDABmem(solver_memory_.get(), which));
    if (solver_was_called_B_)
        force_reinit_postprocess_B_ = true;
    xQB_.copy(yQB0);
    N_VScale(ONE, xQB_.getNVector(), ida_memB->ida_phiQ[0]);
}

void IDASolver::setSensParams(
    realtype const* p, realtype const* pbar, int const* plist
) const {
    int status = IDASetSensParams(
        solver_memory_.get(), const_cast<realtype*>(p),
        const_cast<realtype*>(pbar), const_cast<int*>(plist)
    );
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDASetSensParams");
}

void IDASolver::getDky(realtype const t, int const k) const {
    int status = IDAGetDky(solver_memory_.get(), t, k, dky_.getNVector());
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDAGetDky");
}

void IDASolver::getSens() const {
    realtype tDummy = 0;
    int status
        = IDAGetSens(solver_memory_.get(), &tDummy, sx_.getNVectorArray());
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDAGetSens");
}

void IDASolver::getSensDky(realtype const t, int const k) const {
    int status
        = IDAGetSensDky(solver_memory_.get(), t, k, sx_.getNVectorArray());
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDAGetSens");
}

void IDASolver::getB(int const which) const {
    realtype tDummy = 0;
    int status = IDAGetB(
        solver_memory_.get(), which, &tDummy, xB_.getNVector(),
        dxB_.getNVector()
    );
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDAGetB");
}

void IDASolver::getDkyB(realtype const t, int k, int const which) const {
    int status = IDAGetDky(
        IDAGetAdjIDABmem(solver_memory_.get(), which), t, k, dky_.getNVector()
    );
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDAGetB");
}
void IDASolver::getQuadB(int which) const {
    realtype tDummy = 0;
    int status
        = IDAGetQuadB(solver_memory_.get(), which, &tDummy, xQB_.getNVector());
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDAGetQuadB");
}

void IDASolver::getQuad(realtype& t) const {
    int status = IDAGetQuad(solver_memory_.get(), &t, xQ_.getNVector());
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDAGetQuad");
}

void IDASolver::getQuadDkyB(realtype const t, int k, int const which) const {
    int status = IDAGetQuadDky(
        IDAGetAdjIDABmem(solver_memory_.get(), which), t, k, xQB_.getNVector()
    );
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDAGetB");
}

void IDASolver::getQuadDky(realtype const t, int const k) const {
    int status = IDAGetQuadDky(solver_memory_.get(), t, k, xQ_.getNVector());
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDAGetQuadDky");
}

void IDASolver::adjInit() const {
    int status;
    if (getAdjInitDone()) {
        status = IDAAdjReInit(solver_memory_.get());
    } else {
        status = IDAAdjInit(
            solver_memory_.get(), static_cast<int>(maxsteps_),
            static_cast<int>(interp_type_)
        );
        setAdjInitDone();
    }
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDAAdjInit");
}

void IDASolver::quadInit(AmiVector const& xQ0) const {
    int status;
    xQ_.copy(xQ0);
    if (getQuadInitDone()) {
        status = IDAQuadReInit(
            solver_memory_.get(), const_cast<N_Vector>(xQ0.getNVector())
        );
    } else {
        status = IDAQuadInit(solver_memory_.get(), fqBdot_ss, xQ_.getNVector());
        setQuadInitDone();
    }
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDAQuadInit");
}

void IDASolver::allocateSolverB(int* which) const {
    if (!solver_memory_B_.empty()) {
        *which = 0;
        return;
    }
    int status = IDACreateB(solver_memory_.get(), which);
    if (*which + 1 > static_cast<int>(solver_memory_B_.size()))
        solver_memory_B_.resize(*which + 1);
    solver_memory_B_.at(*which)
        = std::unique_ptr<void, std::function<void(void*)>>(
            getAdjBmem(solver_memory_.get(), *which), [](void* /*ptr*/) {}
        );
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDACreateB");
}

void IDASolver::setSStolerancesB(
    int const which, realtype const relTolB, realtype const absTolB
) const {
    int status
        = IDASStolerancesB(solver_memory_.get(), which, relTolB, absTolB);
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDASStolerancesB");
}

void IDASolver::quadSStolerancesB(
    int const which, realtype const reltolQB, realtype const abstolQB
) const {
    int status
        = IDAQuadSStolerancesB(solver_memory_.get(), which, reltolQB, abstolQB);
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDAQuadSStolerancesB");
}

void IDASolver::quadSStolerances(
    realtype const reltolQB, realtype const abstolQB
) const {
    int status = IDAQuadSStolerances(solver_memory_.get(), reltolQB, abstolQB);
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDAQuadSStolerances");
}

int IDASolver::solve(realtype const tout, int const itask) const {
    if (force_reinit_postprocess_F_)
        reInitPostProcessF(tout);
    int status = IDASolve(
        solver_memory_.get(), tout, &t_, x_.getNVector(), dx_.getNVector(),
        itask
    );
    solver_was_called_F_ = true;
    if (status < 0) // status > 0 is okay and is used for e.g. root return
        throw IntegrationFailure(status, t_);
    return status;
}

int IDASolver::solveF(realtype const tout, int const itask, int* ncheckPtr)
    const {
    if (force_reinit_postprocess_F_)
        reInitPostProcessF(tout);
    int status = IDASolveF(
        solver_memory_.get(), tout, &t_, x_.getNVector(), xB_.getNVector(),
        itask, ncheckPtr
    );
    solver_was_called_F_ = true;
    if (status < 0) // status > 0 is okay and is used for e.g. root return
        throw IntegrationFailure(status, t_);
    return status;
}

void IDASolver::solveB(realtype const tBout, int const itaskB) const {
    if (force_reinit_postprocess_B_)
        reInitPostProcessB(tBout);
    int status = IDASolveB(solver_memory_.get(), tBout, itaskB);
    solver_was_called_B_ = true;
    if (status != IDA_SUCCESS)
        throw IntegrationFailure(status, tBout);
}

void IDASolver::setMaxNumStepsB(int const which, long int const mxstepsB)
    const {
    int status = IDASetMaxNumStepsB(solver_memory_.get(), which, mxstepsB);
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDASetMaxNumStepsB");
}

void IDASolver::diag() const {
    throw AmiException("Diag Solver was not implemented for DAEs");
}

void IDASolver::diagB(int const /*which*/) const {
    throw AmiException("Diag Solver was not implemented for DAEs");
}

void IDASolver::getNumSteps(void const* ami_mem, long int* numsteps) const {
    int status = IDAGetNumSteps(const_cast<void*>(ami_mem), numsteps);
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDAGetNumSteps");
}

void IDASolver::getNumRhsEvals(void const* ami_mem, long int* numrhsevals)
    const {
    int status = IDAGetNumResEvals(const_cast<void*>(ami_mem), numrhsevals);
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDAGetNumResEvals");
}

void IDASolver::getNumErrTestFails(
    void const* ami_mem, long int* numerrtestfails
) const {
    int status
        = IDAGetNumErrTestFails(const_cast<void*>(ami_mem), numerrtestfails);
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDAGetNumErrTestFails");
}

void IDASolver::getNumNonlinSolvConvFails(
    void const* ami_mem, long int* numnonlinsolvconvfails
) const {
    int status = IDAGetNumNonlinSolvConvFails(
        const_cast<void*>(ami_mem), numnonlinsolvconvfails
    );
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDAGetNumNonlinSolvConvFails");
}

void IDASolver::getLastOrder(void const* ami_mem, int* order) const {
    int status = IDAGetLastOrder(const_cast<void*>(ami_mem), order);
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDAGetLastOrder");
}

void* IDASolver::getAdjBmem(void* ami_mem, int which) const {
    return IDAGetAdjIDABmem(ami_mem, which);
}

void IDASolver::calcIC(realtype tout1) const {
    int status = IDACalcIC(solver_memory_.get(), IDA_YA_YDP_INIT, tout1);
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDACalcIC");
    status = IDAGetConsistentIC(
        solver_memory_.get(), x_.getNVector(), dx_.getNVector()
    );
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDACalcIC");
}

void IDASolver::calcICB(int const which, realtype const tout1) const {
    int status = IDACalcICB(
        solver_memory_.get(), which, tout1, xB_.getNVector(), dxB_.getNVector()
    );
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDACalcICB");
}

void IDASolver::setStopTime(realtype const tstop) const {
    int status = IDASetStopTime(solver_memory_.get(), tstop);
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDASetStopTime");
}

void IDASolver::turnOffRootFinding() const {
    int status = IDARootInit(solver_memory_.get(), 0, nullptr);
    if (status != IDA_SUCCESS)
        throw IDAException(status, "IDARootInit");
}

Model const* IDASolver::getModel() const {
    if (!solver_memory_)
        throw AmiException(
            "Solver has not been allocated, information is not available"
        );
    auto ida_mem = static_cast<IDAMem>(solver_memory_.get());
    auto user_data = static_cast<user_data_type*>(ida_mem->ida_user_data);
    if (user_data)
        return user_data->first;
    return nullptr;
}

void IDASolver::setLinearSolver() const {
    int status = IDASetLinearSolver(
        solver_memory_.get(), linear_solver_->get(), linear_solver_->getMatrix()
    );
    if (status != IDA_SUCCESS)
        throw IDAException(status, "setLinearSolver");
}

void IDASolver::setLinearSolverB(int const which) const {
    int status = IDASetLinearSolverB(
        solver_memory_B_[which].get(), which, linear_solver_B_->get(),
        linear_solver_B_->getMatrix()
    );
    if (status != IDA_SUCCESS)
        throw IDAException(status, "setLinearSolverB");
}

void IDASolver::setNonLinearSolver() const {
    int status = IDASetNonlinearSolver(
        solver_memory_.get(), non_linear_solver_->get()
    );
    if (status != IDA_SUCCESS)
        throw CvodeException(status, "CVodeSetNonlinearSolver");
}

void IDASolver::setNonLinearSolverSens() const {
    if (getSensitivityOrder() < SensitivityOrder::first)
        return;
    if (getSensitivityMethod() != SensitivityMethod::forward)
        return;

    int status = IDA_SUCCESS;

    switch (ism_) {
    case InternalSensitivityMethod::staggered:
        status = IDASetNonlinearSolverSensStg(
            solver_memory_.get(), non_linear_solver_sens_->get()
        );
        break;
    case InternalSensitivityMethod::simultaneous:
        status = IDASetNonlinearSolverSensSim(
            solver_memory_.get(), non_linear_solver_sens_->get()
        );
        break;
    case InternalSensitivityMethod::staggered1:
    default:
        throw AmiException(
            "Unsupported internal sensitivity method selected: %d", ism_
        );
    }

    if (status != IDA_SUCCESS)
        throw CvodeException(status, "CVodeSolver::setNonLinearSolverSens");
}

void IDASolver::setNonLinearSolverB(int which) const {
    int status = IDASetNonlinearSolverB(
        solver_memory_.get(), which, non_linear_solver_B_->get()
    );
    if (status != IDA_SUCCESS)
        throw CvodeException(status, "CVodeSetNonlinearSolverB");
}

/**
 * @brief Jacobian of xdot with respect to states x
 * @param N number of state variables
 * @param t timepoint
 * @param cj scaling factor, inverse of the step size
 * @param x Vector with the states
 * @param dx Vector with the derivative states
 * @param xdot Vector with the right hand side
 * @param J Matrix to which the Jacobian will be written
 * @param user_data object with user input
 * @param tmp1 temporary storage vector
 * @param tmp2 temporary storage vector
 * @param tmp3 temporary storage vector
 * @return status flag indicating successful execution
 **/
int fJ(
    realtype t, realtype cj, N_Vector x, N_Vector dx, N_Vector xdot,
    SUNMatrix J, void* user_data, N_Vector /*tmp1*/, N_Vector /*tmp2*/,
    N_Vector /*tmp3*/
) {
    auto typed_udata = static_cast<IDASolver::user_data_type*>(user_data);
    Expects(typed_udata);
    auto model = dynamic_cast<Model_DAE*>(typed_udata->first);
    Expects(model);
    model->fJ(t, cj, x, dx, xdot, J);
    return model->checkFinite(J, ModelQuantity::J, t);
}

/**
 * @brief Jacobian of xBdot with respect to adjoint state xB
 * @param t timepoint
 * @param cj scaling factor, inverse of the step size
 * @param x Vector with the states
 * @param dx Vector with the derivative states
 * @param xB Vector with the adjoint states
 * @param dxB Vector with the adjoint derivative states
 * @param xBdot Vector with the adjoint right hand side
 * @param JB Matrix to which the Jacobian will be written
 * @param user_data object with user input @type Model_DAE
 * @param tmp1B temporary storage vector
 * @param tmp2B temporary storage vector
 * @param tmp3B temporary storage vector
 * @return status flag indicating successful execution
 **/
int fJB(
    realtype t, realtype cj, N_Vector x, N_Vector dx, N_Vector xB, N_Vector dxB,
    N_Vector /*xBdot*/, SUNMatrix JB, void* user_data, N_Vector /*tmp1B*/,
    N_Vector /*tmp2B*/, N_Vector /*tmp3B*/
) {
    auto typed_udata = static_cast<IDASolver::user_data_type*>(user_data);
    Expects(typed_udata);
    auto model = dynamic_cast<Model_DAE*>(typed_udata->first);
    Expects(model);

    model->fJB(t, cj, x, dx, xB, dxB, JB);
    return model->checkFinite(JB, ModelQuantity::JB, t);
}

/**
 * @brief J in sparse form (for sparse solvers from the SuiteSparse Package)
 * @param t timepoint
 * @param cj scalar in Jacobian (inverse stepsize)
 * @param x Vector with the states
 * @param dx Vector with the derivative states
 * @param xdot Vector with the right hand side
 * @param J Matrix to which the Jacobian will be written
 * @param user_data object with user input
 * @param tmp1 temporary storage vector
 * @param tmp2 temporary storage vector
 * @param tmp3 temporary storage vector
 * @return status flag indicating successful execution
 */
int fJSparse(
    realtype t, realtype cj, N_Vector x, N_Vector dx, N_Vector /*xdot*/,
    SUNMatrix J, void* user_data, N_Vector /*tmp1*/, N_Vector /*tmp2*/,
    N_Vector /*tmp3*/
) {
    auto typed_udata = static_cast<IDASolver::user_data_type*>(user_data);
    Expects(typed_udata);
    auto model = dynamic_cast<Model_DAE*>(typed_udata->first);
    Expects(model);

    model->fJSparse(t, cj, x, dx, J);
    return model->checkFinite(J, ModelQuantity::J, t);
}

/**
 * @brief JB in sparse form (for sparse solvers from the SuiteSparse Package)
 * @param t timepoint
 * @param cj scalar in Jacobian
 * @param x Vector with the states
 * @param dx Vector with the derivative states
 * @param xB Vector with the adjoint states
 * @param dxB Vector with the adjoint derivative states
 * @param xBdot Vector with the adjoint right hand side
 * @param JB Matrix to which the Jacobian will be written
 * @param user_data object with user input
 * @param tmp1B temporary storage vector
 * @param tmp2B temporary storage vector
 * @param tmp3B temporary storage vector
 * @return status flag indicating successful execution
 */
int fJSparseB(
    realtype t, realtype cj, N_Vector x, N_Vector dx, N_Vector xB, N_Vector dxB,
    N_Vector /*xBdot*/, SUNMatrix JB, void* user_data, N_Vector /*tmp1B*/,
    N_Vector /*tmp2B*/, N_Vector /*tmp3B*/
) {
    auto typed_udata = static_cast<IDASolver::user_data_type*>(user_data);
    Expects(typed_udata);
    auto model = dynamic_cast<Model_DAE*>(typed_udata->first);
    Expects(model);

    model->fJSparseB(t, cj, x, dx, xB, dxB, JB);
    return model->checkFinite(JB, ModelQuantity::JB, t);
}

/**
 * @brief J in banded form (for banded solvers)
 * @param t timepoint
 * @param cj scalar in Jacobian (inverse stepsize)
 * @param x Vector with the states
 * @param dx Vector with the derivative states
 * @param xdot Vector with the right hand side
 * @param J Matrix to which the Jacobian will be written
 * @param user_data object with user input
 * @param tmp1 temporary storage vector
 * @param tmp2 temporary storage vector
 * @param tmp3 temporary storage vector
 * @return status flag indicating successful execution
 */
int fJBand(
    realtype t, realtype cj, N_Vector x, N_Vector dx, N_Vector xdot,
    SUNMatrix J, void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3
) {
    return fJ(t, cj, x, dx, xdot, J, user_data, tmp1, tmp2, tmp3);
}

/**
 * @brief JB in banded form (for banded solvers)
 * @param t timepoint
 * @param cj scalar in Jacobian (inverse stepsize)
 * @param x Vector with the states
 * @param dx Vector with the derivative states
 * @param xB Vector with the adjoint states
 * @param dxB Vector with the adjoint derivative states
 * @param xBdot Vector with the adjoint right hand side
 * @param JB Matrix to which the Jacobian will be written
 * @param user_data object with user input
 * @param tmp1B temporary storage vector
 * @param tmp2B temporary storage vector
 * @param tmp3B temporary storage vector
 * @return status flag indicating successful execution
 */
int fJBandB(
    realtype t, realtype cj, N_Vector x, N_Vector dx, N_Vector xB, N_Vector dxB,
    N_Vector xBdot, SUNMatrix JB, void* user_data, N_Vector tmp1B,
    N_Vector tmp2B, N_Vector tmp3B
) {
    return fJB(
        t, cj, x, dx, xB, dxB, xBdot, JB, user_data, tmp1B, tmp2B, tmp3B
    );
}

/**
 * @brief Matrix vector product of J with a vector v (for iterative solvers)
 * @param t timepoint @type realtype
 * @param cj scaling factor, inverse of the step size
 * @param x Vector with the states
 * @param dx Vector with the derivative states
 * @param xdot Vector with the right hand side
 * @param v Vector with which the Jacobian is multiplied
 * @param Jv Vector to which the Jacobian vector product will be written
 * @param user_data object with user input
 * @param tmp1 temporary storage vector
 * @param tmp2 temporary storage vector
 * @return status flag indicating successful execution
 **/
int fJv(
    realtype t, N_Vector x, N_Vector dx, N_Vector /*xdot*/, N_Vector v,
    N_Vector Jv, realtype cj, void* user_data, N_Vector /*tmp1*/,
    N_Vector /*tmp2*/
) {

    auto typed_udata = static_cast<IDASolver::user_data_type*>(user_data);
    Expects(typed_udata);
    auto model = dynamic_cast<Model_DAE*>(typed_udata->first);
    Expects(model);

    model->fJv(t, x, dx, v, Jv, cj);
    return model->checkFinite(gsl::make_span(Jv), ModelQuantity::Jv);
}

/**
 * @brief Matrix vector product of JB with a vector v (for iterative solvers)
 * @param t timepoint
 * @param x Vector with the states
 * @param dx Vector with the derivative states
 * @param xB Vector with the adjoint states
 * @param dxB Vector with the adjoint derivative states
 * @param xBdot Vector with the adjoint right hand side
 * @param vB Vector with which the Jacobian is multiplied
 * @param JvB Vector to which the Jacobian vector product will be written
 * @param cj scalar in Jacobian (inverse stepsize)
 * @param user_data object with user input
 * @param tmpB1 temporary storage vector
 * @param tmpB2 temporary storage vector
 * @return status flag indicating successful execution
 **/
int fJvB(
    realtype t, N_Vector x, N_Vector dx, N_Vector xB, N_Vector dxB,
    N_Vector /*xBdot*/, N_Vector vB, N_Vector JvB, realtype cj, void* user_data,
    N_Vector /*tmpB1*/, N_Vector /*tmpB2*/
) {

    auto typed_udata = static_cast<IDASolver::user_data_type*>(user_data);
    Expects(typed_udata);
    auto model = dynamic_cast<Model_DAE*>(typed_udata->first);
    Expects(model);

    model->fJvB(t, x, dx, xB, dxB, vB, JvB, cj);
    return model->checkFinite(gsl::make_span(JvB), ModelQuantity::JvB);
}

/**
 * @brief Event trigger function for events
 * @param t timepoint
 * @param x Vector with the states
 * @param dx Vector with the derivative states
 * @param root array with root function values
 * @param user_data object with user input
 * @return status flag indicating successful execution
 */
int froot(
    realtype t, N_Vector x, N_Vector dx, realtype* root, void* user_data
) {
    auto typed_udata = static_cast<IDASolver::user_data_type*>(user_data);
    Expects(typed_udata);
    auto model = dynamic_cast<Model_DAE*>(typed_udata->first);
    Expects(model);

    model->froot(t, x, dx, gsl::make_span<realtype>(root, model->ne));
    return model->checkFinite(
        gsl::make_span<realtype>(root, model->ne), ModelQuantity::root
    );
}

/**
 * @brief Residual function of the DAE
 * @param t timepoint
 * @param x Vector with the states
 * @param dx Vector with the derivative states
 * @param xdot Vector with the right hand side
 * @param user_data object with user input
 * @return status flag indicating successful execution
 */
int fxdot(realtype t, N_Vector x, N_Vector dx, N_Vector xdot, void* user_data) {
    auto typed_udata = static_cast<IDASolver::user_data_type*>(user_data);
    Expects(typed_udata);
    auto model = dynamic_cast<Model_DAE*>(typed_udata->first);
    Expects(model);
    auto solver = dynamic_cast<IDASolver const*>(typed_udata->second);
    Expects(model);

    if (solver->timeExceeded(500)) {
        return AMICI_MAX_TIME_EXCEEDED;
    }

    if (t > 1e200
        && !model->checkFinite(gsl::make_span(x), ModelQuantity::xdot)) {
        /* when t is large (typically ~1e300), CVODES may pass all NaN x
           to fxdot from which we typically cannot recover. To save time
           on normal execution, we do not always want to check finiteness
           of x, but only do so when t is large and we expect problems. */
        return AMICI_UNRECOVERABLE_ERROR;
    }

    model->fxdot(t, x, dx, xdot);
    return model->checkFinite(gsl::make_span(xdot), ModelQuantity::xdot);
}

/**
 * @brief Right hand side of differential equation for adjoint state xB
 * @param t timepoint
 * @param x Vector with the states
 * @param dx Vector with the derivative states
 * @param xB Vector with the adjoint states
 * @param dxB Vector with the adjoint derivative states
 * @param xBdot Vector with the adjoint right hand side
 * @param user_data object with user input
 * @return status flag indicating successful execution
 */
int fxBdot(
    realtype t, N_Vector x, N_Vector dx, N_Vector xB, N_Vector dxB,
    N_Vector xBdot, void* user_data
) {
    auto typed_udata = static_cast<IDASolver::user_data_type*>(user_data);
    Expects(typed_udata);
    auto model = dynamic_cast<Model_DAE*>(typed_udata->first);
    Expects(model);
    auto solver = dynamic_cast<IDASolver const*>(typed_udata->second);
    Expects(model);

    if (solver->timeExceeded(500)) {
        return AMICI_MAX_TIME_EXCEEDED;
    }

    model->fxBdot(t, x, dx, xB, dxB, xBdot);
    return model->checkFinite(gsl::make_span(xBdot), ModelQuantity::xBdot);
}

/**
 * @brief Right hand side of integral equation for quadrature states qB
 * @param t timepoint
 * @param x Vector with the states
 * @param dx Vector with the derivative states
 * @param xB Vector with the adjoint states
 * @param dxB Vector with the adjoint derivative states
 * @param qBdot Vector with the adjoint quadrature right hand side
 * @param user_data pointer to temp data object
 * @return status flag indicating successful execution
 */
int fqBdot(
    realtype t, N_Vector x, N_Vector dx, N_Vector xB, N_Vector dxB,
    N_Vector qBdot, void* user_data
) {

    auto typed_udata = static_cast<IDASolver::user_data_type*>(user_data);
    Expects(typed_udata);
    auto model = dynamic_cast<Model_DAE*>(typed_udata->first);
    Expects(model);

    model->fqBdot(t, x, dx, xB, dxB, qBdot);
    return model->checkFinite(gsl::make_span(qBdot), ModelQuantity::qBdot);
}

/**
 * @brief Right hand side of differential equation for adjoint state xB
 * when simulating in steadystate mode
 * @param t timepoint
 * @param xB Vector with the adjoint states
 * @param dxB Vector with the adjoint derivative states
 * @param xBdot Vector with the adjoint right hand side
 * @param user_data object with user input
 * @return status flag indicating successful execution
 */
static int fxBdot_ss(
    realtype t, N_Vector xB, N_Vector dxB, N_Vector xBdot, void* user_data
) {
    auto typed_udata = static_cast<IDASolver::user_data_type*>(user_data);
    Expects(typed_udata);
    auto model = dynamic_cast<Model_DAE*>(typed_udata->first);
    Expects(model);

    model->fxBdot_ss(t, xB, dxB, xBdot);
    return model->checkFinite(gsl::make_span(xBdot), ModelQuantity::xBdot_ss);
}

/**
 * @brief Right hand side of integral equation for quadrature states qB
 * when simulating in steadystate mode
 * @param t timepoint
 * @param xB Vector with the adjoint states
 * @param dxB Vector with the adjoint derivative states
 * @param qBdot Vector with the adjoint quadrature right hand side
 * @param user_data pointer to temp data object
 * @return status flag indicating successful execution
 */
static int fqBdot_ss(
    realtype t, N_Vector xB, N_Vector dxB, N_Vector qBdot, void* user_data
) {
    auto typed_udata = static_cast<IDASolver::user_data_type*>(user_data);
    Expects(typed_udata);
    auto model = dynamic_cast<Model_DAE*>(typed_udata->first);
    Expects(model);

    model->fqBdot_ss(t, xB, dxB, qBdot);
    return model->checkFinite(gsl::make_span(qBdot), ModelQuantity::qBdot_ss);
}

/**
 * @brief JB in sparse form for steady state case
 * @param t timepoint
 * @param cj scalar in Jacobian (inverse stepsize)
 * @param x Vector with the states
 * @param dx Vector with the derivative states
 * @param xdot Vector with the right hand side
 * @param J Matrix to which the Jacobian will be written
 * @param user_data object with user input
 * @param tmp1 temporary storage vector
 * @param tmp2 temporary storage vector
 * @param tmp3 temporary storage vector
 * @return status flag indicating successful execution
 */
static int fJSparseB_ss(
    realtype /*t*/, realtype /*cj*/, N_Vector /*x*/, N_Vector /*dx*/,
    N_Vector xBdot, SUNMatrix JB, void* user_data, N_Vector /*tmp1*/,
    N_Vector /*tmp2*/, N_Vector /*tmp3*/
) {
    auto typed_udata = static_cast<IDASolver::user_data_type*>(user_data);
    Expects(typed_udata);
    auto model = dynamic_cast<Model_DAE*>(typed_udata->first);
    Expects(model);

    model->fJSparseB_ss(JB);
    return model->checkFinite(
        gsl::make_span(xBdot), ModelQuantity::JSparseB_ss
    );
}

/**
 * @brief Right hand side of differential equation for state sensitivities sx
 * @param Ns number of parameters
 * @param t timepoint
 * @param x Vector with the states
 * @param dx Vector with the derivative states
 * @param xdot Vector with the right hand side
 * @param sx Vector with the state sensitivities
 * @param sdx Vector with the derivative state sensitivities
 * @param sxdot Vector with the sensitivity right hand side
 * @param user_data object with user input
 * @param tmp1 temporary storage vector
 * @param tmp2 temporary storage vector
 * @param tmp3 temporary storage vector
 * @return status flag indicating successful execution
 */
int fsxdot(
    int /*Ns*/, realtype t, N_Vector x, N_Vector dx, N_Vector /*xdot*/,
    N_Vector* sx, N_Vector* sdx, N_Vector* sxdot, void* user_data,
    N_Vector /*tmp1*/, N_Vector /*tmp2*/, N_Vector /*tmp3*/
) {

    auto typed_udata = static_cast<IDASolver::user_data_type*>(user_data);
    Expects(typed_udata);
    auto model = dynamic_cast<Model_DAE*>(typed_udata->first);
    Expects(model);

    for (int ip = 0; ip < model->nplist(); ip++) {
        model->fsxdot(t, x, dx, ip, sx[ip], sdx[ip], sxdot[ip]);
        if (model->checkFinite(gsl::make_span(sxdot[ip]), ModelQuantity::sxdot)
            != AMICI_SUCCESS)
            return AMICI_RECOVERABLE_ERROR;
    }

    return AMICI_SUCCESS;
}

} // namespace amici
