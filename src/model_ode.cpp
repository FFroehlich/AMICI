#include "amici/model_ode.h"
#include "amici/solver_cvodes.h"
#include <amici/sundials_matrix_wrapper.h>

namespace amici {

void Model_ODE::fJ(
    realtype const t, realtype const /*cj*/, AmiVector const& x,
    AmiVector const& /*dx*/, AmiVector const& xdot, SUNMatrix J
) {
    fJ(t, x.getNVector(), xdot.getNVector(), J);
}

void Model_ODE::fJ(
    realtype t, const_N_Vector x, const_N_Vector /*xdot*/, SUNMatrix J
) {
    fJSparse(t, x, derived_state_.J_);
    derived_state_.J_.refresh();
    auto JDense = SUNMatrixWrapper(J);
    derived_state_.J_.to_dense(JDense);
}

void Model_ODE::fJSparse(
    realtype const t, realtype const /*cj*/, AmiVector const& x,
    AmiVector const& /*dx*/, AmiVector const& /*xdot*/, SUNMatrix J
) {
    fJSparse(t, x.getNVector(), J);
}

void Model_ODE::fJSparse(realtype t, const_N_Vector x, SUNMatrix J) {
    auto x_pos = computeX_pos(x);
    fdwdx(t, N_VGetArrayPointerConst(x_pos), false);
    if (pythonGenerated) {
        auto JSparse = SUNMatrixWrapper(J);
        // python generated
        derived_state_.dxdotdx_explicit.zero();
        derived_state_.dxdotdx_implicit.zero();
        if (derived_state_.dxdotdx_explicit.capacity()) {
            fdxdotdx_explicit_colptrs(derived_state_.dxdotdx_explicit);
            fdxdotdx_explicit_rowvals(derived_state_.dxdotdx_explicit);
            fdxdotdx_explicit(
                derived_state_.dxdotdx_explicit.data(), t,
                N_VGetArrayPointerConst(x_pos),
                state_.unscaledParameters.data(), state_.fixedParameters.data(),
                state_.h.data(), derived_state_.w_.data()
            );
        }
        fdxdotdw(t, x_pos);
        /* Sparse matrix multiplication
         dxdotdx_implicit += dxdotdw * dwdx */
        derived_state_.dxdotdw_.sparse_multiply(
            derived_state_.dxdotdx_implicit, derived_state_.dwdx_
        );

        JSparse.sparse_add(
            derived_state_.dxdotdx_explicit, 1.0,
            derived_state_.dxdotdx_implicit, 1.0
        );
    } else {
        fJSparse(
            static_cast<SUNMatrixContent_Sparse>(SM_CONTENT_S(J)), t,
            N_VGetArrayPointerConst(x_pos), state_.unscaledParameters.data(),
            state_.fixedParameters.data(), state_.h.data(),
            derived_state_.w_.data(), derived_state_.dwdx_.data()
        );
    }
}

void Model_ODE::
    fJv(realtype const t, AmiVector const& x, AmiVector const& /*dx*/,
        AmiVector const& /*xdot*/, AmiVector const& v, AmiVector& Jv,
        realtype const /*cj*/) {
    fJv(v.getNVector(), Jv.getNVector(), t, x.getNVector());
}

void Model_ODE::fJv(
    const_N_Vector v, N_Vector Jv, realtype t, const_N_Vector x
) {
    N_VConst(0.0, Jv);
    fJSparse(t, x, derived_state_.J_);
    derived_state_.J_.refresh();
    derived_state_.J_.multiply(Jv, v);
}

void Model_ODE::froot(
    realtype const t, AmiVector const& x, AmiVector const& /*dx*/,
    gsl::span<realtype> root
) {
    froot(t, x.getNVector(), root);
}

void Model_ODE::froot(realtype t, const_N_Vector x, gsl::span<realtype> root) {
    auto x_pos = computeX_pos(x);
    std::fill(root.begin(), root.end(), 0.0);
    froot(
        root.data(), t, N_VGetArrayPointerConst(x_pos),
        state_.unscaledParameters.data(), state_.fixedParameters.data(),
        state_.h.data(), state_.total_cl.data()
    );
}

void Model_ODE::fxdot(
    realtype const t, AmiVector const& x, AmiVector const& /*dx*/,
    AmiVector& xdot
) {
    fxdot(t, x.getNVector(), xdot.getNVector());
}

void Model_ODE::fxdot(realtype t, const_N_Vector x, N_Vector xdot) {
    auto x_pos = computeX_pos(x);
    fw(t, N_VGetArrayPointerConst(x_pos));
    N_VConst(0.0, xdot);
    fxdot(
        N_VGetArrayPointer(xdot), t, N_VGetArrayPointerConst(x_pos),
        state_.unscaledParameters.data(), state_.fixedParameters.data(),
        state_.h.data(), derived_state_.w_.data()
    );
}

void Model_ODE::fJDiag(
    realtype const t, AmiVector& JDiag, realtype const /*cj*/,
    AmiVector const& x, AmiVector const& /*dx*/
) {
    fJDiag(t, JDiag.getNVector(), x.getNVector());
    if (checkFinite(JDiag.getVector(), ModelQuantity::JDiag, t)
        != AMICI_SUCCESS)
        throw AmiException("Evaluation of fJDiag failed!");
}

void Model_ODE::fdxdotdw(realtype const t, const_N_Vector x) {
    derived_state_.dxdotdw_.zero();
    if (nw > 0 && derived_state_.dxdotdw_.capacity()) {
        auto x_pos = computeX_pos(x);

        fdxdotdw_colptrs(derived_state_.dxdotdw_);
        fdxdotdw_rowvals(derived_state_.dxdotdw_);
        fdxdotdw(
            derived_state_.dxdotdw_.data(), t, N_VGetArrayPointerConst(x_pos),
            state_.unscaledParameters.data(), state_.fixedParameters.data(),
            state_.h.data(), derived_state_.w_.data()
        );
    }
}

void Model_ODE::fdxdotdp(realtype const t, const_N_Vector x) {
    auto x_pos = computeX_pos(x);
    fdwdp(t, N_VGetArrayPointerConst(x_pos), false);

    if (pythonGenerated) {
        // python generated
        derived_state_.dxdotdp_explicit.zero();
        derived_state_.dxdotdp_implicit.zero();
        if (derived_state_.dxdotdp_explicit.capacity()) {
            fdxdotdp_explicit_colptrs(derived_state_.dxdotdp_explicit);
            fdxdotdp_explicit_rowvals(derived_state_.dxdotdp_explicit);
            fdxdotdp_explicit(
                derived_state_.dxdotdp_explicit.data(), t,
                N_VGetArrayPointerConst(x_pos),
                state_.unscaledParameters.data(), state_.fixedParameters.data(),
                state_.h.data(), derived_state_.w_.data()
            );
        }

        fdxdotdw(t, x_pos);
        /* Sparse matrix multiplication
         dxdotdp_implicit += dxdotdw * dwdp */
        derived_state_.dxdotdw_.sparse_multiply(
            derived_state_.dxdotdp_implicit, derived_state_.dwdp_
        );

        derived_state_.dxdotdp_full.sparse_add(
            derived_state_.dxdotdp_explicit, 1.0,
            derived_state_.dxdotdp_implicit, 1.0
        );
    } else {
        // matlab generated
        for (int ip = 0; ip < nplist(); ip++) {
            N_VConst(0.0, derived_state_.dxdotdp.getNVector(ip));
            fdxdotdp(
                derived_state_.dxdotdp.data(ip), t,
                N_VGetArrayPointerConst(x_pos),
                state_.unscaledParameters.data(), state_.fixedParameters.data(),
                state_.h.data(), plist(ip), derived_state_.w_.data(),
                derived_state_.dwdp_.data()
            );
        }
    }
}

void Model_ODE::
    fdxdotdp(realtype const t, AmiVector const& x, AmiVector const& /*dx*/) {
    fdxdotdp(t, x.getNVector());
}

std::unique_ptr<Solver> Model_ODE::getSolver() {
    return std::unique_ptr<Solver>(new amici::CVodeSolver());
}

void Model_ODE::fJSparse(
    SUNMatrixContent_Sparse /*JSparse*/, realtype const /*t*/,
    realtype const* /*x*/, realtype const* /*p*/, realtype const* /*k*/,
    realtype const* /*h*/, realtype const* /*w*/, realtype const* /*dwdx*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s "
        "is not implemented for this model!",
        __func__
    ); // not implemented
}

void Model_ODE::fJSparse(
    realtype* /*JSparse*/, realtype const /*t*/, realtype const* /*x*/,
    realtype const* /*p*/, realtype const* /*k*/, realtype const* /*h*/,
    realtype const* /*w*/, realtype const* /*dwdx*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s "
        "is not implemented for this model!",
        __func__
    ); // not implemented
}

void Model_ODE::fJSparse_colptrs(SUNMatrixWrapper& /*JSparse*/) {
    throw AmiException(
        "Requested functionality is not supported as %s "
        "is not implemented for this model!",
        __func__
    ); // not implemented
}

void Model_ODE::fJSparse_rowvals(SUNMatrixWrapper& /*JSparse*/) {
    throw AmiException(
        "Requested functionality is not supported as %s "
        "is not implemented for this model!",
        __func__
    ); // not implemented
}

void Model_ODE::froot(
    realtype* /*root*/, realtype const /*t*/, realtype const* /*x*/,
    realtype const* /*p*/, realtype const* /*k*/, realtype const* /*h*/,
    realtype const* /*tcl*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s is not "
        "implemented for this model!",
        __func__
    ); // not implemented
}

void Model_ODE::fdxdotdp(
    realtype* /*dxdotdp*/, realtype const /*t*/, realtype const* /*x*/,
    realtype const* /*p*/, realtype const* /*k*/, realtype const* /*h*/,
    int const /*ip*/, realtype const* /*w*/, realtype const* /*dwdp*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s "
        "is not implemented for this model!",
        __func__
    ); // not implemented
}

void Model_ODE::fdxdotdp_explicit(
    realtype* /*dxdotdp_explicit*/, realtype const /*t*/, realtype const* /*x*/,
    realtype const* /*p*/, realtype const* /*k*/, realtype const* /*h*/,
    realtype const* /*w*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s "
        "is not implemented for this model!",
        __func__
    ); // not implemented
}

void Model_ODE::fdxdotdp_explicit_colptrs(SUNMatrixWrapper& /*dxdotdp*/) {
    throw AmiException(
        "Requested functionality is not supported as %s "
        "is not implemented for this model!",
        __func__
    ); // not implemented
}

void Model_ODE::fdxdotdp_explicit_rowvals(SUNMatrixWrapper& /*dxdotdp*/) {
    throw AmiException(
        "Requested functionality is not supported as %s "
        "is not implemented for this model!",
        __func__
    ); // not implemented
}

void Model_ODE::fdxdotdx_explicit(
    realtype* /*dxdotdx_explicit*/, realtype const /*t*/, realtype const* /*x*/,
    realtype const* /*p*/, realtype const* /*k*/, realtype const* /*h*/,
    realtype const* /*w*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s "
        "is not implemented for this model!",
        __func__
    ); // not implemented
}

void Model_ODE::fdxdotdx_explicit_colptrs(SUNMatrixWrapper& /*dxdotdx*/) {
    throw AmiException(
        "Requested functionality is not supported as %s "
        "is not implemented for this model!",
        __func__
    ); // not implemented
}

void Model_ODE::fdxdotdx_explicit_rowvals(SUNMatrixWrapper& /*dxdotdx*/) {
    throw AmiException(
        "Requested functionality is not supported as %s "
        "is not implemented for this model!",
        __func__
    ); // not implemented
}

void Model_ODE::fdxdotdw(
    realtype* /*dxdotdw*/, realtype const /*t*/, realtype const* /*x*/,
    realtype const* /*p*/, realtype const* /*k*/, realtype const* /*h*/,
    realtype const* /*w*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s "
        "is not implemented for this model!",
        __func__
    ); // not implemented
}

void Model_ODE::fdxdotdw_colptrs(SUNMatrixWrapper& /*dxdotdw*/) {
    throw AmiException(
        "Requested functionality is not supported as %s "
        "is not implemented for this model!",
        __func__
    ); // not implemented
}

void Model_ODE::fdxdotdw_rowvals(SUNMatrixWrapper& /*dxdotdw*/) {
    throw AmiException(
        "Requested functionality is not supported as %s "
        "is not implemented for this model!",
        __func__
    ); // not implemented
}

void Model_ODE::fJB(
    realtype const t, realtype /*cj*/, AmiVector const& x,
    AmiVector const& /*dx*/, AmiVector const& xB, AmiVector const& /*dxB*/,
    AmiVector const& xBdot, SUNMatrix JB
) {
    fJB(t, x.getNVector(), xB.getNVector(), xBdot.getNVector(), JB);
}

void Model_ODE::fJB(
    realtype t, const_N_Vector x, const_N_Vector /*xB*/,
    const_N_Vector /*xBdot*/, SUNMatrix JB
) {
    fJSparse(t, x, derived_state_.J_);
    derived_state_.J_.refresh();
    auto JDenseB = SUNMatrixWrapper(JB);
    derived_state_.J_.transpose(JDenseB, -1.0, nxtrue_solver);
}

void Model_ODE::fJSparseB(
    realtype const t, realtype /*cj*/, AmiVector const& x,
    AmiVector const& /*dx*/, AmiVector const& xB, AmiVector const& /*dxB*/,
    AmiVector const& xBdot, SUNMatrix JB
) {
    fJSparseB(t, x.getNVector(), xB.getNVector(), xBdot.getNVector(), JB);
}

void Model_ODE::fJSparseB(
    realtype t, const_N_Vector x, const_N_Vector /*xB*/,
    const_N_Vector /*xBdot*/, SUNMatrix JB
) {
    fJSparse(t, x, derived_state_.J_);
    derived_state_.J_.refresh();
    auto JSparseB = SUNMatrixWrapper(JB);
    derived_state_.J_.transpose(JSparseB, -1.0, nxtrue_solver);
}

void Model_ODE::fJDiag(realtype t, N_Vector JDiag, const_N_Vector x) {
    fJSparse(t, x, derived_state_.J_);
    derived_state_.J_.refresh();
    derived_state_.J_.to_diag(JDiag);
}

void Model_ODE::fJvB(
    const_N_Vector vB, N_Vector JvB, realtype t, const_N_Vector x,
    const_N_Vector xB
) {
    N_VConst(0.0, JvB);
    fJSparseB(t, x, xB, nullptr, derived_state_.JB_);
    derived_state_.JB_.refresh();
    derived_state_.JB_.multiply(JvB, vB);
}

void Model_ODE::fxBdot(
    realtype t, AmiVector const& x, AmiVector const& /*dx*/,
    AmiVector const& xB, AmiVector const& /*dxB*/, AmiVector& xBdot
) {
    fxBdot(t, x.getNVector(), xB.getNVector(), xBdot.getNVector());
}

void Model_ODE::fxBdot(
    realtype t, const_N_Vector x, const_N_Vector xB, N_Vector xBdot
) {
    N_VConst(0.0, xBdot);
    fJSparseB(t, x, xB, nullptr, derived_state_.JB_);
    derived_state_.JB_.refresh();
    derived_state_.JB_.multiply(xBdot, xB);
}

void Model_ODE::fqBdot(
    realtype t, const_N_Vector x, const_N_Vector xB, N_Vector qBdot
) {
    /* initialize with zeros */
    N_VConst(0.0, qBdot);
    fdxdotdp(t, x);

    if (pythonGenerated) {
        /* call multiplication */
        derived_state_.dxdotdp_full.multiply(qBdot, xB, state_.plist, true);
        N_VScale(-1.0, qBdot, qBdot);
    } else {
        /* was matlab generated */
        for (int ip = 0; ip < nplist(); ip++) {
            for (int ix = 0; ix < nxtrue_solver; ix++)
                NV_Ith_S(qBdot, ip * nJ)
                    -= NV_Ith_S(xB, ix) * derived_state_.dxdotdp.at(ix, ip);
            // second order part
            for (int iJ = 1; iJ < nJ; iJ++)
                for (int ix = 0; ix < nxtrue_solver; ix++)
                    NV_Ith_S(qBdot, ip * nJ + iJ)
                        -= NV_Ith_S(xB, ix)
                               * derived_state_.dxdotdp.at(
                                   ix + iJ * nxtrue_solver, ip
                               )
                           + NV_Ith_S(xB, ix + iJ * nxtrue_solver)
                                 * derived_state_.dxdotdp.at(ix, ip);
        }
    }
}

void Model_ODE::fxBdot_ss(
    realtype const t, AmiVector const& xB, AmiVector const& /*dx*/,
    AmiVector& xBdot
) {
    fxBdot_ss(t, xB.getNVector(), xBdot.getNVector());
}

void Model_ODE::fxBdot_ss(realtype /*t*/, const_N_Vector xB, N_Vector xBdot)
    const {
    /* Right hand side of the adjoint state for steady state computations.
       J is fixed (as x remains in steady state), so the RHS becomes simple. */
    N_VConst(0.0, xBdot);
    derived_state_.JB_.multiply(xBdot, xB);
}

void Model_ODE::fqBdot_ss(realtype /*t*/, N_Vector xB, N_Vector qBdot) const {
    /* Quadratures when computing adjoints for steady state. The integrand is
       just the adjoint state itself. */
    N_VScale(1.0, xB, qBdot);
}

void Model_ODE::fJSparseB_ss(SUNMatrix JB) {
    /* Just copy the model Jacobian */
    SUNMatCopy(derived_state_.JB_, JB);
    derived_state_.JB_.refresh();
}

void Model_ODE::writeSteadystateJB(
    realtype const t, realtype /*cj*/, AmiVector const& x,
    AmiVector const& /*dx*/, AmiVector const& xB, AmiVector const& /*dxB*/,
    AmiVector const& xBdot
) {
    /* Get backward Jacobian */
    fJSparseB(
        t, x.getNVector(), xB.getNVector(), xBdot.getNVector(),
        derived_state_.JB_
    );
    derived_state_.JB_.refresh();
    /* Switch sign, as we integrate forward in time, not backward */
    derived_state_.JB_.scale(-1);
}

void Model_ODE::fsxdot(
    realtype const t, AmiVector const& x, AmiVector const& /*dx*/, int const ip,
    AmiVector const& sx, AmiVector const& /*sdx*/, AmiVector& sxdot
) {
    fsxdot(t, x.getNVector(), ip, sx.getNVector(), sxdot.getNVector());
}

void Model_ODE::fsxdot(
    realtype t, const_N_Vector x, int ip, const_N_Vector sx, N_Vector sxdot
) {

    /* sxdot is just the total derivative d(xdot)dp,
     so we just call dxdotdp and copy the stuff over */
    if (ip == 0) {
        // we only need to call this for the first parameter index will be
        // the same for all remaining
        fdxdotdp(t, x);
        fJSparse(t, x, derived_state_.J_);
        derived_state_.J_.refresh();
    }
    if (pythonGenerated) {
        /* copy dxdotdp and the implicit version over */
        // initialize
        N_VConst(0.0, sxdot);
        realtype* sxdot_tmp = N_VGetArrayPointer(sxdot);

        derived_state_.dxdotdp_full.scatter(
            plist(ip), 1.0, nullptr, gsl::make_span(sxdot_tmp, nx_solver), 0,
            nullptr, 0
        );

    } else {
        /* copy dxdotdp over */
        N_VScale(1.0, derived_state_.dxdotdp.getNVector(ip), sxdot);
    }
    derived_state_.J_.multiply(sxdot, sx);
}

} // namespace amici
