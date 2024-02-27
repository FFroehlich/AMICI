#include "amici/model_dae.h"
#include "amici/solver_idas.h"

namespace amici {

void Model_DAE::fJ(
    realtype const t, realtype const cj, AmiVector const& x,
    AmiVector const& dx, AmiVector const& xdot, SUNMatrix J
) {
    fJ(t, cj, x.getNVector(), dx.getNVector(), xdot.getNVector(), J);
}

void Model_DAE::fJ(
    realtype t, realtype cj, const_N_Vector x, const_N_Vector dx,
    const_N_Vector /*xdot*/, SUNMatrix J
) {
    fJSparse(t, cj, x, dx, derived_state_.J_);
    derived_state_.J_.refresh();
    auto JDense = SUNMatrixWrapper(J);
    derived_state_.J_.to_dense(JDense);
}

void Model_DAE::fJSparse(
    realtype const t, realtype const cj, AmiVector const& x,
    AmiVector const& dx, AmiVector const& /*xdot*/, SUNMatrix J
) {
    fJSparse(t, cj, x.getNVector(), dx.getNVector(), J);
}

void Model_DAE::fJSparse(
    realtype t, realtype cj, const_N_Vector x, const_N_Vector dx, SUNMatrix J
) {
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
                state_.h.data(), N_VGetArrayPointerConst(dx),
                derived_state_.w_.data()
            );
        }
        fdxdotdw(t, x_pos, dx);
        /* Sparse matrix multiplication
         dxdotdx_implicit += dxdotdw * dwdx */
        derived_state_.dxdotdw_.sparse_multiply(
            derived_state_.dxdotdx_implicit, derived_state_.dwdx_
        );

        derived_state_.dfdx_.sparse_add(
            derived_state_.dxdotdx_explicit, 1.0,
            derived_state_.dxdotdx_implicit, 1.0
        );
        fM(t, x_pos);
        JSparse.sparse_add(
            derived_state_.MSparse_, -cj, derived_state_.dfdx_, 1.0
        );
    } else {
        fJSparse(
            static_cast<SUNMatrixContent_Sparse>(SM_CONTENT_S(J)), t,
            N_VGetArrayPointerConst(x_pos), state_.unscaledParameters.data(),
            state_.fixedParameters.data(), state_.h.data(), cj,
            N_VGetArrayPointerConst(dx), derived_state_.w_.data(),
            derived_state_.dwdx_.data()
        );
    }
}

void Model_DAE::fJv(
    realtype const t, AmiVector const& x, AmiVector const& dx,
    AmiVector const& /*xdot*/, AmiVector const& v, AmiVector& Jv,
    realtype const cj
) {
    fJv(t, x.getNVector(), dx.getNVector(), v.getNVector(), Jv.getNVector(),
        cj);
}

void Model_DAE::fJv(
    realtype t, const_N_Vector x, const_N_Vector dx, const_N_Vector v,
    N_Vector Jv, realtype cj
) {
    N_VConst(0.0, Jv);
    fJSparse(t, cj, x, dx, derived_state_.J_);
    derived_state_.J_.refresh();
    derived_state_.J_.multiply(Jv, v);
}

void Model_DAE::froot(
    realtype const t, AmiVector const& x, AmiVector const& dx,
    gsl::span<realtype> root
) {
    froot(t, x.getNVector(), dx.getNVector(), root);
}

void Model_DAE::froot(
    realtype t, const_N_Vector x, const_N_Vector dx, gsl::span<realtype> root
) {
    std::fill(root.begin(), root.end(), 0.0);
    auto x_pos = computeX_pos(x);
    froot(
        root.data(), t, N_VGetArrayPointerConst(x_pos),
        state_.unscaledParameters.data(), state_.fixedParameters.data(),
        state_.h.data(), N_VGetArrayPointerConst(dx)
    );
}

void Model_DAE::fxdot(
    realtype const t, AmiVector const& x, AmiVector const& dx, AmiVector& xdot
) {
    fxdot(t, x.getNVector(), dx.getNVector(), xdot.getNVector());
}

void Model_DAE::fxdot(
    realtype t, const_N_Vector x, const_N_Vector dx, N_Vector xdot
) {
    auto x_pos = computeX_pos(x);
    fw(t, N_VGetArrayPointerConst(x), false);
    N_VConst(0.0, xdot);
    fxdot(
        N_VGetArrayPointer(xdot), t, N_VGetArrayPointerConst(x_pos),
        state_.unscaledParameters.data(), state_.fixedParameters.data(),
        state_.h.data(), N_VGetArrayPointerConst(dx), derived_state_.w_.data()
    );
}

void Model_DAE::fJDiag(
    realtype const t, AmiVector& JDiag, realtype const /*cj*/,
    AmiVector const& x, AmiVector const& dx
) {
    fJSparse(t, 0.0, x.getNVector(), dx.getNVector(), derived_state_.J_);
    derived_state_.J_.refresh();
    derived_state_.J_.to_diag(JDiag.getNVector());
    if (checkFinite(JDiag.getVector(), ModelQuantity::JDiag) != AMICI_SUCCESS)
        throw AmiException("Evaluation of fJDiag failed!");
}

void Model_DAE::fdxdotdw(
    realtype const t, const_N_Vector x, const_N_Vector const dx
) {
    derived_state_.dxdotdw_.zero();
    if (nw > 0 && derived_state_.dxdotdw_.capacity()) {
        auto x_pos = computeX_pos(x);

        fdxdotdw_colptrs(derived_state_.dxdotdw_);
        fdxdotdw_rowvals(derived_state_.dxdotdw_);
        fdxdotdw(
            derived_state_.dxdotdw_.data(), t, N_VGetArrayPointerConst(x_pos),
            state_.unscaledParameters.data(), state_.fixedParameters.data(),
            state_.h.data(), N_VGetArrayPointerConst(dx),
            derived_state_.w_.data()
        );
    }
}

void Model_DAE::fdxdotdp(
    realtype const t, const_N_Vector const x, const_N_Vector const dx
) {
    auto x_pos = computeX_pos(x);

    if (pythonGenerated) {
        // python generated, not yet implemented for DAEs
        throw AmiException("Wrapping of DAEs is not yet implemented from Python"
        );
    } else {
        // matlab generated
        fdwdp(t, N_VGetArrayPointerConst(x_pos), false);

        for (int ip = 0; ip < nplist(); ip++) {
            N_VConst(0.0, derived_state_.dxdotdp.getNVector(ip));
            fdxdotdp(
                derived_state_.dxdotdp.data(ip), t,
                N_VGetArrayPointerConst(x_pos),
                state_.unscaledParameters.data(), state_.fixedParameters.data(),
                state_.h.data(), plist(ip), N_VGetArrayPointerConst(dx),
                derived_state_.w_.data(), derived_state_.dwdp_.data()
            );
        }
    }
}

void Model_DAE::fM(realtype t, const_N_Vector x) {
    derived_state_.M_.zero();
    if (pythonGenerated) {
        /*
         * non-algebraic states in python generated code always have factor
         * 1 in the mass matrix, so we can easily construct this matrix here
         * and avoid having to generate c++ code
         */
        int ndiff = 0;
        for (int ix = 0; ix < nx_solver; ix++) {
            derived_state_.MSparse_.set_indexptr(ix, ndiff);
            if (this->idlist.at(ix) == 1.0) {
                derived_state_.MSparse_.set_data(ndiff, 1.0);
                derived_state_.MSparse_.set_indexval(ndiff, ix);
                ndiff++;
            }
        }
        derived_state_.MSparse_.set_indexptr(nx_solver, ndiff);
        assert(ndiff == derived_state_.MSparse_.capacity());
    } else {
        auto x_pos = computeX_pos(x);
        fM(derived_state_.M_.data(), t, N_VGetArrayPointerConst(x_pos),
           state_.unscaledParameters.data(), state_.fixedParameters.data());
    }
}

std::unique_ptr<Solver> Model_DAE::getSolver() {
    return std::unique_ptr<Solver>(new amici::IDASolver());
}

void Model_DAE::fJSparse(
    SUNMatrixContent_Sparse /*JSparse*/, realtype /*t*/, realtype const* /*x*/,
    double const* /*p*/, double const* /*k*/, realtype const* /*h*/,
    realtype /*cj*/, realtype const* /*dx*/, realtype const* /*w*/,
    realtype const* /*dwdx*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s "
        "is not implemented for this model!",
        __func__
    ); // not implemented
}

void Model_DAE::froot(
    realtype* /*root*/, realtype const /*t*/, realtype const* /*x*/,
    double const* /*p*/, double const* /*k*/, realtype const* /*h*/,
    realtype const* /*dx*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s is not "
        "implemented for this model!",
        __func__
    ); // not implemented
}

void Model_DAE::fdxdotdp(
    realtype* /*dxdotdp*/, realtype const /*t*/, realtype const* /*x*/,
    realtype const* /*p*/, realtype const* /*k*/, realtype const* /*h*/,
    int const /*ip*/, realtype const* /*dx*/, realtype const* /*w*/,
    realtype const* /*dwdp*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s is not "
        "implemented for this model!",
        __func__
    );
}

void Model_DAE::fdxdotdp_explicit(
    realtype* /*dxdotdp_explicit*/, realtype const /*t*/, realtype const* /*x*/,
    realtype const* /*p*/, realtype const* /*k*/, realtype const* /*h*/,
    realtype const* /*dx*/, realtype const* /*w*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s "
        "is not implemented for this model!",
        __func__
    ); // not implemented
}

void Model_DAE::fdxdotdp_explicit_colptrs(SUNMatrixWrapper& /*dxdotdp*/) {
    throw AmiException(
        "Requested functionality is not supported as %s "
        "is not implemented for this model!",
        __func__
    ); // not implemented
}

void Model_DAE::fdxdotdp_explicit_rowvals(SUNMatrixWrapper& /*dxdotdp*/) {
    throw AmiException(
        "Requested functionality is not supported as %s "
        "is not implemented for this model!",
        __func__
    ); // not implemented
}

void Model_DAE::fdxdotdx_explicit(
    realtype* /*dxdotdx_explicit*/, realtype const /*t*/, realtype const* /*x*/,
    realtype const* /*p*/, realtype const* /*k*/, realtype const* /*h*/,
    realtype const* /*dx*/, realtype const* /*w*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s "
        "is not implemented for this model!",
        __func__
    ); // not implemented
}

void Model_DAE::fdxdotdx_explicit_colptrs(SUNMatrixWrapper& /*dxdotdx*/) {
    throw AmiException(
        "Requested functionality is not supported as %s "
        "is not implemented for this model!",
        __func__
    ); // not implemented
}

void Model_DAE::fdxdotdx_explicit_rowvals(SUNMatrixWrapper& /*dxdotdx*/) {
    throw AmiException(
        "Requested functionality is not supported as %s "
        "is not implemented for this model!",
        __func__
    ); // not implemented
}

void Model_DAE::fdxdotdw(
    realtype* /*dxdotdw*/, realtype const /*t*/, realtype const* /*x*/,
    realtype const* /*p*/, realtype const* /*k*/, realtype const* /*h*/,
    realtype const* /*dx*/, realtype const* /*w*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s "
        "is not implemented for this model!",
        __func__
    ); // not implemented
}

void Model_DAE::fdxdotdw_colptrs(SUNMatrixWrapper& /*dxdotdw*/) {
    throw AmiException(
        "Requested functionality is not supported as %s "
        "is not implemented for this model!",
        __func__
    ); // not implemented
}

void Model_DAE::fdxdotdw_rowvals(SUNMatrixWrapper& /*dxdotdw*/) {
    throw AmiException(
        "Requested functionality is not supported as %s "
        "is not implemented for this model!",
        __func__
    ); // not implemented
}

void Model_DAE::
    fM(realtype* /*M*/, realtype const /*t*/, realtype const* /*x*/,
       realtype const* /*p*/, realtype const* /*k*/) {}

void Model_DAE::fJB(
    realtype const t, realtype cj, AmiVector const& x, AmiVector const& dx,
    AmiVector const& xB, AmiVector const& /*dxB*/, AmiVector const& /*xBdot*/,
    SUNMatrix JB
) {
    fJB(t, cj, x.getNVector(), dx.getNVector(), xB.getNVector(),
        dx.getNVector(), JB);
}

void Model_DAE::fJB(
    realtype t, realtype cj, const_N_Vector x, const_N_Vector dx,
    const_N_Vector /*xB*/, const_N_Vector /*dxB*/, SUNMatrix JB
) {
    fJSparse(t, cj, x, dx, derived_state_.J_);
    derived_state_.J_.refresh();
    auto JBDense = SUNMatrixWrapper(JB);
    derived_state_.J_.transpose(JBDense, -1.0, nxtrue_solver);
}

void Model_DAE::fJSparseB(
    realtype const t, realtype cj, AmiVector const& x, AmiVector const& dx,
    AmiVector const& xB, AmiVector const& dxB, AmiVector const& /*xBdot*/,
    SUNMatrix JB
) {
    fJSparseB(
        t, cj, x.getNVector(), dx.getNVector(), xB.getNVector(),
        dxB.getNVector(), JB
    );
}

void Model_DAE::fJSparseB(
    realtype t, realtype cj, const_N_Vector x, const_N_Vector dx,
    const_N_Vector /*xB*/, const_N_Vector /*dxB*/, SUNMatrix JB
) {
    fJSparse(t, cj, x, dx, derived_state_.J_);
    derived_state_.J_.refresh();
    auto JSparseB = SUNMatrixWrapper(JB);
    derived_state_.J_.transpose(JSparseB, -1.0, nxtrue_solver);
}

void Model_DAE::fJvB(
    realtype t, const_N_Vector x, const_N_Vector dx, const_N_Vector xB,
    const_N_Vector dxB, const_N_Vector vB, N_Vector JvB, realtype cj
) {
    N_VConst(0.0, JvB);
    fJSparseB(t, cj, x, dx, xB, dxB, derived_state_.JB_);
    derived_state_.JB_.refresh();
    derived_state_.JB_.multiply(JvB, vB);
}

void Model_DAE::fxBdot(
    realtype t, const_N_Vector x, const_N_Vector dx, const_N_Vector xB,
    const_N_Vector dxB, N_Vector xBdot
) {
    N_VConst(0.0, xBdot);
    fJSparseB(t, 1.0, x, dx, xB, dxB, derived_state_.JB_);
    derived_state_.JB_.refresh();
    fM(t, x);
    derived_state_.JB_.multiply(xBdot, xB);
}

void Model_DAE::fqBdot(
    realtype t, const_N_Vector x, const_N_Vector dx, const_N_Vector xB,
    const_N_Vector /*dxB*/, N_Vector qBdot
) {
    N_VConst(0.0, qBdot);
    fdxdotdp(t, x, dx);
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

void Model_DAE::fxBdot_ss(
    realtype const t, AmiVector const& xB, AmiVector const& dxB,
    AmiVector& xBdot
) {
    fxBdot_ss(t, xB.getNVector(), dxB.getNVector(), xBdot.getNVector());
}

void Model_DAE::fxBdot_ss(
    realtype /*t*/, const_N_Vector xB, const_N_Vector /*dxB*/, N_Vector xBdot
) const {
    /* Right hand side of the adjoint state for steady state computations.
     J is fixed (as x remains in steady state), so the RHS becomes simple. */
    N_VConst(0.0, xBdot);
    derived_state_.JB_.multiply(xBdot, xB);
    /* Mind the minus sign... */
    N_VScale(-1.0, xBdot, xBdot);
}

void Model_DAE::fqBdot_ss(
    realtype /*t*/, const_N_Vector xB, const_N_Vector /*dxB*/, N_Vector qBdot
) const {
    /* Quadratures when computing adjoints for steady state. The integrand is
     just the adjoint state itself. */
    N_VScale(1.0, const_cast<N_Vector>(xB), qBdot);
}

void Model_DAE::fJSparseB_ss(SUNMatrix JB) {
    /* Just pass the model Jacobian on to JB */
    SUNMatCopy(derived_state_.JB_, JB);
    derived_state_.JB_.refresh();
}

void Model_DAE::writeSteadystateJB(
    realtype const t, realtype cj, AmiVector const& x, AmiVector const& dx,
    AmiVector const& xB, AmiVector const& dxB, AmiVector const& /*xBdot*/
) {
    /* Get backward Jacobian */
    fJSparseB(
        t, cj, x.getNVector(), dx.getNVector(), xB.getNVector(),
        dxB.getNVector(), derived_state_.JB_
    );
    derived_state_.JB_.refresh();
    /* Switch sign, as we integrate forward in time, not backward */
    derived_state_.JB_.scale(-1);
}

void Model_DAE::fsxdot(
    realtype const t, AmiVector const& x, AmiVector const& dx, int const ip,
    AmiVector const& sx, AmiVector const& sdx, AmiVector& sxdot
) {
    fsxdot(
        t, x.getNVector(), dx.getNVector(), ip, sx.getNVector(),
        sdx.getNVector(), sxdot.getNVector()
    );
}

void Model_DAE::fsxdot(
    realtype t, const_N_Vector x, const_N_Vector dx, int ip, const_N_Vector sx,
    const_N_Vector sdx, N_Vector sxdot
) {
    if (ip == 0) {
        // we only need to call this for the first parameter index will be
        // the same for all remaining
        fM(t, x);
        fdxdotdp(t, x, dx);
        fJSparse(t, 0.0, x, dx, derived_state_.J_);
        derived_state_.J_.refresh();
    }

    if (pythonGenerated) {
        // python generated, not yet implemented for DAEs
        throw AmiException("Wrapping of DAEs is not yet implemented from Python"
        );
    } else {
        /* copy dxdotdp over */
        N_VScale(1.0, derived_state_.dxdotdp.getNVector(ip), sxdot);
    }

    derived_state_.J_.multiply(sxdot, sx);
    derived_state_.M_.multiply(sxdot, sdx, -1.0);
}

} // namespace amici
