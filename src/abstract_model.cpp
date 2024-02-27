#include "amici/abstract_model.h"

namespace amici {

std::string AbstractModel::getAmiciVersion() const {
    throw AmiException("Version not set during code generation");
}

std::string AbstractModel::getAmiciCommit() const {
    throw AmiException("Commit not set during code generation");
}

void AbstractModel::
    fx0(realtype* /*x0*/, realtype const /*t*/, realtype const* /*p*/,
        realtype const* /*k*/) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

bool AbstractModel::isFixedParameterStateReinitializationAllowed() const {
    return false;
}

void AbstractModel::fx0_fixedParameters(
    realtype* /*x0*/, realtype const /*t*/, realtype const* /*p*/,
    realtype const* /*k*/, gsl::span<int const> /*reinitialization_state_idxs*/
) {
    // no-op default implementation
}

void AbstractModel::fsx0_fixedParameters(
    realtype* /*sx0*/, realtype const /*t*/, realtype const* /*x0*/,
    realtype const* /*p*/, realtype const* /*k*/, int const /*ip*/,
    gsl::span<int const> /*reinitialization_state_idxs*/
) {
    // no-op default implementation
}

void AbstractModel::fsx0(
    realtype* /*sx0*/, realtype const /*t*/, realtype const* /*x0*/,
    realtype const* /*p*/, realtype const* /*k*/, int const /*ip*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::fdx0(AmiVector& /*x0*/, AmiVector& /*dx0*/) {
    // no-op default implementation
}

void AbstractModel::fstau(
    realtype* /*stau*/, realtype const /*t*/, realtype const* /*x*/,
    realtype const* /*p*/, realtype const* /*k*/, realtype const* /*h*/,
    realtype const* /*tcl*/, realtype const* /*sx*/, int const /*ip*/,
    int const /*ie*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::
    fy(realtype* /*y*/, realtype const /*t*/, realtype const* /*x*/,
       realtype const* /*p*/, realtype const* /*k*/, realtype const* /*h*/,
       realtype const* /*w*/) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::fdydp(
    realtype* /*dydp*/, realtype const /*t*/, realtype const* /*x*/,
    realtype const* /*p*/, realtype const* /*k*/, realtype const* /*h*/,
    int const /*ip*/, realtype const* /*w*/, realtype const* /*dwdp*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::fdydp(
    realtype* /*dydp*/, realtype const /*t*/, realtype const* /*x*/,
    realtype const* /*p*/, realtype const* /*k*/, realtype const* /*h*/,
    int /*ip*/, realtype const* /*w*/, realtype const* /*tcl*/,
    realtype const* /*dtcldp*/, realtype const* /*spl*/,
    realtype const* /*sspl*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::fdydx(
    realtype* /*dydx*/, realtype const /*t*/, realtype const* /*x*/,
    realtype const* /*p*/, realtype const* /*k*/, realtype const* /*h*/,
    realtype const* /*w*/, realtype const* /*dwdx*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::
    fz(realtype* /*z*/, int const /*ie*/, realtype const /*t*/,
       realtype const* /*x*/, realtype const* /*p*/, realtype const* /*k*/,
       realtype const* /*h*/) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::
    fsz(realtype* /*sz*/, int const /*ie*/, realtype const /*t*/,
        realtype const* /*x*/, realtype const* /*p*/, realtype const* /*k*/,
        realtype const* /*h*/, realtype const* /*sx*/, int const /*ip*/) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::
    frz(realtype* /*rz*/, int const /*ie*/, realtype const /*t*/,
        realtype const* /*x*/, realtype const* /*p*/, realtype const* /*k*/,
        realtype const* /*h*/) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::fsrz(
    realtype* /*srz*/, int const /*ie*/, realtype const /*t*/,
    realtype const* /*x*/, realtype const* /*p*/, realtype const* /*k*/,
    realtype const* /*h*/, realtype const* /*sx*/, int const /*ip*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::fdzdp(
    realtype* /*dzdp*/, int const /*ie*/, realtype const /*t*/,
    realtype const* /*x*/, realtype const* /*p*/, realtype const* /*k*/,
    realtype const* /*h*/, int const /*ip*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::fdzdx(
    realtype* /*dzdx*/, int const /*ie*/, realtype const /*t*/,
    realtype const* /*x*/, realtype const* /*p*/, realtype const* /*k*/,
    realtype const* /*h*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::fdrzdp(
    realtype* /*drzdp*/, int const /*ie*/, realtype const /*t*/,
    realtype const* /*x*/, realtype const* /*p*/, realtype const* /*k*/,
    realtype const* /*h*/, int const /*ip*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::fdrzdx(
    realtype* /*drzdx*/, int const /*ie*/, realtype const /*t*/,
    realtype const* /*x*/, realtype const* /*p*/, realtype const* /*k*/,
    realtype const* /*h*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::fdeltax(
    realtype* /*deltax*/, realtype const /*t*/, realtype const* /*x*/,
    realtype const* /*p*/, realtype const* /*k*/, realtype const* /*h*/,
    int const /*ie*/, realtype const* /*xdot*/, realtype const* /*xdot_old*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::fdeltasx(
    realtype* /*deltasx*/, realtype const /*t*/, realtype const* /*x*/,
    realtype const* /*p*/, realtype const* /*k*/, realtype const* /*h*/,
    realtype const* /*w*/, int const /*ip*/, int const /*ie*/,
    realtype const* /*xdot*/, realtype const* /*xdot_old*/,
    realtype const* /*sx*/, realtype const* /*stau*/, realtype const* /*tcl*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::fdeltaxB(
    realtype* /*deltaxB*/, realtype const /*t*/, realtype const* /*x*/,
    realtype const* /*p*/, realtype const* /*k*/, realtype const* /*h*/,
    int const /*ie*/, realtype const* /*xdot*/, realtype const* /*xdot_old*/,
    realtype const* /*xB*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::fdeltaqB(
    realtype* /*deltaqB*/, realtype const /*t*/, realtype const* /*x*/,
    realtype const* /*p*/, realtype const* /*k*/, realtype const* /*h*/,
    int const /*ip*/, int const /*ie*/, realtype const* /*xdot*/,
    realtype const* /*xdot_old*/, realtype const* /*xB*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::fsigmay(
    realtype* /*sigmay*/, realtype const /*t*/, realtype const* /*p*/,
    realtype const* /*k*/, realtype const* /*y*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::fdsigmaydp(
    realtype* /*dsigmaydp*/, realtype const /*t*/, realtype const* /*p*/,
    realtype const* /*k*/, realtype const* /*y*/, int const /*ip*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::fdsigmaydy(
    realtype* /*dsigmaydy*/, realtype const /*t*/, realtype const* /*p*/,
    realtype const* /*k*/, realtype const* /*y*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::fsigmaz(
    realtype* /*sigmaz*/, realtype const /*t*/, realtype const* /*p*/,
    realtype const* /*k*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::fdsigmazdp(
    realtype* /*dsigmazdp*/, realtype const /*t*/, realtype const* /*p*/,
    realtype const* /*k*/, int const /*ip*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::
    fJy(realtype* /*nllh*/, int const /*iy*/, realtype const* /*p*/,
        realtype const* /*k*/, realtype const* /*y*/,
        realtype const* /*sigmay*/, realtype const* /*my*/) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::
    fJz(realtype* /*nllh*/, int const /*iz*/, realtype const* /*p*/,
        realtype const* /*k*/, realtype const* /*z*/,
        realtype const* /*sigmaz*/, realtype const* /*mz*/) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::fJrz(
    realtype* /*nllh*/, int const /*iz*/, realtype const* /*p*/,
    realtype const* /*k*/, realtype const* /*z*/, realtype const* /*sigmaz*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::fdJydy(
    realtype* /*dJydy*/, int const /*iy*/, realtype const* /*p*/,
    realtype const* /*k*/, realtype const* /*y*/, realtype const* /*sigmay*/,
    realtype const* /*my*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::
    fdJydy_colptrs(SUNMatrixWrapper& /*indexptrs*/, int /*index*/) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::
    fdJydy_rowvals(SUNMatrixWrapper& /*indexptrs*/, int /*index*/) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::fdJydsigma(
    realtype* /*dJydsigma*/, int const /*iy*/, realtype const* /*p*/,
    realtype const* /*k*/, realtype const* /*y*/, realtype const* /*sigmay*/,
    realtype const* /*my*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::fdJzdz(
    realtype* /*dJzdz*/, int const /*iz*/, realtype const* /*p*/,
    realtype const* /*k*/, realtype const* /*z*/, realtype const* /*sigmaz*/,
    realtype const* /*mz*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::fdJzdsigma(
    realtype* /*dJzdsigma*/, int const /*iz*/, realtype const* /*p*/,
    realtype const* /*k*/, realtype const* /*z*/, realtype const* /*sigmaz*/,
    realtype const* /*mz*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::fdJrzdz(
    realtype* /*dJrzdz*/, int const /*iz*/, realtype const* /*p*/,
    realtype const* /*k*/, realtype const* /*rz*/, realtype const* /*sigmaz*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::fdJrzdsigma(
    realtype* /*dJrzdsigma*/, int const /*iz*/, realtype const* /*p*/,
    realtype const* /*k*/, realtype const* /*rz*/, realtype const* /*sigmaz*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::fw(
    realtype* /*w*/, realtype const /*t*/, realtype const* /*x*/,
    realtype const* /*p*/, realtype const* /*k*/, realtype const* /*h*/,
    realtype const* /*tcl*/, realtype const* /*spl*/, bool /*include_static*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::fdwdp(
    realtype* /*dwdp*/, realtype const /*t*/, realtype const* /*x*/,
    realtype const* /*p*/, realtype const* /*k*/, realtype const* /*h*/,
    realtype const* /*w*/, realtype const* /*tcl*/, realtype const* /*stcl*/,
    realtype const* /*spl*/, realtype const* /*sspl*/, bool /*include_static*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::fdwdp_colptrs(SUNMatrixWrapper& /*dwdp*/) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::fdwdp_rowvals(SUNMatrixWrapper& /*dwdp*/) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::fdwdx(
    realtype* /*dwdx*/, realtype const /*t*/, realtype const* /*x*/,
    realtype const* /*p*/, realtype const* /*k*/, realtype const* /*h*/,
    realtype const* /*w*/, realtype const* /*tcl*/, realtype const* /*spl*/,
    bool /*include_static*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::fdwdx_colptrs(SUNMatrixWrapper& /*dwdx*/) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::fdwdx_rowvals(SUNMatrixWrapper& /*dwdx*/) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::fdwdw(
    realtype* /*dwdw*/, realtype /*t*/, realtype const* /*x*/,
    realtype const* /*p*/, realtype const* /*k*/, realtype const* /*h*/,
    realtype const* /*w*/, realtype const* /*tcl*/, bool /*include_static*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s "
        "is not implemented for this model!",
        __func__
    );
}

void AbstractModel::fdwdw_colptrs(SUNMatrixWrapper& /*dwdw*/) {
    throw AmiException(
        "Requested functionality is not supported as %s "
        "is not implemented for this model!",
        __func__
    );
}

void AbstractModel::fdwdw_rowvals(SUNMatrixWrapper& /*dwdw*/) {
    throw AmiException(
        "Requested functionality is not supported as %s "
        "is not implemented for this model!",
        __func__
    );
}

void AbstractModel::fdx_rdatadx_solver(
    realtype* /*dx_rdatadx_solver*/, realtype const* /*x*/,
    realtype const* /*tcl*/, realtype const* /*p*/, realtype const* /*k*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::fdx_rdatadx_solver_rowvals(SUNMatrixWrapper& /*dxrdxs*/) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::fdx_rdatadx_solver_colptrs(SUNMatrixWrapper& /*dxrdxs*/) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::fdx_rdatadp(
    realtype* /*dx_rdatadp*/, realtype const* /*x*/, realtype const* /*tcl*/,
    realtype const* /*p*/, realtype const* /*k*/, int const /*ip*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::fdx_rdatadtcl(
    realtype* /*dx_rdatadtcl*/, realtype const* /*x*/, realtype const* /*tcl*/,
    realtype const* /*p*/, realtype const* /*k*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::fdx_rdatadtcl_rowvals(SUNMatrixWrapper& /*dxrdtcl*/) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::fdx_rdatadtcl_colptrs(SUNMatrixWrapper& /*dxrdtcl*/) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::fdtotal_cldp(
    realtype* /*dtotal_cldp*/, realtype const* /*x_rdata*/,
    realtype const* /*p*/, realtype const* /*k*/, int const /*ip*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::fdtotal_cldx_rdata(
    realtype* /*dtotal_cldx_rdata*/, realtype const* /*x_rdata*/,
    realtype const* /*p*/, realtype const* /*k*/, realtype const* /*tcl*/
) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::
    fdtotal_cldx_rdata_colptrs(SUNMatrixWrapper& /*dtotal_cldx_rdata*/) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

void AbstractModel::
    fdtotal_cldx_rdata_rowvals(SUNMatrixWrapper& /*dtotal_cldx_rdata*/) {
    throw AmiException(
        "Requested functionality is not supported as %s is "
        "not implemented for this model!",
        __func__
    );
}

std::vector<HermiteSpline>
AbstractModel::fcreate_splines(realtype const* /*p*/, realtype const* /*k*/) {
    return std::vector<HermiteSpline>();
}

void AbstractModel::fdspline_valuesdp(
    realtype* /*dspline_valuesdp*/, realtype const* /*p*/,
    realtype const* /*k*/, int const /*ip*/
) {
    // no-op default implementation
}

void AbstractModel::fdspline_slopesdp(
    realtype* /*dspline_slopesdp*/, realtype const* /*p*/,
    realtype const* /*k*/, int const /*ip*/
) {
    // no-op default implementation
}

} // namespace amici
