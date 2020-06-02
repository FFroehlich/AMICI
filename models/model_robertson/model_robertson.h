#ifndef _amici_model_robertson_h
#define _amici_model_robertson_h
/* Generated by amiwrap (R2017b) 2cf671b663c340dc79a72e68f12e0fa7c9501158 */
#include <cmath>
#include <memory>
#include "amici/defines.h"
#include <sunmatrix/sunmatrix_sparse.h> //SUNMatrixContent_Sparse definition
#include "amici/solver_idas.h"
#include "amici/model_dae.h"

namespace amici {

class Solver;

namespace model_model_robertson{

extern void J_model_robertson(realtype *J, const realtype t, const realtype *x, const double *p, const double *k, const realtype *h, const realtype cj, const realtype *dx, const realtype *w, const realtype *dwdx);
extern void JB_model_robertson(realtype *JB, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype cj, const realtype *xB, const realtype *dx, const realtype *dxB, const realtype *w, const realtype *dwdx);
extern void JDiag_model_robertson(realtype *JDiag, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype cj, const realtype *dx, const realtype *w, const realtype *dwdx);
extern void JSparse_model_robertson(SUNMatrixContent_Sparse JSparse, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype cj, const realtype *dx, const realtype *w, const realtype *dwdx);
extern void JSparseB_model_robertson(SUNMatrixContent_Sparse JSparseB, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype cj, const realtype *xB, const realtype *dx, const realtype *dxB, const realtype *w, const realtype *dwdx);
extern void Jy_model_robertson(double *nllh, const int iy, const realtype *p, const realtype *k, const double *y, const double *sigmay, const double *my);
extern void M_model_robertson(realtype *M, const realtype t, const realtype *x, const realtype *p, const realtype *k);
extern void dJydsigma_model_robertson(double *dJydsigma, const int iy, const realtype *p, const realtype *k, const double *y, const double *sigmay, const double *my);
extern void dJydy_model_robertson(double *dJydy, const int iy, const realtype *p, const realtype *k, const double *y, const double *sigmay, const double *my);
extern void dwdp_model_robertson(realtype *dwdp, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *tcl, const realtype *stcl);
extern void dwdx_model_robertson(realtype *dwdx, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *tcl);
extern void dxdotdp_model_robertson(realtype *dxdotdp, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ip, const realtype *dx, const realtype *w, const realtype *dwdp);
extern void dydx_model_robertson(double *dydx, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *dwdx);
extern void sigmay_model_robertson(double *sigmay, const realtype t, const realtype *p, const realtype *k);
extern void w_model_robertson(realtype *w, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *tcl);
extern void x0_model_robertson(realtype *x0, const realtype t, const realtype *p, const realtype *k);
extern void xdot_model_robertson(realtype *xdot, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *dx, const realtype *w);
extern void y_model_robertson(double *y, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w);

class Model_model_robertson : public amici::Model_DAE {
public:
    Model_model_robertson() : amici::Model_DAE(3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    0,
                    0,
                    0,
                    1,
                    1,
                    2,
                    1,
                    0,
                    {},
                    9,
                    2,
                    2,
                    amici::SecondOrderMode::none,
                    std::vector<realtype>(3,1.0),
                    std::vector<realtype>(1,1.0),
                    std::vector<int>(),
                    std::vector<realtype>{1, 1, 0},
                    std::vector<int>{})
                    {};

    virtual amici::Model* clone() const override { return new Model_model_robertson(*this); };

    const  std::string getAmiciCommit() const override { return "2cf671b663c340dc79a72e68f12e0fa7c9501158"; };

    virtual void fJ(realtype *J, const realtype t, const realtype *x, const double *p, const double *k, const realtype *h, const realtype cj, const realtype *dx, const realtype *w, const realtype *dwdx) override {
        J_model_robertson(J, t, x, p, k, h, cj, dx, w, dwdx);
    }

    virtual void fJB(realtype *JB, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype cj, const realtype *xB, const realtype *dx, const realtype *dxB, const realtype *w, const realtype *dwdx) override {
        JB_model_robertson(JB, t, x, p, k, h, cj, xB, dx, dxB, w, dwdx);
    }

    virtual void fJDiag(realtype *JDiag, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype cj, const realtype *dx, const realtype *w, const realtype *dwdx) override {
        JDiag_model_robertson(JDiag, t, x, p, k, h, cj, dx, w, dwdx);
    }

    virtual void fJSparse(SUNMatrixContent_Sparse JSparse, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype cj, const realtype *dx, const realtype *w, const realtype *dwdx) override {
        JSparse_model_robertson(JSparse, t, x, p, k, h, cj, dx, w, dwdx);
    }

    virtual void fJSparseB(SUNMatrixContent_Sparse JSparseB, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype cj, const realtype *xB, const realtype *dx, const realtype *dxB, const realtype *w, const realtype *dwdx) override {
        JSparseB_model_robertson(JSparseB, t, x, p, k, h, cj, xB, dx, dxB, w, dwdx);
    }

    virtual void fJrz(double *nllh, const int iz, const realtype *p, const realtype *k, const double *rz, const double *sigmaz) override {
    }

    virtual void fJy(double *nllh, const int iy, const realtype *p, const realtype *k, const double *y, const double *sigmay, const double *my) override {
        Jy_model_robertson(nllh, iy, p, k, y, sigmay, my);
    }

    virtual void fJz(double *nllh, const int iz, const realtype *p, const realtype *k, const double *z, const double *sigmaz, const double *mz) override {
    }

    virtual void fM(realtype *M, const realtype t, const realtype *x, const realtype *p, const realtype *k) override {
        M_model_robertson(M, t, x, p, k);
    }

    virtual void fdJrzdsigma(double *dJrzdsigma, const int iz, const realtype *p, const realtype *k, const double *rz, const double *sigmaz) override {
    }

    virtual void fdJrzdz(double *dJrzdz, const int iz, const realtype *p, const realtype *k, const double *rz, const double *sigmaz) override {
    }

    virtual void fdJydsigma(double *dJydsigma, const int iy, const realtype *p, const realtype *k, const double *y, const double *sigmay, const double *my) override {
        dJydsigma_model_robertson(dJydsigma, iy, p, k, y, sigmay, my);
    }

    virtual void fdJydy(double *dJydy, const int iy, const realtype *p, const realtype *k, const double *y, const double *sigmay, const double *my) override {
        dJydy_model_robertson(dJydy, iy, p, k, y, sigmay, my);
    }

    virtual void fdJzdsigma(double *dJzdsigma, const int iz, const realtype *p, const realtype *k, const double *z, const double *sigmaz, const double *mz) override {
    }

    virtual void fdJzdz(double *dJzdz, const int iz, const realtype *p, const realtype *k, const double *z, const double *sigmaz, const double *mz) override {
    }

    virtual void fdeltaqB(double *deltaqB, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ip, const int ie, const realtype *xdot, const realtype *xdot_old, const realtype *xB) override {
    }

    virtual void fdeltasx(double *deltasx, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const int ip, const int ie, const realtype *xdot, const realtype *xdot_old, const realtype *sx, const realtype *stau) override {
    }

    virtual void fdeltax(double *deltax, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ie, const realtype *xdot, const realtype *xdot_old) override {
    }

    virtual void fdeltaxB(double *deltaxB, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ie, const realtype *xdot, const realtype *xdot_old, const realtype *xB) override {
    }

    virtual void fdrzdp(double *drzdp, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ip) override {
    }

    virtual void fdrzdx(double *drzdx, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h) override {
    }

    virtual void fdsigmaydp(double *dsigmaydp, const realtype t, const realtype *p, const realtype *k, const int ip) override {
    }

    virtual void fdsigmazdp(double *dsigmazdp, const realtype t, const realtype *p, const realtype *k, const int ip) override {
    }

    virtual void fdwdp(realtype *dwdp, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *tcl, const realtype *stcl) override {
        dwdp_model_robertson(dwdp, t, x, p, k, h, w, tcl, stcl);
    }

    virtual void fdwdx(realtype *dwdx, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *tcl) override {
        dwdx_model_robertson(dwdx, t, x, p, k, h, w, tcl);
    }

    virtual void fdxdotdp(realtype *dxdotdp, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ip, const realtype *dx, const realtype *w, const realtype *dwdp) override {
        dxdotdp_model_robertson(dxdotdp, t, x, p, k, h, ip, dx, w, dwdp);
    }

    virtual void fdydp(double *dydp, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ip, const realtype *w, const realtype *dwdp) override {
    }

    virtual void fdydx(double *dydx, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *dwdx) override {
        dydx_model_robertson(dydx, t, x, p, k, h, w, dwdx);
    }

    virtual void fdzdp(double *dzdp, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ip) override {
    }

    virtual void fdzdx(double *dzdx, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h) override {
    }

    virtual void froot(realtype *root, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *dx) override {
    }

    virtual void frz(double *rz, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h) override {
    }

    virtual void fsigmay(double *sigmay, const realtype t, const realtype *p, const realtype *k) override {
        sigmay_model_robertson(sigmay, t, p, k);
    }

    virtual void fsigmaz(double *sigmaz, const realtype t, const realtype *p, const realtype *k) override {
    }

    virtual void fsrz(double *srz, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *sx, const int ip) override {
    }

    virtual void fstau(double *stau, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *sx, const int ip, const int ie) override {
    }

    virtual void fsx0(realtype *sx0, const realtype t,const realtype *x0, const realtype *p, const realtype *k, const int ip) override {
    }

    virtual void fsz(double *sz, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *sx, const int ip) override {
    }

    virtual void fw(realtype *w, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *tcl) override {
        w_model_robertson(w, t, x, p, k, h, tcl);
    }

    virtual void fx0(realtype *x0, const realtype t, const realtype *p, const realtype *k) override {
        x0_model_robertson(x0, t, p, k);
    }

    virtual void fxdot(realtype *xdot, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *dx, const realtype *w) override {
        xdot_model_robertson(xdot, t, x, p, k, h, dx, w);
    }

    virtual void fy(double *y, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w) override {
        y_model_robertson(y, t, x, p, k, h, w);
    }

    virtual void fz(double *z, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h) override {
    }

};

} // namespace model_model_robertson

} // namespace amici 

#endif /* _amici_model_robertson_h */
