#ifndef _amici_wrapfunctions_h
#define _amici_wrapfunctions_h
#include <cmath>
#include <memory>
#include "amici/defines.h"
#include <sundials/sundials_sparse.h> //SlsMat definition
#include "amici/solver_cvodes.h"
#include "amici/model_ode.h"

namespace amici {
class Solver;
}


#define pi M_PI

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

std::unique_ptr<amici::Model> getModel();
extern void J_model_events(realtype *J, const realtype t, const realtype *x, const double *p, const double *k, const realtype *h, const realtype *w, const realtype *dwdx);
extern void JB_model_events(realtype *JB, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *xB, const realtype *w, const realtype *dwdx);
extern void JDiag_model_events(realtype *JDiag, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *dwdx);
extern void JSparse_model_events(SlsMat JSparse, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *dwdx);
extern void JSparseB_model_events(SlsMat JSparseB, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *xB, const realtype *w, const realtype *dwdx);
extern void Jrz_model_events(double *nllh, const int iz, const realtype *p, const realtype *k, const double *rz, const double *sigmaz);
extern void Jv_model_events(realtype *Jv, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *v, const realtype *w, const realtype *dwdx);
extern void JvB_model_events(realtype *JvB, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *xB, const realtype *vB, const realtype *w, const realtype *dwdx);
extern void Jy_model_events(double *nllh, const int iy, const realtype *p, const realtype *k, const double *y, const double *sigmay, const double *my);
extern void Jz_model_events(double *nllh, const int iz, const realtype *p, const realtype *k, const double *z, const double *sigmaz, const double *mz);
extern void dJrzdsigma_model_events(double *dJrzdsigma, const int iz, const realtype *p, const realtype *k, const double *rz, const double *sigmaz);
extern void dJrzdz_model_events(double *dJrzdz, const int iz, const realtype *p, const realtype *k, const double *rz, const double *sigmaz);
extern void dJydsigma_model_events(double *dJydsigma, const int iy, const realtype *p, const realtype *k, const double *y, const double *sigmay, const double *my);
extern void dJydy_model_events(double *dJydy, const int iy, const realtype *p, const realtype *k, const double *y, const double *sigmay, const double *my);
extern void dJzdsigma_model_events(double *dJzdsigma, const int iz, const realtype *p, const realtype *k, const double *z, const double *sigmaz, const double *mz);
extern void dJzdz_model_events(double *dJzdz, const int iz, const realtype *p, const realtype *k, const double *z, const double *sigmaz, const double *mz);
extern void deltasx_model_events(double *deltasx, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const int ip, const int ie, const realtype *xdot, const realtype *xdot_old, const realtype *sx, const realtype *stau);
extern void drzdx_model_events(double *drzdx, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h);
extern void dxdotdp_model_events(realtype *dxdotdp, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ip, const realtype *w, const realtype *dwdp);
extern void dydp_model_events(double *dydp, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ip);
extern void dydx_model_events(double *dydx, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h);
extern void dzdx_model_events(double *dzdx, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h);
extern void qBdot_model_events(realtype *qBdot, const int ip, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *xB, const realtype *w, const realtype *dwdp);
extern void root_model_events(realtype *root, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h);
extern void rz_model_events(double *rz, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h);
extern void sigma_y_model_events(double *sigmay, const realtype t, const realtype *p, const realtype *k);
extern void sigma_z_model_events(double *sigmaz, const realtype t, const realtype *p, const realtype *k);
extern void srz_model_events(double *srz, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *sx, const int ip);
extern void stau_model_events(double *stau, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *sx, const int ip, const int ie);
extern void sxdot_model_events(realtype *sxdot, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ip, const realtype *sx, const realtype *w, const realtype *dwdx, const realtype *J, const realtype *dxdotdp);
extern void sz_model_events(double *sz, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *sx, const int ip);
extern void x0_model_events(realtype *x0, const realtype t, const realtype *p, const realtype *k);
extern void xBdot_model_events(realtype *xBdot, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *xB, const realtype *w, const realtype *dwdx);
extern void xdot_model_events(realtype *xdot, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w);
extern void y_model_events(double *y, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h);
extern void z_model_events(double *z, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h);

class Model_model_events : public amici::Model_ODE {
public:
    Model_model_events() : amici::Model_ODE(3,
                    3,
                    1,
                    1,
                    2,
                    2,
                    4,
                    1,
                    0,
                    0,
                    0,
                    4,
                    0,
                    1,
                    amici::AMICI_O2MODE_NONE,
                    std::vector<realtype>(4),
                    std::vector<realtype>(4),
                    std::vector<int>(),
                    std::vector<realtype>{0, 0, 0},
                    std::vector<int>{1, 2})
                    {};

    virtual amici::Model* clone() const override { return new Model_model_events(*this); };

    virtual void fJ(realtype *J, const realtype t, const realtype *x, const double *p, const double *k, const realtype *h, const realtype *w, const realtype *dwdx) override {
        J_model_events(J, t, x, p, k, h, w, dwdx);
    }

    virtual void fJB(realtype *JB, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *xB, const realtype *w, const realtype *dwdx) override {
        JB_model_events(JB, t, x, p, k, h, xB, w, dwdx);
    }

    virtual void fJDiag(realtype *JDiag, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *dwdx) override {
        JDiag_model_events(JDiag, t, x, p, k, h, w, dwdx);
    }

    virtual void fJSparse(SlsMat JSparse, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *dwdx) override {
        JSparse_model_events(JSparse, t, x, p, k, h, w, dwdx);
    }

    virtual void fJSparseB(SlsMat JSparseB, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *xB, const realtype *w, const realtype *dwdx) override {
        JSparseB_model_events(JSparseB, t, x, p, k, h, xB, w, dwdx);
    }

    virtual void fJrz(double *nllh, const int iz, const realtype *p, const realtype *k, const double *rz, const double *sigmaz) override {
        Jrz_model_events(nllh, iz, p, k, rz, sigmaz);
    }

    virtual void fJv(realtype *Jv, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *v, const realtype *w, const realtype *dwdx) override {
        Jv_model_events(Jv, t, x, p, k, h, v, w, dwdx);
    }

    virtual void fJvB(realtype *JvB, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *xB, const realtype *vB, const realtype *w, const realtype *dwdx) override {
        JvB_model_events(JvB, t, x, p, k, h, xB, vB, w, dwdx);
    }

    virtual void fJy(double *nllh, const int iy, const realtype *p, const realtype *k, const double *y, const double *sigmay, const double *my) override {
        Jy_model_events(nllh, iy, p, k, y, sigmay, my);
    }

    virtual void fJz(double *nllh, const int iz, const realtype *p, const realtype *k, const double *z, const double *sigmaz, const double *mz) override {
        Jz_model_events(nllh, iz, p, k, z, sigmaz, mz);
    }

    virtual void fdJrzdsigma(double *dJrzdsigma, const int iz, const realtype *p, const realtype *k, const double *rz, const double *sigmaz) override {
        dJrzdsigma_model_events(dJrzdsigma, iz, p, k, rz, sigmaz);
    }

    virtual void fdJrzdz(double *dJrzdz, const int iz, const realtype *p, const realtype *k, const double *rz, const double *sigmaz) override {
        dJrzdz_model_events(dJrzdz, iz, p, k, rz, sigmaz);
    }

    virtual void fdJydsigma(double *dJydsigma, const int iy, const realtype *p, const realtype *k, const double *y, const double *sigmay, const double *my) override {
        dJydsigma_model_events(dJydsigma, iy, p, k, y, sigmay, my);
    }

    virtual void fdJydy(double *dJydy, const int iy, const realtype *p, const realtype *k, const double *y, const double *sigmay, const double *my) override {
        dJydy_model_events(dJydy, iy, p, k, y, sigmay, my);
    }

    virtual void fdJzdsigma(double *dJzdsigma, const int iz, const realtype *p, const realtype *k, const double *z, const double *sigmaz, const double *mz) override {
        dJzdsigma_model_events(dJzdsigma, iz, p, k, z, sigmaz, mz);
    }

    virtual void fdJzdz(double *dJzdz, const int iz, const realtype *p, const realtype *k, const double *z, const double *sigmaz, const double *mz) override {
        dJzdz_model_events(dJzdz, iz, p, k, z, sigmaz, mz);
    }

    virtual void fdeltaqB(double *deltaqB, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ip, const int ie, const realtype *xdot, const realtype *xdot_old, const realtype *xB) override {
    }

    virtual void fdeltasx(double *deltasx, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const int ip, const int ie, const realtype *xdot, const realtype *xdot_old, const realtype *sx, const realtype *stau) override {
        deltasx_model_events(deltasx, t, x, p, k, h, w, ip, ie, xdot, xdot_old, sx, stau);
    }

    virtual void fdeltax(double *deltax, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ie, const realtype *xdot, const realtype *xdot_old) override {
    }

    virtual void fdeltaxB(double *deltaxB, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ie, const realtype *xdot, const realtype *xdot_old, const realtype *xB) override {
    }

    virtual void fdrzdp(double *drzdp, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ip) override {
    }

    virtual void fdrzdx(double *drzdx, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h) override {
        drzdx_model_events(drzdx, ie, t, x, p, k, h);
    }

    virtual void fdsigma_ydp(double *dsigmaydp, const realtype t, const realtype *p, const realtype *k, const int ip) override {
    }

    virtual void fdsigma_zdp(double *dsigmazdp, const realtype t, const realtype *p, const realtype *k, const int ip) override {
    }

    virtual void fdwdp(realtype *dwdp, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w) override {
    }

    virtual void fdwdx(realtype *dwdx, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w) override {
    }

    virtual void fdxdotdp(realtype *dxdotdp, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ip, const realtype *w, const realtype *dwdp) override {
        dxdotdp_model_events(dxdotdp, t, x, p, k, h, ip, w, dwdp);
    }

    virtual void fdydp(double *dydp, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ip) override {
        dydp_model_events(dydp, t, x, p, k, h, ip);
    }

    virtual void fdydx(double *dydx, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h) override {
        dydx_model_events(dydx, t, x, p, k, h);
    }

    virtual void fdzdp(double *dzdp, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ip) override {
    }

    virtual void fdzdx(double *dzdx, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h) override {
        dzdx_model_events(dzdx, ie, t, x, p, k, h);
    }

    virtual void fqBdot(realtype *qBdot, const int ip, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *xB, const realtype *w, const realtype *dwdp) override {
        qBdot_model_events(qBdot, ip, t, x, p, k, h, xB, w, dwdp);
    }

    virtual void froot(realtype *root, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h) override {
        root_model_events(root, t, x, p, k, h);
    }

    virtual void frz(double *rz, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h) override {
        rz_model_events(rz, ie, t, x, p, k, h);
    }

    virtual void fsigma_y(double *sigmay, const realtype t, const realtype *p, const realtype *k) override {
        sigma_y_model_events(sigmay, t, p, k);
    }

    virtual void fsigma_z(double *sigmaz, const realtype t, const realtype *p, const realtype *k) override {
        sigma_z_model_events(sigmaz, t, p, k);
    }

    virtual void fsrz(double *srz, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *sx, const int ip) override {
        srz_model_events(srz, ie, t, x, p, k, h, sx, ip);
    }

    virtual void fstau(double *stau, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *sx, const int ip, const int ie) override {
        stau_model_events(stau, t, x, p, k, h, sx, ip, ie);
    }

    virtual void fsx0(realtype *sx0, const realtype t,const realtype *x0, const realtype *p, const realtype *k, const int ip) override {
    }

    virtual void fsxdot(realtype *sxdot, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ip, const realtype *sx, const realtype *w, const realtype *dwdx, const realtype *J, const realtype *dxdotdp) override {
        sxdot_model_events(sxdot, t, x, p, k, h, ip, sx, w, dwdx, J, dxdotdp);
    }

    virtual void fsz(double *sz, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *sx, const int ip) override {
        sz_model_events(sz, ie, t, x, p, k, h, sx, ip);
    }

    virtual void fw(realtype *w, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h) override {
    }

    virtual void fx0(realtype *x0, const realtype t, const realtype *p, const realtype *k) override {
        x0_model_events(x0, t, p, k);
    }

    virtual void fxBdot(realtype *xBdot, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *xB, const realtype *w, const realtype *dwdx) override {
        xBdot_model_events(xBdot, t, x, p, k, h, xB, w, dwdx);
    }

    virtual void fxdot(realtype *xdot, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w) override {
        xdot_model_events(xdot, t, x, p, k, h, w);
    }

    virtual void fy(double *y, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h) override {
        y_model_events(y, t, x, p, k, h);
    }

    virtual void fz(double *z, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h) override {
        z_model_events(z, ie, t, x, p, k, h);
    }

};

#endif /* _amici_wrapfunctions_h */
