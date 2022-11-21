#ifndef _amici_model_nested_events_h
#define _amici_model_nested_events_h
/* Generated by amiwrap (R2017b) f250595a4da28c2ef7dea61db44b1bef1606f644 */
#include <cmath>
#include <memory>
#include "amici/defines.h"
#include <sunmatrix/sunmatrix_sparse.h> //SUNMatrixContent_Sparse definition
#include "amici/solver_cvodes.h"
#include "amici/model_ode.h"

namespace amici {

class Solver;

namespace model_model_nested_events{

extern void JSparse_model_nested_events(SUNMatrixContent_Sparse JSparse, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *dwdx);
extern void Jy_model_nested_events(double *nllh, const int iy, const realtype *p, const realtype *k, const double *y, const double *sigmay, const double *my);
extern void dJydsigma_model_nested_events(double *dJydsigma, const int iy, const realtype *p, const realtype *k, const double *y, const double *sigmay, const double *my);
extern void dJydy_model_nested_events(double *dJydy, const int iy, const realtype *p, const realtype *k, const double *y, const double *sigmay, const double *my);
extern void deltaqB_model_nested_events(double *deltaqB, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ip, const int ie, const realtype *xdot, const realtype *xdot_old, const realtype *xB);
extern void deltasx_model_nested_events(double *deltasx, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const int ip, const int ie, const realtype *xdot, const realtype *xdot_old, const realtype *sx, const realtype *stau, const realtype *tcl);
extern void deltax_model_nested_events(double *deltax, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ie, const realtype *xdot, const realtype *xdot_old);
extern void dxdotdp_model_nested_events(realtype *dxdotdp, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ip, const realtype *w, const realtype *dwdp);
extern void dydx_model_nested_events(double *dydx, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *dwdx);
extern void root_model_nested_events(realtype *root, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *tcl);
extern void sigmay_model_nested_events(double *sigmay, const realtype t, const realtype *p, const realtype *k, const realtype *y);
extern void stau_model_nested_events(double *stau, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *tcl, const realtype *sx, const int ip, const int ie);
extern void sx0_model_nested_events(realtype *sx0, const realtype t,const realtype *x0, const realtype *p, const realtype *k, const int ip);
extern void x0_model_nested_events(realtype *x0, const realtype t, const realtype *p, const realtype *k);
extern void xdot_model_nested_events(realtype *xdot, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w);
extern void y_model_nested_events(double *y, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w);

class Model_model_nested_events : public amici::Model_ODE {
public:
    Model_model_nested_events()
        : amici::Model_ODE(
              amici::ModelDimensions(
                  1,
                  1,
                  1,
                  1,
                  0,
                  5,
                  0,
                  1,
                  1,
                  0,
                  0,
                  4,
                  0,
                  1,
                  0,
                  0,
                  0,
                  0,
                  0,
                  {},
                  0,
                  0,
                  0,
                  1,
                  0,
                  0
              ),
              amici::SimulationParameters(
                  std::vector<realtype>(0, 1.0),
                  std::vector<realtype>(5, 1.0)
              ),
              amici::SecondOrderMode::none,
              std::vector<realtype>{0},
              std::vector<int>{})
              {};

    amici::Model* clone() const override { return new Model_model_nested_events(*this); };

    std::string getAmiciCommit() const override { return "f250595a4da28c2ef7dea61db44b1bef1606f644"; };

    void fJSparse(SUNMatrixContent_Sparse JSparse, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *dwdx) override {
        JSparse_model_nested_events(JSparse, t, x, p, k, h, w, dwdx);
    }

    void fJrz(double *nllh, const int iz, const realtype *p, const realtype *k, const double *rz, const double *sigmaz) override {
    }

    void fJy(double *nllh, const int iy, const realtype *p, const realtype *k, const double *y, const double *sigmay, const double *my) override {
        Jy_model_nested_events(nllh, iy, p, k, y, sigmay, my);
    }

    void fJz(double *nllh, const int iz, const realtype *p, const realtype *k, const double *z, const double *sigmaz, const double *mz) override {
    }

    void fdJrzdsigma(double *dJrzdsigma, const int iz, const realtype *p, const realtype *k, const double *rz, const double *sigmaz) override {
    }

    void fdJrzdz(double *dJrzdz, const int iz, const realtype *p, const realtype *k, const double *rz, const double *sigmaz) override {
    }

    void fdJydsigma(double *dJydsigma, const int iy, const realtype *p, const realtype *k, const double *y, const double *sigmay, const double *my) override {
        dJydsigma_model_nested_events(dJydsigma, iy, p, k, y, sigmay, my);
    }

    void fdJydy(double *dJydy, const int iy, const realtype *p, const realtype *k, const double *y, const double *sigmay, const double *my) override {
        dJydy_model_nested_events(dJydy, iy, p, k, y, sigmay, my);
    }

    void fdJzdsigma(double *dJzdsigma, const int iz, const realtype *p, const realtype *k, const double *z, const double *sigmaz, const double *mz) override {
    }

    void fdJzdz(double *dJzdz, const int iz, const realtype *p, const realtype *k, const double *z, const double *sigmaz, const double *mz) override {
    }

    void fdeltaqB(double *deltaqB, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ip, const int ie, const realtype *xdot, const realtype *xdot_old, const realtype *xB) override {
        deltaqB_model_nested_events(deltaqB, t, x, p, k, h, ip, ie, xdot, xdot_old, xB);
    }

    void fdeltasx(double *deltasx, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const int ip, const int ie, const realtype *xdot, const realtype *xdot_old, const realtype *sx, const realtype *stau, const realtype *tcl) override {
        deltasx_model_nested_events(deltasx, t, x, p, k, h, w, ip, ie, xdot, xdot_old, sx, stau, tcl);
    }

    void fdeltax(double *deltax, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ie, const realtype *xdot, const realtype *xdot_old) override {
        deltax_model_nested_events(deltax, t, x, p, k, h, ie, xdot, xdot_old);
    }

    void fdeltaxB(double *deltaxB, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ie, const realtype *xdot, const realtype *xdot_old, const realtype *xB) override {
    }

    void fdrzdp(double *drzdp, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ip) override {
    }

    void fdrzdx(double *drzdx, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h) override {
    }

    void fdsigmaydp(double *dsigmaydp, const realtype t, const realtype *p, const realtype *k, const realtype *y, const int ip) override {
    }

    void fdsigmazdp(double *dsigmazdp, const realtype t, const realtype *p, const realtype *k, const int ip) override {
    }

    void fdwdp(realtype *dwdp, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *tcl, const realtype *stcl, const realtype *spl, const realtype *sspl) override {
    }

    void fdwdx(realtype *dwdx, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *tcl, const realtype *spl) override {
    }

    void fdxdotdp(realtype *dxdotdp, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ip, const realtype *w, const realtype *dwdp) override {
        dxdotdp_model_nested_events(dxdotdp, t, x, p, k, h, ip, w, dwdp);
    }

    void fdydp(double *dydp, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ip, const realtype *w, const realtype *dwdp) override {
    }

    void fdydx(double *dydx, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *dwdx) override {
        dydx_model_nested_events(dydx, t, x, p, k, h, w, dwdx);
    }

    void fdzdp(double *dzdp, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ip) override {
    }

    void fdzdx(double *dzdx, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h) override {
    }

    void froot(realtype *root, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *tcl) override {
        root_model_nested_events(root, t, x, p, k, h, tcl);
    }

    void frz(double *rz, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h) override {
    }

    void fsigmay(double *sigmay, const realtype t, const realtype *p, const realtype *k, const realtype *y) override {
        sigmay_model_nested_events(sigmay, t, p, k, y);
    }

    void fsigmaz(double *sigmaz, const realtype t, const realtype *p, const realtype *k) override {
    }

    void fsrz(double *srz, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *sx, const int ip) override {
    }

    void fstau(double *stau, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *tcl, const realtype *sx, const int ip, const int ie) override {
        stau_model_nested_events(stau, t, x, p, k, h, tcl, sx, ip, ie);
    }

    void fsx0(realtype *sx0, const realtype t,const realtype *x0, const realtype *p, const realtype *k, const int ip) override {
        sx0_model_nested_events(sx0, t, x0, p, k, ip);
    }

    void fsz(double *sz, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *sx, const int ip) override {
    }

    void fw(realtype *w, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *tcl, const realtype *spl) override {
    }

    void fx0(realtype *x0, const realtype t, const realtype *p, const realtype *k) override {
        x0_model_nested_events(x0, t, p, k);
    }

    void fxdot(realtype *xdot, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w) override {
        xdot_model_nested_events(xdot, t, x, p, k, h, w);
    }

    void fy(double *y, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w) override {
        y_model_nested_events(y, t, x, p, k, h, w);
    }

    void fz(double *z, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h) override {
    }

};

} // namespace model_model_nested_events

} // namespace amici 

#endif /* _amici_model_nested_events_h */
