#ifndef _amici_model_neuron_o2_h
#define _amici_model_neuron_o2_h
/* Generated by amiwrap (R2017b) 31279f3c8293dd4ce1db9dabf03202740751bf5e */
#include <cmath>
#include <memory>
#include "amici/defines.h"
#include <sunmatrix/sunmatrix_sparse.h> //SUNMatrixContent_Sparse definition
#include "amici/solver_cvodes.h"
#include "amici/model_ode.h"

namespace amici {

class Solver;

namespace model_model_neuron_o2{

extern void JSparse_model_neuron_o2(SUNMatrixContent_Sparse JSparse, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *dwdx);
extern void Jrz_model_neuron_o2(double *nllh, const int iz, const realtype *p, const realtype *k, const double *rz, const double *sigmaz);
extern void Jy_model_neuron_o2(double *nllh, const int iy, const realtype *p, const realtype *k, const double *y, const double *sigmay, const double *my);
extern void Jz_model_neuron_o2(double *nllh, const int iz, const realtype *p, const realtype *k, const double *z, const double *sigmaz, const double *mz);
extern void dJrzdsigma_model_neuron_o2(double *dJrzdsigma, const int iz, const realtype *p, const realtype *k, const double *rz, const double *sigmaz);
extern void dJrzdz_model_neuron_o2(double *dJrzdz, const int iz, const realtype *p, const realtype *k, const double *rz, const double *sigmaz);
extern void dJydsigma_model_neuron_o2(double *dJydsigma, const int iy, const realtype *p, const realtype *k, const double *y, const double *sigmay, const double *my);
extern void dJydy_model_neuron_o2(double *dJydy, const int iy, const realtype *p, const realtype *k, const double *y, const double *sigmay, const double *my);
extern void dJzdsigma_model_neuron_o2(double *dJzdsigma, const int iz, const realtype *p, const realtype *k, const double *z, const double *sigmaz, const double *mz);
extern void dJzdz_model_neuron_o2(double *dJzdz, const int iz, const realtype *p, const realtype *k, const double *z, const double *sigmaz, const double *mz);
extern void deltaqB_model_neuron_o2(double *deltaqB, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ip, const int ie, const realtype *xdot, const realtype *xdot_old, const realtype *xB);
extern void deltasx_model_neuron_o2(double *deltasx, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const int ip, const int ie, const realtype *xdot, const realtype *xdot_old, const realtype *sx, const realtype *stau, const realtype *tcl);
extern void deltax_model_neuron_o2(double *deltax, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ie, const realtype *xdot, const realtype *xdot_old);
extern void deltaxB_model_neuron_o2(double *deltaxB, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ie, const realtype *xdot, const realtype *xdot_old, const realtype *xB);
extern void drzdx_model_neuron_o2(double *drzdx, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h);
extern void dwdx_model_neuron_o2(realtype *dwdx, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *tcl, const realtype *spl);
extern void dxdotdp_model_neuron_o2(realtype *dxdotdp, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ip, const realtype *w, const realtype *dwdp);
extern void dydx_model_neuron_o2(double *dydx, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *dwdx);
extern void dzdx_model_neuron_o2(double *dzdx, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h);
extern void root_model_neuron_o2(realtype *root, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *tcl);
extern void rz_model_neuron_o2(double *rz, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h);
extern void sigmay_model_neuron_o2(double *sigmay, const realtype t, const realtype *p, const realtype *k, const realtype *y);
extern void sigmaz_model_neuron_o2(double *sigmaz, const realtype t, const realtype *p, const realtype *k);
extern void srz_model_neuron_o2(double *srz, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *sx, const int ip);
extern void stau_model_neuron_o2(double *stau, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *tcl, const realtype *sx, const int ip, const int ie);
extern void sx0_model_neuron_o2(realtype *sx0, const realtype t,const realtype *x0, const realtype *p, const realtype *k, const int ip);
extern void sz_model_neuron_o2(double *sz, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *sx, const int ip);
extern void w_model_neuron_o2(realtype *w, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *tcl, const realtype *spl);
extern void x0_model_neuron_o2(realtype *x0, const realtype t, const realtype *p, const realtype *k);
extern void xdot_model_neuron_o2(realtype *xdot, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w);
extern void y_model_neuron_o2(double *y, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w);
extern void z_model_neuron_o2(double *z, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h);

class Model_model_neuron_o2 : public amici::Model_ODE {
public:
    Model_model_neuron_o2()
        : amici::Model_ODE(
              amici::ModelDimensions(
                  10,
                  2,
                  10,
                  2,
                  0,
                  4,
                  2,
                  5,
                  1,
                  5,
                  1,
                  1,
                  0,
                  5,
                  2,
                  2,
                  0,
                  0,
                  0,
                  {},
                  0,
                  0,
                  0,
                  27,
                  1,
                  8
              ),
              amici::SimulationParameters(
                  std::vector<realtype>(2, 1.0),
                  std::vector<realtype>(4, 1.0)
              ),
              amici::SecondOrderMode::full,
              std::vector<realtype>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
              std::vector<int>{1, 1, 1, 1, 1})
              {};

    amici::Model* clone() const override { return new Model_model_neuron_o2(*this); };

    std::string getAmiciCommit() const override { return "31279f3c8293dd4ce1db9dabf03202740751bf5e"; };

    void fJSparse(SUNMatrixContent_Sparse JSparse, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *dwdx) override {
        JSparse_model_neuron_o2(JSparse, t, x, p, k, h, w, dwdx);
    }

    void fJrz(double *nllh, const int iz, const realtype *p, const realtype *k, const double *rz, const double *sigmaz) override {
        Jrz_model_neuron_o2(nllh, iz, p, k, rz, sigmaz);
    }

    void fJy(double *nllh, const int iy, const realtype *p, const realtype *k, const double *y, const double *sigmay, const double *my) override {
        Jy_model_neuron_o2(nllh, iy, p, k, y, sigmay, my);
    }

    void fJz(double *nllh, const int iz, const realtype *p, const realtype *k, const double *z, const double *sigmaz, const double *mz) override {
        Jz_model_neuron_o2(nllh, iz, p, k, z, sigmaz, mz);
    }

    void fdJrzdsigma(double *dJrzdsigma, const int iz, const realtype *p, const realtype *k, const double *rz, const double *sigmaz) override {
        dJrzdsigma_model_neuron_o2(dJrzdsigma, iz, p, k, rz, sigmaz);
    }

    void fdJrzdz(double *dJrzdz, const int iz, const realtype *p, const realtype *k, const double *rz, const double *sigmaz) override {
        dJrzdz_model_neuron_o2(dJrzdz, iz, p, k, rz, sigmaz);
    }

    void fdJydsigma(double *dJydsigma, const int iy, const realtype *p, const realtype *k, const double *y, const double *sigmay, const double *my) override {
        dJydsigma_model_neuron_o2(dJydsigma, iy, p, k, y, sigmay, my);
    }

    void fdJydy(double *dJydy, const int iy, const realtype *p, const realtype *k, const double *y, const double *sigmay, const double *my) override {
        dJydy_model_neuron_o2(dJydy, iy, p, k, y, sigmay, my);
    }

    void fdJzdsigma(double *dJzdsigma, const int iz, const realtype *p, const realtype *k, const double *z, const double *sigmaz, const double *mz) override {
        dJzdsigma_model_neuron_o2(dJzdsigma, iz, p, k, z, sigmaz, mz);
    }

    void fdJzdz(double *dJzdz, const int iz, const realtype *p, const realtype *k, const double *z, const double *sigmaz, const double *mz) override {
        dJzdz_model_neuron_o2(dJzdz, iz, p, k, z, sigmaz, mz);
    }

    void fdeltaqB(double *deltaqB, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ip, const int ie, const realtype *xdot, const realtype *xdot_old, const realtype *xB) override {
        deltaqB_model_neuron_o2(deltaqB, t, x, p, k, h, ip, ie, xdot, xdot_old, xB);
    }

    void fdeltasx(double *deltasx, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const int ip, const int ie, const realtype *xdot, const realtype *xdot_old, const realtype *sx, const realtype *stau, const realtype *tcl) override {
        deltasx_model_neuron_o2(deltasx, t, x, p, k, h, w, ip, ie, xdot, xdot_old, sx, stau, tcl);
    }

    void fdeltax(double *deltax, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ie, const realtype *xdot, const realtype *xdot_old) override {
        deltax_model_neuron_o2(deltax, t, x, p, k, h, ie, xdot, xdot_old);
    }

    void fdeltaxB(double *deltaxB, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ie, const realtype *xdot, const realtype *xdot_old, const realtype *xB) override {
        deltaxB_model_neuron_o2(deltaxB, t, x, p, k, h, ie, xdot, xdot_old, xB);
    }

    void fdrzdp(double *drzdp, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ip) override {
    }

    void fdrzdx(double *drzdx, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h) override {
        drzdx_model_neuron_o2(drzdx, ie, t, x, p, k, h);
    }

    void fdsigmaydp(double *dsigmaydp, const realtype t, const realtype *p, const realtype *k, const realtype *y, const int ip) override {
    }

    void fdsigmazdp(double *dsigmazdp, const realtype t, const realtype *p, const realtype *k, const int ip) override {
    }

    void fdwdp(realtype *dwdp, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *tcl, const realtype *stcl, const realtype *spl, const realtype *sspl) override {
    }

    void fdwdx(realtype *dwdx, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *tcl, const realtype *spl) override {
        dwdx_model_neuron_o2(dwdx, t, x, p, k, h, w, tcl, spl);
    }

    void fdxdotdp(realtype *dxdotdp, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ip, const realtype *w, const realtype *dwdp) override {
        dxdotdp_model_neuron_o2(dxdotdp, t, x, p, k, h, ip, w, dwdp);
    }

    void fdydp(double *dydp, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ip, const realtype *w, const realtype *dwdp) override {
    }

    void fdydx(double *dydx, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *dwdx) override {
        dydx_model_neuron_o2(dydx, t, x, p, k, h, w, dwdx);
    }

    void fdzdp(double *dzdp, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ip) override {
    }

    void fdzdx(double *dzdx, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h) override {
        dzdx_model_neuron_o2(dzdx, ie, t, x, p, k, h);
    }

    void froot(realtype *root, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *tcl) override {
        root_model_neuron_o2(root, t, x, p, k, h, tcl);
    }

    void frz(double *rz, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h) override {
        rz_model_neuron_o2(rz, ie, t, x, p, k, h);
    }

    void fsigmay(double *sigmay, const realtype t, const realtype *p, const realtype *k, const realtype *y) override {
        sigmay_model_neuron_o2(sigmay, t, p, k, y);
    }

    void fsigmaz(double *sigmaz, const realtype t, const realtype *p, const realtype *k) override {
        sigmaz_model_neuron_o2(sigmaz, t, p, k);
    }

    void fsrz(double *srz, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *sx, const int ip) override {
        srz_model_neuron_o2(srz, ie, t, x, p, k, h, sx, ip);
    }

    void fstau(double *stau, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *tcl, const realtype *sx, const int ip, const int ie) override {
        stau_model_neuron_o2(stau, t, x, p, k, h, tcl, sx, ip, ie);
    }

    void fsx0(realtype *sx0, const realtype t,const realtype *x0, const realtype *p, const realtype *k, const int ip) override {
        sx0_model_neuron_o2(sx0, t, x0, p, k, ip);
    }

    void fsz(double *sz, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *sx, const int ip) override {
        sz_model_neuron_o2(sz, ie, t, x, p, k, h, sx, ip);
    }

    void fw(realtype *w, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *tcl, const realtype *spl) override {
        w_model_neuron_o2(w, t, x, p, k, h, tcl, spl);
    }

    void fx0(realtype *x0, const realtype t, const realtype *p, const realtype *k) override {
        x0_model_neuron_o2(x0, t, p, k);
    }

    void fxdot(realtype *xdot, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w) override {
        xdot_model_neuron_o2(xdot, t, x, p, k, h, w);
    }

    void fy(double *y, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w) override {
        y_model_neuron_o2(y, t, x, p, k, h, w);
    }

    void fz(double *z, const int ie, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h) override {
        z_model_neuron_o2(z, ie, t, x, p, k, h);
    }

};

} // namespace model_model_neuron_o2

} // namespace amici

#endif /* _amici_model_neuron_o2_h */
