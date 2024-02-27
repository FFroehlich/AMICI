#ifndef AMICI_ABSTRACT_MODEL_H
#define AMICI_ABSTRACT_MODEL_H

#include "amici/defines.h"
#include "amici/splinefunctions.h"
#include "amici/sundials_matrix_wrapper.h"
#include "amici/vector.h"

#include <sunmatrix/sunmatrix_band.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunmatrix/sunmatrix_sparse.h>

#include <memory>

namespace amici {

class Solver;

/**
 * @brief Abstract base class of amici::Model defining functions that need to
 * be implemented in an AMICI model.
 *
 * Some functions have empty default implementations or throw.
 * This class shall not have any data members.
 */
class AbstractModel {
  public:
    virtual ~AbstractModel() = default;

    /**
     * @brief Retrieves the solver object
     * @return The Solver instance
     */
    virtual std::unique_ptr<Solver> getSolver() = 0;

    /**
     * @brief Root function
     * @param t time
     * @param x state
     * @param dx time derivative of state (DAE only)
     * @param root array to which values of the root function will be written
     */
    virtual void froot(
        realtype const t, AmiVector const& x, AmiVector const& dx,
        gsl::span<realtype> root
    ) = 0;

    /**
     * @brief Residual function
     * @param t time
     * @param x state
     * @param dx time derivative of state (DAE only)
     * @param xdot array to which values of the residual function will be
     * written
     */
    virtual void fxdot(
        realtype const t, AmiVector const& x, AmiVector const& dx,
        AmiVector& xdot
    ) = 0;

    /**
     * @brief Sensitivity Residual function
     * @param t time
     * @param x state
     * @param dx time derivative of state (DAE only)
     * @param ip parameter index
     * @param sx sensitivity state
     * @param sdx time derivative of sensitivity state (DAE only)
     * @param sxdot array to which values of the sensitivity residual function
     * will be written
     */
    virtual void fsxdot(
        realtype const t, AmiVector const& x, AmiVector const& dx, int ip,
        AmiVector const& sx, AmiVector const& sdx, AmiVector& sxdot
    ) = 0;

    /**
     * @brief Residual function backward when running in steady state mode
     * @param t time
     * @param xB adjoint state
     * @param dxB time derivative of state (DAE only)
     * @param xBdot array to which values of the residual function will be
     * written
     */
    virtual void fxBdot_ss(
        realtype const t, AmiVector const& xB, AmiVector const& dxB,
        AmiVector& xBdot
    ) = 0;

    /**
     * @brief Sparse Jacobian function backward, steady state case
     * @param JB sparse matrix to which values of the Jacobian will be written
     */
    virtual void fJSparseB_ss(SUNMatrix JB) = 0;

    /**
     * @brief Computes the sparse backward Jacobian for steadystate integration
     * and writes it to the model member
     * @param t timepoint
     * @param cj scalar in Jacobian
     * @param x Vector with the states
     * @param dx Vector with the derivative states
     * @param xB Vector with the adjoint states
     * @param dxB Vector with the adjoint derivative states
     * @param xBdot Vector with the adjoint state right hand side
     */
    virtual void writeSteadystateJB(
        realtype const t, realtype cj, AmiVector const& x, AmiVector const& dx,
        AmiVector const& xB, AmiVector const& dxB, AmiVector const& xBdot
    ) = 0;

    /**
     * @brief Dense Jacobian function
     * @param t time
     * @param cj scaling factor (inverse of timestep, DAE only)
     * @param x state
     * @param dx time derivative of state (DAE only)
     * @param xdot values of residual function (unused)
     * @param J dense matrix to which values of the jacobian will be written
     */
    virtual void
    fJ(realtype const t, realtype cj, AmiVector const& x, AmiVector const& dx,
       AmiVector const& xdot, SUNMatrix J)
        = 0;

    /**
     * @brief Dense Jacobian function
     * @param t time
     * @param cj scaling factor (inverse of timestep, DAE only)
     * @param x state
     * @param dx time derivative of state (DAE only)
     * @param xB Vector with the adjoint states
     * @param dxB Vector with the adjoint derivative states
     * @param xBdot Vector with the adjoint right hand side (unused)
     * @param JB dense matrix to which values of the jacobian will be written
     */
    virtual void
    fJB(realtype const t, realtype cj, AmiVector const& x, AmiVector const& dx,
        AmiVector const& xB, AmiVector const& dxB, AmiVector const& xBdot,
        SUNMatrix JB)
        = 0;

    /**
     * @brief Sparse Jacobian function
     * @param t time
     * @param cj scaling factor (inverse of timestep, DAE only)
     * @param x state
     * @param dx time derivative of state (DAE only)
     * @param xdot values of residual function (unused)
     * @param J sparse matrix to which values of the Jacobian will be written
     */
    virtual void fJSparse(
        realtype const t, realtype cj, AmiVector const& x, AmiVector const& dx,
        AmiVector const& xdot, SUNMatrix J
    ) = 0;

    /**
     * @brief Sparse Jacobian function
     * @param t time
     * @param cj scaling factor (inverse of timestep, DAE only)
     * @param x state
     * @param dx time derivative of state (DAE only)
     * @param xB Vector with the adjoint states
     * @param dxB Vector with the adjoint derivative states
     * @param xBdot Vector with the adjoint right hand side (unused)
     * @param JB dense matrix to which values of the jacobian will be written
     */
    virtual void fJSparseB(
        realtype const t, realtype cj, AmiVector const& x, AmiVector const& dx,
        AmiVector const& xB, AmiVector const& dxB, AmiVector const& xBdot,
        SUNMatrix JB
    ) = 0;

    /**
     * @brief Diagonal Jacobian function
     * @param t time
     * @param Jdiag array to which the diagonal of the Jacobian will be written
     * @param cj scaling factor (inverse of timestep, DAE only)
     * @param x state
     * @param dx time derivative of state (DAE only)
     */
    virtual void fJDiag(
        realtype const t, AmiVector& Jdiag, realtype cj, AmiVector const& x,
        AmiVector const& dx
    ) = 0;

    /**
     * @brief Model-specific sparse implementation of explicit parameter
     * derivative of right hand side
     * @param t time
     * @param x state
     * @param dx time derivative of state (DAE only)
     */
    virtual void
    fdxdotdp(realtype const t, AmiVector const& x, AmiVector const& dx)
        = 0;

    /**
     * @brief Jacobian multiply function
     * @param t time
     * @param x state
     * @param dx time derivative of state (DAE only)
     * @param xdot values of residual function (unused)
     * @param v multiplication vector (unused)
     * @param nJv array to which result of multiplication will be written
     * @param cj scaling factor (inverse of timestep, DAE only)
     */
    virtual void
    fJv(realtype const t, AmiVector const& x, AmiVector const& dx,
        AmiVector const& xdot, AmiVector const& v, AmiVector& nJv, realtype cj)
        = 0;

    /**
     * @brief Returns the AMICI version that was used to generate the model
     * @return AMICI version string
     */
    virtual std::string getAmiciVersion() const;

    /**
     * @brief Returns the AMICI commit that was used to generate the model
     * @return AMICI commit string
     */
    virtual std::string getAmiciCommit() const;

    /**
     * @brief Model-specific implementation of fx0
     * @param x0 initial state
     * @param t initial time
     * @param p parameter vector
     * @param k constant vector
     */
    virtual void
    fx0(realtype* x0, realtype const t, realtype const* p, realtype const* k);

    /**
     * @brief Function indicating whether reinitialization of states depending
     * on fixed parameters is permissible
     * @return flag indicating whether reinitialization of states depending on
     * fixed parameters is permissible
     */
    virtual bool isFixedParameterStateReinitializationAllowed() const;

    /**
     * @brief Model-specific implementation of fx0_fixedParameters
     * @param x0 initial state
     * @param t initial time
     * @param p parameter vector
     * @param k constant vector
     * @param reinitialization_state_idxs Indices of states to be reinitialized
     * based on provided constants / fixed parameters.
     */
    virtual void fx0_fixedParameters(
        realtype* x0, realtype const t, realtype const* p, realtype const* k,
        gsl::span<int const> reinitialization_state_idxs
    );

    /**
     * @brief Model-specific implementation of fsx0_fixedParameters
     * @param sx0 initial state sensitivities
     * @param t initial time
     * @param x0 initial state
     * @param p parameter vector
     * @param k constant vector
     * @param ip sensitivity index
     * @param reinitialization_state_idxs Indices of states to be reinitialized
     * based on provided constants / fixed parameters.
     */
    virtual void fsx0_fixedParameters(
        realtype* sx0, realtype const t, realtype const* x0, realtype const* p,
        realtype const* k, int ip,
        gsl::span<int const> reinitialization_state_idxs
    );

    /**
     * @brief Model-specific implementation of fsx0
     * @param sx0 initial state sensitivities
     * @param t initial time
     * @param x0 initial state
     * @param p parameter vector
     * @param k constant vector
     * @param ip sensitivity index
     */
    virtual void fsx0(
        realtype* sx0, realtype const t, realtype const* x0, realtype const* p,
        realtype const* k, int ip
    );

    /**
     * @brief Initial value for time derivative of states (only necessary for
     * DAEs)
     * @param x0 Vector with the initial states
     * @param dx0 Vector to which the initial derivative states will be written
     * (only DAE)
     */
    virtual void fdx0(AmiVector& x0, AmiVector& dx0);

    /**
     * @brief Model-specific implementation of fstau
     * @param stau total derivative of event timepoint
     * @param t current time
     * @param x current state
     * @param p parameter vector
     * @param k constant vector
     * @param h Heaviside vector
     * @param tcl total abundances for conservation laws
     * @param sx current state sensitivity
     * @param ip sensitivity index
     * @param ie event index
     */
    virtual void fstau(
        realtype* stau, realtype const t, realtype const* x, realtype const* p,
        realtype const* k, realtype const* h, realtype const* tcl,
        realtype const* sx, int ip, int ie
    );

    /**
     * @brief Model-specific implementation of fy
     * @param y model output at current timepoint
     * @param t current time
     * @param x current state
     * @param p parameter vector
     * @param k constant vector
     * @param h Heaviside vector
     * @param w repeating elements vector
     */
    virtual void
    fy(realtype* y, realtype const t, realtype const* x, realtype const* p,
       realtype const* k, realtype const* h, realtype const* w);

    /**
     * @brief Model-specific implementation of fdydp (MATLAB-only)
     * @param dydp partial derivative of observables y w.r.t. model parameters p
     * @param t current time
     * @param x current state
     * @param p parameter vector
     * @param k constant vector
     * @param h Heaviside vector
     * @param ip parameter index w.r.t. which the derivative is requested
     * @param w repeating elements vector
     * @param dwdp Recurring terms in xdot, parameter derivative
     */
    virtual void fdydp(
        realtype* dydp, realtype const t, realtype const* x, realtype const* p,
        realtype const* k, realtype const* h, int ip, realtype const* w,
        realtype const* dwdp
    );

    /**
     * @brief Model-specific implementation of fdydp (Python)
     * @param dydp partial derivative of observables y w.r.t. model parameters p
     * @param t current time
     * @param x current state
     * @param p parameter vector
     * @param k constant vector
     * @param h Heaviside vector
     * @param ip parameter index w.r.t. which the derivative is requested
     * @param w repeating elements vector
     * @param tcl total abundances for conservation laws
     * @param dtcldp Sensitivities of total abundances for conservation laws
     * @param spl spline value vector
     * @param sspl sensitivities of spline values vector w.r.t. parameters \f$ p
     * \f$
     */
    virtual void fdydp(
        realtype* dydp, realtype const t, realtype const* x, realtype const* p,
        realtype const* k, realtype const* h, int ip, realtype const* w,
        realtype const* tcl, realtype const* dtcldp, realtype const* spl,
        realtype const* sspl
    );

    /**
     * @brief Model-specific implementation of fdydx
     * @param dydx partial derivative of observables y w.r.t. model states x
     * @param t current time
     * @param x current state
     * @param p parameter vector
     * @param k constant vector
     * @param h Heaviside vector
     * @param w repeating elements vector
     * @param dwdx Recurring terms in xdot, state derivative
     */
    virtual void fdydx(
        realtype* dydx, realtype const t, realtype const* x, realtype const* p,
        realtype const* k, realtype const* h, realtype const* w,
        realtype const* dwdx
    );

    /**
     * @brief Model-specific implementation of fz
     * @param z value of event output
     * @param ie event index
     * @param t current time
     * @param x current state
     * @param p parameter vector
     * @param k constant vector
     * @param h Heaviside vector
     */
    virtual void
    fz(realtype* z, int ie, realtype const t, realtype const* x,
       realtype const* p, realtype const* k, realtype const* h);

    /**
     * @brief Model-specific implementation of fsz
     * @param sz Sensitivity of rz, total derivative
     * @param ie event index
     * @param t current time
     * @param x current state
     * @param p parameter vector
     * @param k constant vector
     * @param h Heaviside vector
     * @param sx current state sensitivity
     * @param ip sensitivity index
     */
    virtual void
    fsz(realtype* sz, int ie, realtype const t, realtype const* x,
        realtype const* p, realtype const* k, realtype const* h,
        realtype const* sx, int ip);

    /**
     * @brief Model-specific implementation of frz
     * @param rz value of root function at current timepoint (non-output events
     * not included)
     * @param ie event index
     * @param t current time
     * @param x current state
     * @param p parameter vector
     * @param k constant vector
     * @param h Heaviside vector
     */
    virtual void
    frz(realtype* rz, int ie, realtype const t, realtype const* x,
        realtype const* p, realtype const* k, realtype const* h);

    /**
     * @brief Model-specific implementation of fsrz
     * @param srz Sensitivity of rz, total derivative
     * @param ie event index
     * @param t current time
     * @param x current state
     * @param p parameter vector
     * @param k constant vector
     * @param sx current state sensitivity
     * @param h Heaviside vector
     * @param ip sensitivity index
     */
    virtual void fsrz(
        realtype* srz, int ie, realtype const t, realtype const* x,
        realtype const* p, realtype const* k, realtype const* h,
        realtype const* sx, int ip
    );

    /**
     * @brief Model-specific implementation of fdzdp
     * @param dzdp partial derivative of event-resolved output z w.r.t. model
     * parameters p
     * @param ie event index
     * @param t current time
     * @param x current state
     * @param p parameter vector
     * @param k constant vector
     * @param h Heaviside vector
     * @param ip parameter index w.r.t. which the derivative is requested
     */
    virtual void fdzdp(
        realtype* dzdp, int ie, realtype const t, realtype const* x,
        realtype const* p, realtype const* k, realtype const* h, int ip
    );

    /**
     * @brief Model-specific implementation of fdzdx
     * @param dzdx partial derivative of event-resolved output z w.r.t. model
     * states x
     * @param ie event index
     * @param t current time
     * @param x current state
     * @param p parameter vector
     * @param k constant vector
     * @param h Heaviside vector
     */
    virtual void fdzdx(
        realtype* dzdx, int ie, realtype const t, realtype const* x,
        realtype const* p, realtype const* k, realtype const* h
    );

    /**
     * @brief Model-specific implementation of fdrzdp
     * @param drzdp partial derivative of root output rz w.r.t. model parameters
     * p
     * @param ie event index
     * @param t current time
     * @param x current state
     * @param p parameter vector
     * @param k constant vector
     * @param h Heaviside vector
     * @param ip parameter index w.r.t. which the derivative is requested
     */
    virtual void fdrzdp(
        realtype* drzdp, int ie, realtype const t, realtype const* x,
        realtype const* p, realtype const* k, realtype const* h, int ip
    );

    /**
     * @brief Model-specific implementation of fdrzdx
     * @param drzdx partial derivative of root output rz w.r.t. model states x
     * @param ie event index
     * @param t current time
     * @param x current state
     * @param p parameter vector
     * @param k constant vector
     * @param h Heaviside vector
     */
    virtual void fdrzdx(
        realtype* drzdx, int ie, realtype const t, realtype const* x,
        realtype const* p, realtype const* k, realtype const* h
    );

    /**
     * @brief Model-specific implementation of fdeltax
     * @param deltax state update
     * @param t current time
     * @param x current state
     * @param p parameter vector
     * @param k constant vector
     * @param h Heaviside vector
     * @param ie event index
     * @param xdot new model right hand side
     * @param xdot_old previous model right hand side
     */
    virtual void fdeltax(
        realtype* deltax, realtype const t, realtype const* x,
        realtype const* p, realtype const* k, realtype const* h, int ie,
        realtype const* xdot, realtype const* xdot_old
    );

    /**
     * @brief Model-specific implementation of fdeltasx
     * @param deltasx sensitivity update
     * @param t current time
     * @param x current state
     * @param p parameter vector
     * @param k constant vector
     * @param h Heaviside vector
     * @param w repeating elements vector
     * @param ip sensitivity index
     * @param ie event index
     * @param xdot new model right hand side
     * @param xdot_old previous model right hand side
     * @param sx state sensitivity
     * @param stau event-time sensitivity
     * @param tcl total abundances for conservation laws
     */
    virtual void fdeltasx(
        realtype* deltasx, realtype const t, realtype const* x,
        realtype const* p, realtype const* k, realtype const* h,
        realtype const* w, int ip, int ie, realtype const* xdot,
        realtype const* xdot_old, realtype const* sx, realtype const* stau,
        realtype const* tcl
    );

    /**
     * @brief Model-specific implementation of fdeltaxB
     * @param deltaxB adjoint state update
     * @param t current time
     * @param x current state
     * @param p parameter vector
     * @param k constant vector
     * @param h Heaviside vector
     * @param ie event index
     * @param xdot new model right hand side
     * @param xdot_old previous model right hand side
     * @param xB current adjoint state
     */
    virtual void fdeltaxB(
        realtype* deltaxB, realtype const t, realtype const* x,
        realtype const* p, realtype const* k, realtype const* h, int ie,
        realtype const* xdot, realtype const* xdot_old, realtype const* xB
    );

    /**
     * @brief Model-specific implementation of fdeltaqB
     * @param deltaqB sensitivity update
     * @param t current time
     * @param x current state
     * @param p parameter vector
     * @param k constant vector
     * @param h Heaviside vector
     * @param ip sensitivity index
     * @param ie event index
     * @param xdot new model right hand side
     * @param xdot_old previous model right hand side
     * @param xB adjoint state
     */
    virtual void fdeltaqB(
        realtype* deltaqB, realtype const t, realtype const* x,
        realtype const* p, realtype const* k, realtype const* h, int ip, int ie,
        realtype const* xdot, realtype const* xdot_old, realtype const* xB
    );

    /**
     * @brief Model-specific implementation of fsigmay
     * @param sigmay standard deviation of measurements
     * @param t current time
     * @param p parameter vector
     * @param k constant vector
     * @param y model output at timepoint t
     */
    virtual void fsigmay(
        realtype* sigmay, realtype const t, realtype const* p,
        realtype const* k, realtype const* y
    );

    /**
     * @brief Model-specific implementation of fdsigmaydp
     * @param dsigmaydp partial derivative of standard deviation of measurements
     * @param t current time
     * @param p parameter vector
     * @param k constant vector
     * @param y model output at timepoint t
     * @param ip sensitivity index
     */
    virtual void fdsigmaydp(
        realtype* dsigmaydp, realtype const t, realtype const* p,
        realtype const* k, realtype const* y, int ip
    );
    /**
     * @brief Model-specific implementation of fsigmay
     * @param dsigmaydy partial derivative of standard deviation of measurements
     * w.r.t. model outputs
     * @param t current time
     * @param p parameter vector
     * @param k constant vector
     * @param y model output at timepoint t
     */
    virtual void fdsigmaydy(
        realtype* dsigmaydy, realtype const t, realtype const* p,
        realtype const* k, realtype const* y
    );

    /**
     * @brief Model-specific implementation of fsigmaz
     * @param sigmaz standard deviation of event measurements
     * @param t current time
     * @param p parameter vector
     * @param k constant vector
     */
    virtual void fsigmaz(
        realtype* sigmaz, realtype const t, realtype const* p, realtype const* k
    );

    /**
     * @brief Model-specific implementation of fsigmaz
     * @param dsigmazdp partial derivative of standard deviation of event
     * measurements
     * @param t current time
     * @param p parameter vector
     * @param k constant vector
     * @param ip sensitivity index
     */
    virtual void fdsigmazdp(
        realtype* dsigmazdp, realtype const t, realtype const* p,
        realtype const* k, int ip
    );

    /**
     * @brief Model-specific implementation of fJy
     * @param nllh negative log-likelihood for measurements y
     * @param iy output index
     * @param p parameter vector
     * @param k constant vector
     * @param y model output at timepoint
     * @param sigmay measurement standard deviation at timepoint
     * @param my measurements at timepoint
     */
    virtual void
    fJy(realtype* nllh, int iy, realtype const* p, realtype const* k,
        realtype const* y, realtype const* sigmay, realtype const* my);

    /**
     * @brief Model-specific implementation of fJz
     * @param nllh negative log-likelihood for event measurements z
     * @param iz event output index
     * @param p parameter vector
     * @param k constant vector
     * @param z model event output at timepoint
     * @param sigmaz event measurement standard deviation at timepoint
     * @param mz event measurements at timepoint
     */
    virtual void
    fJz(realtype* nllh, int iz, realtype const* p, realtype const* k,
        realtype const* z, realtype const* sigmaz, realtype const* mz);

    /**
     * @brief Model-specific implementation of fJrz
     * @param nllh regularization for event measurements z
     * @param iz event output index
     * @param p parameter vector
     * @param k constant vector
     * @param z model event output at timepoint
     * @param sigmaz event measurement standard deviation at timepoint
     */
    virtual void fJrz(
        realtype* nllh, int iz, realtype const* p, realtype const* k,
        realtype const* z, realtype const* sigmaz
    );

    /**
     * @brief Model-specific implementation of fdJydy
     * @param dJydy partial derivative of time-resolved measurement negative
     * log-likelihood Jy
     * @param iy output index
     * @param p parameter vector
     * @param k constant vector
     * @param y model output at timepoint
     * @param sigmay measurement standard deviation at timepoint
     * @param my measurement at timepoint
     */
    virtual void fdJydy(
        realtype* dJydy, int iy, realtype const* p, realtype const* k,
        realtype const* y, realtype const* sigmay, realtype const* my
    );

    /**
     * @brief Model-specific implementation of fdJydy colptrs
     * @param dJydy sparse matrix to which colptrs will be written
     * @param index ytrue index
     */
    virtual void fdJydy_colptrs(SUNMatrixWrapper& dJydy, int index);

    /**
     * @brief Model-specific implementation of fdJydy rowvals
     * @param dJydy sparse matrix to which rowvals will be written
     * @param index `ytrue` index
     */
    virtual void fdJydy_rowvals(SUNMatrixWrapper& dJydy, int index);

    /**
     * @brief Model-specific implementation of fdJydsigma
     * @param dJydsigma Sensitivity of time-resolved measurement negative
     * log-likelihood Jy w.r.t. standard deviation sigmay
     * @param iy output index
     * @param p parameter vector
     * @param k constant vector
     * @param y model output at timepoint
     * @param sigmay measurement standard deviation at timepoint
     * @param my measurement at timepoint
     */
    virtual void fdJydsigma(
        realtype* dJydsigma, int iy, realtype const* p, realtype const* k,
        realtype const* y, realtype const* sigmay, realtype const* my
    );

    /**
     * @brief Model-specific implementation of fdJzdz
     * @param dJzdz partial derivative of event measurement negative
     * log-likelihood Jz
     * @param iz event output index
     * @param p parameter vector
     * @param k constant vector
     * @param z model event output at timepoint
     * @param sigmaz event measurement standard deviation at timepoint
     * @param mz event measurement at timepoint
     */
    virtual void fdJzdz(
        realtype* dJzdz, int iz, realtype const* p, realtype const* k,
        realtype const* z, realtype const* sigmaz, realtype const* mz
    );

    /**
     * @brief Model-specific implementation of fdJzdsigma
     * @param dJzdsigma Sensitivity of event measurement negative log-likelihood
     * Jz w.r.t. standard deviation sigmaz
     * @param iz event output index
     * @param p parameter vector
     * @param k constant vector
     * @param z model event output at timepoint
     * @param sigmaz event measurement standard deviation at timepoint
     * @param mz event measurement at timepoint
     */
    virtual void fdJzdsigma(
        realtype* dJzdsigma, int iz, realtype const* p, realtype const* k,
        realtype const* z, realtype const* sigmaz, realtype const* mz
    );

    /**
     * @brief Model-specific implementation of fdJrzdz
     * @param dJrzdz partial derivative of event penalization Jrz
     * @param iz event output index
     * @param p parameter vector
     * @param k constant vector
     * @param rz model root output at timepoint
     * @param sigmaz event measurement standard deviation at timepoint
     */
    virtual void fdJrzdz(
        realtype* dJrzdz, int iz, realtype const* p, realtype const* k,
        realtype const* rz, realtype const* sigmaz
    );

    /**
     * @brief Model-specific implementation of fdJrzdsigma
     * @param dJrzdsigma Sensitivity of event penalization Jrz w.r.t. standard
     * deviation sigmaz
     * @param iz event output index
     * @param p parameter vector
     * @param k constant vector
     * @param rz model root output at timepoint
     * @param sigmaz event measurement standard deviation at timepoint
     */
    virtual void fdJrzdsigma(
        realtype* dJrzdsigma, int iz, realtype const* p, realtype const* k,
        realtype const* rz, realtype const* sigmaz
    );

    /**
     * @brief Model-specific implementation of fw
     * @param w Recurring terms in xdot
     * @param t timepoint
     * @param x vector with the states
     * @param p parameter vector
     * @param k constants vector
     * @param h Heaviside vector
     * @param tcl total abundances for conservation laws
     * @param spl spline value vector
     * @param include_static Whether to (re-)evaluate only dynamic expressions
     * (false) or also static expressions (true).
     * Dynamic expressions are those that depend directly or indirectly on time,
     * static expressions are those that don't.
     */
    virtual void
    fw(realtype* w, realtype const t, realtype const* x, realtype const* p,
       realtype const* k, realtype const* h, realtype const* tcl,
       realtype const* spl, bool include_static = true);

    /**
     * @brief Model-specific sparse implementation of dwdp
     * @param dwdp Recurring terms in xdot, parameter derivative
     * @param t timepoint
     * @param x vector with the states
     * @param p parameter vector
     * @param k constants vector
     * @param h Heaviside vector
     * @param w vector with helper variables
     * @param tcl total abundances for conservation laws
     * @param stcl sensitivities of total abundances for conservation laws
     * @param spl spline value vector
     * @param sspl sensitivities of spline values vector w.r.t. parameters \f$ p
     * \f$
     * @param include_static Whether to (re-)evaluate only dynamic expressions
     * (false) or also static expressions (true).
     * Dynamic expressions are those that depend directly or indirectly on time,
     * static expressions are those that don't.
     */
    virtual void fdwdp(
        realtype* dwdp, realtype const t, realtype const* x, realtype const* p,
        realtype const* k, realtype const* h, realtype const* w,
        realtype const* tcl, realtype const* stcl, realtype const* spl,
        realtype const* sspl, bool include_static = true
    );

    /**
     * @brief Model-specific implementation for dwdp, column pointers
     * @param dwdp sparse matrix to which colptrs will be written
     */
    virtual void fdwdp_colptrs(SUNMatrixWrapper& dwdp);

    /**
     * @brief Model-specific implementation for dwdp, row values
     * @param dwdp sparse matrix to which rowvals will be written
     */
    virtual void fdwdp_rowvals(SUNMatrixWrapper& dwdp);

    /**
     * @brief Model-specific implementation of dwdx, data part
     * @param dwdx Recurring terms in xdot, state derivative
     * @param t timepoint
     * @param x vector with the states
     * @param p parameter vector
     * @param k constants vector
     * @param h Heaviside vector
     * @param w vector with helper variables
     * @param tcl total abundances for conservation laws
     * @param spl spline value vector
     * @param include_static Whether to (re-)evaluate only dynamic expressions
     * (false) or also static expressions (true).
     * Dynamic expressions are those that depend directly or indirectly on time,
     * static expressions are those that don't.
     */
    virtual void fdwdx(
        realtype* dwdx, realtype const t, realtype const* x, realtype const* p,
        realtype const* k, realtype const* h, realtype const* w,
        realtype const* tcl, realtype const* spl, bool include_static = true
    );

    /**
     * @brief Model-specific implementation for dwdx, column pointers
     * @param dwdx sparse matrix to which colptrs will be written
     */
    virtual void fdwdx_colptrs(SUNMatrixWrapper& dwdx);

    /**
     * @brief Model-specific implementation for dwdx, row values
     * @param dwdx sparse matrix to which rowvals will be written
     */
    virtual void fdwdx_rowvals(SUNMatrixWrapper& dwdx);

    /**
     * @brief Model-specific implementation of fdwdw, no w chainrule (Py)
     * @param dwdw partial derivative w wrt w
     * @param t timepoint
     * @param x Vector with the states
     * @param p parameter vector
     * @param k constants vector
     * @param h Heaviside vector
     * @param w vector with helper variables
     * @param tcl Total abundances for conservation laws
     * @param include_static Whether to (re-)evaluate only dynamic expressions
     * (false) or also static expressions (true).
     * Dynamic expressions are those that depend directly or indirectly on time,
     * static expressions are those that don't.
     */
    virtual void fdwdw(
        realtype* dwdw, realtype t, realtype const* x, realtype const* p,
        realtype const* k, realtype const* h, realtype const* w,
        realtype const* tcl, bool include_static = true
    );

    /**
     * @brief Model-specific implementation of fdwdw, colptrs part
     * @param dwdw sparse matrix to which colptrs will be written
     */
    virtual void fdwdw_colptrs(SUNMatrixWrapper& dwdw);

    /**
     * @brief Model-specific implementation of fdwdw, rowvals part
     * @param dwdw sparse matrix to which rowvals will be written
     */
    virtual void fdwdw_rowvals(SUNMatrixWrapper& dwdw);

    /**
     * @brief Compute dx_rdata / dx_solver
     * @param dx_rdatadx_solver dx_rdata / dx_solver
     * @param p parameter vector
     * @param k constant vector
     * @param x State variables with conservation laws applied
     * @param tcl Total abundances for conservation laws
     */
    virtual void fdx_rdatadx_solver(
        realtype* dx_rdatadx_solver, realtype const* x, realtype const* tcl,
        realtype const* p, realtype const* k
    );

    /**
     * @brief Model-specific implementation of fdx_rdatadx_solver, colptrs part
     * @param dxrdatadxsolver sparse matrix to which colptrs will be written
     */
    virtual void fdx_rdatadx_solver_colptrs(SUNMatrixWrapper& dxrdatadxsolver);

    /**
     * @brief Model-specific implementation of fdx_rdatadx_solver, rowvals part
     * @param dxrdatadxsolver sparse matrix to which rowvals will be written
     */
    virtual void fdx_rdatadx_solver_rowvals(SUNMatrixWrapper& dxrdatadxsolver);

    /**
     * @brief Compute dx_rdata / dp
     * @param dx_rdatadp dx_rdata / dp
     * @param p parameter vector
     * @param k constant vector
     * @param x State variables with conservation laws applied
     * @param tcl Total abundances for conservation laws
     * @param ip Sensitivity index
     */
    virtual void fdx_rdatadp(
        realtype* dx_rdatadp, realtype const* x, realtype const* tcl,
        realtype const* p, realtype const* k, int const ip
    );

    /**
     * @brief Compute dx_rdata / dtcl
     * @param dx_rdatadtcl dx_rdata / dtcl
     * @param p parameter vector
     * @param k constant vector
     * @param x State variables with conservation laws applied
     * @param tcl Total abundances for conservation laws
     */
    virtual void fdx_rdatadtcl(
        realtype* dx_rdatadtcl, realtype const* x, realtype const* tcl,
        realtype const* p, realtype const* k
    );

    /**
     * @brief Model-specific implementation of fdx_rdatadtcl, colptrs part
     * @param dx_rdatadtcl sparse matrix to which colptrs will be written
     */
    virtual void fdx_rdatadtcl_colptrs(SUNMatrixWrapper& dx_rdatadtcl);

    /**
     * @brief Model-specific implementation of fdx_rdatadtcl, rowvals part
     * @param dx_rdatadtcl sparse matrix to which rowvals will be written
     */
    virtual void fdx_rdatadtcl_rowvals(SUNMatrixWrapper& dx_rdatadtcl);

    /**
     * @brief Compute dtotal_cl / dp
     * @param dtotal_cldp dtotal_cl / dp
     * @param x_rdata State variables with conservation laws applied
     * @param p parameter vector
     * @param k constant vector
     * @param ip Sensitivity index
     */
    virtual void fdtotal_cldp(
        realtype* dtotal_cldp, realtype const* x_rdata, realtype const* p,
        realtype const* k, int const ip
    );

    /**
     * @brief Compute dtotal_cl / dx_rdata
     * @param dtotal_cldx_rdata dtotal_cl / dx_rdata
     * @param x_rdata State variables with conservation laws applied
     * @param p parameter vector
     * @param k constant vector
     * @param tcl Total abundances for conservation laws
     */
    virtual void fdtotal_cldx_rdata(
        realtype* dtotal_cldx_rdata, realtype const* x_rdata, realtype const* p,
        realtype const* k, realtype const* tcl
    );

    /**
     * @brief Model-specific implementation of fdtotal_cldx_rdata, colptrs part
     * @param dtotal_cldx_rdata sparse matrix to which colptrs will be written
     */
    virtual void fdtotal_cldx_rdata_colptrs(SUNMatrixWrapper& dtotal_cldx_rdata
    );

    /**
     * @brief Model-specific implementation of fdtotal_cldx_rdata, rowvals part
     * @param dtotal_cldx_rdata sparse matrix to which rowvals will be written
     */
    virtual void fdtotal_cldx_rdata_rowvals(SUNMatrixWrapper& dtotal_cldx_rdata
    );

    /**
     * @brief Model-specific implementation of spline creation
     * @param p parameter vector
     * @param k constants vector
     * @return Vector of splines used in the model
     */
    virtual std::vector<HermiteSpline>
    fcreate_splines(realtype const* p, realtype const* k);

    /**
     * @brief Model-specific implementation the parametric derivatives
     * of spline node values
     * @param dspline_valuesdp vector to which derivatives will be written
     * @param p parameter vector
     * @param k constants vector
     * @param ip Sensitivity index
     */
    virtual void fdspline_valuesdp(
        realtype* dspline_valuesdp, realtype const* p, realtype const* k,
        int const ip
    );

    /**
     * @brief Model-specific implementation the parametric derivatives
     * of slopevalues at spline nodes
     * @param dspline_slopesdp vector to which derivatives will be written
     * @param p parameter vector
     * @param k constants vector
     * @param ip Sensitivity index
     */
    virtual void fdspline_slopesdp(
        realtype* dspline_slopesdp, realtype const* p, realtype const* k,
        int const ip
    );
};

} // namespace amici

#endif // AMICI_ABSTRACT_MODEL_H
