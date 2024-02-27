#ifndef AMICI_MODEL_ODE_H
#define AMICI_MODEL_ODE_H

#include "amici/model.h"

#include <nvector/nvector_serial.h>

#include <sundials/sundials_matrix.h>
#include <sunmatrix/sunmatrix_band.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunmatrix/sunmatrix_sparse.h>

#include <vector>

namespace amici {

class CVodeSolver;

/**
 * @brief The Model class represents an AMICI ODE model.
 *
 * The model does not contain any data, but represents the state
 * of the model at a specific time t. The states must not always be
 * in sync, but may be updated asynchronously.
 */
class Model_ODE : public Model {
  public:
    /** default constructor */
    Model_ODE() = default;

    /**
     * @brief Constructor with model dimensions
     * @param model_dimensions Model dimensions
     * @param simulation_parameters Simulation parameters
     * @param o2mode second order sensitivity mode
     * @param idlist indexes indicating algebraic components (DAE only)
     * @param z2event mapping of event outputs to events
     * @param pythonGenerated flag indicating matlab or python wrapping
     * @param ndxdotdp_explicit number of nonzero elements dxdotdp_explicit
     * @param ndxdotdx_explicit number of nonzero elements dxdotdx_explicit
     * @param w_recursion_depth Recursion depth of fw
     * @param state_independent_events Map of events with state-independent
     * triggers functions, mapping trigger timepoints to event indices.
     */
    Model_ODE(
        ModelDimensions const& model_dimensions,
        SimulationParameters simulation_parameters,
        SecondOrderMode const o2mode, std::vector<realtype> const& idlist,
        std::vector<int> const& z2event, bool const pythonGenerated = false,
        int const ndxdotdp_explicit = 0, int const ndxdotdx_explicit = 0,
        int const w_recursion_depth = 0,
        std::map<realtype, std::vector<int>> state_independent_events = {}
    )
        : Model(
            model_dimensions, simulation_parameters, o2mode, idlist, z2event,
            pythonGenerated, ndxdotdp_explicit, ndxdotdx_explicit,
            w_recursion_depth, state_independent_events
        ) {}

    void
    fJ(realtype t, realtype cj, AmiVector const& x, AmiVector const& dx,
       AmiVector const& xdot, SUNMatrix J) override;

    /**
     * @brief Implementation of fJ at the N_Vector level
     *
     * This function provides an
     * interface to the model specific routines for the solver
     * implementation as well as the AmiVector level implementation
     * @param t timepoint
     * @param x Vector with the states
     * @param xdot Vector with the right hand side
     * @param J Matrix to which the Jacobian will be written
     **/
    void fJ(realtype t, const_N_Vector x, const_N_Vector xdot, SUNMatrix J);

    void
    fJB(realtype const t, realtype cj, AmiVector const& x, AmiVector const& dx,
        AmiVector const& xB, AmiVector const& dxB, AmiVector const& xBdot,
        SUNMatrix JB) override;

    /**
     * @brief Implementation of fJB at the N_Vector level, this function
     *provides an interface to the model specific routines for the solver
     *implementation
     * @param t timepoint
     * @param x Vector with the states
     * @param xB Vector with the adjoint states
     * @param xBdot Vector with the adjoint right hand side
     * @param JB Matrix to which the Jacobian will be written
     **/
    void
    fJB(realtype t, const_N_Vector x, const_N_Vector xB, const_N_Vector xBdot,
        SUNMatrix JB);

    void fJSparse(
        realtype t, realtype cj, AmiVector const& x, AmiVector const& dx,
        AmiVector const& xdot, SUNMatrix J
    ) override;

    /**
     * @brief Implementation of fJSparse at the N_Vector level, this function
     * provides an interface to the model specific routines for the solver
     * implementation as well as the AmiVector level implementation
     * @param t timepoint
     * @param x Vector with the states
     * @param J Matrix to which the Jacobian will be written
     */
    void fJSparse(realtype t, const_N_Vector x, SUNMatrix J);

    void fJSparseB(
        realtype const t, realtype cj, AmiVector const& x, AmiVector const& dx,
        AmiVector const& xB, AmiVector const& dxB, AmiVector const& xBdot,
        SUNMatrix JB
    ) override;

    /**
     * @brief Implementation of fJSparseB at the N_Vector level, this function
     * provides an interface to the model specific routines for the solver
     * implementation
     * @param t timepoint
     * @param x Vector with the states
     * @param xB Vector with the adjoint states
     * @param xBdot Vector with the adjoint right hand side
     * @param JB Matrix to which the Jacobian will be written
     */
    void fJSparseB(
        realtype t, const_N_Vector x, const_N_Vector xB, const_N_Vector xBdot,
        SUNMatrix JB
    );

    /**
     * @brief Implementation of fJDiag at the N_Vector level, this function
     *provides an interface to the model specific routines for the solver
     *implementation
     * @param t timepoint
     * @param JDiag Vector to which the Jacobian diagonal will be written
     * @param x Vector with the states
     **/
    void fJDiag(realtype t, N_Vector JDiag, const_N_Vector x);

    /**
     * @brief Diagonal of the Jacobian (for preconditioning)
     * @param t timepoint
     * @param JDiag Vector to which the Jacobian diagonal will be written
     * @param cj scaling factor, inverse of the step size
     * @param x Vector with the states
     * @param dx Vector with the derivative states
     **/
    void fJDiag(
        realtype t, AmiVector& JDiag, realtype cj, AmiVector const& x,
        AmiVector const& dx
    ) override;

    void
    fJv(realtype t, AmiVector const& x, AmiVector const& dx,
        AmiVector const& xdot, AmiVector const& v, AmiVector& nJv,
        realtype cj) override;

    /**
     * @brief Implementation of fJv at the N_Vector level.
     * @param t timepoint
     * @param x Vector with the states
     * @param v Vector with which the Jacobian is multiplied
     * @param Jv Vector to which the Jacobian vector product will be
     * written
     **/
    void fJv(const_N_Vector v, N_Vector Jv, realtype t, const_N_Vector x);

    /**
     * @brief Implementation of fJvB at the N_Vector level
     * @param t timepoint
     * @param x Vector with the states
     * @param xB Vector with the adjoint states
     * @param vB Vector with which the Jacobian is multiplied
     * @param JvB Vector to which the Jacobian vector product will be written
     **/
    void fJvB(
        const_N_Vector vB, N_Vector JvB, realtype t, const_N_Vector x,
        const_N_Vector xB
    );

    void froot(
        realtype t, AmiVector const& x, AmiVector const& dx,
        gsl::span<realtype> root
    ) override;

    /**
     * @brief Implementation of froot at the N_Vector level
     * This function provides an interface to the model specific routines for
     * the solver implementation as well as the AmiVector level implementation
     * @param t timepoint
     * @param x Vector with the states
     * @param root array with root function values
     */
    void froot(realtype t, const_N_Vector x, gsl::span<realtype> root);

    void fxdot(
        realtype t, AmiVector const& x, AmiVector const& dx, AmiVector& xdot
    ) override;

    /**
     * @brief Implementation of fxdot at the N_Vector level, this function
     * provides an interface to the model specific routines for the solver
     * implementation as well as the AmiVector level implementation
     * @param t timepoint
     * @param x Vector with the states
     * @param xdot Vector with the right hand side
     */
    void fxdot(realtype t, const_N_Vector x, N_Vector xdot);

    /**
     * @brief Implementation of fxBdot at the N_Vector level
     * @param t timepoint
     * @param x Vector with the states
     * @param xB Vector with the adjoint states
     * @param xBdot Vector with the adjoint right hand side
     */
    void fxBdot(realtype t, N_Vector x, N_Vector xB, N_Vector xBdot);

    /**
     * @brief Implementation of fqBdot at the N_Vector level
     * @param t timepoint
     * @param x Vector with the states
     * @param xB Vector with the adjoint states
     * @param qBdot Vector with the adjoint quadrature right hand side
     */
    void
    fqBdot(realtype t, const_N_Vector x, const_N_Vector xB, N_Vector qBdot);

    void fxBdot_ss(
        realtype const t, AmiVector const& xB, AmiVector const& /*dxB*/,
        AmiVector& xBdot
    ) override;

    /**
     * @brief Implementation of fxBdot for steady state at the N_Vector level
     * @param t timepoint
     * @param xB Vector with the states
     * @param xBdot Vector with the adjoint right hand side
     */
    void fxBdot_ss(realtype t, const_N_Vector xB, N_Vector xBdot) const;

    /**
     * @brief Implementation of fqBdot for steady state case at the N_Vector
     * level
     * @param t timepoint
     * @param xB Vector with the adjoint states
     * @param qBdot Vector with the adjoint quadrature right hand side
     */
    void fqBdot_ss(realtype t, N_Vector xB, N_Vector qBdot) const;

    /**
     * @brief Sparse Jacobian function backward, steady state case
     * @param JB sparse matrix to which values of the Jacobian will be written
     */
    void fJSparseB_ss(SUNMatrix JB) override;

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
    void writeSteadystateJB(
        realtype const t, realtype cj, AmiVector const& x, AmiVector const& dx,
        AmiVector const& xB, AmiVector const& dxB, AmiVector const& xBdot
    ) override;

    void fsxdot(
        realtype t, AmiVector const& x, AmiVector const& dx, int ip,
        AmiVector const& sx, AmiVector const& sdx, AmiVector& sxdot
    ) override;

    /**
     * @brief Implementation of fsxdot at the N_Vector level
     * @param t timepoint
     * @param x Vector with the states
     * @param ip parameter index
     * @param sx Vector with the state sensitivities
     * @param sxdot Vector with the sensitivity right hand side
     */
    void fsxdot(
        realtype t, const_N_Vector x, int ip, const_N_Vector sx, N_Vector sxdot
    );

    std::unique_ptr<Solver> getSolver() override;

  protected:
    /**
     * @brief Model specific implementation for fJSparse (Matlab)
     * @param JSparse Matrix to which the Jacobian will be written
     * @param t timepoint
     * @param x Vector with the states
     * @param p parameter vector
     * @param k constants vector
     * @param h Heaviside vector
     * @param w vector with helper variables
     * @param dwdx derivative of w wrt x
     **/
    virtual void fJSparse(
        SUNMatrixContent_Sparse JSparse, realtype t, realtype const* x,
        realtype const* p, realtype const* k, realtype const* h,
        realtype const* w, realtype const* dwdx
    );

    /**
     * @brief Model specific implementation for fJSparse, data only (Py)
     * @param JSparse Matrix to which the Jacobian will be written
     * @param t timepoint
     * @param x Vector with the states
     * @param p parameter vector
     * @param k constants vector
     * @param h Heaviside vector
     * @param w vector with helper variables
     * @param dwdx derivative of w wrt x
     **/
    virtual void fJSparse(
        realtype* JSparse, realtype t, realtype const* x, realtype const* p,
        realtype const* k, realtype const* h, realtype const* w,
        realtype const* dwdx
    );

    /**
     * @brief Model specific implementation for fJSparse, column pointers
     * @param JSparse sparse matrix to which colptrs will be written
     **/
    virtual void fJSparse_colptrs(SUNMatrixWrapper& JSparse);

    /**
     * @brief Model specific implementation for fJSparse, row values
     * @param JSparse sparse matrix to which rowvals will be written
     **/
    virtual void fJSparse_rowvals(SUNMatrixWrapper& JSparse);

    /**
     * @brief Model specific implementation for froot
     * @param root values of the trigger function
     * @param t timepoint
     * @param x Vector with the states
     * @param p parameter vector
     * @param k constants vector
     * @param h Heaviside vector
     * @param tcl total abundances for conservation laws
     **/
    virtual void froot(
        realtype* root, realtype t, realtype const* x, realtype const* p,
        realtype const* k, realtype const* h, realtype const* tcl
    );

    /**
     * @brief Model specific implementation for fxdot
     * @param xdot residual function
     * @param t timepoint
     * @param x Vector with the states
     * @param p parameter vector
     * @param k constants vector
     * @param h Heaviside vector
     * @param w vector with helper variables
     **/
    virtual void fxdot(
        realtype* xdot, realtype t, realtype const* x, realtype const* p,
        realtype const* k, realtype const* h, realtype const* w
    ) = 0;

    /**
     * @brief Model specific implementation of fdxdotdp, with w chainrule
     * (Matlab)
     * @param dxdotdp partial derivative xdot wrt p
     * @param t timepoint
     * @param x Vector with the states
     * @param p parameter vector
     * @param k constants vector
     * @param h Heaviside vector
     * @param ip parameter index
     * @param w vector with helper variables
     * @param dwdp derivative of w wrt p
     */
    virtual void fdxdotdp(
        realtype* dxdotdp, realtype t, realtype const* x, realtype const* p,
        realtype const* k, realtype const* h, int ip, realtype const* w,
        realtype const* dwdp
    );

    /**
     * @brief Model specific implementation of fdxdotdp_explicit, no w chainrule
     * (Py)
     * @param dxdotdp_explicit partial derivative xdot wrt p
     * @param t timepoint
     * @param x Vector with the states
     * @param p parameter vector
     * @param k constants vector
     * @param h Heaviside vector
     * @param w vector with helper variables
     */
    virtual void fdxdotdp_explicit(
        realtype* dxdotdp_explicit, realtype t, realtype const* x,
        realtype const* p, realtype const* k, realtype const* h,
        realtype const* w
    );

    /**
     * @brief Model specific implementation of fdxdotdp_explicit, colptrs part
     * @param dxdotdp sparse matrix to which colptrs will be written
     */
    virtual void fdxdotdp_explicit_colptrs(SUNMatrixWrapper& dxdotdp);

    /**
     * @brief Model specific implementation of fdxdotdp_explicit, rowvals part
     * @param dxdotdp sparse matrix to which rowvals will be written
     */
    virtual void fdxdotdp_explicit_rowvals(SUNMatrixWrapper& dxdotdp);

    /**
     * @brief Model specific implementation of fdxdotdx_explicit, no w chainrule
     * (Py)
     * @param dxdotdx_explicit partial derivative xdot wrt x
     * @param t timepoint
     * @param x Vector with the states
     * @param p parameter vector
     * @param k constants vector
     * @param h heavyside vector
     * @param w vector with helper variables
     */
    virtual void fdxdotdx_explicit(
        realtype* dxdotdx_explicit, realtype t, realtype const* x,
        realtype const* p, realtype const* k, realtype const* h,
        realtype const* w
    );

    /**
     * @brief Model specific implementation of fdxdotdx_explicit, colptrs part
     * @param dxdotdx sparse matrix to which colptrs will be written
     */
    virtual void fdxdotdx_explicit_colptrs(SUNMatrixWrapper& dxdotdx);

    /**
     * @brief Model specific implementation of fdxdotdx_explicit, rowvals part
     * @param dxdotdx sparse matrix to which rowvals will be written
     */
    virtual void fdxdotdx_explicit_rowvals(SUNMatrixWrapper& dxdotdx);

    /**
     * @brief Model specific implementation of fdxdotdw, data part
     * @param dxdotdw partial derivative xdot wrt w
     * @param t timepoint
     * @param x Vector with the states
     * @param p parameter vector
     * @param k constants vector
     * @param h Heaviside vector
     * @param w vector with helper variables
     */
    virtual void fdxdotdw(
        realtype* dxdotdw, realtype t, realtype const* x, realtype const* p,
        realtype const* k, realtype const* h, realtype const* w
    );

    /**
     * @brief Model specific implementation of fdxdotdw, colptrs part
     * @param dxdotdw sparse matrix to which colptrs will be written
     */
    virtual void fdxdotdw_colptrs(SUNMatrixWrapper& dxdotdw);

    /**
     * @brief Model specific implementation of fdxdotdw, rowvals part
     * @param dxdotdw sparse matrix to which rowvals will be written
     */
    virtual void fdxdotdw_rowvals(SUNMatrixWrapper& dxdotdw);

    /**
     * @brief Sensitivity of dx/dt wrt model parameters w
     * @param t timepoint
     * @param x Vector with the states
     */
    void fdxdotdw(realtype t, const_N_Vector x);

    /** Explicit sensitivity of dx/dt wrt model parameters p
     * @param t timepoint
     * @param x Vector with the states
     */
    void fdxdotdp(realtype t, const_N_Vector x);

    void fdxdotdp(realtype t, AmiVector const& x, AmiVector const& dx) override;
};
} // namespace amici

#endif // MODEL_H
