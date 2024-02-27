#ifndef AMICI_MODEL_DIMENSIONS_H
#define AMICI_MODEL_DIMENSIONS_H

#include <gsl/gsl-lite.hpp>
#include <vector>

namespace amici {

/**
 * @brief Container for model dimensions.
 *
 * Holds number of states, observables, etc.
 */
struct ModelDimensions {
    /** Default ctor */
    ModelDimensions() = default;

    /**
     * @brief Constructor with model dimensions
     * @param nx_rdata Number of state variables
     * @param nxtrue_rdata Number of state variables of the non-augmented model
     * @param nx_solver Number of state variables with conservation laws applied
     * @param nxtrue_solver Number of state variables of the non-augmented model
     * with conservation laws applied
     * @param nx_solver_reinit Number of state variables with conservation laws
     * subject to reinitialization
     * @param np Number of parameters
     * @param nk Number of constants
     * @param ny Number of observables
     * @param nytrue Number of observables of the non-augmented model
     * @param nz Number of event observables
     * @param nztrue Number of event observables of the non-augmented model
     * @param ne Number of events
     * @param ne_solver Number of events that require root-finding
     * @param nspl Number of splines
     * @param nJ Number of objective functions
     * @param nw Number of repeating elements
     * @param ndwdx Number of nonzero elements in the `x` derivative of the
     * repeating elements
     * @param ndwdp Number of nonzero elements in the `p` derivative of the
     * repeating elements
     * @param ndwdw Number of nonzero elements in the `w` derivative of the
     * repeating elements
     * @param ndxdotdw Number of nonzero elements in the \f$ w\f$ derivative of
     * \f$ xdot\f$
     * @param ndJydy Number of nonzero elements in the \f$ y\f$ derivative of
     * \f$ dJy\f$ (shape `nytrue`)
     * @param ndxrdatadxsolver Number of nonzero elements in the \f$ x\f$
     * derivative of \f$ x_rdata\f$
     * @param ndxrdatadtcl Number of nonzero elements in the \f$ tcl\f$
     * derivative of \f$ x_rdata\f$
     * @param ndtotal_cldx_rdata Number of nonzero elements in the
     * \f$ x_rdata \f$  derivative of \f$ total_cl \f$
     * @param nnz Number of nonzero elements in Jacobian
     * @param ubw Upper matrix bandwidth in the Jacobian
     * @param lbw Lower matrix bandwidth in the Jacobian
     */
    ModelDimensions(
        int const nx_rdata, int const nxtrue_rdata, int const nx_solver,
        int const nxtrue_solver, int const nx_solver_reinit, int const np,
        int const nk, int const ny, int const nytrue, int const nz,
        int const nztrue, int const ne, int const ne_solver, int const nspl,
        int const nJ, int const nw, int const ndwdx, int const ndwdp,
        int const ndwdw, int const ndxdotdw, std::vector<int> ndJydy,
        int const ndxrdatadxsolver, int const ndxrdatadtcl,
        int const ndtotal_cldx_rdata, int const nnz, int const ubw,
        int const lbw
    )
        : nx_rdata(nx_rdata)
        , nxtrue_rdata(nxtrue_rdata)
        , nx_solver(nx_solver)
        , nxtrue_solver(nxtrue_solver)
        , nx_solver_reinit(nx_solver_reinit)
        , np(np)
        , nk(nk)
        , ny(ny)
        , nytrue(nytrue)
        , nz(nz)
        , nztrue(nztrue)
        , ne(ne)
        , ne_solver(ne_solver)
        , nspl(nspl)
        , nw(nw)
        , ndwdx(ndwdx)
        , ndwdp(ndwdp)
        , ndwdw(ndwdw)
        , ndxdotdw(ndxdotdw)
        , ndJydy(std::move(ndJydy))
        , ndxrdatadxsolver(ndxrdatadxsolver)
        , ndxrdatadtcl(ndxrdatadtcl)
        , ndtotal_cldx_rdata(ndtotal_cldx_rdata)
        , nnz(nnz)
        , nJ(nJ)
        , ubw(ubw)
        , lbw(lbw) {
        Expects(nxtrue_rdata >= 0);
        Expects(nxtrue_rdata <= nx_rdata);
        Expects(nxtrue_solver >= 0);
        Expects(nx_solver <= nx_rdata);
        Expects(nxtrue_solver <= nx_solver);
        Expects(nx_solver_reinit >= 0);
        Expects(nx_solver_reinit <= nx_solver);
        Expects(np >= 0);
        Expects(nk >= 0);
        Expects(nytrue <= ny);
        Expects(nytrue >= 0);
        Expects(nztrue >= 0);
        Expects(nztrue <= nz);
        Expects(ne >= 0);
        Expects(ne_solver >= 0);
        Expects(ne >= ne_solver);
        Expects(nspl >= 0);
        Expects(nw >= 0);
        Expects(ndwdx >= 0);
        Expects(ndwdx <= nw * nx_solver);
        Expects(ndwdp >= 0);
        Expects(ndwdp <= nw * np);
        Expects(ndwdw >= 0);
        Expects(ndwdw <= nw * nw);
        Expects(ndxdotdw >= 0);
        Expects(ndxrdatadxsolver >= 0);
        Expects(ndxrdatadxsolver <= nx_rdata * nx_solver);
        Expects(ndxrdatadtcl >= 0);
        Expects(ndxrdatadtcl <= nx_rdata * (nx_rdata - nx_solver));
        Expects(ndtotal_cldx_rdata >= 0);
        Expects(ndtotal_cldx_rdata <= (nx_rdata - nx_solver) * nx_rdata);
        Expects(nnz >= 0);
        Expects(nJ >= 0);
        Expects(ubw >= 0);
        Expects(lbw >= 0);
    }

    /** Number of states */
    int nx_rdata{0};

    /** Number of states in the unaugmented system */
    int nxtrue_rdata{0};

    /** Number of states with conservation laws applied */
    int nx_solver{0};

    /**
     * Number of states in the unaugmented system with conservation laws
     * applied
     */
    int nxtrue_solver{0};

    /** Number of solver states subject to reinitialization */
    int nx_solver_reinit{0};

    /** Number of parameters */
    int np{0};

    /** Number of constants */
    int nk{0};

    /** Number of observables */
    int ny{0};

    /** Number of observables in the unaugmented system */
    int nytrue{0};

    /** Number of event outputs */
    int nz{0};

    /** Number of event outputs in the unaugmented system */
    int nztrue{0};

    /** Number of events */
    int ne{0};

    /** Number of events that require root-finding */
    int ne_solver{0};

    /** Number of spline functions in the model */
    int nspl{0};

    /** Number of common expressions */
    int nw{0};

    /**
     * Number of nonzero elements in the `x` derivative of the
     * repeating elements
     */
    int ndwdx{0};

    /**
     * Number of nonzero elements in the `p` derivative of the
     * repeating elements
     */
    int ndwdp{0};

    /**
     * Number of nonzero elements in the `w` derivative of the
     * repeating elements
     */
    int ndwdw{0};

    /** Number of nonzero elements in the \f$ w \f$ derivative of \f$ xdot \f$
     */
    int ndxdotdw{0};

    /**
     * Number of nonzero elements in the \f$ y \f$ derivative of
     * \f$ dJy \f$ (dimension `nytrue`)
     */
    std::vector<int> ndJydy;

    /** Number of nonzero elements in the \f$ x \f$ derivative of \f$ x_rdata
     * \f$ */
    int ndxrdatadxsolver{0};

    /** Number of nonzero elements in the \f$ tcl\f$ derivative of \f$ x_rdata
     * \f$ */
    int ndxrdatadtcl{0};

    /** Number of nonzero elements in the \f$ x_rdata\f$ derivative of
     *  \f$ total_cl \f$ */
    int ndtotal_cldx_rdata{0};

    /** Number of nonzero entries in Jacobian */
    int nnz{0};

    /** Dimension of the augmented objective function for 2nd order ASA */
    int nJ{0};

    /** Upper bandwidth of the Jacobian */
    int ubw{0};

    /** Lower bandwidth of the Jacobian */
    int lbw{0};
};

} // namespace amici

#endif // AMICI_MODEL_DIMENSIONS_H
