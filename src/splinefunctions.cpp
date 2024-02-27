#include "amici/splinefunctions.h"
#include "amici/amici.h"
#include "amici/defines.h"
#include "amici/exception.h"

#include <algorithm> // std::min
#include <cassert>
#include <cmath>
#include <vector>

namespace amici {

static realtype
evaluate_polynomial(realtype const x, gsl::span<realtype const> coeff) {
    /* Use Horner's method (https://en.wikipedia.org/wiki/Horner%27s_method)
     * for numerical efficiency:
     *
     * spline(t) = a * t**3 + b * t**2 + c * t + d
     *           = d + t * (c + t * (b + t * a))
     * with coeff[0, 1, 2, 3] = [d, c, b, a]
     */
    assert(coeff.size() >= 4);
    auto coeff_p = coeff.data();
    return coeff_p[0] + x * (coeff_p[1] + x * (coeff_p[2] + x * coeff_p[3]));
}

AbstractSpline::AbstractSpline(
    std::vector<realtype> nodes, std::vector<realtype> node_values,
    bool equidistant_spacing, bool logarithmic_parametrization
)
    : nodes_(std::move(nodes))
    , node_values_(std::move(node_values))
    , equidistant_spacing_(equidistant_spacing)
    , logarithmic_parametrization_(logarithmic_parametrization) {

    /* we want to set the number of nodes */
    auto n_nodes_ = static_cast<int>(node_values_.size());

    /* In case we have equidistant spacing, compute node locations */
    if (equidistant_spacing_) {
        if (nodes_.size() != 2)
            throw AmiException("Splines with equidistant spacing need a nodes "
                               "vector with two elements (first/last node).");
        realtype node_start = nodes_[0];
        realtype node_step = (nodes_[1] - nodes_[0]) / (n_nodes_ - 1);
        nodes_.resize(n_nodes_);
        nodes_[n_nodes_ - 1] = nodes_[1];
        for (int i_node = 0; i_node < n_nodes_ - 1; i_node++)
            nodes_[i_node] = node_start + i_node * node_step;
    } else if (nodes_.size() != node_values_.size()) {
        throw std::invalid_argument(
            "Number of nodes and number of node_values do not match."
        );
    }
}

realtype AbstractSpline::get_value(realtype const t) const {
    auto y = get_value_scaled(t);
    return logarithmic_parametrization_ ? std::exp(y) : y;
}

realtype AbstractSpline::get_sensitivity(realtype const t, int const ip) const {
    auto s = get_sensitivity_scaled(t, ip);
    return logarithmic_parametrization_ ? s * get_value(t) : s;
}

realtype AbstractSpline::get_sensitivity(
    realtype const t, int const ip, realtype const value
) const {
    auto s = get_sensitivity_scaled(t, ip);
    return logarithmic_parametrization_ ? s * value : s;
}

realtype AbstractSpline::get_node_value(int const i) const {
    return node_values_[i];
}

realtype AbstractSpline::get_node_value_scaled(int const i) const {
    // TODO It could be precomputed and stored in the object.
    //      Not sure if its worth the effort.
    if (logarithmic_parametrization_)
        return std::log(node_values_[i]);
    else
        return node_values_[i];
}

realtype AbstractSpline::get_final_value_scaled() const {
    return final_value_scaled_;
}

realtype AbstractSpline::get_final_value() const {
    auto y = get_final_value_scaled();
    return logarithmic_parametrization_ ? std::exp(y) : y;
}

void AbstractSpline::set_final_value_scaled(realtype finalValue) {
    final_value_scaled_ = finalValue;
}

realtype AbstractSpline::get_final_sensitivity_scaled(int const ip) const {
    return final_sensitivity_scaled_[ip];
}

realtype AbstractSpline::get_final_sensitivity(int const ip) const {
    auto s = get_final_sensitivity_scaled(ip);
    if (logarithmic_parametrization_) {
        auto v = get_final_value();
        if (std::isinf(v)) {
            assert(
                v > 0
            ); // logarithmic parameterization means positive values only
            assert(
                std::isnan(s) || s == 0
            ); // in the case the limit is +inf, sensitivity in log-scale will
               // either be NaN or zero
            return s;
        } else {
            return s * v;
        }
    } else {
        return s;
    }
}

void AbstractSpline::set_final_sensitivity_scaled(
    std::vector<realtype> finalSensitivity
) {
    final_sensitivity_scaled_ = std::move(finalSensitivity);
}

bool AbstractSpline::get_equidistant_spacing() const {
    return equidistant_spacing_;
}

bool AbstractSpline::get_logarithmic_parametrization() const {
    return logarithmic_parametrization_;
}

HermiteSpline::HermiteSpline(
    std::vector<realtype> nodes, std::vector<realtype> node_values,
    std::vector<realtype> node_values_derivative,
    SplineBoundaryCondition firstNodeBC, SplineBoundaryCondition lastNodeBC,
    SplineExtrapolation firstNodeExtrapol, SplineExtrapolation lastNodeExtrapol,
    bool node_derivative_by_FD, bool equidistant_spacing,
    bool logarithmic_parametrization
)
    : AbstractSpline(
        std::move(nodes), std::move(node_values), equidistant_spacing,
        logarithmic_parametrization
    )
    , node_values_derivative_(std::move(node_values_derivative))
    , first_node_bc_(firstNodeBC)
    , last_node_bc_(lastNodeBC)
    , first_node_ep_(firstNodeExtrapol)
    , last_node_ep_(lastNodeExtrapol)
    , node_derivative_by_FD_(node_derivative_by_FD) {
    if (!node_derivative_by_FD_
        && node_values_derivative_.size() != nodes_.size()) {
        throw std::invalid_argument(
            "Size of node_values_derivative does not match number of nodes."
        );
    }

    /* We may have to compute the derivatives at the nodes */
    handle_inner_derivatives();
    /* First and last node need to be handled separately */
    handle_boundary_conditions();
}

void HermiteSpline::handle_inner_derivatives() {
    /* If values of the derivative at the nodes are to be computed by finite
     * differences, we have to fill up node_values_derivative_ */
    if (node_derivative_by_FD_) {
        node_values_derivative_.resize(n_nodes(), 0.0);
        if (get_equidistant_spacing()) {
            realtype hx2 = 2 * (nodes_[1] - nodes_[0]);
            for (int i_node = 1; i_node < n_nodes() - 1; i_node++)
                node_values_derivative_[i_node]
                    = (node_values_[i_node + 1] - node_values_[i_node - 1])
                      / hx2;
        } else {
            for (int i_node = 1; i_node < n_nodes() - 1; i_node++) {
                realtype dleft
                    = (node_values_[i_node] - node_values_[i_node - 1])
                      / (nodes_[i_node] - nodes_[i_node - 1]);
                realtype dright
                    = (node_values_[i_node + 1] - node_values_[i_node])
                      / (nodes_[i_node + 1] - nodes_[i_node]);
                node_values_derivative_[i_node] = (dleft + dright) / 2;
            }
        }
    }
}

void HermiteSpline::handle_boundary_conditions() {
    int last = n_nodes() - 1;

    if ((first_node_bc_ == SplineBoundaryCondition::periodic
         || last_node_bc_ == SplineBoundaryCondition::periodic)
        && first_node_bc_ != last_node_bc_)
        throw AmiException("If one of the boundary conditions is periodic, "
                           "the other one must be periodic too.");

    /* We have to take special care of the first node */
    switch (first_node_bc_) {
    case SplineBoundaryCondition::given:
        if (node_derivative_by_FD_)
            /* 1-sided FD */
            node_values_derivative_[0]
                = (node_values_[1] - node_values_[0]) / (nodes_[1] - nodes_[0]);
        break;

    case SplineBoundaryCondition::zeroDerivative:
        node_values_derivative_[0] = 0;
        break;

    case SplineBoundaryCondition::natural:
        node_values_derivative_[0] = -0.5 * node_values_derivative_[1]
                                     + 1.5 * (node_values_[1] - node_values_[0])
                                           / (nodes_[1] - nodes_[0]);
        break;

    case SplineBoundaryCondition::naturalZeroDerivative:
        throw AmiException("Natural boundary condition with zero "
                           "derivative is not allowed for Hermite splines.");

    case SplineBoundaryCondition::periodic:
        if (node_derivative_by_FD_) {
            if (get_equidistant_spacing()) {
                realtype hx2 = 2 * (nodes_[1] - nodes_[0]);
                node_values_derivative_[0]
                    = (node_values_[1] - node_values_[last - 1]) / hx2;
            } else {
                realtype dleft = (node_values_[last] - node_values_[last - 1])
                                 / (nodes_[last] - nodes_[last - 1]);
                realtype dright = (node_values_[1] - node_values_[0])
                                  / (nodes_[1] - nodes_[0]);
                node_values_derivative_[0] = (dleft + dright) / 2;
            }
        }
        break;

    default:
        throw AmiException("Invalid value for boundary condition.");
    }

    /* ...and the last node (1-sided FD). */
    switch (last_node_bc_) {
    case SplineBoundaryCondition::given:
        if (node_derivative_by_FD_)
            /* 1-sided FD */
            node_values_derivative_[last]
                = (node_values_[last] - node_values_[last - 1])
                  / (nodes_[last] - nodes_[last - 1]);
        break;

    case SplineBoundaryCondition::zeroDerivative:
        node_values_derivative_[last] = 0;
        break;

    case SplineBoundaryCondition::natural:
        node_values_derivative_[last]
            = -0.5 * node_values_derivative_[last - 1]
              + 1.5 * (node_values_[last] - node_values_[last - 1])
                    / (nodes_[last] - nodes_[last - 1]);
        break;

    case SplineBoundaryCondition::naturalZeroDerivative:
        throw AmiException("Natural boundary condition with zero "
                           "derivative is not allowed for Hermite splines.");

    case SplineBoundaryCondition::periodic:
        if (node_derivative_by_FD_)
            // if one bc is periodic, the other is periodic too
            node_values_derivative_[last] = node_values_derivative_[0];
        break;

    default:
        throw AmiException("Invalid value for boundary condition.");
    }
}

realtype HermiteSpline::get_node_derivative(int const i) const {
    return node_values_derivative_[i];
}

realtype HermiteSpline::get_node_derivative_scaled(int const i) const {
    // TODO It could be precomputed and stored in the object.
    //      Not sure if its worth the effort.
    if (get_logarithmic_parametrization())
        return node_values_derivative_[i] / node_values_[i];
    else
        return node_values_derivative_[i];
}

void HermiteSpline::compute_coefficients() {
    /* Allocate space for the coefficients for Horner's method.
     * They are stored in the vector as
     * [d_0, c_0, b_0, a_0, d_1, c_1, ... , b_{n_nodes-1}, a_{n_nodes-1}] */
    coefficients.resize(4 * (n_nodes() - 1), 0.0);

    /* Compute the coefficients of the spline polynomials:
     * spline(t) = a * t**3 + b * t**2 + c * t + d
     *           = d + t * (c + t * (b + t * a))
     * with coefficients[4 * i_node + (0, 1, 2, 3)] = (d, c, b, a)
     */

    for (int i_node = 0; i_node < n_nodes() - 1; i_node++) {
        /* Get the length of the interval. Yes, we could save computation time
         * by exploiting equidistant spacing, but we're talking about <1k FLOPs
         * for sure, no matter what model. Screw it. */
        realtype len = nodes_[i_node + 1] - nodes_[i_node];

        /* Coefficients for cubic Hermite polynomials */
        coefficients[4 * i_node] = get_node_value_scaled(i_node);
        coefficients[4 * i_node + 1] = len * get_node_derivative_scaled(i_node);
        coefficients[4 * i_node + 2]
            = -3 * get_node_value_scaled(i_node)
              - 2 * len * get_node_derivative_scaled(i_node)
              + 3 * get_node_value_scaled(i_node + 1)
              - len * get_node_derivative_scaled(i_node + 1);
        coefficients[4 * i_node + 3]
            = 2 * get_node_value_scaled(i_node)
              + len * get_node_derivative_scaled(i_node)
              - 2 * get_node_value_scaled(i_node + 1)
              + len * get_node_derivative_scaled(i_node + 1);
    }

    /* Take care of coefficients for extrapolation */
    compute_coefficients_extrapolation();
}

void HermiteSpline::compute_coefficients_extrapolation() {
    /* Do we want to extrapolate at all? */
    bool needExtrapolationCoefficients
        = first_node_ep_ == SplineExtrapolation::constant
          || first_node_ep_ == SplineExtrapolation::linear
          || last_node_ep_ == SplineExtrapolation::constant
          || last_node_ep_ == SplineExtrapolation::linear;
    if (!needExtrapolationCoefficients)
        return;

    coefficients_extrapolate.resize(4, 0.0);

    int last = n_nodes() - 1;

    /* Beyond the spline nodes, we need to extrapolate using a * t + b.
     * Those coefficients are stored as [b_first, a_first, b_last, a_last] */
    switch (first_node_ep_) {
    case SplineExtrapolation::constant:
        coefficients_extrapolate[0] = get_node_value_scaled(0);
        coefficients_extrapolate[1] = 0;
        break;

    case SplineExtrapolation::linear:
        coefficients_extrapolate[0]
            = get_node_value_scaled(0)
              - nodes_[0] * get_node_derivative_scaled(0);
        coefficients_extrapolate[1] = get_node_derivative_scaled(0);
        break;

    default:
        /* We don't need specific coefficients in the cases of:
         * noExtrapolation, polynomial, periodic*/
        break;
    }
    switch (last_node_ep_) {
    case SplineExtrapolation::constant:
        coefficients_extrapolate[2] = get_node_value_scaled(last);
        coefficients_extrapolate[3] = 0;
        break;

    case SplineExtrapolation::linear:
        coefficients_extrapolate[2]
            = get_node_value_scaled(last)
              - nodes_[last] * get_node_derivative_scaled(last);
        coefficients_extrapolate[3] = get_node_derivative_scaled(last);
        break;

    default:
        /* We don't need specific coefficients in the cases of:
         * noExtrapolation, polynomial, periodic*/
        break;
    }
}

#ifdef DVALUESDP
#error "Preprocessor macro DVALUESDP already defined?!"
#else
#define DVALUESDP(i_node) dvaluesdp[node_offset + (i_node) * nplist]
#endif
#ifdef DSLOPESDP
#error "Preprocessor macro DSLOPESDP already defined?!"
#else
#define DSLOPESDP(i_node) dslopesdp[node_offset + (i_node) * nplist]
#endif

void HermiteSpline::compute_coefficients_sensi(
    int nplist, int spline_offset, gsl::span<realtype> dvaluesdp,
    gsl::span<realtype> dslopesdp
) {
    // If slopes are computed by finite differences,
    // we need to autocompute the slope sensitivities
    if (node_derivative_by_FD_) {
        assert(dvaluesdp.size() == dslopesdp.size());
        for (int ip = 0; ip < nplist; ip++)
            compute_slope_sensitivities_by_fd(
                nplist, spline_offset, ip, dvaluesdp, dslopesdp
            );
    }

    /* Ensure that dslopesdp satisfies the BC */
    if (first_node_bc_ == SplineBoundaryCondition::zeroDerivative) {
        for (int ip = 0; ip < nplist; ip++)
            dslopesdp[spline_offset + ip] = 0.0;
    }
    if (last_node_bc_ == SplineBoundaryCondition::zeroDerivative) {
        int last = n_nodes() - 1;
        for (int ip = 0; ip < nplist; ip++)
            dslopesdp[spline_offset + ip + last * nplist] = 0.0;
    }

    // If necessary, translate sensitivities to logarithmic parametrization
    if (get_logarithmic_parametrization()) {
        for (int i_node = 0; i_node < n_nodes(); i_node++) {
            for (int ip = 0; ip < nplist; ip++) {
                int node_offset = spline_offset + ip;
                realtype value = get_node_value(i_node);
                realtype slope = node_values_derivative_[i_node];
                realtype dvaluedp = DVALUESDP(i_node);
                realtype dslopedp = DSLOPESDP(i_node);
                DVALUESDP(i_node) = dvaluedp / value;
                DSLOPESDP(i_node)
                    = (dslopedp - dvaluedp * slope / value) / value;
            }
        }
    }

    /*
     * Allocate space for the coefficients
     * They are stored in the vector as
     * [ D[d_0, p0], D[c_0, p0], D[b_0, p0], D[a_0, p0], D[d_1, p0],
     *   ... ,
     *   D[b_{n_nodes-1}, p0], D[a_{n_nodes-1}, p0],
     *   D[d_0, p1], D[c_0, p1], ...
     *   ..., D[b_{n_nodes-1}, p{nplist-1}, D[a_{n_nodes-1}, p{nplist-1}]
     * ]
     */
    int n_spline_coefficients = 4 * (n_nodes() - 1);
    coefficients_sensi.resize(n_spline_coefficients * nplist, 0.0);

    /*
     * We're using short hand notation for some node values or slopes, based on
     * the notation used on https://en.wikipedia.org/wiki/Cubic_Hermite_spline
     * In brief: "p" denotes the current (k-th) spline node value,
     * "m" its tangent or slope, "s" in front the sensitivity, "1" at the end
     * means the following node (" + 1"), so "smk1" is the sensitivity of the
     * slope at node k + 1, w.r.t. to the current parameter (looping index).
     */

    /* Parametric derivatives of splines are splines again.
     * We compute the coefficients for those polynomials now. */
    for (int i_node = 0; i_node < n_nodes() - 1; i_node++) {
        /* Get the length of the interval. */
        realtype len = nodes_[i_node + 1] - nodes_[i_node];

        /* As computing the coefficient is a mess, it's in another function */
        for (int ip = 0; ip < nplist; ip++)
            get_coeffs_sensi_lowlevel(
                ip, i_node, nplist, n_spline_coefficients, spline_offset, len,
                dvaluesdp, dslopesdp, coefficients_sensi
            );
    }

    /* We need the coefficients for extrapolating beyond the spline domain */
    compute_coefficients_extrapolation_sensi(
        nplist, spline_offset, dvaluesdp, dslopesdp
    );
}

void HermiteSpline::compute_slope_sensitivities_by_fd(
    int nplist, int spline_offset, int ip, gsl::span<realtype> dvaluesdp,
    gsl::span<realtype> dslopesdp
) {
    int last = n_nodes() - 1;
    int node_offset = spline_offset + ip;

    // Left boundary (first node)
    switch (first_node_bc_) {
    case SplineBoundaryCondition::given:
        DSLOPESDP(0) = (DVALUESDP(1) - DVALUESDP(0)) / (nodes_[1] - nodes_[0]);
        break;

    case SplineBoundaryCondition::zeroDerivative:
        DSLOPESDP(0) = 0;
        break;

    case SplineBoundaryCondition::natural:
        throw AmiException("Natural boundary condition for Hermite "
                           "splines is not implemented yet.");

    case SplineBoundaryCondition::periodic:
        if (get_equidistant_spacing()) {
            realtype hx2 = 2 * (nodes_[1] - nodes_[0]);
            DSLOPESDP(0) = (DVALUESDP(1) - DVALUESDP(last - 1)) / hx2;
        } else {
            realtype dleft = (DVALUESDP(last) - DVALUESDP(last - 1))
                             / (nodes_[last] - nodes_[last - 1]);
            realtype dright
                = (DVALUESDP(1) - DVALUESDP(0)) / (nodes_[1] - nodes_[0]);
            DSLOPESDP(0) = (dleft + dright) / 2;
        }
        break;

    default:
        throw AmiException("Unexpected value for boundary condition.");
    }

    // Inner nodes
    if (get_equidistant_spacing()) {
        realtype hx2 = 2 * (nodes_[1] - nodes_[0]);
        for (int i_node = 1; i_node < n_nodes() - 1; i_node++)
            DSLOPESDP(i_node)
                = (DVALUESDP(i_node + 1) - DVALUESDP(i_node - 1)) / hx2;
    } else {
        for (int i_node = 1; i_node < n_nodes() - 1; i_node++) {
            realtype dleft = (DVALUESDP(i_node) - DVALUESDP(i_node - 1))
                             / (nodes_[i_node] - nodes_[i_node - 1]);
            realtype dright = (DVALUESDP(i_node + 1) - DVALUESDP(i_node))
                              / (nodes_[i_node + 1] - nodes_[i_node]);
            DSLOPESDP(i_node) = (dleft + dright) / 2;
        }
    }

    // Right boundary (last nodes)
    switch (last_node_bc_) {
    case SplineBoundaryCondition::given:
        DSLOPESDP(last) = (DVALUESDP(last) - DVALUESDP(last - 1))
                          / (nodes_[last] - nodes_[last - 1]);
        break;

    case SplineBoundaryCondition::zeroDerivative:
        DSLOPESDP(last) = 0;
        break;

    case SplineBoundaryCondition::natural:
        throw AmiException("Natural boundary condition for Hermite "
                           "splines is not implemented yet.");

    case SplineBoundaryCondition::naturalZeroDerivative:
        throw AmiException("Natural boundary condition with zero "
                           "derivative is not allowed for Hermite splines.");

    case SplineBoundaryCondition::periodic:
        // if one bc is periodic, the other is periodic too
        DSLOPESDP(last) = DSLOPESDP(0);
        break;

    default:
        throw AmiException("Unexpected value for boundary condition.");
    }
}

#undef DVALUESDP
#undef DSLOPESDP

void HermiteSpline::compute_coefficients_extrapolation_sensi(
    int nplist, int spline_offset, gsl::span<realtype> dvaluesdp,
    gsl::span<realtype> dslopesdp
) {

    /* Do we want to extrapolate at all? */
    bool needExtrapolationCoefficients
        = first_node_ep_ == SplineExtrapolation::constant
          || first_node_ep_ == SplineExtrapolation::linear
          || last_node_ep_ == SplineExtrapolation::constant
          || last_node_ep_ == SplineExtrapolation::linear;
    if (!needExtrapolationCoefficients)
        return;

    /* Beyond the spline nodes, we need to extrapolate using a * t + b.
     * Those coefficients are stored as
     * [
     *   D[b_first, p0], D[a_first, p0], D[b_last, p0], D[a_last, p0],
     *   D[b_first, p1], ... D[a_last, p{nplist-1}]
     * ]
     */
    coefficients_extrapolate_sensi.resize(4 * nplist, 0.0);

    realtype sm0;
    for (int ip = 0; ip < nplist; ip++) {
        realtype sp0 = dvaluesdp[spline_offset + ip];
        switch (first_node_ep_) {
        /* This whole switch-case-if-else-if-thing could be moved
         * outside the loop, I know. Yet, it's at most some thousand
         * if's done once in the program for saving many lines of code
         * and getting a much clearer code structure. */
        case SplineExtrapolation::constant:
            sm0 = 0;
            break;

        case SplineExtrapolation::linear:
            if (first_node_bc_ == SplineBoundaryCondition::zeroDerivative) {
                sm0 = 0;
            } else if (get_node_derivative_by_fd() && first_node_bc_ == SplineBoundaryCondition::given) {
                sm0 = (dvaluesdp[spline_offset + ip + nplist] - sp0)
                      / (nodes_[1] - nodes_[0]);

            } else if (get_node_derivative_by_fd() && first_node_bc_ == SplineBoundaryCondition::natural) {
                throw AmiException(
                    "Natural boundary condition for "
                    "Hermite splines with linear extrapolation is "
                    "not yet implemented."
                );

            } else if (!get_node_derivative_by_fd() && first_node_bc_ == SplineBoundaryCondition::given) {
                sm0 = dslopesdp[spline_offset + ip];

            } else if (!get_node_derivative_by_fd() && first_node_bc_ == SplineBoundaryCondition::natural) {
                throw AmiException(
                    "Natural boundary condition for "
                    "Hermite splines with linear extrapolation is "
                    "not yet implemented."
                );

            } else {
                throw AmiException(
                    "Some weird combination of spline boundary "
                    "condition, extrapolation and finite differences was "
                    "passed which should not be allowed."
                );
            }
            break;

        default:
            /* We don't need specific coefficients in the cases of:
             * noExtrapolation, polynomial, periodic
             * NB the corresponding values in coefficients_extrapolate_sensi
             *    will never be accessed, so it's safe to leave them
             *    undefined.
             */
            continue;
        }
        /* Write them to the vector */
        coefficients_extrapolate_sensi[4 * ip] = sp0 - sm0 * nodes_[0];
        coefficients_extrapolate_sensi[4 * ip + 1] = sm0;
    }

    realtype sm_end;
    for (int ip = 0; ip < nplist; ip++) {
        realtype sp_end
            = dvaluesdp[spline_offset + ip + (n_nodes() - 1) * nplist];
        switch (last_node_ep_) {
        /* This whole switch-case-if-else-if-thing could be moved
         * outside the loop, I know. Yet, it's at most some thousand
         * if's done once in the program for saving many lines of code
         * and getting a much clearer code structure. */
        case SplineExtrapolation::constant:
            sm_end = 0;
            break;

        case SplineExtrapolation::linear:
            if (last_node_bc_ == SplineBoundaryCondition::zeroDerivative) {
                sm_end = 0;
            } else if (get_node_derivative_by_fd() && last_node_bc_ == SplineBoundaryCondition::given) {
                sm_end = (sp_end
                          - dvaluesdp
                              [spline_offset + ip + (n_nodes() - 2) * nplist])
                         / (nodes_[n_nodes() - 1] - nodes_[n_nodes() - 2]);

            } else if (get_node_derivative_by_fd() && last_node_bc_ == SplineBoundaryCondition::natural) {
                throw AmiException(
                    "Natural boundary condition for "
                    "Hermite splines with linear extrapolation is "
                    "not yet implemented."
                );

            } else if (!get_node_derivative_by_fd() && last_node_bc_ == SplineBoundaryCondition::given) {
                sm_end
                    = dslopesdp[spline_offset + ip + (n_nodes() - 1) * nplist];

            } else if (!get_node_derivative_by_fd() && last_node_bc_ == SplineBoundaryCondition::natural) {
                throw AmiException(
                    "Natural boundary condition for "
                    "Hermite splines with linear extrapolation is "
                    "not yet implemented."
                );

            } else {
                throw AmiException(
                    "Some weird combination of spline boundary "
                    "condition, extrapolation and finite differences was "
                    "passed which should not be allowed."
                );
            }
            break;

        default:
            /* We don't need specific coefficients in the cases of:
             * noExtrapolation, polynomial, periodic
             * NB the corresponding values in coefficients_extrapolate_sensi
             *    will never be accessed, so it's safe to leave them
             *    undefined.
             */
            continue;
        }
        /* Write them to the vector */
        coefficients_extrapolate_sensi[4 * ip + 2]
            = sp_end - sm_end * nodes_[n_nodes() - 1];
        coefficients_extrapolate_sensi[4 * ip + 3] = sm_end;
    }
}

void HermiteSpline::get_coeffs_sensi_lowlevel(
    int ip, int i_node, int nplist, int n_spline_coefficients,
    int spline_offset, realtype len, gsl::span<realtype> dnodesdp,
    gsl::span<realtype> dslopesdp, gsl::span<realtype> coeffs
) const {
    /* We're using the short hand notation for node values and slopes from
     * compute_coefficients_sensi() here. See this function for documentation.
     */
    int node_offset = spline_offset + ip;
    realtype spk = dnodesdp[node_offset + i_node * nplist];
    realtype spk1 = dnodesdp[node_offset + (i_node + 1) * nplist];
    realtype smk = dslopesdp[node_offset + i_node * nplist];
    realtype smk1 = dslopesdp[node_offset + (i_node + 1) * nplist];

    /* Compute the actual coefficients */
    coeffs[ip * n_spline_coefficients + 4 * i_node] = spk;
    coeffs[ip * n_spline_coefficients + 4 * i_node + 1] = len * smk;
    coeffs[ip * n_spline_coefficients + 4 * i_node + 2]
        = 3 * (spk1 - spk) - len * (2 * smk + smk1);
    coeffs[ip * n_spline_coefficients + 4 * i_node + 3]
        = 2 * (spk - spk1) + len * (smk + smk1);
}

void HermiteSpline::compute_final_value() {
    /* We need to compute the final value of the spline, depending on its
     * boundary condition and the extrapolation option. */
    realtype finalValue;
    if (last_node_ep_ == SplineExtrapolation::constant) {
        finalValue = coefficients_extrapolate[2];
    } else if (last_node_ep_ == SplineExtrapolation::linear) {
        if (last_node_bc_ == SplineBoundaryCondition::zeroDerivative) {
            finalValue = coefficients_extrapolate[2];
        } else if (coefficients_extrapolate[3] < 0) {
            finalValue = -INFINITY;
        } else if (coefficients_extrapolate[3] > 0) {
            finalValue = INFINITY;
        } else {
            finalValue = coefficients_extrapolate[2];
        }
    } else if (last_node_ep_ == SplineExtrapolation::polynomial) {
        int last = 4 * (n_nodes() - 1) - 1;
        if (coefficients[last] < 0) {
            finalValue = -INFINITY;
        } else if (coefficients[last] > 0) {
            finalValue = INFINITY;
        } else if (coefficients[last - 1] < 0) {
            finalValue = -INFINITY;
        } else if (coefficients[last - 1] > 0) {
            finalValue = INFINITY;
        } else if (coefficients[last - 2] < 0) {
            finalValue = -INFINITY;
        } else if (coefficients[last - 2] > 0) {
            finalValue = INFINITY;
        } else {
            finalValue = coefficients[last - 3];
        }
    } else {
        /* Periodic: will not yield a steady state, unless the spline is the
         * constant function */
        finalValue = get_node_value_scaled(0);
        for (int i = 0; i < n_nodes(); i++) {
            if (get_node_value_scaled(i) != finalValue
                || get_node_derivative_scaled(i) != 0) {
                finalValue = NAN;
                break;
            }
        }
    }
    set_final_value_scaled(finalValue);
}

void HermiteSpline::compute_final_sensitivity(
    int nplist, int /*spline_offset*/, gsl::span<realtype> /*dvaluesdp*/,
    gsl::span<realtype> /*dslopesdp*/
) {
    /* We need to compute the final value of the spline, depending on its
     * boundary condition and the extrapolation option. */
    std::vector<realtype> finalSensitivity(nplist, 0);
    if ((last_node_ep_ == SplineExtrapolation::constant)
        || (last_node_bc_ == SplineBoundaryCondition::zeroDerivative
            && last_node_ep_ == SplineExtrapolation::linear)) {
        for (int ip = 0; ip < nplist; ip++)
            finalSensitivity[ip] = coefficients_extrapolate_sensi[4 * ip + 2];
    } else if (last_node_ep_ == SplineExtrapolation::linear) {
        /* If steady state is infinity, sensitivity must be 0
         * (unless the derivative is zero and the final value will change
         * abruptly from finite to +-inf) (if the derivative is constant zero in
         * a neighbourhood, then the final value will not change, but this is
         * impossible to determine just from the sensitivity of the derivative)
         */
        int last = n_nodes() - 1;
        if (get_node_derivative_scaled(last) == 0)
            std::fill(finalSensitivity.begin(), finalSensitivity.end(), NAN);
    } else if (last_node_ep_ == SplineExtrapolation::polynomial) {
        /* Yes, that's not correct. But I don't see any good reason for
         * implementing a case, which anybody with more than a dead fish
         * between the ears will never use. */
        std::fill(finalSensitivity.begin(), finalSensitivity.end(), NAN);
    } else {
        /* Periodic: will not yield a steady state
         * (unless the spline is the constant funtion,
         * but even in that case sensitivity information is not able to tell us
         * whether the steady state continues to exist in a neighbourhood of the
         * current parameters
         */
        std::fill(finalSensitivity.begin(), finalSensitivity.end(), NAN);
    }
    set_final_sensitivity_scaled(finalSensitivity);
}

realtype HermiteSpline::get_value_scaled(realtype const t) const {
    /* Is this a steady state computation? */
    if (std::isinf(t))
        return get_final_value_scaled();

    /* Compute the spline value */
    int i_node;
    realtype len;

    /* Are we past the last node? Extrapolate! */
    if (t > nodes_[n_nodes() - 1]) {
        switch (last_node_ep_) {
        case SplineExtrapolation::noExtrapolation:
            throw AmiException(
                "Trying to evaluate spline after last "
                "spline node, but spline has been specified not to allow "
                "extrapolation."
            );

        case SplineExtrapolation::constant:
            return coefficients_extrapolate[2];

        case SplineExtrapolation::linear:
            return coefficients_extrapolate[2]
                   + t * coefficients_extrapolate[3];

        case SplineExtrapolation::polynomial:
            /* Evaluate last interpolation polynomial */
            i_node = n_nodes() - 2;
            len = nodes_[i_node + 1] - nodes_[i_node];
            return evaluate_polynomial(
                (t - nodes_[i_node]) / len,
                gsl::make_span(coefficients).subspan(i_node * 4)
            );

        case SplineExtrapolation::periodic:
            len = nodes_[n_nodes() - 1] - nodes_[0];
            return get_value(nodes_[0] + std::fmod(t - nodes_[0], len));

        default:
            throw AmiException("Unsupported SplineExtrapolation type");
        }
    }

    /* Are we before the first node? Extrapolate! */
    if (t < nodes_[0]) {
        switch (first_node_ep_) {
        case SplineExtrapolation::noExtrapolation:
            throw AmiException(
                "Trying to evaluate spline before first "
                "spline node, but spline has been specified not to allow "
                "extrapolation."
            );

        case SplineExtrapolation::constant:
            return coefficients_extrapolate[0];

        case SplineExtrapolation::linear:
            return coefficients_extrapolate[0]
                   + t * coefficients_extrapolate[1];

        case SplineExtrapolation::polynomial:
            /* Evaluate last interpolation polynomial */
            len = nodes_[1] - nodes_[0];
            return evaluate_polynomial((t - nodes_[0]) / len, coefficients);

        case SplineExtrapolation::periodic:
            len = nodes_[n_nodes() - 1] - nodes_[0];
            return get_value(
                nodes_[n_nodes() - 1] + std::fmod(t - nodes_[0], len)
            );
        default:
            throw AmiException("Unsupported SplineExtrapolation type");
        }
    }

    /* Get the spline interval which we need */
    if (get_equidistant_spacing()) {
        /* equidistant spacing: just compute the interval */
        len = nodes_[1] - nodes_[0];
        i_node = static_cast<int>(std::trunc((t - nodes_[0]) / len));
        i_node = std::min(i_node, n_nodes() - 2);
    } else {
        /* no equidistant spacing: we need to iterate */
        i_node = 0;
        while (nodes_[i_node + 1] < t) {
            i_node++;
        }
        if (t == nodes_[i_node + 1])
            return get_node_value_scaled(i_node + 1); // make it exact on nodes
        len = nodes_[i_node + 1] - nodes_[i_node];
    }

    /* Evaluate the interpolation polynomial */
    return evaluate_polynomial(
        (t - nodes_[i_node]) / len,
        gsl::make_span(coefficients).subspan(i_node * 4)
    );
}

realtype
HermiteSpline::get_sensitivity_scaled(realtype const t, int const ip) const {
    /* Is this a steady state computation? */
    if (std::isinf(t))
        return get_final_sensitivity_scaled(ip);

    /* Compute the parametric derivative of the spline value */
    int i_node;
    realtype len;

    if (t > nodes_[n_nodes() - 1]) {
        /* Are we past the last node? Extrapolate! */
        switch (last_node_ep_) {
        case SplineExtrapolation::noExtrapolation:
            throw AmiException(
                "Trying to evaluate spline sensitivity "
                "after last spline node, but spline has been specified "
                "to not allow extrapolation."
            );

        case SplineExtrapolation::constant:
            return coefficients_extrapolate_sensi[4 * ip + 2];

        case SplineExtrapolation::linear:
            return coefficients_extrapolate_sensi[4 * ip + 2]
                   + t * coefficients_extrapolate_sensi[4 * ip + 3];

        case SplineExtrapolation::polynomial:
            /* Evaluate last interpolation polynomial */
            i_node = n_nodes() - 2;
            len = nodes_[i_node + 1] - nodes_[i_node];
            return evaluate_polynomial(
                (t - nodes_[i_node]) / len,
                gsl::make_span(coefficients_sensi)
                    .subspan(ip * (n_nodes() - 1) * 4 + i_node * 4)
            );

        case SplineExtrapolation::periodic:
            len = nodes_[n_nodes() - 1] - nodes_[0];
            return get_sensitivity(
                nodes_[0] + std::fmod(t - nodes_[0], len), ip
            );
        default:
            throw AmiException("Unsupported SplineExtrapolation type");
        }
    }

    if (t < nodes_[0]) {
        /* Are we before the first node? Extrapolate! */
        switch (first_node_ep_) {
        case SplineExtrapolation::noExtrapolation:
            throw AmiException(
                "Trying to evaluate spline before first "
                "spline node, but spline has been specified to not allow "
                "extrapolation."
            );

        case SplineExtrapolation::constant:
            return coefficients_extrapolate_sensi[4 * ip + 0];

        case SplineExtrapolation::linear:
            return coefficients_extrapolate_sensi[4 * ip + 0]
                   + t * coefficients_extrapolate_sensi[4 * ip + 1];

        case SplineExtrapolation::polynomial:
            /* Evaluate last interpolation polynomial */
            len = nodes_[1] - nodes_[0];
            return evaluate_polynomial(
                (t - nodes_[0]) / len, gsl::make_span(coefficients_sensi)
                                           .subspan(ip * (n_nodes() - 1) * 4)
            );

        case SplineExtrapolation::periodic:
            len = nodes_[n_nodes() - 1] - nodes_[0];
            return get_sensitivity(
                nodes_[n_nodes() - 1] + std::fmod(t - nodes_[0], len), ip
            );
        default:
            throw AmiException("Unsupported SplineExtrapolation type");
        }
    }

    /* Get the spline interval which we need */
    if (get_equidistant_spacing()) {
        /* equidistant spacing: just compute the interval */
        len = nodes_[1] - nodes_[0];
        i_node = static_cast<int>(std::trunc((t - nodes_[0]) / len));
        i_node = std::min(i_node, n_nodes() - 2);
    } else {
        /* no equidistant spacing: we need to iterate */
        i_node = 0;
        while (nodes_[i_node + 1] < t) {
            i_node++;
        }
        len = nodes_[i_node + 1] - nodes_[i_node];
    }

    /* Evaluate the interpolation polynomial */
    return evaluate_polynomial(
        (t - nodes_[i_node]) / len,
        gsl::make_span(coefficients_sensi)
            .subspan(ip * (n_nodes() - 1) * 4 + i_node * 4)
    );
}

} // namespace amici
