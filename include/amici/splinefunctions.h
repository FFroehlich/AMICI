#ifndef AMICI_SPLINEFUNCTIONS_H
#define AMICI_SPLINEFUNCTIONS_H

#include "amici/defines.h"

#include <vector>

#include <gsl/gsl-lite.hpp>

namespace amici {

class Model;
/**
 * @brief AMICI spline base class.
 *
 * Instances of this class are created upon solver setup and the needed splines
 * are set up (e.g., interpolation of the nodes is performed).
 * Upon call to a spline function, only the evaluation of the spline polynomial
 * is carried out.
 */
class AbstractSpline {
  public:
    /** default constructor */
    AbstractSpline() = default;

    /**
     * @brief Common constructor for `AbstractSpline` instances.
     * @param nodes the nodes defining the position at which the value of
     * the spline is known
     * (if `equidistant_spacing` is true, it must contain only the first and
     * the last node; the other nodes will be automatically inserted,
     * assuming they are uniformly spaced)
     * @param node_values the values assumed by the spline at the nodes
     * @param equidistant_spacing whether equidistant nodes are to be computed
     * @param logarithmic_parametrization if true, the spline interpolation
     * will occur in log-space in order to ensure positivity of the interpolant
     * (which strictly speaking will no longer be a spline)
     */
    AbstractSpline(
        std::vector<realtype> nodes, std::vector<realtype> node_values,
        bool equidistant_spacing, bool logarithmic_parametrization
    );

    virtual ~AbstractSpline() = default;

    /**
     * @brief Compute the coefficients for all polynomial segments of this
     * spline
     */
    virtual void compute_coefficients() = 0;

    /**
     * @brief Compute the coefficients for all polynomial segments of
     * the derivatives of this spline with respect to the parameters
     * @param nplist number of parameters
     * @param spline_offset offset of this spline inside `dvaluesdp`
     * and `dslopesdp`
     * @param dvaluesdp derivatives of the spline values with respect to the
     * parameters (for all splines in the model, not just this one)
     * @param dslopesdp derivatives of the spline derivatives with respect
     * to the parameters (for all splines in the model, not just this one)
     * @remark The contents of `dvaluesdp` and `dslopesdp` may be modified
     * by this function.
     */
    virtual void compute_coefficients_sensi(
        int nplist, int spline_offset, gsl::span<realtype> dvaluesdp,
        gsl::span<realtype> dslopesdp
    ) = 0;

    /**
     * @brief Get the value of this spline at a given point
     * @param t point at which the spline is to be evaluated
     * @return value of the spline at `t`
     */
    realtype get_value(realtype const t) const;

    /**
     * @brief Get the value of this spline at a given point
     * in the scale in which interpolation is carried out (e.g., log-scale)
     * @param t point at which the spline is to be evaluated
     * @return scaled value of the spline at `t`
     */
    virtual realtype get_value_scaled(realtype const t) const = 0;

    /**
     * @brief Get the value of this spline at a given node
     * @param i index of the node at which the spline is to be evaluated
     * @return value of the spline at the `i`-th node
     */
    realtype get_node_value(int const i) const;

    /**
     * @brief Get the value of this spline at a given node
     * in the scale in which interpolation is carried out (e.g., log-scale)
     * @param i index of the node at which the spline is to be evaluated
     * @return scaled value of the spline at the `i`-th node
     */
    realtype get_node_value_scaled(int const i) const;

    /**
     * @brief Get the derivative of this spline with respect to a given
     * parameter at a given point
     * @param t point at which the sensitivity is to be evaluated
     * @param ip index of the parameter
     * @return sensitivity of the spline with respect to the `ip`th parameter
     * at `t`
     */
    realtype get_sensitivity(realtype const t, int const ip) const;

    /**
     * @brief Get the derivative of this spline with respect to a given
     * parameter at a given point
     * @param t point at which the sensitivity is to be evaluated
     * @param ip index of the parameter
     * @param value value of the spline at the given time point.
     *        It is used e.g. when interpolation is carried out in log-space.
     *        If omitted it will be computed.
     * @return sensitivity of the spline with respect to the `ip`th parameter
     * at `t`
     */
    realtype
    get_sensitivity(realtype const t, int const ip, realtype const value) const;

    /**
     * @brief Get the derivative of this spline with respect to a given
     * parameter at a given point
     * in the scale in which interpolation is carried out (e.g., log-scale)
     * @param t point at which the sensitivity is to be evaluated
     * @param ip index of the parameter
     * @return scaled sensitivity of the spline with respect to the `ip`th
     * parameter at `t`
     */
    virtual realtype
    get_sensitivity_scaled(realtype const t, int const ip) const
        = 0;

    /**
     * @brief Compute the limit value of the spline
     * as the evaluation point tends to positive infinity.
     */
    virtual void compute_final_value() = 0;

    /**
     * @brief Compute the limit of the value of the sensitivity
     * as the evaluation point tends to positive infinity.
     * @param nplist number of parameters
     * @param spline_offset offset of this spline inside `dspline_valuesdp`
     * and `dspline_slopesdp`
     * @param dspline_valuesdp derivatives of the spline values with respect to
     * the parameters (for all splines in the model, not just this one)
     * @param dspline_slopesdp derivatives of the spline derivatives with
     * respect to the parameters (for all splines in the model, not just this
     * one)
     */
    virtual void compute_final_sensitivity(
        int nplist, int spline_offset, gsl::span<realtype> dspline_valuesdp,
        gsl::span<realtype> dspline_slopesdp
    ) = 0;

    /**
     * @brief Get the limit value of the spline
     * as the evaluation point tends to positive infinity.
     * @return limit value
     */
    realtype get_final_value() const;

    /**
     * @brief Get the limit value of the spline
     * (in the scale in which interpolation is carried out)
     * as the evaluation point tends to positive infinity.
     * @return limit value
     */
    realtype get_final_value_scaled() const;

    /**
     * @brief Get the limit value of the sensitivity
     * with respect to the given parameter
     * as the evaluation point tends to positive infinity.
     * @param ip parameter index
     * @return limit value
     */
    realtype get_final_sensitivity(int const ip) const;

    /**
     * @brief Get the limit value of the sensitivity
     * with respect to the given parameter
     * (in the scale in which interpolation is carried out)
     * as the evaluation point tends to positive infinity.
     * @param ip parameter index
     * @return limit value
     */
    realtype get_final_sensitivity_scaled(int const ip) const;

    /**
     * @brief Whether nodes are uniformly spaced
     * @return boolean flag
     */
    bool get_equidistant_spacing() const;

    /**
     * @brief Whether spline interpolation is carried out in log-space
     * @return boolean flag
     */
    bool get_logarithmic_parametrization() const;

    /**
     * @brief The number of interpolation nodes for this spline
     * @return number of nodes
     */
    int n_nodes() const { return static_cast<int>(nodes_.size()); }

  protected:
    /**
     * @brief The nodes at which this spline is interpolated
     */
    std::vector<realtype> nodes_;

    /**
     * @brief The values the spline assumes at the nodes
     */
    std::vector<realtype> node_values_;

    /**
     * @brief Coefficients for each polynomial segment of the spline
     */
    std::vector<realtype> coefficients;

    /**
     * @brief Polynomial coefficients for the extrapolating the spline values
     */
    std::vector<realtype> coefficients_extrapolate;

    /**
     * @brief Coefficients for each polynomial segment of the sensitivities
     * with respect to the parameters
     */
    std::vector<realtype> coefficients_sensi;

    /**
     * @brief Polynomial coefficients for the extrapolating the sensitivities
     */
    std::vector<realtype> coefficients_extrapolate_sensi;

    /**
     * @brief Set the limit value of the spline
     * (in the scale in which interpolation is carried out)
     * as the evaluation point tends to positive infinity.
     * @param finalValue final value
     */
    void set_final_value_scaled(realtype finalValue);

    /**
     * @brief Set the limit value of the sensitivity
     * (in the scale in which interpolation is carried out)
     * as the evaluation point tends to positive infinity.
     * @param finalSensitivity final value of the sensitivity
     * for each parameter
     */
    void set_final_sensitivity_scaled(std::vector<realtype> finalSensitivity);

  private:
    realtype final_value_scaled_;

    std::vector<realtype> final_sensitivity_scaled_;

    bool equidistant_spacing_ = false;

    bool logarithmic_parametrization_ = false;

}; // class SplineFunction

/**
 * @brief AMICI Hermite spline class.
 *
 * Instances of this class represent Hermite splines,
 * which are uniquely determined by their nodes,
 * the values at their nodes, the derivatives at their nodes
 * (defaulting to finite difference approximations from the node values),
 * boundary conditions and extrapolation conditions.
 * Optionally, the spline can be defined in log-space in order
 * to ensure positivity.
 */
class HermiteSpline : public AbstractSpline {
  public:
    HermiteSpline() = default;

    /**
     * @brief Construct a `HermiteSpline`.
     * @param nodes the nodes defining the position at which the value of
     * the spline is known
     * (if `equidistant_spacing` is true, it must contain only the first and
     * the last node; the other nodes will be automatically inserted,
     * assuming they are uniformly spaced)
     * @param node_values the values assumed by the spline at the nodes
     * @param node_values_derivative the derivatives of the spline at the nodes
     * (if `node_derivative_by_FD` is true, it will resized and filled with
     * finite difference approximations computed from `node_values`)
     * @param firstNodeBC boundary condition at the first node
     * @param lastNodeBC boundary condition at the last node
     * @param firstNodeExtrapol extrapolation method on the left side
     * @param lastNodeExtrapol extrapolation method on the right side
     * @param node_derivative_by_FD whether derivatives are to be computed by
     * finite differences
     * @param equidistant_spacing whether equidistant nodes are to be computed
     * @param logarithmic_parametrization if true, the spline interpolation
     * will occur in log-space in order to ensure positivity of the interpolant
     * (which strictly speaking will no longer be a spline)
     */
    HermiteSpline(
        std::vector<realtype> nodes, std::vector<realtype> node_values,
        std::vector<realtype> node_values_derivative,
        SplineBoundaryCondition firstNodeBC, SplineBoundaryCondition lastNodeBC,
        SplineExtrapolation firstNodeExtrapol,
        SplineExtrapolation lastNodeExtrapol, bool node_derivative_by_FD,
        bool equidistant_spacing, bool logarithmic_parametrization
    );

    void compute_coefficients() override;

    void compute_coefficients_sensi(
        int nplist, int spline_offset, gsl::span<realtype> dvaluesdp,
        gsl::span<realtype> dslopesdp
    ) override;

    void compute_final_value() override;

    void compute_final_sensitivity(
        int nplist, int spline_offset, gsl::span<realtype> dspline_valuesdp,
        gsl::span<realtype> dspline_slopesdp
    ) override;

    realtype get_value_scaled(realtype const t) const override;

    /**
     * @brief Get the derivative of the spline at a given node
     * @param i index of the node at which the spline is to be evaluated
     * @return value of the derivative at the `i`-th node
     */
    realtype get_node_derivative(int const i) const;

    /**
     * @brief Get the derivative of the spline at a given node
     * in the scale in which interpolation is carried out (e.g., log-scale)
     * @param i index of the node at which the spline is to be evaluated
     * @return scaled value of the derivative at the `i`-th node
     */
    realtype get_node_derivative_scaled(int const i) const;

    realtype
    get_sensitivity_scaled(realtype const t, int const ip) const override;

    /**
     * @brief Whether derivatives of this spline are computed
     * by finite differences
     * @return boolean flag
     */
    bool get_node_derivative_by_fd() const { return node_derivative_by_FD_; }

  private:
    void compute_slope_sensitivities_by_fd(
        int nplist, int spline_offset, int ip, gsl::span<realtype> dvaluesdp,
        gsl::span<realtype> dslopesdp
    );

    void get_coeffs_sensi_lowlevel(
        int ip, int i_node, int nplist, int n_spline_coefficients,
        int spline_offset, realtype len, gsl::span<realtype> dnodesdp,
        gsl::span<realtype> dslopesdp, gsl::span<realtype> coeffs
    ) const;

    void handle_inner_derivatives();

    void handle_boundary_conditions();

    void compute_coefficients_extrapolation();

    void compute_coefficients_extrapolation_sensi(
        int nplist, int spline_offset, gsl::span<realtype> dspline_valuesdp,
        gsl::span<realtype> dspline_slopesdp
    );

    std::vector<realtype> node_values_derivative_;

    SplineBoundaryCondition first_node_bc_ = SplineBoundaryCondition::given;

    SplineBoundaryCondition last_node_bc_ = SplineBoundaryCondition::given;

    SplineExtrapolation first_node_ep_ = SplineExtrapolation::linear;

    SplineExtrapolation last_node_ep_ = SplineExtrapolation::linear;

    bool node_derivative_by_FD_ = false;

}; // class HermiteSpline

} // namespace amici

#endif /* AMICI_SPLINEFUNCTIONS_H */
