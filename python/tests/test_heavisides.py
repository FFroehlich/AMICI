"""Tests for SBML events, including piecewise expressions."""

import numpy as np
import pytest
from util import (
    check_trajectories_with_adjoint_sensitivities,
    check_trajectories_with_forward_sensitivities,
    check_trajectories_without_sensitivities,
    create_amici_model,
    create_sbml_model,
)

pytestmark = pytest.mark.filterwarnings(
    # https://github.com/AMICI-dev/AMICI/issues/18
    "ignore:Adjoint sensitivity analysis for models with discontinuous "
    "right hand sides .*:UserWarning",
)


@pytest.fixture(
    params=[
        "state_and_param_dep_heavisides",
        "piecewise_with_boolean_operations",
        "piecewise_many_conditions",
    ]
)
def model(request):
    """Returns the requested AMICI model and analytical expressions."""
    (
        initial_assignments,
        parameters,
        rate_rules,
        species,
        events,
        timepoints,
        x_expected,
        sx_expected,
    ) = get_model_definition(request.param)

    # SBML model
    sbml_document, sbml_model = create_sbml_model(
        initial_assignments=initial_assignments,
        parameters=parameters,
        rate_rules=rate_rules,
        species=species,
        events=events,
        # uncomment `to_file` to save SBML model to file for inspection
        # to_file=sbml_test_models / (model_name + '.sbml'),
    )

    # AMICI model
    amici_model = create_amici_model(
        sbml_model=sbml_model,
        model_name=request.param,
    )
    amici_model.setTimepoints(timepoints)

    return amici_model, parameters, timepoints, x_expected, sx_expected


def test_models(model):
    amici_model, parameters, timepoints, x_expected, sx_expected = model

    result_expected_x = np.array(
        [x_expected(t, **parameters) for t in timepoints]
    )
    result_expected_sx = np.array(
        [sx_expected(t, **parameters) for t in timepoints]
    )

    # Does the AMICI simulation match the analytical solution?
    check_trajectories_without_sensitivities(amici_model, result_expected_x)
    check_trajectories_with_forward_sensitivities(
        amici_model, result_expected_x, result_expected_sx
    )
    check_trajectories_with_adjoint_sensitivities(amici_model)


def get_model_definition(model_name):
    if model_name == "state_and_param_dep_heavisides":
        return model_definition_state_and_parameter_dependent_heavisides()
    elif model_name == "piecewise_with_boolean_operations":
        return model_definition_piecewise_with_boolean_operations()
    elif model_name == "piecewise_many_conditions":
        return model_definition_piecewise_many_conditions()
    else:
        raise NotImplementedError(
            f"Model with name {model_name} is not implemented."
        )


def model_definition_state_and_parameter_dependent_heavisides():
    """Test model for state- and parameter-dependent heavisides.

    ODEs
    ----
    d/dt x_1:
        - { alpha * x_1,    t <  x_2
        - { -beta * x_1,    t >= x_2
    d/dt x_2:
        - { gamma * x_2,    t <  delta
        - {         eta,    t >= delta
    """
    # Model components
    species = ["x_1", "x_2"]
    initial_assignments = {
        "x_1": "zeta",
    }
    rate_rules = {
        "x_1": "piecewise( alpha * x_1, time < x_2,   -beta * x_1 )",
        "x_2": "piecewise( gamma * x_2, time < delta,  eta        )",
    }
    parameters = {
        "alpha": float(np.log(2)),
        "beta": float(np.log(4)),
        "gamma": float(np.log(3)),
        "delta": 1,
        "eta": 0.5,
        "zeta": 0.25,
    }
    timepoints = np.linspace(0, 10, 100)
    events = {}

    # Analytical solution
    def x_expected(t, alpha, beta, gamma, delta, eta, zeta):
        # get x_1
        tau_1 = (np.exp(gamma * delta) - delta * eta) / (1 - eta)
        if t < tau_1:
            x_1 = zeta * np.exp(alpha * t)
        else:
            x_1 = zeta * np.exp(alpha * tau_1 - beta * (t - tau_1))

        # get x_2
        tau_2 = delta
        if t < tau_2:
            x_2 = np.exp(gamma * t)
        else:
            x_2 = np.exp(gamma * delta) + eta * (t - delta)

        return x_1, x_2

    def sx_expected(t, alpha, beta, gamma, delta, eta, zeta):
        # get sx_1, w.r.t. parameters
        tau_1 = (np.exp(gamma * delta) - delta * eta) / (1 - eta)
        if t < tau_1:
            sx_1_alpha = zeta * t * np.exp(alpha * t)
            sx_1_beta = 0
            sx_1_gamma = 0
            sx_1_delta = 0
            sx_1_eta = 0
            sx_1_zeta = np.exp(alpha * t)
        else:
            # Never trust Wolfram Alpha...
            sx_1_alpha = (
                zeta * tau_1 * np.exp(alpha * tau_1 - beta * (t - tau_1))
            )
            sx_1_beta = (
                zeta * (tau_1 - t) * np.exp(alpha * tau_1 - beta * (t - tau_1))
            )
            sx_1_gamma = (
                zeta
                * (alpha + beta)
                * delta
                * np.exp(gamma * delta)
                / (1 - eta)
                * np.exp(alpha * tau_1 - beta * (t - tau_1))
            )
            sx_1_delta = (
                zeta
                * (alpha + beta)
                * np.exp(alpha * tau_1 - beta * (t - tau_1))
                * (gamma * np.exp(gamma * delta) - eta)
                / (1 - eta)
            )
            sx_1_eta = (
                zeta
                * (alpha + beta)
                * (-delta * (1 - eta) + np.exp(gamma * delta) - delta * eta)
                / (1 - eta) ** 2
                * np.exp(alpha * tau_1 - beta * (t - tau_1))
            )
            sx_1_zeta = np.exp(alpha * tau_1 - beta * (t - tau_1))

        # get sx_2, w.r.t. parameters
        tau_2 = delta
        sx_2_alpha = 0
        sx_2_beta = 0
        sx_2_zeta = 0
        if t < tau_2:
            sx_2_gamma = t * np.exp(gamma * t)
            sx_2_delta = 0
            sx_2_eta = 0
        else:
            sx_2_gamma = delta * np.exp(gamma * delta)
            sx_2_delta = gamma * np.exp(gamma * delta) - eta
            sx_2_eta = t - delta

        sx_1 = (
            sx_1_alpha,
            sx_1_beta,
            sx_1_gamma,
            sx_1_delta,
            sx_1_eta,
            sx_1_zeta,
        )
        sx_2 = (
            sx_2_alpha,
            sx_2_beta,
            sx_2_gamma,
            sx_2_delta,
            sx_2_eta,
            sx_2_zeta,
        )

        return np.array((sx_1, sx_2)).transpose()

    return (
        initial_assignments,
        parameters,
        rate_rules,
        species,
        events,
        timepoints,
        x_expected,
        sx_expected,
    )


def model_definition_piecewise_with_boolean_operations():
    """Test model for boolean operations in a piecewise condition.

    ODEs
    ----
    d/dt x_1:
        - { 1,    (alpha <= t and t < beta) or (gamma <= t and t < delta)
        - { 0,    otherwise
    """
    # Model components
    species = ["x_1"]
    initial_assignments = {"x_1": "x_1_0"}
    rate_rules = {
        "x_1": (
            "piecewise("
            "1, "  # noqa
            "(alpha <= time && time < beta) || "  # noqa
            "(gamma <= time && time < delta), "
            "0"
            ")"
        ),
    }
    parameters = {
        "alpha": 1,
        "beta": 2,
        "gamma": 3,
        "delta": 4,
        "x_1_0": 1,
    }
    timepoints = np.linspace(0, 5, 100)
    events = {}

    # Analytical solution
    def x_expected(t, x_1_0, alpha, beta, gamma, delta):
        if t < alpha:
            return (x_1_0,)
        elif alpha <= t < beta:
            return (x_1_0 + (t - alpha),)
        elif beta <= t < gamma:
            return (x_1_0 + (beta - alpha),)
        elif gamma <= t < delta:
            return (x_1_0 + (beta - alpha) + (t - gamma),)
        else:
            return (x_1_0 + (beta - alpha) + (delta - gamma),)

    def sx_expected(t, x_1_0, alpha, beta, gamma, delta):
        # x0 is very simple...
        sx_x0 = 1
        sx_alpha = -1 if t >= alpha else 0
        sx_beta = 1 if t >= beta else 0
        sx_gamma = -1 if t >= gamma else 0
        sx_delta = 1 if t >= delta else 0
        sx = (sx_alpha, sx_beta, sx_gamma, sx_delta, sx_x0)
        return np.array((sx,)).transpose()

    return (
        initial_assignments,
        parameters,
        rate_rules,
        species,
        events,
        timepoints,
        x_expected,
        sx_expected,
    )


def model_definition_piecewise_many_conditions():
    """Test model for piecewise functions with many pieces.

    ODEs
    ----
    d/dt x_1:
        - { 1,    floor(t) is odd
        - { 0,    otherwise
    """
    # Model components
    species = ["x_1"]
    initial_assignments = {"x_1": "x_1_0"}
    t_final = 5

    pieces = "piecewise("
    for t in range(t_final):
        if t > 0:
            pieces += ", "
        if t % 2 == 1:
            pieces += f"1, time < {t + 1}"
        else:
            pieces += f"0, time < {t + 1}"
    pieces += ", 0)"
    rate_rules = {
        "x_1": pieces,
    }

    parameters = {
        "x_1_0": 1,
    }
    timepoints = np.linspace(0, t_final, 100)
    events = {}

    # Analytical solution
    def x_expected(t, x_1_0):
        if np.floor(t) % 2 == 1:
            return (x_1_0 + (np.floor(t) - 1) / 2 + (t - np.floor(t)),)
        else:
            return (x_1_0 + np.floor(t) / 2,)

    def sx_expected(t, x_1_0):
        return np.array(
            [
                [
                    1,
                ],
            ]
        )

    return (
        initial_assignments,
        parameters,
        rate_rules,
        species,
        events,
        timepoints,
        x_expected,
        sx_expected,
    )
