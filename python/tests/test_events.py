"""Tests for SBML events, including piecewise expressions."""
from copy import deepcopy

import amici
import numpy as np
import pytest
from amici.antimony_import import antimony2amici
from amici.gradient_check import check_derivatives
from amici.testing import TemporaryDirectoryWinSafe as TemporaryDirectory
from amici.testing import skip_on_valgrind
from util import (
    check_trajectories_with_forward_sensitivities,
    check_trajectories_without_sensitivities,
    create_amici_model,
    create_sbml_model,
)


@pytest.fixture(
    params=[
        pytest.param("events_plus_heavisides", marks=skip_on_valgrind),
        pytest.param(
            "piecewise_plus_event_simple_case", marks=skip_on_valgrind
        ),
        pytest.param(
            "piecewise_plus_event_semi_complicated", marks=skip_on_valgrind
        ),
        pytest.param(
            "piecewise_plus_event_trigger_depends_on_state",
            marks=skip_on_valgrind,
        ),
        pytest.param("nested_events", marks=skip_on_valgrind),
        pytest.param("event_state_dep_ddeltax_dtpx", marks=skip_on_valgrind),
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


def get_model_definition(model_name):
    if model_name == "piecewise_plus_event_simple_case":
        return model_definition_piecewise_plus_event_simple_case()
    if model_name == "piecewise_plus_event_semi_complicated":
        return model_definition_piecewise_plus_event_semi_complicated()
    if model_name == "piecewise_plus_event_trigger_depends_on_state":
        return model_definition_piecewise_plus_event_trigger_depends_on_state()
    if model_name == "events_plus_heavisides":
        return model_definition_events_plus_heavisides()
    if model_name == "nested_events":
        return model_definition_nested_events()
    if model_name == "event_state_dep_ddeltax_dtpx":
        return model_definition_event_state_dep_ddeltax_dtpx()

    raise NotImplementedError(
        f"Model with name {model_name} is not implemented."
    )


def model_definition_events_plus_heavisides():
    """Test model for state- and parameter-dependent Heavisides.

    ODEs
    ----
    d/dt x_1:
        - {            0,    t <  delta
        - { -alpha * x_1,    t >= delta
    d/dt x_2:
        - beta * x_1 - gamma * x_2
    d/dt x_3:
        - {     -eta * x_3,    t <  zeta
        - { -eta * x_3 + 1,    t >= zeta

    Events:
    -------
    event_1:
        trigger: k1 - x_3
        bolus: [[ -x_3 / 2],
                [        0],
                [        0]]
    event_2:
        trigger: t - zeta
        bolus: [[        0],
                [        0],
                [ zeta / 3]]
    """
    # Model components
    species = ["x_1", "x_2", "x_3"]
    initial_assignments = {
        "x_1": "k1",
        "x_2": "k2",
        "x_3": "k3",
    }
    rate_rules = {
        "x_1": "piecewise( -alpha * x_1, time >= delta, 0)",
        "x_2": "beta * x_1 - gamma * x_2",
        "x_3": "-eta * x_3 + piecewise( 1, time >= zeta, 0)",
    }
    parameters = {
        "k1": 2,
        "k2": 0.01,
        "k3": 5,
        "alpha": 2,
        "beta": 3,
        "gamma": 2,
        "delta": 3,
        "eta": 1,
        "zeta": 5,
    }
    events = {
        "event_1": {
            "trigger": "x_3 < k1",
            "target": "x_1",
            "assignment": "x_1 - x_3 / 2",
        },
        "event_2": {
            "trigger": "time >= zeta",
            "target": "x_3",
            "assignment": "x_3 + zeta / 3",
        },
    }
    timepoints = np.linspace(0, 8, 400)

    # Analytical solution
    def x_expected(t, k1, k2, k3, alpha, beta, gamma, delta, eta, zeta):
        # The system reads dx/dt = Ax + b
        # x0 = (k1, k2, k3)
        x0 = np.array([[k1], [k2], [k3]])

        # gather event time points
        event_1_time = (np.log(k3) - np.log(k1)) / eta  # k1 > x3
        event_2_time = delta
        event_3_time = zeta

        def get_early_x(t):
            # compute dynamics
            if t < event_1_time:
                # Define A
                A = np.array([[0, 0, 0], [beta, -gamma, 0], [0, 0, -eta]])
                tmp_x = expm(t * A)
                return np.matmul(tmp_x, x0)

            elif t <= event_2_time:
                # "simulate" until first event
                A = np.array([[0, 0, 0], [beta, -gamma, 0], [0, 0, -eta]])
                tmp_x = expm(event_1_time * A)
                x1 = np.matmul(tmp_x, x0)
                # apply bolus
                delta_x = np.array([[float(-x1[2, 0] / 2)], [0], [0]])
                x1 += delta_x
                # "simulate" on
                tmp_x = expm((t - event_1_time) * A)
                return np.matmul(tmp_x, x1)

        if t < event_2_time:
            x = get_early_x(t).flatten()
        elif t < event_3_time:
            x2 = get_early_x(event_2_time)

            A = np.array([[-alpha, 0, 0], [beta, -gamma, 0], [0, 0, -eta]])
            tmp_x = expm((t - event_2_time) * A)
            x = np.matmul(tmp_x, x2).flatten()
        else:
            x2 = get_early_x(event_2_time)

            A = np.array([[-alpha, 0, 0], [beta, -gamma, 0], [0, 0, -eta]])
            tmp_x = expm((event_3_time - event_2_time) * A)
            x3 = np.matmul(tmp_x, x2)
            # apply bolus
            x3 += np.array([[0], [0], [zeta / 3]])

            hom_x = np.matmul(expm((t - event_3_time) * A), x3)
            inhom_x = [
                [0],
                [0],
                [-np.exp(-eta * (t - event_3_time)) / eta + 1 / eta],
            ]

            x = (hom_x + inhom_x).flatten()

        return np.array(x)

    def sx_expected(t, parameters):
        """get sx, w.r.t. parameters, via finite differences"""
        sx = []
        eps = 1e-6

        for ip in parameters:
            perturbed_params = deepcopy(parameters)
            perturbed_params[ip] += eps
            sx_p = x_expected(t, **perturbed_params)
            perturbed_params[ip] -= 2 * eps
            sx_m = x_expected(t, **perturbed_params)
            sx.append((sx_p - sx_m) / (2 * eps))

        return np.array(sx)

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


def model_definition_nested_events():
    """Test model for state- and parameter-dependent heavisides.

    ODEs
    ----
    d/dt x_1:
        inflow_1 - decay_1 * x1
    d/dt x_2:
        - decay_2 * x_2

    Events:
    -------
    event_1:
        trigger: x_1 > inflow_1 / decay_2
        bolus: [[          0],
                [ -1 / time]]
    event_2:
        trigger: x_2 > 0.5
        bolus: [[ bolus],
                [ bolus]]
    """
    # Model components
    species = ["x_1", "x_2"]
    initial_assignments = {
        "x_1": "k1",
        "x_2": "k2",
    }
    rate_rules = {
        "x_1": "inflow_1 - decay_1 * x_1",
        "x_2": "- decay_2 * x_2",
    }
    parameters = {
        "k1": 0,
        "k2": 0,
        "inflow_1": 4,
        "decay_1": 2,
        "decay_2": 5,
        "bolus": 0,  # for bolus != 0, nested event sensitivities are off!
    }
    events = {
        "event_1": {
            "trigger": "x_1 > inflow_1 / decay_2",
            "target": "x_2",
            "assignment": "x_2 - 1 / time",
        },
        "event_2": {
            "trigger": "x_2 < - 0.5",
            "target": ["x_1", "x_2"],
            "assignment": ["x_1 + bolus", "x_2 + bolus"],
        },
    }
    timepoints = np.linspace(0, 1, 101)

    # Analytical solution
    def x_expected(t, k1, k2, inflow_1, decay_1, decay_2, bolus):
        # gather temporary variables
        # event_time = x_1 > inflow_1 / decay_2
        equil = inflow_1 / decay_1
        tmp1 = inflow_1 / decay_2 - inflow_1 / decay_1
        tmp2 = k1 - inflow_1 / decay_1
        event_time = (-1 / decay_1) * np.log(tmp1 / tmp2)

        def get_early_x(t):
            # compute dynamics before event
            x_1 = equil * (1 - np.exp(-decay_1 * t)) + k1 * np.exp(
                -decay_1 * t
            )
            x_2 = k2 * np.exp(-decay_2 * t)
            return np.array([[x_1], [x_2]])

        if t < event_time:
            x = get_early_x(t).flatten()
        else:
            # compute state after event
            x_tau = get_early_x(event_time)
            tau_x1 = x_tau[0] + bolus
            tau_x2 = x_tau[1] - 1 / event_time + bolus

            # compute dynamics after event
            inhom = np.exp(decay_1 * event_time) * tau_x1
            x_1 = equil * (
                1 - np.exp(decay_1 * (event_time - t))
            ) + inhom * np.exp(-decay_1 * t)
            x_2 = tau_x2 * np.exp(decay_2 * event_time) * np.exp(-decay_2 * t)

            x = np.array([[x_1], [x_2]])

        return x.flatten()

    def sx_expected(t, parameters):
        """get sx, w.r.t. parameters, via finite differences"""
        sx = []
        eps = 1e-6

        for ip in parameters:
            perturbed_params = deepcopy(parameters)
            perturbed_params[ip] += eps
            sx_p = x_expected(t, **perturbed_params)
            perturbed_params[ip] -= 2 * eps
            sx_m = x_expected(t, **perturbed_params)
            sx.append((sx_p - sx_m) / (2 * eps))

        return np.array(sx)

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


def model_definition_piecewise_plus_event_simple_case():
    """Test model for boolean operations in a piecewise condition.

    ODEs
    ----
    d/dt x_1:
        - { 1,    (alpha <= t and t < beta)
        - { 0,    otherwise
    """
    # Model components
    species = ["x_1"]
    initial_assignments = {"x_1": "x_1_0"}
    rate_rules = {"x_1": "piecewise(1, (alpha < time && time < beta), 0)"}
    parameters = {
        "alpha": 2,
        "beta": 3,
        "gamma": 4.5,
        "x_1_0": 1,
    }
    timepoints = np.linspace(0.0, 5.0, 100)  # np.array((0.0, 4.0,))
    events = {
        "event_1": {
            "trigger": "time > alpha",
            "target": "x_1",
            "assignment": "gamma",
        },
        "event_2": {
            "trigger": "time > beta",
            "target": "x_1",
            "assignment": "x_1 + 2.5",
        },
    }

    # Analytical solution
    def x_expected(t, x_1_0, alpha, beta, gamma):
        t_event_1 = alpha
        t_event_2 = beta

        if t < t_event_1:
            x = x_1_0
        elif t < t_event_2:
            x = gamma + t - t_event_1
        else:
            x = gamma + t_event_2 - t_event_1 + 2.5

        return np.array((x,))

    def sx_expected(t, parameters):
        """get sx, w.r.t. parameters, via finite differences"""
        sx = []
        eps = 1e-6

        for ip in parameters:
            perturbed_params = deepcopy(parameters)
            perturbed_params[ip] += eps
            sx_p = np.array(x_expected(t, **perturbed_params))
            perturbed_params[ip] -= 2 * eps
            sx_m = np.array(x_expected(t, **perturbed_params))
            sx.append((sx_p - sx_m) / (2 * eps))

        return np.array(sx)

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


def model_definition_event_state_dep_ddeltax_dtpx():
    """Test model with state-dependent partial derivatives of update functions wrt parameters, time, and states."""
    # Model components
    species = ["x_1"]
    initial_assignments = {"x_1": "x_1_0"}
    rate_rules = {"x_1": "1"}
    parameters = {
        "alpha": 1.5,
        "beta": 2.5,
        "gamma": 3.5,
        "delta": 5.5,
        "x_1_0": 1,
    }
    timepoints = np.linspace(0.0, 5.0, 100)
    events = {
        # state-dependent ddeltaxdt
        "event_1": {
            "trigger": "time > alpha",
            "target": "x_1",
            "assignment": "x_1 * time",
        },
        # state-dependent ddeltaxdp
        "event_2": {
            "trigger": "time > beta",
            "target": "x_1",
            "assignment": "x_1 * delta",
        },
        # state-dependent ddeltaxdx
        "event_3": {
            "trigger": "time > gamma",
            "target": "x_1",
            "assignment": "2 * x_1 * x_1",
        },
    }

    # Analytical solution
    def x_expected(t, x_1_0, alpha, beta, gamma, delta):
        if t < alpha:
            # before first event triggered
            x = x_1_0 + t
        elif t < beta:
            # after first event triggered
            x = (x_1_0 + alpha) * alpha + (t - alpha)
        elif t < gamma:
            # after second event triggered
            x = ((x_1_0 + alpha) * alpha + (beta - alpha)) * delta + (t - beta)
        else:
            # after third event triggered
            x = (
                ((x_1_0 + alpha) * alpha + (beta - alpha)) * delta
                + (gamma - beta)
            ) ** 2 * 2 + (t - gamma)

        return np.array((x,))

    def sx_expected(t, parameters):
        """get sx, w.r.t. parameters, via finite differences"""
        sx = []
        eps = 1e-6

        for ip in parameters:
            perturbed_params = deepcopy(parameters)
            perturbed_params[ip] += eps
            sx_p = np.array(x_expected(t, **perturbed_params))
            perturbed_params[ip] -= 2 * eps
            sx_m = np.array(x_expected(t, **perturbed_params))
            sx.append((sx_p - sx_m) / (2 * eps))

        return np.array(sx)

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


def model_definition_piecewise_plus_event_semi_complicated():
    """Test model for boolean operations in a piecewise condition, discrete
    events and a non-vanishing quadrature for the adjoint state.
    """
    # Model components
    species = ["x_1", "x_2"]
    initial_assignments = {"x_1": "x_1_0", "x_2": "x_2_0"}
    rate_rules = {
        "x_1": "piecewise(delta * x_1, (alpha < time && time < beta), - x_1)",
        "x_2": "- eta * x_2",
    }
    parameters = {
        "alpha": 2,
        "beta": 3,
        "gamma": 4.5,
        "x_1_0": 1,
        "x_2_0": 5,
        "delta": 2.5,
        "eta": 1.4,
    }
    timepoints = np.linspace(0.0, 5.0, 100)
    events = {
        "event_1": {
            "trigger": "time > alpha / 2",
            "target": "x_1",
            "assignment": "gamma",
        },
        "event_2": {
            "trigger": "time > beta",
            "target": "x_1",
            "assignment": "x_1 + x_2",
        },
    }

    # Analytical solution
    def x_expected(t, x_1_0, x_2_0, alpha, beta, gamma, delta, eta):
        t_event_1 = alpha / 2
        t_event_2 = beta
        heaviside_1 = alpha

        x_2 = x_2_0 * np.exp(-eta * t)

        if t < t_event_1:
            x_1 = x_1_0 * np.exp(-t)
        elif t < heaviside_1:
            x_1 = gamma * np.exp(-(t - t_event_1))
        elif t < t_event_2:
            x_1_heaviside_1 = gamma * np.exp(-(heaviside_1 - t_event_1))
            x_1 = x_1_heaviside_1 * np.exp(delta * (t - heaviside_1))
        else:
            x_1_heaviside_1 = gamma * np.exp(-(heaviside_1 - t_event_1))
            x_1_at_event_2 = x_1_heaviside_1 * np.exp(
                delta * (t_event_2 - heaviside_1)
            )
            x_2_at_event_2 = x_2_0 * np.exp(-eta * t_event_2)
            x1_after_event_2 = x_1_at_event_2 + x_2_at_event_2
            x_1 = x1_after_event_2 * np.exp(-(t - t_event_2))

        return np.array((x_1, x_2))

    def sx_expected(t, parameters):
        """get sx, w.r.t. parameters, via finite differences"""
        sx = []
        eps = 1e-6

        for ip in parameters:
            perturbed_params = deepcopy(parameters)
            perturbed_params[ip] += eps
            sx_p = np.array(x_expected(t, **perturbed_params))
            perturbed_params[ip] -= 2 * eps
            sx_m = np.array(x_expected(t, **perturbed_params))
            sx.append((sx_p - sx_m) / (2 * eps))

        return np.array(sx)

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


def model_definition_piecewise_plus_event_trigger_depends_on_state():
    """Test model for boolean operations in a piecewise condition.

    ODEs
    ----
    d/dt x_1:
        - { 1,    (alpha <= t and t < beta)
        - { 0,    otherwise
    """
    # Model components
    species = ["x_1", "x_2"]
    initial_assignments = {"x_1": "x_1_0", "x_2": "x_2_0"}
    rate_rules = {
        "x_1": "piecewise(1, (alpha < time && time < beta), 0)",
        "x_2": "- x_2",
    }
    parameters = {
        "alpha": 2,
        "beta": 3,
        "gamma": 4.5,
        "x_1_0": 1,
        "x_2_0": 5,
    }
    timepoints = np.linspace(0.0, 5.0, 100)
    events = {
        "event_1": {
            "trigger": "x_1 > 1.4",
            "target": "x_1",
            "assignment": "x_1 + gamma",
        },
        "event_2": {
            "trigger": "time > beta",
            "target": "x_1",
            "assignment": "x_1 + x_2",
        },
    }

    # Analytical solution
    def x_expected(t, x_1_0, x_2_0, alpha, beta, gamma):
        heaviside_1 = alpha
        t_event_1 = alpha + 1.4 - x_1_0
        t_event_2 = beta
        # This should hold in order that the analytical solution is correct
        assert heaviside_1 < t_event_1

        # x_2 never gets perturbed
        x_2 = x_2_0 * np.exp(-t)

        if t < heaviside_1:
            x_1 = x_1_0
        elif t < t_event_1:
            x_1 = (t - heaviside_1) + x_1_0
        elif t < t_event_2:
            x_1 = gamma + (t - heaviside_1) + x_1_0
        else:
            x_2_at_event_2 = x_2_0 * np.exp(-t_event_2)
            x_1_at_event_2 = gamma + (t_event_2 - heaviside_1) + x_1_0
            x_1 = x_1_at_event_2 + x_2_at_event_2

        return np.array((x_1, x_2))

    def sx_expected(t, parameters):
        """get sx, w.r.t. parameters, via finite differences"""
        sx = []
        eps = 1e-6

        for ip in parameters:
            perturbed_params = deepcopy(parameters)
            perturbed_params[ip] += eps
            sx_p = np.array(x_expected(t, **perturbed_params))
            perturbed_params[ip] -= 2 * eps
            sx_m = np.array(x_expected(t, **perturbed_params))
            sx.append((sx_p - sx_m) / (2 * eps))

        return np.array(sx)

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


def test_models(model):
    amici_model, parameters, timepoints, x_expected, sx_expected = model

    result_expected_x = np.array(
        [x_expected(t, **parameters) for t in timepoints]
    )
    result_expected_sx = np.array(
        [sx_expected(t, parameters) for t in timepoints]
    )

    # assert correctness of trajectories
    check_trajectories_without_sensitivities(amici_model, result_expected_x)
    check_trajectories_with_forward_sensitivities(
        amici_model, result_expected_x, result_expected_sx
    )


def expm(x):
    """``expm`` wrapper

    Uses ``expm`` from ``mpmath``. *Something* changed in scipy's ``expm`` in
    version 1.9.0 breaking these tests"""
    from mpmath import expm

    return np.array(expm(x).tolist()).astype(float)


def test_handling_of_fixed_time_point_event_triggers():
    """Test handling of events without solver-tracked root functions."""
    ant_model = """
    model test_events_time_based
        event_target = 0
        bolus = 1
        at (time > 1): event_target = 1
        at (time > 2): event_target = event_target + bolus
        at (time > 3): event_target = 3
    end
    """
    module_name = "test_events_time_based"
    with TemporaryDirectory(prefix=module_name, delete=False) as outdir:
        antimony2amici(
            ant_model,
            model_name=module_name,
            output_dir=outdir,
            verbose=True,
        )
        model_module = amici.import_model_module(
            module_name=module_name, module_path=outdir
        )
        amici_model = model_module.getModel()
        assert amici_model.ne == 3
        assert amici_model.ne_solver == 0
        amici_model.setTimepoints(np.linspace(0, 4, 200))
        amici_solver = amici_model.getSolver()
        rdata = amici.runAmiciSimulation(amici_model, amici_solver)
        assert rdata.status == amici.AMICI_SUCCESS
        assert (rdata.x[rdata.ts < 1] == 0).all()
        assert (rdata.x[(rdata.ts >= 1) & (rdata.ts < 2)] == 1).all()
        assert (rdata.x[(rdata.ts >= 2) & (rdata.ts < 3)] == 2).all()
        assert (rdata.x[(rdata.ts >= 3)] == 3).all()

        check_derivatives(amici_model, amici_solver, edata=None)
