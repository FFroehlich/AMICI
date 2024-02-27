"""Tests for petab_objective.py."""

from functools import partial
from pathlib import Path

import amici
import numpy as np
import pandas as pd
import petab
import pytest
from amici.petab.petab_import import import_petab_problem
from amici.petab.simulations import SLLH, simulate_petab
from amici.testing import skip_on_valgrind

# Absolute and relative tolerances for finite difference gradient checks.
ATOL: float = 1e-3
RTOL: float = 1e-3


@pytest.fixture
def lotka_volterra() -> petab.Problem:
    return petab.Problem.from_yaml(
        str(
            Path(__file__).parent
            / "petab_test_problems"
            / "lotka_volterra"
            / "petab"
            / "problem.yaml"
        )
    )


@skip_on_valgrind
def test_simulate_petab_sensitivities(lotka_volterra):
    petab_problem = lotka_volterra
    amici_model = import_petab_problem(petab_problem)
    amici_solver = amici_model.getSolver()

    amici_solver.setSensitivityOrder(amici.SensitivityOrder_first)
    amici_solver.setMaxSteps(int(1e5))

    problem_parameters = dict(
        zip(
            petab_problem.x_ids,
            petab_problem.x_nominal,
        )
    )

    results = {}
    for scaled_parameters in [True, False]:
        for scaled_gradients in [True, False]:
            _problem_parameters = problem_parameters.copy()
            if scaled_parameters:
                _problem_parameters = petab_problem.scale_parameters(
                    problem_parameters
                )
            results[(scaled_parameters, scaled_gradients)] = pd.Series(
                simulate_petab(
                    petab_problem=petab_problem,
                    amici_model=amici_model,
                    solver=amici_solver,
                    problem_parameters=_problem_parameters,
                    scaled_parameters=scaled_parameters,
                    scaled_gradients=scaled_gradients,
                )[SLLH]
            )

    # Computed previously, is the same as a central difference gradient
    # check, to >4 s.f.
    expected_results_scaled = pd.Series(
        {
            "alpha": -2.112626,
            "gamma": 21.388535,
        }
    )
    expected_results_unscaled = pd.Series(
        {
            "alpha": -0.458800,
            "gamma": 3.096308,
        }
    )

    assert_equal = partial(pd.testing.assert_series_equal, rtol=1e-3)

    # `scaled_gradients` affects gradients, `scaled_parameters` does not.
    assert_equal(results[(True, True)], expected_results_scaled)
    assert_equal(results[(False, True)], expected_results_scaled)

    assert_equal(results[(True, False)], expected_results_unscaled)
    assert_equal(results[(False, False)], expected_results_unscaled)

    # The test gradients are scaled correctly.
    assert_equal(
        results[(True, True)],
        results[(True, False)] * pd.Series(problem_parameters) * np.log(10),
    )
