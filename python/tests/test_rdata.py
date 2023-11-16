"""Test amici.ReturnData(View)-related functionality"""
import amici
import numpy as np
import pytest
from amici.numpy import evaluate
from numpy.testing import assert_almost_equal, assert_array_equal


@pytest.fixture(scope="session")
def rdata_by_id_fixture(sbml_example_presimulation_module):
    model_module = sbml_example_presimulation_module
    model = model_module.getModel()
    model.setTimepoints(np.linspace(0, 60, 61))
    solver = model.getSolver()
    solver.setSensitivityMethod(amici.SensitivityMethod.forward)
    solver.setSensitivityOrder(amici.SensitivityOrder.first)
    rdata = amici.runAmiciSimulation(model, solver)
    assert rdata.status == amici.AMICI_SUCCESS
    return model, rdata


def test_rdata_by_id(rdata_by_id_fixture):
    model, rdata = rdata_by_id_fixture

    assert_array_equal(rdata.by_id(model.getStateIds()[1]), rdata.x[:, 1])
    assert_array_equal(rdata.by_id(model.getStateIds()[1], "x"), rdata.x[:, 1])
    assert_array_equal(
        rdata.by_id(model.getStateIds()[1], "x", model), rdata.x[:, 1]
    )

    assert_array_equal(
        rdata.by_id(model.getObservableIds()[0], "y", model), rdata.y[:, 0]
    )

    assert_array_equal(rdata.by_id(model.getExpressionIds()[1]), rdata.w[:, 1])
    assert_array_equal(
        rdata.by_id(model.getExpressionIds()[1], "w", model), rdata.w[:, 1]
    )

    assert_array_equal(
        rdata.by_id(model.getStateIds()[1], "sx", model), rdata.sx[:, :, 1]
    )


def test_evaluate(rdata_by_id_fixture):
    # get IDs of model components
    model, rdata = rdata_by_id_fixture
    expr0_id = model.getExpressionIds()[0]
    state1_id = model.getStateIds()[1]
    observable0_id = model.getObservableIds()[0]

    # ensure `evaluate` works for atoms
    expr0 = rdata.by_id(expr0_id)
    assert_array_equal(expr0, evaluate(expr0_id, rdata=rdata))

    state1 = rdata.by_id(state1_id)
    assert_array_equal(state1, evaluate(state1_id, rdata=rdata))

    observable0 = rdata.by_id(observable0_id)
    assert_array_equal(observable0, evaluate(observable0_id, rdata=rdata))

    # ensure `evaluate` works for expressions
    assert_almost_equal(
        expr0 + state1 * observable0,
        evaluate(f"{expr0_id} + {state1_id} * {observable0_id}", rdata=rdata),
    )
