"""Test functions that require special treatment in amici.sbml_import.

In particular, some of the functions tested here require an installation of
boost.
"""

import amici
import numpy as np
import pytest
from amici.antimony_import import antimony2amici
from amici.gradient_check import check_derivatives
from amici.testing import TemporaryDirectoryWinSafe, skip_on_valgrind
from numpy.testing import (
    assert_approx_equal,
    assert_array_almost_equal_nulp,
    assert_allclose,
)
from conftest import MODEL_STEADYSTATE_SCALED_XML
from scipy.special import loggamma


@pytest.fixture(scope="session")
def model_special_likelihoods():
    """Test model for special likelihood functions."""
    # load sbml model
    sbml_importer = amici.SbmlImporter(MODEL_STEADYSTATE_SCALED_XML)

    # define observables
    observables = {
        "o1": {"formula": "100*10^x1"},
        "o2": {"formula": "100*10^x1"},
    }

    # define different noise models
    noise_distributions = {
        "o1": "binomial",
        "o2": "negative-binomial",
    }

    module_name = "test_special_likelihoods"
    with TemporaryDirectoryWinSafe(prefix=module_name) as outdir:
        sbml_importer.sbml2amici(
            model_name=module_name,
            output_dir=outdir,
            observables=observables,
            constant_parameters=["k0"],
            noise_distributions=noise_distributions,
        )

        yield amici.import_model_module(
            module_name=module_name, module_path=outdir
        )


@skip_on_valgrind
# FD check fails occasionally, so give some extra tries
@pytest.mark.flaky(reruns=5)
def test_special_likelihoods(model_special_likelihoods):
    """Test special likelihood functions."""
    model = model_special_likelihoods.getModel()
    model.setTimepoints(np.linspace(0, 60, 10))
    solver = model.getSolver()
    solver.setSensitivityOrder(amici.SensitivityOrder.first)

    # Test in region with positive density

    # run model once to create an edata
    rdata = amici.runAmiciSimulation(model, solver)
    edata = amici.ExpData(rdata, 0.001, 0)

    # make sure measurements are smaller for non-degenerate probability
    y = edata.getObservedData()
    y = tuple(val * np.random.uniform(0, 1) for val in y)
    edata.setObservedData(y)

    # set sigmas
    sigma = 0.2
    sigmas = sigma * np.ones(len(y))
    edata.setObservedDataStdDev(sigmas)

    # and now run for real and also compute likelihood values
    rdata = amici.runAmiciSimulations(model, solver, [edata])[0]

    # check if the values make overall sense
    assert np.isfinite(rdata["llh"])
    assert np.all(np.isfinite(rdata["sllh"]))
    assert np.any(rdata["sllh"])

    rdata_df = amici.getSimulationObservablesAsDataFrame(
        model, edata, rdata, by_id=True
    )
    edata_df = amici.getDataObservablesAsDataFrame(model, edata, by_id=True)

    # check correct likelihood value
    llh_exp = -sum(
        [
            binomial_nllh(edata_df["o1"], rdata_df["o1"], sigma),
            negative_binomial_nllh(edata_df["o2"], rdata_df["o2"], sigma),
        ]
    )
    assert np.isclose(rdata["llh"], llh_exp)

    # check gradient
    for sensi_method in [
        amici.SensitivityMethod.forward,
        amici.SensitivityMethod.adjoint,
    ]:
        solver = model.getSolver()
        solver.setSensitivityMethod(sensi_method)
        solver.setSensitivityOrder(amici.SensitivityOrder.first)
        check_derivatives(
            model,
            solver,
            edata,
            atol=1e-4,
            rtol=1e-3,
            check_least_squares=False,
        )

    # Test for m > y, i.e. in region with 0 density

    rdata = amici.runAmiciSimulation(model, solver)
    edata = amici.ExpData(rdata, 0.001, 0)

    # make sure measurements are smaller for non-degenerate probability
    y = edata.getObservedData()
    y = tuple(val * np.random.uniform(0.5, 3) for val in y)
    edata.setObservedData(y)
    edata.setObservedDataStdDev(sigmas)

    # and now run for real and also compute likelihood values
    rdata = amici.runAmiciSimulations(model, solver, [edata])[0]

    # m > y -> outside binomial domain -> 0 density
    assert rdata["llh"] == -np.inf
    # check for non-informative gradient
    assert all(np.isnan(rdata["sllh"]))


def binomial_nllh(m: np.ndarray, y: np.ndarray, p: float):
    if any(m > y):
        return np.inf
    return sum(
        -loggamma(y + 1)
        + loggamma(m + 1)
        + loggamma(y - m + 1)
        - m * np.log(p)
        - (y - m) * np.log(1 - p)
    )


def negative_binomial_nllh(m: np.ndarray, y: np.ndarray, p: float):
    r = y * (1 - p) / p
    return sum(
        -loggamma(m + r)
        + loggamma(m + 1)
        + loggamma(r)
        - r * np.log(1 - p)
        - m * np.log(p)
    )


@skip_on_valgrind
def test_rateof():
    """Test chained rateOf to verify that model expressions are evaluated in the correct order."""
    ant_model = """
    model test_chained_rateof
        species S1, S2, S3, S4;
        S1 = 0;
        S3 = 0;
        p2 = 1;
        rate = 1;
        S4 = 0.5 * rateOf(S3);
        S2' = 2 * rateOf(S3);
        S1' = S2 + rateOf(S2);
        S3' = rate;
        p1 = 2 * rateOf(S1);
        p2' = rateOf(S1);
        p3 = rateOf(rate);
    end
    """
    module_name = "test_chained_rateof"
    with TemporaryDirectoryWinSafe(prefix=module_name) as outdir:
        antimony2amici(
            ant_model,
            model_name=module_name,
            output_dir=outdir,
        )
        model_module = amici.import_model_module(
            module_name=module_name, module_path=outdir
        )
        amici_model = model_module.getModel()
        t = np.linspace(0, 10, 11)
        amici_model.setTimepoints(t)
        amici_solver = amici_model.getSolver()
        rdata = amici.runAmiciSimulation(amici_model, amici_solver)

        state_ids_solver = amici_model.getStateIdsSolver()
        i_S1 = state_ids_solver.index("S1")
        i_S2 = state_ids_solver.index("S2")
        i_S3 = state_ids_solver.index("S3")
        i_p2 = state_ids_solver.index("p2")
        assert_approx_equal(rdata["xdot"][i_S3], 1)
        assert_approx_equal(rdata["xdot"][i_S2], 2)
        assert_approx_equal(
            rdata["xdot"][i_S1], rdata.by_id("S2")[-1] + rdata["xdot"][i_S2]
        )
        assert_approx_equal(rdata["xdot"][i_S1], rdata["xdot"][i_p2])

        assert_array_almost_equal_nulp(rdata.by_id("S3"), t, 10)
        assert_array_almost_equal_nulp(
            rdata.by_id("S2"), 2 * rdata.by_id("S3")
        )
        assert_array_almost_equal_nulp(
            rdata.by_id("S4")[1:], 0.5 * np.diff(rdata.by_id("S3")), 10
        )
        assert_array_almost_equal_nulp(rdata.by_id("p3"), 0)
        assert_array_almost_equal_nulp(
            rdata.by_id("p2"), 1 + rdata.by_id("S1")
        )


@skip_on_valgrind
def test_rateof_with_expression_dependent_rate():
    """Test rateOf, where the rateOf argument depends on `w` and requires
    toposorting."""
    ant_model = """
    model test_rateof_with_expression_dependent_rate
        S1 = 0;
        S2 = 0;
        S1' = rate;
        S2' = 2 * rateOf(S1);
        # the id of the following expression must be alphabetically before
        #  `rate`, so that toposort is required to evaluate the expressions
        #  in the correct order
        e1 := 2 * rateOf(S1);
        rate := time
    end
    """
    module_name = "test_rateof_with_expression_dependent_rate"
    with TemporaryDirectoryWinSafe(prefix=module_name) as outdir:
        antimony2amici(
            ant_model,
            model_name=module_name,
            output_dir=outdir,
        )
        model_module = amici.import_model_module(
            module_name=module_name, module_path=outdir
        )
        amici_model = model_module.getModel()
        t = np.linspace(0, 10, 11)
        amici_model.setTimepoints(t)
        amici_solver = amici_model.getSolver()
        rdata = amici.runAmiciSimulation(amici_model, amici_solver)

        state_ids_solver = amici_model.getStateIdsSolver()

        assert_array_almost_equal_nulp(rdata.by_id("e1"), 2 * t, 1)

        i_S1 = state_ids_solver.index("S1")
        i_S2 = state_ids_solver.index("S2")
        assert_approx_equal(rdata["xdot"][i_S1], t[-1])
        assert_approx_equal(rdata["xdot"][i_S2], 2 * t[-1])

        assert_allclose(np.diff(rdata.by_id("S1")), t[:-1] + 0.5, atol=1e-9)
        assert_array_almost_equal_nulp(
            rdata.by_id("S2"), 2 * rdata.by_id("S1"), 10
        )
