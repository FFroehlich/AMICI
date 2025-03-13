"""Tests related to amici.sbml_import"""

import os
import re
import sys
from numbers import Number
from pathlib import Path

import amici
import libsbml
import numpy as np
import pytest
from amici.gradient_check import check_derivatives
from amici.sbml_import import SbmlImporter
from amici.testing import TemporaryDirectoryWinSafe as TemporaryDirectory
from amici.testing import skip_on_valgrind
from numpy.testing import assert_allclose, assert_array_equal

from conftest import MODEL_STEADYSTATE_SCALED_XML


def simple_sbml_model():
    """Some testmodel"""
    document = libsbml.SBMLDocument(3, 1)
    model = document.createModel()
    model.setTimeUnits("second")
    model.setExtentUnits("mole")
    model.setSubstanceUnits("mole")
    c1 = model.createCompartment()
    c1.setId("C1")
    model.addCompartment(c1)
    s1 = model.createSpecies()
    s1.setId("S1")
    s1.setCompartment("C1")
    model.addSpecies(s1)
    p1 = model.createParameter()
    p1.setId("p1")
    p1.setValue(2.0)
    model.addParameter(p1)

    return document, model


def test_sbml2amici_no_observables():
    """Test model generation works for model without observables"""
    sbml_doc, sbml_model = simple_sbml_model()
    sbml_importer = SbmlImporter(sbml_source=sbml_model, from_file=False)
    model_name = "test_sbml2amici_no_observables"
    with TemporaryDirectory() as tmpdir:
        sbml_importer.sbml2amici(
            model_name=model_name,
            output_dir=tmpdir,
            observables=None,
            compute_conservation_laws=False,
        )

        # Ensure import succeeds (no missing symbols)
        module_module = amici.import_model_module(model_name, tmpdir)
        assert hasattr(module_module, "getModel")


@skip_on_valgrind
def test_sbml2amici_nested_observables_fail():
    """Test model generation works for model without observables"""
    sbml_doc, sbml_model = simple_sbml_model()
    sbml_importer = SbmlImporter(sbml_source=sbml_model, from_file=False)
    model_name = "test_sbml2amici_nested_observables_fail"
    with TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="(?i)nested"):
            sbml_importer.sbml2amici(
                model_name=model_name,
                output_dir=tmpdir,
                observables={
                    "outer": {"formula": "inner"},
                    "inner": {"formula": "S1"},
                },
                compute_conservation_laws=False,
                generate_sensitivity_code=False,
                compile=False,
            )


def test_nosensi():
    sbml_doc, sbml_model = simple_sbml_model()
    sbml_importer = SbmlImporter(sbml_source=sbml_model, from_file=False)
    model_name = "test_nosensi"
    with TemporaryDirectory() as tmpdir:
        sbml_importer.sbml2amici(
            model_name=model_name,
            output_dir=tmpdir,
            observables=None,
            compute_conservation_laws=False,
            generate_sensitivity_code=False,
        )

        model_module = amici.import_model_module(
            module_name=model_name, module_path=tmpdir
        )

        model = model_module.getModel()
        model.setTimepoints(np.linspace(0, 60, 61))
        solver = model.getSolver()
        solver.setSensitivityOrder(amici.SensitivityOrder.first)
        solver.setSensitivityMethod(amici.SensitivityMethod.forward)
        rdata = amici.runAmiciSimulation(model, solver)
        assert rdata.status == amici.AMICI_ERROR


@pytest.fixture(scope="session")
def observable_dependent_error_model():
    sbml_doc, sbml_model = simple_sbml_model()
    # add parameter and rate rule
    sbml_model.getSpecies("S1").setInitialConcentration(1.0)
    sbml_model.getParameter("p1").setValue(0.2)
    rr = sbml_model.createRateRule()
    rr.setVariable("S1")
    rr.setMath(libsbml.parseL3Formula("p1"))
    relative_sigma = sbml_model.createParameter()
    relative_sigma.setId("relative_sigma")
    relative_sigma.setValue(0.05)

    sbml_importer = SbmlImporter(sbml_source=sbml_model, from_file=False)

    model_name = "observable_dependent_error_model"
    with TemporaryDirectory() as tmpdir:
        sbml_importer.sbml2amici(
            model_name=model_name,
            output_dir=tmpdir,
            observables={
                "observable_s1": {"formula": "S1"},
                "observable_s1_scaled": {"formula": "0.5 * S1"},
            },
            sigmas={
                "observable_s1": "0.1 + relative_sigma * observable_s1",
                "observable_s1_scaled": "0.02 * observable_s1_scaled",
            },
        )
        yield amici.import_model_module(
            module_name=model_name, module_path=tmpdir
        )


@skip_on_valgrind
def test_sbml2amici_observable_dependent_error(
    observable_dependent_error_model,
):
    """Check gradients for model with observable-dependent error"""
    model_module = observable_dependent_error_model
    model = model_module.getModel()
    model.setTimepoints(np.linspace(0, 60, 61))
    solver = model.getSolver()

    # generate artificial data
    rdata = amici.runAmiciSimulation(model, solver)
    assert_allclose(
        rdata.sigmay[:, 0],
        0.1 + 0.05 * rdata.y[:, 0],
        rtol=1.0e-5,
        atol=1.0e-8,
    )
    assert_allclose(
        rdata.sigmay[:, 1], 0.02 * rdata.y[:, 1], rtol=1.0e-5, atol=1.0e-8
    )
    edata = amici.ExpData(rdata, 1.0, 0.0)
    edata.setObservedDataStdDev(np.nan)

    # check sensitivities
    solver.setSensitivityOrder(amici.SensitivityOrder.first)
    # FSA
    solver.setSensitivityMethod(amici.SensitivityMethod.forward)
    rdata = amici.runAmiciSimulation(model, solver, edata)
    assert np.any(rdata.ssigmay != 0.0)
    check_derivatives(model, solver, edata)
    # ASA
    solver.setSensitivityMethod(amici.SensitivityMethod.adjoint)
    check_derivatives(model, solver, edata)


@skip_on_valgrind
def test_logging_works(observable_dependent_error_model, caplog):
    """Check that warnings are forwarded to Python logging"""
    model_module = observable_dependent_error_model
    model = model_module.getModel()
    model.setTimepoints(np.linspace(0, 60, 61))
    solver = model.getSolver()

    # this will prematurely stop the simulation
    solver.setMaxSteps(1)

    rdata = amici.runAmiciSimulation(model, solver)
    assert rdata.status != amici.AMICI_SUCCESS
    assert "mxstep steps taken" in caplog.text


@skip_on_valgrind
def test_model_module_is_set(observable_dependent_error_model):
    model_module = observable_dependent_error_model
    assert model_module.getModel().module is model_module
    assert isinstance(model_module.getModel().module, amici.ModelModule)


@pytest.fixture(scope="session")
def model_steadystate_module():
    sbml_file = MODEL_STEADYSTATE_SCALED_XML
    sbml_importer = amici.SbmlImporter(sbml_file)

    observables = amici.assignmentRules2observables(
        sbml_importer.sbml,
        filter_function=lambda variable: variable.getId().startswith(
            "observable_"
        )
        and not variable.getId().endswith("_sigma"),
    )

    module_name = "test_model_steadystate_scaled"
    with TemporaryDirectory(prefix=module_name) as outdir:
        sbml_importer.sbml2amici(
            model_name=module_name,
            output_dir=outdir,
            observables=observables,
            constant_parameters=["k0"],
            sigmas={"observable_x1withsigma": "observable_x1withsigma_sigma"},
        )

        yield amici.import_model_module(
            module_name=module_name, module_path=outdir
        )


def test_presimulation(sbml_example_presimulation_module):
    """Test 'presimulation' test model"""
    model = sbml_example_presimulation_module.getModel()
    solver = model.getSolver()
    model.setTimepoints(np.linspace(0, 60, 61))
    model.setSteadyStateSensitivityMode(
        amici.SteadyStateSensitivityMode.integrationOnly
    )
    solver.setSensitivityOrder(amici.SensitivityOrder.first)
    model.setReinitializeFixedParameterInitialStates(True)

    rdata = amici.runAmiciSimulation(model, solver)
    edata = amici.ExpData(rdata, 0.1, 0.0)
    edata.fixedParameters = [10, 2]
    edata.fixedParametersPresimulation = [10, 2]
    edata.fixedParametersPreequilibration = [3, 0]
    assert isinstance(
        amici.runAmiciSimulation(model, solver, edata), amici.ReturnDataView
    )

    solver.setRelativeTolerance(1e-12)
    solver.setAbsoluteTolerance(1e-12)
    check_derivatives(model, solver, edata, epsilon=1e-4)


def test_steadystate_simulation(model_steadystate_module):
    model = model_steadystate_module.getModel()
    model.setTimepoints(np.linspace(0, 60, 60))
    solver = model.getSolver()
    solver.setSensitivityOrder(amici.SensitivityOrder.first)
    rdata = amici.runAmiciSimulation(model, solver)
    edata = [amici.ExpData(rdata, 1, 0)]
    edata[0].id = "some condition ID"
    rdata = amici.runAmiciSimulations(model, solver, edata)

    assert rdata[0].status == amici.AMICI_SUCCESS
    assert rdata[0].id == edata[0].id

    # check roundtripping of DataFrame conversion
    df_edata = amici.getDataObservablesAsDataFrame(model, edata)
    edata_reconstructed = amici.getEdataFromDataFrame(model, df_edata)

    assert_allclose(
        amici.ExpDataView(edata[0])["observedData"],
        amici.ExpDataView(edata_reconstructed[0])["observedData"],
        rtol=1.0e-5,
        atol=1.0e-8,
    )

    assert_allclose(
        amici.ExpDataView(edata[0])["observedDataStdDev"],
        amici.ExpDataView(edata_reconstructed[0])["observedDataStdDev"],
        rtol=1.0e-5,
        atol=1.0e-8,
    )

    if len(edata[0].fixedParameters):
        assert list(edata[0].fixedParameters) == list(
            edata_reconstructed[0].fixedParameters
        )

    else:
        assert list(model.getFixedParameters()) == list(
            edata_reconstructed[0].fixedParameters
        )

    assert list(edata[0].fixedParametersPreequilibration) == list(
        edata_reconstructed[0].fixedParametersPreequilibration
    )

    df_state = amici.getSimulationStatesAsDataFrame(model, edata, rdata)
    assert_allclose(
        rdata[0]["x"],
        df_state[list(model.getStateIds())].values,
        rtol=1.0e-5,
        atol=1.0e-8,
    )

    df_obs = amici.getSimulationObservablesAsDataFrame(model, edata, rdata)
    assert_allclose(
        rdata[0]["y"],
        df_obs[list(model.getObservableIds())].values,
        rtol=1.0e-5,
        atol=1.0e-8,
    )
    amici.getResidualsAsDataFrame(model, edata, rdata)

    df_expr = amici.pandas.get_expressions_as_dataframe(model, edata, rdata)
    assert_allclose(
        rdata[0]["w"],
        df_expr[list(model.getExpressionIds())].values,
        rtol=1.0e-5,
        atol=1.0e-8,
    )

    solver.setRelativeTolerance(1e-12)
    solver.setAbsoluteTolerance(1e-12)
    check_derivatives(
        model, solver, edata[0], atol=1e-3, rtol=1e-3, epsilon=1e-4
    )

    # Run some additional tests which need a working Model,
    # but don't need precomputed expectations.
    _test_set_parameters_by_dict(model_steadystate_module)


def test_solver_reuse(model_steadystate_module):
    model = model_steadystate_module.getModel()
    model.setTimepoints(np.linspace(0, 60, 60))
    solver = model.getSolver()
    solver.setSensitivityOrder(amici.SensitivityOrder.first)
    rdata = amici.runAmiciSimulation(model, solver)
    edata = amici.ExpData(rdata, 1, 0)

    for sensi_method in (
        amici.SensitivityMethod.forward,
        amici.SensitivityMethod.adjoint,
    ):
        solver.setSensitivityMethod(sensi_method)
        rdata1 = amici.runAmiciSimulation(model, solver, edata)
        rdata2 = amici.runAmiciSimulation(model, solver, edata)

        assert rdata1.status == amici.AMICI_SUCCESS

        for attr in rdata1:
            if "time" in attr or attr == "messages":
                continue

            val1 = getattr(rdata1, attr)
            val2 = getattr(rdata2, attr)
            msg = (
                f"Values for {attr} do not match for sensitivity "
                f"method {sensi_method}"
            )
            if isinstance(val1, np.ndarray):
                assert_array_equal(val1, val2, err_msg=msg)
            elif isinstance(val1, Number) and np.isnan(val1):
                assert np.isnan(val2)
            else:
                assert val1 == val2, msg


@pytest.fixture
def model_test_likelihoods():
    """Test model for various likelihood functions."""
    # load sbml model
    sbml_file = MODEL_STEADYSTATE_SCALED_XML
    sbml_importer = amici.SbmlImporter(sbml_file)

    # define observables
    observables = {
        "o1": {"formula": "x1"},
        "o2": {"formula": "10^x1"},
        "o3": {"formula": "10^x1"},
        "o4": {"formula": "x1"},
        "o5": {"formula": "10^x1"},
        "o6": {"formula": "10^x1"},
        "o7": {"formula": "x1"},
    }

    # define different noise models
    noise_distributions = {
        "o1": "normal",
        "o2": "log-normal",
        "o3": "log10-normal",
        "o4": "laplace",
        "o5": "log-laplace",
        "o6": "log10-laplace",
        "o7": lambda str_symbol: f"Abs({str_symbol} - m{str_symbol}) "
        f"/ sigma{str_symbol}",
    }

    module_name = "model_test_likelihoods"
    with TemporaryDirectory(prefix=module_name) as outdir:
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
def test_likelihoods(model_test_likelihoods):
    """Test the custom noise distributions used to define cost functions."""
    model = model_test_likelihoods.getModel()
    model.setTimepoints(np.linspace(0, 60, 60))
    solver = model.getSolver()
    solver.setSensitivityOrder(amici.SensitivityOrder.first)

    # run model once to create an edata

    rdata = amici.runAmiciSimulation(model, solver)
    sigmas = rdata["y"].max(axis=0) * 0.05
    edata = amici.ExpData(rdata, sigmas, [])
    # just make all observables positive since some are logarithmic
    while min(edata.getObservedData()) < 0:
        edata = amici.ExpData(rdata, sigmas, [])

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
            normal_nllh(edata_df["o1"], rdata_df["o1"], sigmas[0]),
            log_normal_nllh(edata_df["o2"], rdata_df["o2"], sigmas[1]),
            log10_normal_nllh(edata_df["o3"], rdata_df["o3"], sigmas[2]),
            laplace_nllh(edata_df["o4"], rdata_df["o4"], sigmas[3]),
            log_laplace_nllh(edata_df["o5"], rdata_df["o5"], sigmas[4]),
            log10_laplace_nllh(edata_df["o6"], rdata_df["o6"], sigmas[5]),
            custom_nllh(edata_df["o7"], rdata_df["o7"], sigmas[6]),
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
        solver.setRelativeTolerance(1e-12)
        solver.setAbsoluteTolerance(1e-12)
        check_derivatives(
            model,
            solver,
            edata,
            atol=1e-2,
            rtol=1e-2,
            epsilon=1e-5,
            check_least_squares=False,
        )


@skip_on_valgrind
def test_likelihoods_error():
    """Test whether wrong inputs lead to expected errors."""
    sbml_file = MODEL_STEADYSTATE_SCALED_XML
    sbml_importer = amici.SbmlImporter(sbml_file)

    # define observables
    observables = {"o1": {"formula": "x1"}}

    # define different noise models
    noise_distributions = {"o1": "nörmal"}

    module_name = "test_likelihoods_error"
    outdir = "test_likelihoods_error"
    with pytest.raises(ValueError):
        sbml_importer.sbml2amici(
            model_name=module_name,
            output_dir=outdir,
            observables=observables,
            constant_parameters=["k0"],
            noise_distributions=noise_distributions,
        )


@skip_on_valgrind
@pytest.mark.usefixtures("model_units_module")
def test_units(model_units_module):
    """
    Test whether SBML import works for models using sbml:units annotations.
    """
    model = model_units_module.getModel()
    model.setTimepoints(np.linspace(0, 1, 101))
    solver = model.getSolver()

    rdata = amici.runAmiciSimulation(model, solver)
    assert rdata["status"] == amici.AMICI_SUCCESS


@skip_on_valgrind
@pytest.mark.skipif(
    os.name == "nt", reason="Avoid `CERTIFICATE_VERIFY_FAILED` error"
)
def test_sympy_exp_monkeypatch():
    """
    This model contains a removeable discontinuity at t=0 that requires
    monkeypatching sympy.Pow._eval_derivative in order to be able to compute
    non-nan sensitivities
    """
    import pooch

    model_file = pooch.retrieve(
        url="https://www.ebi.ac.uk/biomodels/model/download/BIOMD0000000529.2?filename=BIOMD0000000529_url.xml",
        known_hash="md5:c6e0b298397485b93d7acfab80b21fd4",
    )
    importer = amici.SbmlImporter(model_file)
    module_name = "BIOMD0000000529"

    with TemporaryDirectory() as outdir:
        importer.sbml2amici(module_name, outdir)
        model_module = amici.import_model_module(
            module_name=module_name, module_path=outdir
        )

        model = model_module.getModel()
        model.setTimepoints(np.linspace(0, 8, 250))
        model.requireSensitivitiesForAllParameters()
        model.setAlwaysCheckFinite(True)
        model.setParameterScale(
            amici.parameterScalingFromIntVector(
                [
                    amici.ParameterScaling.none
                    if re.match(r"n[0-9]+$", par_id)
                    else amici.ParameterScaling.log10
                    for par_id in model.getParameterIds()
                ]
            )
        )

        solver = model.getSolver()
        solver.setSensitivityMethod(amici.SensitivityMethod.forward)
        solver.setSensitivityOrder(amici.SensitivityOrder.first)

        rdata = amici.runAmiciSimulation(model, solver)

        # print sensitivity-related results
        assert rdata["status"] == amici.AMICI_SUCCESS
        check_derivatives(
            model, solver, None, atol=1e-2, rtol=1e-2, epsilon=1e-3
        )


def normal_nllh(m, y, sigma):
    return sum(0.5 * (np.log(2 * np.pi * sigma**2) + ((y - m) / sigma) ** 2))


def log_normal_nllh(m, y, sigma):
    return sum(
        0.5
        * (
            np.log(2 * np.pi * sigma**2 * m**2)
            + ((np.log(y) - np.log(m)) / sigma) ** 2
        )
    )


def log10_normal_nllh(m, y, sigma):
    return sum(
        0.5
        * (
            np.log(2 * np.pi * sigma**2 * m**2 * np.log(10) ** 2)
            + ((np.log10(y) - np.log10(m)) / sigma) ** 2
        )
    )


def laplace_nllh(m, y, sigma):
    return sum(np.log(2 * sigma) + np.abs(y - m) / sigma)


def log_laplace_nllh(m, y, sigma):
    return sum(np.log(2 * sigma * m) + np.abs(np.log(y) - np.log(m)) / sigma)


def log10_laplace_nllh(m, y, sigma):
    return sum(
        np.log(2 * sigma * m * np.log(10))
        + np.abs(np.log10(y) - np.log10(m)) / sigma
    )


def custom_nllh(m, y, sigma):
    return sum(np.abs(m - y) / sigma)


def _test_set_parameters_by_dict(model_module):
    """Test setting parameter via id/name => value dicts"""
    model = model_module.getModel()
    old_parameter_values = model.getParameters()
    parameter_ids = model.getParameterIds()
    change_par_id = parameter_ids[-1]
    new_par_val = 0.1234
    old_par_val = model.getParameterById(change_par_id)

    assert model.getParameterById(change_par_id) != new_par_val
    model.setParameterById({change_par_id: new_par_val})
    assert model.getParameterById(change_par_id) == new_par_val
    # reset and check we are back to original
    model.setParameterById(change_par_id, old_par_val)
    assert model.getParameters() == old_parameter_values

    # Same for by-name
    parameter_names = model.getParameterNames()
    change_par_name = parameter_names[-1]
    model.setParameterByName({change_par_name: new_par_val})
    assert model.getParameterByName(change_par_name) == new_par_val
    model.setParameterByName(change_par_name, old_par_val)
    assert model.getParameters() == old_parameter_values


@skip_on_valgrind
@pytest.mark.parametrize("extract_cse", [True, False])
def test_code_gen_uses_cse(extract_cse):
    """Check that code generation honors AMICI_EXTRACT_CSE"""
    old_environ = os.environ.copy()
    try:
        os.environ["AMICI_EXTRACT_CSE"] = str(extract_cse)
        sbml_importer = amici.SbmlImporter(MODEL_STEADYSTATE_SCALED_XML)
        model_name = "test_code_gen_uses_cse"
        with TemporaryDirectory() as tmpdir:
            sbml_importer.sbml2amici(
                model_name=model_name,
                compile=False,
                generate_sensitivity_code=False,
                output_dir=tmpdir,
            )
            xdot = Path(tmpdir, "xdot.cpp").read_text()
        assert ("__amici_cse_0 = " in xdot) == extract_cse
    finally:
        os.environ = old_environ


@skip_on_valgrind
def test_code_gen_uses_lhs_symbol_ids():
    """Check that code generation uses symbol IDs instead of plain array
    indices"""
    sbml_importer = amici.SbmlImporter(MODEL_STEADYSTATE_SCALED_XML)
    model_name = "test_code_gen_uses_lhs_symbol_ids"
    with TemporaryDirectory() as tmpdir:
        sbml_importer.sbml2amici(
            model_name=model_name,
            compile=False,
            generate_sensitivity_code=False,
            output_dir=tmpdir,
        )
        dwdx = Path(tmpdir, "dwdx.cpp").read_text()
    assert "dobservable_x1_dx1 = " in dwdx


@skip_on_valgrind
def test_hardcode_parameters():
    """Test model generation works for model without observables"""
    sbml_doc, sbml_model = simple_sbml_model()
    sbml_importer = SbmlImporter(sbml_source=sbml_model, from_file=False)
    r = sbml_model.createRateRule()
    r.setVariable("S1")
    r.setFormula("p1")
    assert sbml_model.getParameter("p1").getValue() != 0

    ode_model = sbml_importer._build_ode_model()
    assert str(ode_model.parameters()) == "[p1]"
    assert ode_model.differential_states()[0].get_dt().name == "p1"

    ode_model = sbml_importer._build_ode_model(
        constant_parameters=[],
        hardcode_symbols=["p1"],
    )
    assert str(ode_model.parameters()) == "[]"
    assert (
        ode_model.differential_states()[0].get_dt()
        == sbml_model.getParameter("p1").getValue()
    )

    with pytest.raises(ValueError):
        sbml_importer._build_ode_model(
            # mutually exclusive
            constant_parameters=["p1"],
            hardcode_symbols=["p1"],
        )


def test_constraints():
    """Test non-negativity constraint handling."""
    from amici.antimony_import import antimony2amici
    from amici import Constraint

    ant_model = """
    model test_non_negative_species
        species A = 10
        species B = 0
        # R1: A => B; k1f * sqrt(A)
        R1: A => B; k1f * max(0, A)
        k1f = 1e10
    end
    """
    module_name = "test_non_negative_species"
    with TemporaryDirectory(prefix=module_name) as outdir:
        antimony2amici(
            ant_model,
            model_name=module_name,
            output_dir=outdir,
            compute_conservation_laws=False,
        )
        model_module = amici.import_model_module(
            module_name=module_name, module_path=outdir
        )
        amici_model = model_module.getModel()
        amici_model.setTimepoints(np.linspace(0, 100, 200))
        amici_solver = amici_model.getSolver()
        rdata = amici.runAmiciSimulation(amici_model, amici_solver)
        assert rdata.status == amici.AMICI_SUCCESS
        # should be non-negative in theory, but is expected to become negative
        #  in practice
        assert np.any(rdata.x < 0)

        amici_solver.setRelativeTolerance(1e-13)
        amici_solver.setConstraints(
            [Constraint.non_negative, Constraint.non_negative]
        )
        rdata = amici.runAmiciSimulation(amici_model, amici_solver)
        assert rdata.status == amici.AMICI_SUCCESS
        assert np.all(rdata.x >= 0)
        assert np.all(
            np.sum(rdata.x, axis=1) - np.sum(rdata.x[0])
            < max(
                np.sum(rdata.x[0]) * amici_solver.getRelativeTolerance(),
                amici_solver.getAbsoluteTolerance(),
            )
        )


@skip_on_valgrind
def test_import_same_model_name():
    """Test for error when loading a model with the same extension name as an
    already loaded model."""
    from amici.antimony_import import antimony2amici
    from amici import import_model_module

    # create three versions of a toy model with different parameter values
    #  to detect which model was loaded
    ant_model_1 = """
    model test_same_extension_error
        species A = 0
        p = 1
        A' = p
    end
    """
    ant_model_2 = ant_model_1.replace("1", "2")
    ant_model_3 = ant_model_1.replace("1", "3")

    module_name = "test_same_extension"
    with TemporaryDirectory(prefix=module_name) as outdir:
        outdir_1 = Path(outdir, "model_1")
        outdir_2 = Path(outdir, "model_2")

        # import the first two models, with the same name,
        #  but in different location (this is now supported)
        antimony2amici(
            ant_model_1,
            model_name=module_name,
            output_dir=outdir_1,
            compute_conservation_laws=False,
        )

        antimony2amici(
            ant_model_2,
            model_name=module_name,
            output_dir=outdir_2,
            compute_conservation_laws=False,
        )

        model_module_1 = import_model_module(
            module_name=module_name, module_path=outdir_1
        )
        assert model_module_1.get_model().getParameters()[0] == 1.0

        # no error if the same model is loaded again without changes on disk
        model_module_1b = import_model_module(
            module_name=module_name, module_path=outdir_1
        )
        # downside: the modules will compare as different
        assert (model_module_1 == model_module_1b) is False
        assert model_module_1.__file__ == model_module_1b.__file__
        assert model_module_1b.get_model().getParameters()[0] == 1.0

        model_module_2 = import_model_module(
            module_name=module_name, module_path=outdir_2
        )
        assert model_module_1.get_model().getParameters()[0] == 1.0
        assert model_module_2.get_model().getParameters()[0] == 2.0

        # import the third model, with the same name and location as the second
        #  model -- this is not supported, because there is some caching at
        #  the C level we cannot control (or don't know how to)

        # On Windows, this will give "permission denied" when building the
        #  extension, because we cannot delete a shared library that is in use

        if sys.platform == "win32":
            return

        antimony2amici(
            ant_model_3,
            model_name=module_name,
            output_dir=outdir_2,
        )

        with pytest.raises(RuntimeError, match="in the same location"):
            import_model_module(module_name=module_name, module_path=outdir_2)

        # this should not affect the previously loaded models
        assert model_module_1.get_model().getParameters()[0] == 1.0
        assert model_module_2.get_model().getParameters()[0] == 2.0

        # test that we can still import the model classically if we wanted to:
        with amici.set_path(outdir_1):
            import test_same_extension as model_module_1c  # noqa: F401

            assert model_module_1c.get_model().getParameters()[0] == 1.0
            assert model_module_1c.get_model().module is model_module_1c


@skip_on_valgrind
def test_regression_2642():
    sbml_file = Path(__file__).parent / "sbml_models" / "regression_2642.xml"
    sbml_importer = amici.SbmlImporter(sbml_file)
    model_name = "regression_2642"
    with TemporaryDirectory(prefix="regression_2642") as outdir:
        sbml_importer.sbml2amici(
            model_name=model_name,
            output_dir=outdir,
        )
        module = amici.import_model_module(
            module_name=model_name, module_path=outdir
        )
        model = module.getModel()
        solver = model.getSolver()
        model.setTimepoints(np.linspace(0, 1, 3))
        r = amici.runAmiciSimulation(model, solver)
        assert (
            len(np.unique(r.w[:, model.getExpressionIds().index("binding")]))
            == 1
        )
