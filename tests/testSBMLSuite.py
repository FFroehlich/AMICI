#!/usr/bin/env python3
"""
Run SBML Test Suite and verify simulation results
[https://github.com/sbmlteam/sbml-test-suite/releases]

Usage:
    pytest tests.testSBMLSuite -n CORES --cases=SELECTION
        CORES can be an integer or `auto` for all available cores.
        SELECTION can be e.g.: `1`, `1,3`, `-3,4,6-7`, or `100-` to select
        specific test cases. If `--cases` is omitted, all cases are run.
"""

import copy
import os
import shutil
import sys
from pathlib import Path

import amici
import libsbml as sbml
import numpy as np
import pandas as pd
import pytest
from amici.constants import SymbolId
from amici.gradient_check import check_derivatives
from numpy.testing import assert_allclose


@pytest.fixture(scope="session")
def result_path() -> Path:
    return Path(__file__).parent / "amici-semantic-results"


@pytest.fixture(scope="function", autouse=True)
def sbml_test_dir():
    # setup
    old_cwd = os.getcwd()
    old_path = copy.copy(sys.path)

    yield

    # teardown
    os.chdir(old_cwd)
    sys.path = old_path


def test_sbml_testsuite_case(
    test_number, result_path, sbml_semantic_cases_dir
):
    test_id = format_test_id(test_number)
    model_dir = None

    if test_id == "01395":
        pytest.skip("NaNs in the Jacobian")

    # test cases for which sensitivities are to be checked
    #  key: case ID; value: epsilon for finite differences
    sensitivity_check_cases = {
        # parameter-dependent conservation laws
        "00783": 1.5e-2,
        # initial events
        "00995": 1e-3,
    }

    try:
        current_test_path = sbml_semantic_cases_dir / test_id

        # parse expected results
        results_file = current_test_path / f"{test_id}-results.csv"
        results = pd.read_csv(results_file, delimiter=",")
        results.rename(
            columns={c: c.replace(" ", "") for c in results.columns},
            inplace=True,
        )

        # setup model
        model_dir = Path(__file__).parent / "SBMLTestModels" / test_id
        model, solver, wrapper = compile_model(
            current_test_path,
            test_id,
            model_dir,
            generate_sensitivity_code=True,
        )
        settings = read_settings_file(current_test_path, test_id)

        atol, rtol = apply_settings(settings, solver, model, test_id)

        solver.setSensitivityOrder(amici.SensitivityOrder.first)
        solver.setSensitivityMethod(amici.SensitivityMethod.forward)

        # simulate model
        rdata = amici.runAmiciSimulation(model, solver)
        if rdata["status"] != amici.AMICI_SUCCESS:
            if test_id in ("00748", "00374", "00369"):
                pytest.skip("Simulation Failed expectedly")
            else:
                raise RuntimeError("Simulation failed unexpectedly")

        # verify
        simulated = verify_results(
            settings, rdata, results, wrapper, model, atol, rtol
        )

        # record results
        write_result_file(simulated, test_id, result_path)

        # check sensitivities for selected models
        if epsilon := sensitivity_check_cases.get(test_id):
            check_derivatives(model, solver, epsilon=epsilon)

    except amici.sbml_import.SBMLException as err:
        pytest.skip(str(err))
    finally:
        if model_dir:
            shutil.rmtree(model_dir, ignore_errors=True)


def verify_results(settings, rdata, expected, wrapper, model, atol, rtol):
    """Verify test results"""
    amount_species, variables = get_amount_and_variables(settings)

    # collect states
    simulated = pd.DataFrame(
        rdata["y"],
        columns=[
            obs["name"]
            for obs in wrapper.symbols[SymbolId.OBSERVABLE].values()
        ],
    )
    simulated["time"] = rdata["ts"]
    # collect parameters
    for par in model.getParameterIds():
        simulated[par] = rdata["ts"] * 0 + model.getParameterById(par)
    # collect fluxes
    for expr_idx, expr_id in enumerate(model.getExpressionIds()):
        if expr_id.startswith("flux_"):
            simulated[expr_id.removeprefix("flux_")] = rdata.w[:, expr_idx]
    # handle renamed reserved symbols
    simulated.rename(
        columns={c: c.replace("amici_", "") for c in simulated.columns},
        inplace=True,
    )

    # SBML test suite case 01308 defines species with initialAmount and
    # hasOnlySubstanceUnits="true", but then request results as concentrations.
    requested_concentrations = [
        s
        for s in settings["concentration"]
        .replace(" ", "")
        .replace("\n", "")
        .split(",")
        if s
    ]
    # We only need to convert species that have only substance units
    concentration_species = [
        str(state_id)
        for state_id, state in {
            **wrapper.symbols[SymbolId.SPECIES],
            **wrapper.symbols[SymbolId.ALGEBRAIC_STATE],
        }.items()
        if str(state_id) in requested_concentrations
        and state.get("amount", False)
    ]
    amounts_to_concentrations(
        concentration_species, wrapper, simulated, requested_concentrations
    )

    concentrations_to_amounts(
        amount_species, wrapper, simulated, requested_concentrations
    )

    # simulated may contain `object` dtype columns and `expected` may
    # contain `np.int64` columns, so we cast everything to `np.float64`.
    for variable in variables:
        expectation = expected[variable].astype(np.float64).values
        try:
            actual = simulated[variable].astype(np.float64).values
        except KeyError as e:
            raise KeyError(f"Missing simulated value for `{variable}`") from e
        assert_allclose(
            actual,
            expectation,
            atol,
            rtol,
            equal_nan=True,
            err_msg=f"Mismatch for {variable}",
        )

    return simulated[variables + ["time"]]


def amounts_to_concentrations(
    amount_species, wrapper, simulated, requested_concentrations
):
    """
    Convert AMICI simulated amounts to concentrations

    Convert from concentration to amount:
    C=n/V
    n=CV (multiply by V)
    Convert from amount to concentration:
    n=CV
    C=n/V (divide by V)
    Dividing by V is equivalent to multiplying the reciprocal by V, then taking
    the reciprocal.
    This allows for the reuse of the concentrations_to_amounts method...
    """
    for species in amount_species:
        if species != "":
            simulated.loc[:, species] = 1 / simulated.loc[:, species]
            concentrations_to_amounts(
                [species], wrapper, simulated, requested_concentrations
            )
            simulated.loc[:, species] = 1 / simulated.loc[:, species]


def concentrations_to_amounts(
    amount_species, wrapper, simulated, requested_concentrations
):
    """Convert AMICI simulated concentrations to amounts"""
    for species in amount_species:
        s = wrapper.sbml.getElementBySId(species)
        # Skip species that are marked to only have substance units since
        # they are already simulated as amounts
        if not isinstance(s, sbml.Species):
            continue

        is_amt = s.getHasOnlySubstanceUnits()
        comp = s.getCompartment()
        # Compartments and parameters that are treated as species do not
        # exist within a compartment.
        # Species with OnlySubstanceUnits don't have to be converted as long
        # as we don't request concentrations for them. Only applies when
        # called from amounts_to_concentrations.
        if (
            is_amt and species not in requested_concentrations
        ) or comp is None:
            continue

        simulated.loc[:, species] *= simulated.loc[
            :, comp if comp in simulated.columns else f"amici_{comp}"
        ]


def write_result_file(
    simulated: pd.DataFrame, test_id: str, result_path: Path
):
    """
    Create test result file for upload to
    http://raterule.caltech.edu/Facilities/Database

    Requires csv file with test ID in name and content of [time, Species, ...]
    """
    # TODO: only states are reported here, not compartments or parameters

    filename = result_path / f"{test_id}.csv"
    simulated.to_csv(filename, index=False)


def get_amount_and_variables(settings):
    """Read amount and species from settings file"""

    # species for which results are expected as amounts
    amount_species = (
        settings["amount"].replace(" ", "").replace("\n", "").split(",")
    )

    # IDs of all variables for which results are expected/provided
    variables = (
        settings["variables"].replace(" ", "").replace("\n", "").split(",")
    )

    return amount_species, variables


def apply_settings(settings, solver, model, test_id: str):
    """Apply model and solver settings as specified in the test case"""
    # start/duration/steps may be empty
    ts = np.linspace(
        float(settings["start"] or 0),
        float(settings["start"] or 0) + float(settings["duration"] or 0),
        int(settings["steps"] or 0) + 1,
    )
    atol = float(settings["absolute"])
    rtol = float(settings["relative"])

    model.setTimepoints(ts)
    solver.setMaxSteps(int(1e6))
    solver.setRelativeTolerance(rtol / 1e4)

    if test_id == "01148":
        solver.setAbsoluteTolerance(atol / 1e6)
    else:
        solver.setAbsoluteTolerance(atol / 1e4)

    return atol, rtol


def compile_model(
    sbml_dir: Path,
    test_id: str,
    model_dir: Path,
    generate_sensitivity_code: bool = False,
):
    """Import the given test model to AMICI"""
    model_dir.mkdir(parents=True, exist_ok=True)

    sbml_file = find_model_file(sbml_dir, test_id)
    sbml_importer = amici.SbmlImporter(sbml_file)

    model_name = f"SBMLTest{test_id}"
    sbml_importer.sbml2amici(
        model_name,
        output_dir=model_dir,
        generate_sensitivity_code=generate_sensitivity_code,
    )

    # settings
    model_module = amici.import_model_module(model_name, model_dir)

    model = model_module.getModel()
    solver = model.getSolver()

    return model, solver, sbml_importer


def find_model_file(current_test_path: Path, test_id: str) -> Path:
    """Find model file for the given test (guess filename extension)"""

    sbml_file = current_test_path / f"{test_id}-sbml-l3v2.xml"

    if not sbml_file.is_file():
        # fallback l3v1
        sbml_file = current_test_path / f"{test_id}-sbml-l3v1.xml"

    if not sbml_file.is_file():
        # fallback l2v5
        sbml_file = current_test_path / f"{test_id}-sbml-l2v5.xml"

    return sbml_file


def read_settings_file(current_test_path: Path, test_id: str):
    """Read settings for the given test"""
    settings_file = current_test_path / f"{test_id}-settings.txt"
    settings = {}
    with open(settings_file) as f:
        for line in f:
            if line != "\n":
                (key, val) = line.split(":")
                settings[key] = val.strip()
    return settings


def format_test_id(test_id) -> str:
    """Format numeric to 0-padded string"""
    return f"{test_id:0>5}"
