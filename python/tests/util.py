"""Tests for SBML events, including piecewise expressions."""

import sys
import tempfile
from pathlib import Path

import libsbml
import numpy as np
import pandas as pd

from amici import (
    AmiciModel,
    ExpData,
    SbmlImporter,
    SensitivityMethod,
    SensitivityOrder,
    import_model_module,
    runAmiciSimulation,
)
from amici.gradient_check import _check_close
from numpy.testing import assert_allclose


def create_amici_model(sbml_model, model_name, **kwargs) -> AmiciModel:
    """
    Import an sbml file and create an AMICI model from it
    """
    sbml_test_models_output_dir = Path("amici_models")
    sbml_test_models_output_dir.mkdir(parents=True, exist_ok=True)

    sbml_importer = SbmlImporter(sbml_model)
    # try not to exceed the stupid maximum path length on windows 💩
    output_dir = (
        sbml_test_models_output_dir / model_name
        if sys.platform != "win32"
        else tempfile.mkdtemp()
    )

    sbml_importer.sbml2amici(
        model_name=model_name, output_dir=output_dir, **kwargs
    )

    model_module = import_model_module(model_name, output_dir)
    return model_module.getModel()


def create_sbml_model(
    initial_assignments,
    parameters,
    rate_rules,
    species,
    events,
    to_file: str = None,
):
    """Create an SBML model from simple definitions.

    See the model definitions and usage in :py:func:`model` for example input.

    The default initial concentration of species is `1.0`. This can currently
    be changed by specifying an initial assignment.
    """
    document = libsbml.SBMLDocument(3, 1)
    model = document.createModel()

    compartment = model.createCompartment()
    compartment.setId("compartment")
    compartment.setConstant(True)
    compartment.setSize(1)
    compartment.setSpatialDimensions(3)
    compartment.setUnits("dimensionless")

    for species_id in species:
        species = model.createSpecies()
        species.setId(species_id)
        species.setCompartment("compartment")
        species.setConstant(False)
        species.setSubstanceUnits("dimensionless")
        species.setBoundaryCondition(False)
        species.setHasOnlySubstanceUnits(False)
        species.setInitialConcentration(1.0)

    for target, formula in initial_assignments.items():
        initial_assignment = model.createInitialAssignment()
        initial_assignment.setSymbol(target)
        initial_assignment.setMath(libsbml.parseL3Formula(formula))

    for target, formula in rate_rules.items():
        rate_rule = model.createRateRule()
        rate_rule.setVariable(target)
        rate_rule.setMath(libsbml.parseL3Formula(formula))

    for parameter_id, parameter_value in parameters.items():
        parameter = model.createParameter()
        parameter.setId(parameter_id)
        parameter.setConstant(True)
        parameter.setValue(parameter_value)
        parameter.setUnits("dimensionless")

    for event_id, event_def in events.items():
        event = model.createEvent()
        event.setId(event_id)
        event.setName(event_id)
        event.setUseValuesFromTriggerTime(False)
        trigger = event.createTrigger()
        trigger.setMath(libsbml.parseL3Formula(event_def["trigger"]))
        trigger.setPersistent(True)
        trigger.setInitialValue(True)

        def create_event_assignment(target, assignment):
            ea = event.createEventAssignment()
            ea.setVariable(target)
            ea.setMath(libsbml.parseL3Formula(assignment))

        if isinstance(event_def["target"], list):
            for event_target, event_assignment in zip(
                event_def["target"], event_def["assignment"], strict=True
            ):
                create_event_assignment(event_target, event_assignment)

        else:
            create_event_assignment(
                event_def["target"], event_def["assignment"]
            )

    if to_file:
        libsbml.writeSBMLToFile(document, to_file)

    # Need to return document, else AMICI throws an error.
    # (possibly due to garbage collection?)
    return document, model


def check_trajectories_without_sensitivities(
    amici_model: AmiciModel,
    result_expected_x: np.ndarray,
):
    """
    Check whether the AMICI simulation matches a known solution
    (ideally an analytically calculated one).
    """
    solver = amici_model.getSolver()
    solver.setAbsoluteTolerance(1e-15)
    solver.setRelativeTolerance(1e-12)
    rdata = runAmiciSimulation(amici_model, solver=solver)
    _check_close(
        rdata["x"], result_expected_x, field="x", rtol=5e-9, atol=1e-13
    )


def check_trajectories_with_forward_sensitivities(
    amici_model: AmiciModel,
    result_expected_x: np.ndarray,
    result_expected_sx: np.ndarray,
):
    """
    Check whether the forward sensitivities of the AMICI simulation match
    a known solution (ideally an analytically calculated one).
    """
    solver = amici_model.getSolver()
    solver.setSensitivityOrder(SensitivityOrder.first)
    solver.setSensitivityMethod(SensitivityMethod.forward)
    solver.setAbsoluteTolerance(1e-15)
    solver.setRelativeTolerance(1e-13)
    solver.setAbsoluteToleranceFSA(1e-15)
    solver.setRelativeToleranceFSA(1e-13)
    rdata = runAmiciSimulation(amici_model, solver=solver)
    _check_close(
        rdata["x"], result_expected_x, field="x", rtol=1e-10, atol=1e-12
    )
    _check_close(
        rdata["sx"], result_expected_sx, field="sx", rtol=1e-7, atol=1e-9
    )


def check_trajectories_with_adjoint_sensitivities(amici_model: AmiciModel):
    """
    Check whether adjoint sensitivities match forward sensitivities and finite
    differences.
    """
    # First compute dummy experimental data to use adjoints
    solver = amici_model.getSolver()
    solver.setAbsoluteTolerance(1e-15)
    rdata = runAmiciSimulation(amici_model, solver=solver)
    edata = ExpData(rdata, 1.0, 1.0)

    # Show that we can do arbitrary precision here (test 8 digits)
    solver = amici_model.getSolver()
    solver.setSensitivityOrder(SensitivityOrder.first)
    solver.setSensitivityMethod(SensitivityMethod.forward)
    solver.setAbsoluteTolerance(1e-16)
    solver.setRelativeTolerance(1e-14)
    rdata_fsa = runAmiciSimulation(amici_model, solver=solver, edata=edata)

    # Show that we can do arbitrary precision here (test 8 digits)
    solver = amici_model.getSolver()
    solver.setSensitivityOrder(SensitivityOrder.first)
    solver.setSensitivityMethod(SensitivityMethod.adjoint)
    solver.setAbsoluteTolerance(1e-16)
    solver.setRelativeTolerance(1e-14)
    solver.setAbsoluteToleranceB(1e-16)
    solver.setRelativeToleranceB(1e-15)
    solver.setAbsoluteToleranceQuadratures(1e-14)
    solver.setRelativeToleranceQuadratures(1e-8)
    rdata_asa = runAmiciSimulation(amici_model, solver=solver, edata=edata)

    assert_allclose(rdata_fsa.x, rdata_asa.x, atol=1e-14, rtol=1e-10)
    assert_allclose(rdata_fsa.llh, rdata_asa.llh, atol=1e-14, rtol=1e-10)
    df = pd.DataFrame(
        {
            "fsa": rdata_fsa["sllh"],
            "asa": rdata_asa["sllh"],
            "fd": np.nan,
        },
        index=list(amici_model.getParameterIds()),
    )
    df["abs_diff"] = df["fsa"] - df["asa"]
    df["rel_diff"] = df["abs_diff"] / df["fsa"]

    # Also test against finite differences
    parameters = amici_model.getUnscaledParameters()
    sllh_fd = []
    eps = 1e-5
    for i_par, par in enumerate(parameters):
        solver = amici_model.getSolver()
        solver.setSensitivityOrder(SensitivityOrder.first)
        solver.setSensitivityMethod(SensitivityMethod.adjoint)
        solver.setAbsoluteTolerance(1e-15)
        solver.setRelativeTolerance(1e-13)
        tmp_par = np.array(parameters[:])
        tmp_par[i_par] += eps
        amici_model.setParameters(tmp_par)
        rdata_p = runAmiciSimulation(amici_model, solver=solver, edata=edata)
        tmp_par = np.array(parameters[:])
        tmp_par[i_par] -= eps
        amici_model.setParameters(tmp_par)
        rdata_m = runAmiciSimulation(amici_model, solver=solver, edata=edata)
        sllh_fd.append((rdata_p["llh"] - rdata_m["llh"]) / (2 * eps))
    df["fd"] = sllh_fd
    print(df)

    # test less strict in terms of absolute error, as the gradient are
    # typically in the order of 1e3
    assert_allclose(
        rdata_fsa["sllh"],
        rdata_asa["sllh"],
        rtol=1e-5,
        atol=1e-3,
        err_msg="Adjoint and forward sensitivities do not match.",
    )
    assert_allclose(
        sllh_fd,
        rdata_fsa["sllh"],
        rtol=1e-5,
        atol=1e-3,
        err_msg="Finite differences and forward sensitivities do not match.",
    )
    assert_allclose(
        sllh_fd,
        rdata_asa["sllh"],
        rtol=1e-5,
        atol=1e-3,
        err_msg="Finite differences and adjoint sensitivities do not match.",
    )
