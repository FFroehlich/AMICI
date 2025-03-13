import os
import warnings

import amici
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from amici.testing import skip_on_valgrind
from conftest import MODEL_CONSTANT_SPECIES_XML


@pytest.fixture
def edata_fixture():
    """edata is generated to test pre- and postequilibration"""
    edata_pre = amici.ExpData(
        2, 0, 0, np.array([0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
    )
    edata_pre.setObservedData([1.5] * 16)
    edata_pre.fixedParameters = np.array([5.0, 20.0])
    edata_pre.fixedParametersPreequilibration = np.array([0.0, 10.0])
    edata_pre.reinitializeFixedParameterInitialStates = True

    # edata for postequilibration
    edata_post = amici.ExpData(2, 0, 0, np.array([float("inf")] * 3))
    edata_post.setObservedData([0.75] * 6)
    edata_post.fixedParameters = np.array([7.5, 30.0])

    # edata with both equilibrations
    edata_full = amici.ExpData(
        2,
        0,
        0,
        np.array(
            [0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 4.0, float("inf"), float("inf")]
        ),
    )
    edata_full.setObservedData([3.14] * 18)
    edata_full.fixedParameters = np.array([1.0, 2.0])
    edata_full.fixedParametersPreequilibration = np.array([3.0, 4.0])
    edata_full.reinitializeFixedParameterInitialStates = True

    return edata_pre, edata_post, edata_full


@pytest.fixture(scope="session")
def models():
    # SBML model we want to import
    sbml_importer = amici.SbmlImporter(MODEL_CONSTANT_SPECIES_XML)

    # Name of the model that will also be the name of the python module
    model_name = model_output_dir = "model_constant_species"
    model_name_cl = model_output_dir_cl = "model_constant_species_cl"

    # Define constants, observables, sigmas
    constant_parameters = ["synthesis_substrate", "init_enzyme"]
    observables = {
        "observable_product": {"name": "", "formula": "product"},
        "observable_substrate": {"name": "", "formula": "substrate"},
    }
    sigmas = {"observable_product": 1.0, "observable_substrate": 1.0}

    # wrap models with and without conservations laws
    sbml_importer.sbml2amici(
        model_name_cl,
        model_output_dir_cl,
        observables=observables,
        constant_parameters=constant_parameters,
        sigmas=sigmas,
    )
    sbml_importer.sbml2amici(
        model_name,
        model_output_dir,
        observables=observables,
        constant_parameters=constant_parameters,
        sigmas=sigmas,
        compute_conservation_laws=False,
    )

    # load both models
    model_without_cl_module = amici.import_model_module(
        model_name, module_path=os.path.abspath(model_name)
    )
    model_with_cl_module = amici.import_model_module(
        model_name_cl, module_path=os.path.abspath(model_name_cl)
    )

    # get the models and return
    model_without_cl = model_without_cl_module.getModel()
    model_with_cl = model_with_cl_module.getModel()
    return model_with_cl, model_without_cl


def get_results(
    model,
    edata=None,
    sensi_order=0,
    sensi_meth=amici.SensitivityMethod.forward,
    sensi_meth_preeq=amici.SensitivityMethod.forward,
    stst_mode=amici.SteadyStateComputationMode.integrateIfNewtonFails,
    stst_sensi_mode=amici.SteadyStateSensitivityMode.newtonOnly,
    reinitialize_states=False,
):
    # set model and data properties
    model.setReinitializeFixedParameterInitialStates(reinitialize_states)

    # get the solver, set the properties
    solver = model.getSolver()
    solver.setNewtonMaxSteps(20)
    solver.setSensitivityOrder(sensi_order)
    solver.setSensitivityMethodPreequilibration(sensi_meth_preeq)
    solver.setSensitivityMethod(sensi_meth)
    model.setSteadyStateSensitivityMode(stst_sensi_mode)
    model.setSteadyStateComputationMode(stst_mode)
    if edata is None:
        model.setTimepoints(np.linspace(0, 5, 101))
    else:
        edata.reinitializeFixedParameterInitialStates = reinitialize_states

    # return simulation results
    return amici.runAmiciSimulation(model, solver, edata)


@skip_on_valgrind
def test_compare_conservation_laws_sbml(models, edata_fixture):
    # first, create the model
    model_with_cl, model_without_cl = models

    assert model_with_cl.ncl() > 0
    assert model_without_cl.nx_rdata == model_with_cl.nx_rdata
    assert model_with_cl.nx_solver < model_without_cl.nx_solver
    assert len(model_with_cl.getStateIdsSolver()) == model_with_cl.nx_solver
    assert (
        len(model_without_cl.getStateIdsSolver()) == model_without_cl.nx_solver
    )

    # ----- compare simulations wo edata, sensi = 0, states ------------------
    # run simulations
    rdata_cl = get_results(model_with_cl)
    assert rdata_cl["status"] == amici.AMICI_SUCCESS
    rdata = get_results(model_without_cl)
    assert rdata["status"] == amici.AMICI_SUCCESS

    # compare state trajectories
    assert_allclose(
        rdata["x"],
        rdata_cl["x"],
        rtol=1.0e-5,
        atol=1.0e-8,
        err_msg="rdata.x mismatch",
    )

    # ----- compare simulations wo edata, sensi = 1, states and sensis -------
    # run simulations
    rdata_cl = get_results(model_with_cl, sensi_order=1)
    assert rdata_cl["status"] == amici.AMICI_SUCCESS
    rdata = get_results(model_without_cl, sensi_order=1)
    assert rdata["status"] == amici.AMICI_SUCCESS

    # compare state trajectories
    for field in ["x", "sx"]:
        assert_allclose(
            rdata[field],
            rdata_cl[field],
            rtol=1.0e-5,
            atol=1.0e-8,
            err_msg=f"rdata.{field} mismatch",
        )

    # ----- compare simulations wo edata, sensi = 0, states and sensis -------

    # run simulations
    edata, _, _ = edata_fixture
    rdata_cl = get_results(model_with_cl, edata=edata)
    assert rdata_cl["status"] == amici.AMICI_SUCCESS
    rdata = get_results(model_without_cl, edata=edata)
    assert rdata["status"] == amici.AMICI_SUCCESS

    # compare preequilibrated states
    for field in ["x", "x_ss", "llh"]:
        assert_allclose(
            rdata[field],
            rdata_cl[field],
            rtol=1.0e-5,
            atol=1.0e-8,
            err_msg=f"rdata.{field} mismatch",
        )

    # ----- compare simulations wo edata, sensi = 1, states and sensis -------

    # run simulations
    rdata_cl = get_results(model_with_cl, edata=edata, sensi_order=1)
    assert rdata_cl["status"] == amici.AMICI_SUCCESS
    rdata = get_results(
        model_without_cl,
        edata=edata,
        sensi_order=1,
        stst_sensi_mode=amici.SteadyStateSensitivityMode.integrateIfNewtonFails,
    )
    assert rdata["status"] == amici.AMICI_SUCCESS
    # check that steady state computation succeeded only by sim in full model
    assert_array_equal(rdata["preeq_status"], np.array([[-3, 1, 0]]))
    # check that steady state computation succeeded by Newton in reduced model
    assert_array_equal(rdata_cl["preeq_status"], np.array([[1, 0, 0]]))

    # compare state sensitivities with edata and preequilibration
    for field in ["x", "x_ss", "sx", "llh", "sllh"]:
        assert_allclose(
            rdata[field],
            rdata_cl[field],
            rtol=1.0e-5,
            atol=1.0e-8,
            err_msg=f"rdata.{field} mismatch",
        )

    # ----- check failure st.st. sensi computation if run wo CLs -------------
    # check failure of steady state sensitivity computation if run wo CLs
    model_without_cl.setSteadyStateSensitivityMode(
        amici.SteadyStateSensitivityMode.newtonOnly
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        rdata = get_results(model_without_cl, edata=edata, sensi_order=1)
        assert rdata["status"] == amici.AMICI_ERROR


def test_adjoint_pre_and_post_equilibration(models, edata_fixture):
    # get both models
    model_cl, model = models

    # check gradient with and without state reinitialization
    for edata in edata_fixture:
        for reinit in [False, True]:
            # compare different ways of preequilibration, full rank Jacobian
            # forward preequilibration, forward simulation
            rff_cl = get_results(
                model_cl,
                edata=edata,
                sensi_order=1,
                sensi_meth=amici.SensitivityMethod.forward,
                sensi_meth_preeq=amici.SensitivityMethod.forward,
                reinitialize_states=reinit,
            )
            # forward preequilibration, adjoint simulation
            rfa_cl = get_results(
                model_cl,
                edata=edata,
                sensi_order=1,
                sensi_meth=amici.SensitivityMethod.adjoint,
                sensi_meth_preeq=amici.SensitivityMethod.forward,
                reinitialize_states=reinit,
            )
            # adjoint preequilibration, adjoint simulation
            raa_cl = get_results(
                model_cl,
                edata=edata,
                sensi_order=1,
                sensi_meth=amici.SensitivityMethod.adjoint,
                sensi_meth_preeq=amici.SensitivityMethod.adjoint,
                reinitialize_states=reinit,
            )

            # assert all are close
            assert_allclose(
                rff_cl["sllh"], rfa_cl["sllh"], rtol=1.0e-5, atol=1.0e-8
            )
            assert_allclose(
                rfa_cl["sllh"], raa_cl["sllh"], rtol=1.0e-5, atol=1.0e-8
            )
            assert_allclose(
                raa_cl["sllh"], rff_cl["sllh"], rtol=1.0e-5, atol=1.0e-8
            )

            # compare fully adjoint approach to simulation with singular
            #  Jacobian
            raa = get_results(
                model,
                edata=edata,
                sensi_order=1,
                sensi_meth=amici.SensitivityMethod.adjoint,
                sensi_meth_preeq=amici.SensitivityMethod.adjoint,
                stst_sensi_mode=amici.SteadyStateSensitivityMode.integrateIfNewtonFails,
                reinitialize_states=reinit,
            )

            # assert gradients are close (quadrature tolerances are laxer)
            assert_allclose(raa_cl["sllh"], raa["sllh"], 1e-5, 1e-5)


@skip_on_valgrind
def test_get_set_model_settings(models):
    """test amici.(get|set)_model_settings cycles for models with and without
    conservation laws"""

    for model in models:
        amici.set_model_settings(model, amici.get_model_settings(model))
