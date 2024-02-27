"""Tests for pre- and post-equilibration"""

import itertools

import amici
import numpy as np
import pytest
from amici.debugging import get_model_for_preeq
from numpy.testing import assert_allclose, assert_equal
from test_pysb import get_data


@pytest.fixture
def preeq_fixture(pysb_example_presimulation_module):
    model = pysb_example_presimulation_module.getModel()
    model.setReinitializeFixedParameterInitialStates(True)

    solver = model.getSolver()
    solver.setSensitivityOrder(amici.SensitivityOrder.first)
    solver.setSensitivityMethod(amici.SensitivityMethod.forward)

    edata = get_data(model)
    edata.t_presim = 2
    edata.fixedParameters = [10, 2]
    edata.fixedParametersPresimulation = [3, 2]
    edata.fixedParametersPreequilibration = [3, 0]
    edata.setTimepoints([1, 5])

    edata_preeq = amici.ExpData(edata)
    edata_preeq.t_presim = 0
    edata_preeq.setTimepoints([np.infty])
    edata_preeq.fixedParameters = edata.fixedParametersPreequilibration
    edata_preeq.fixedParametersPresimulation = ()
    edata_preeq.fixedParametersPreequilibration = ()

    edata_presim = amici.ExpData(edata)
    edata_presim.t_presim = 0
    edata_presim.setTimepoints([edata.t_presim])
    edata_presim.fixedParameters = edata.fixedParametersPresimulation
    edata_presim.fixedParametersPresimulation = ()
    edata_presim.fixedParametersPreequilibration = ()

    edata_sim = amici.ExpData(edata)
    edata_sim.t_presim = 0
    edata_sim.setTimepoints(edata.getTimepoints())
    edata_sim.fixedParameters = edata.fixedParameters
    edata_sim.fixedParametersPresimulation = ()
    edata_sim.fixedParametersPreequilibration = ()

    pscales = [
        amici.ParameterScaling.log10,
        amici.ParameterScaling.ln,
        amici.ParameterScaling.none,
        amici.parameterScalingFromIntVector(
            [
                amici.ParameterScaling.log10,
                amici.ParameterScaling.ln,
                amici.ParameterScaling.none,
                amici.ParameterScaling.log10,
                amici.ParameterScaling.ln,
                amici.ParameterScaling.none,
            ]
        ),
    ]

    plists = [
        [3, 1, 2, 4],
        [0, 1, 2, 3, 4, 5],
        [5, 3, 2, 0, 4, 1],
        [1, 2, 3, 4, 5],
        [1, 1, 1],
    ]

    return (
        model,
        solver,
        edata,
        edata_preeq,
        edata_presim,
        edata_sim,
        pscales,
        plists,
    )


def test_manual_preequilibration(preeq_fixture):
    """Manual preequilibration"""

    (
        model,
        solver,
        edata,
        edata_preeq,
        edata_presim,
        edata_sim,
        pscales,
        plists,
    ) = preeq_fixture

    settings = itertools.product(pscales, plists)

    for pscale, plist in settings:
        model.setInitialStates([])
        model.setInitialStateSensitivities([])
        model.setParameterList(plist)
        model.setParameterScale(pscale)

        # combined
        rdata_auto = amici.runAmiciSimulation(model, solver, edata)
        assert rdata_auto.status == amici.AMICI_SUCCESS

        # manual preequilibration
        rdata_preeq = amici.runAmiciSimulation(model, solver, edata_preeq)
        assert rdata_preeq.status == amici.AMICI_SUCCESS

        # manual reinitialization + presimulation
        x0 = rdata_preeq["x"][0, :]
        x0[1] = edata_presim.fixedParameters[0]
        x0[2] = edata_presim.fixedParameters[1]
        sx0 = rdata_preeq["sx"][0, :, :]
        sx0[:, 1] = 0
        sx0[:, 2] = 0
        model.setInitialStates(x0)
        model.setInitialStateSensitivities(sx0.flatten())
        rdata_presim = amici.runAmiciSimulation(model, solver, edata_presim)
        assert rdata_presim.status == amici.AMICI_SUCCESS

        # manual reinitialization + simulation
        x0 = rdata_presim["x"][0, :]
        x0[1] = edata_sim.fixedParameters[0]
        x0[2] = edata_sim.fixedParameters[1]
        sx0 = rdata_presim["sx"][0, :, :]
        sx0[:, 1] = 0
        sx0[:, 2] = 0
        model.setInitialStates(x0)
        model.setInitialStateSensitivities(sx0.flatten())
        rdata_sim = amici.runAmiciSimulation(model, solver, edata_sim)
        assert rdata_sim.status == amici.AMICI_SUCCESS

        for variable in ["x", "sx"]:
            assert_allclose(
                rdata_auto[variable],
                rdata_sim[variable],
                atol=1e-6,
                rtol=1e-6,
                err_msg=str(
                    dict(pscale=pscale, plist=plist, variable=variable)
                ),
            )


def test_parameter_reordering(preeq_fixture):
    """Test parameter reordering"""

    (
        model,
        solver,
        edata,
        edata_preeq,
        edata_presim,
        edata_sim,
        pscales,
        plists,
    ) = preeq_fixture

    rdata_ordered = amici.runAmiciSimulation(model, solver, edata)

    for plist in plists:
        model.setParameterList(plist)
        rdata_reordered = amici.runAmiciSimulation(model, solver, edata)

        for ip, p_index in enumerate(plist):
            assert_allclose(
                rdata_ordered["sx"][:, p_index, :],
                rdata_reordered["sx"][:, ip, :],
                atol=1e-6,
                rtol=1e-6,
                err_msg=str(dict(variable="sx", plist=plist, p_index=p_index)),
            )


def test_data_replicates(preeq_fixture):
    """Test data replicates"""

    (
        model,
        solver,
        edata,
        edata_preeq,
        edata_presim,
        edata_sim,
        pscales,
        plists,
    ) = preeq_fixture

    sensi_meth = amici.SensitivityMethod.forward
    solver.setSensitivityMethod(sensi_meth)

    # add infty timepoint
    y = edata.getObservedData()
    stdy = edata.getObservedDataStdDev()
    ts = np.hstack([*edata.getTimepoints(), np.inf])
    edata.setTimepoints(sorted(ts))
    edata.setObservedData(np.hstack([y, y[0]]))
    edata.setObservedDataStdDev(np.hstack([stdy, stdy[0]]))
    rdata_single = amici.runAmiciSimulation(model, solver, edata)

    # duplicate data and timepoints
    y = edata.getObservedData()
    stdy = edata.getObservedDataStdDev()
    ts = np.hstack([*edata.getTimepoints(), *edata.getTimepoints()])
    idx = np.argsort(ts)
    edata.setTimepoints(sorted(ts))
    edata.setObservedData(np.hstack([y, y])[idx])
    edata.setObservedDataStdDev(np.hstack([stdy, stdy])[idx])

    rdata_double = amici.runAmiciSimulation(model, solver, edata)

    for variable in ["llh", "sllh"]:
        assert_allclose(
            2 * rdata_single[variable],
            rdata_double[variable],
            atol=1e-6,
            rtol=1e-6,
            err_msg=str(dict(variable=variable, sensi_meth=sensi_meth)),
        )


def test_parameter_in_expdata(preeq_fixture):
    """Test parameter in ExpData"""

    (
        model,
        solver,
        edata,
        edata_preeq,
        edata_presim,
        edata_sim,
        pscales,
        plists,
    ) = preeq_fixture

    rdata = amici.runAmiciSimulation(model, solver, edata)

    # get initial states will compute initial states if nothing is set,
    # this needs go first as we need unmodified model. Also set to
    # preequilibration fixpars first as this is where initial states would be
    # computed otherwise
    model.setFixedParameters(edata.fixedParametersPreequilibration)
    edata.x0 = model.getInitialStates()
    edata.sx0 = model.getInitialStateSensitivities()

    # perturb model initial states
    model.setInitialStates(rdata["x_ss"] * 4)
    model.setInitialStateSensitivities(rdata["sx_ss"].flatten() / 2)

    # set ExpData plist
    edata.plist = model.getParameterList()
    # perturb model parameter list
    model.setParameterList([i for i in reversed(model.getParameterList())])

    # set ExpData parameters
    edata.parameters = model.getParameters()
    # perturb model parameters
    model.setParameters(tuple(p * 2 for p in model.getParameters()))

    # set ExpData pscale
    edata.pscale = model.getParameterScale()
    # perturb model pscale, needs to be done after getting parameters,
    # otherwise we will mess up parameter value
    model.setParameterScale(
        amici.parameterScalingFromIntVector(
            [
                amici.ParameterScaling.log10
                if scaling == amici.ParameterScaling.none
                else amici.ParameterScaling.none
                for scaling in model.getParameterScale()
            ]
        )
    )

    rdata_edata = amici.runAmiciSimulation(model, solver, edata)
    for variable in ["x", "sx"]:
        assert_allclose(
            rdata[variable][0, :],
            rdata_edata[variable][0, :],
            atol=1e-6,
            rtol=1e-6,
            err_msg=str(dict(variable=variable)),
        )


def test_raise_presimulation_with_adjoints(preeq_fixture):
    """Test simulation failures with adjoin+presimulation"""

    (
        model,
        solver,
        edata,
        edata_preeq,
        edata_presim,
        edata_sim,
        pscales,
        plists,
    ) = preeq_fixture

    # preequilibration and presimulation with adjoints:
    # this needs to fail unless we remove presimulation
    solver.setSensitivityMethod(amici.SensitivityMethod.adjoint)

    rdata = amici.runAmiciSimulation(model, solver, edata)
    assert rdata["status"] == amici.AMICI_ERROR

    # add postequilibration
    y = edata.getObservedData()
    stdy = edata.getObservedDataStdDev()
    ts = np.hstack([*edata.getTimepoints(), np.inf])
    edata.setTimepoints(ts)
    edata.setObservedData(np.hstack([y, y[0]]))
    edata.setObservedDataStdDev(np.hstack([stdy, stdy[0]]))

    # remove presimulation
    edata.t_presim = 0
    edata.fixedParametersPresimulation = ()

    # no presim any more, this should work
    rdata = amici.runAmiciSimulation(model, solver, edata)
    assert rdata["status"] == amici.AMICI_SUCCESS


def test_equilibration_methods_with_adjoints(preeq_fixture):
    """Test different combinations of equilibration and simulation
    sensitivity methods"""

    (
        model,
        solver,
        edata,
        edata_preeq,
        edata_presim,
        edata_sim,
        pscales,
        plists,
    ) = preeq_fixture

    # we don't want presim
    edata.t_presim = 0.0
    edata.fixedParametersPresimulation = ()

    # add infty timepoint
    y = edata.getObservedData()
    stdy = edata.getObservedDataStdDev()
    ts = np.hstack([*edata.getTimepoints(), np.inf])
    edata.setTimepoints(sorted(ts))
    edata.setObservedData(np.hstack([y, y[0]]))
    edata.setObservedDataStdDev(np.hstack([stdy, stdy[0]]))

    rdatas = {}
    equil_meths = [
        amici.SteadyStateSensitivityMode.newtonOnly,
        amici.SteadyStateSensitivityMode.integrationOnly,
        amici.SteadyStateSensitivityMode.integrateIfNewtonFails,
    ]
    sensi_meths = [
        amici.SensitivityMethod.forward,
        amici.SensitivityMethod.adjoint,
    ]
    settings = itertools.product(equil_meths, sensi_meths)

    for setting in settings:
        # unpack, solver settings
        equil_meth, sensi_meth = setting
        model.setSteadyStateSensitivityMode(equil_meth)
        solver.setSensitivityMethod(sensi_meth)
        solver.setNewtonMaxSteps(0)

        # add rdatas
        rdatas[setting] = amici.runAmiciSimulation(model, solver, edata)
        # assert successful simulation

        assert rdatas[setting]["status"] == amici.AMICI_SUCCESS

    for setting1, setting2 in itertools.product(settings, settings):
        # assert correctness of result
        for variable in ["llh", "sllh"]:
            assert_allclose(
                rdatas[setting1][variable],
                rdatas[setting2][variable],
                atol=1e-6,
                rtol=1e-6,
                err_msg=str(
                    dict(
                        variable=variable, setting1=setting1, setting2=setting2
                    )
                ),
            )


def test_newton_solver_equilibration(preeq_fixture):
    """Test newton solver for equilibration"""

    (
        model,
        solver,
        edata,
        edata_preeq,
        edata_presim,
        edata_sim,
        pscales,
        plists,
    ) = preeq_fixture

    # we don't want presim
    edata.t_presim = 0.0
    edata.fixedParametersPresimulation = ()

    # add infty timepoint
    y = edata.getObservedData()
    stdy = edata.getObservedDataStdDev()
    ts = np.hstack([*edata.getTimepoints(), np.inf])
    edata.setTimepoints(sorted(ts))
    edata.setObservedData(np.hstack([y, y[0]]))
    edata.setObservedDataStdDev(np.hstack([stdy, stdy[0]]))

    rdatas = {}
    settings = [
        amici.SteadyStateSensitivityMode.integrationOnly,
        amici.SteadyStateSensitivityMode.newtonOnly,
    ]

    solver.setNewtonStepSteadyStateCheck(True)
    solver.setRelativeToleranceSteadyState(1e-12)

    for equil_meth in settings:
        # set sensi method
        sensi_meth = amici.SensitivityMethod.forward
        solver.setSensitivityMethod(sensi_meth)
        model.setSteadyStateSensitivityMode(equil_meth)
        if equil_meth == amici.SteadyStateSensitivityMode.newtonOnly:
            solver.setNewtonMaxSteps(10)

        # add rdatas
        rdatas[equil_meth] = amici.runAmiciSimulation(model, solver, edata)

        # assert successful simulation
        assert rdatas[equil_meth]["status"] == amici.AMICI_SUCCESS

    # assert correct results
    for variable in ["llh", "sllh", "sx0", "sx_ss", "x_ss"]:
        assert_allclose(
            rdatas[settings[0]][variable],
            rdatas[settings[1]][variable],
            atol=1e-5,
            rtol=1e-5,
            err_msg=str(dict(variable=variable)),
        )


def test_newton_steadystate_check(preeq_fixture):
    """Test NewtonStepSteadyStateCheck solver flag"""

    (
        model,
        solver,
        edata,
        edata_preeq,
        edata_presim,
        edata_sim,
        pscales,
        plists,
    ) = preeq_fixture

    # we don't want presim
    edata.t_presim = 0.0
    edata.fixedParametersPresimulation = ()

    # add infty timepoint
    y = edata.getObservedData()
    stdy = edata.getObservedDataStdDev()
    ts = np.hstack([*edata.getTimepoints(), np.inf])
    edata.setTimepoints(sorted(ts))
    edata.setObservedData(np.hstack([y, y[0]]))
    edata.setObservedDataStdDev(np.hstack([stdy, stdy[0]]))

    # set sensi method
    sensi_meth = amici.SensitivityMethod.forward
    solver.setSensitivityMethod(sensi_meth)

    solver.setNewtonMaxSteps(100)

    rdatas = {}
    for newton_check in [True, False]:
        solver.setNewtonStepSteadyStateCheck(newton_check)

        # add rdatas
        rdatas[newton_check] = amici.runAmiciSimulation(model, solver, edata)

        # assert successful simulation
        assert rdatas[newton_check]["status"] == amici.AMICI_SUCCESS

    # assert correct results
    for variable in ["llh", "sllh", "sx0", "sx_ss", "x_ss"]:
        assert_allclose(
            rdatas[True][variable],
            rdatas[False][variable],
            atol=1e-6,
            rtol=1e-6,
            err_msg=str(dict(variable=variable, sensi_meth=sensi_meth)),
        )


def test_steadystate_computation_mode(preeq_fixture):
    """Test newtonOnly and integrationOnly steady-state computation modes"""
    (
        model,
        solver,
        edata,
        edata_preeq,
        edata_presim,
        edata_sim,
        pscales,
        plists,
    ) = preeq_fixture

    sensi_meth = amici.SensitivityMethod.forward
    solver.setSensitivityOrder(amici.SensitivityOrder.first)
    solver.setSensitivityMethodPreequilibration(sensi_meth)
    solver.setNewtonMaxSteps(10)

    rdatas = {}
    stst_computation_modes = [
        amici.SteadyStateComputationMode.integrationOnly,
        amici.SteadyStateComputationMode.newtonOnly,
    ]
    for mode in stst_computation_modes:
        model.setSteadyStateComputationMode(mode)
        rdatas[mode] = amici.runAmiciSimulation(model, solver, edata)

        # assert successful simulation
        assert rdatas[mode]["status"] == amici.AMICI_SUCCESS

    assert np.all(
        rdatas[amici.SteadyStateComputationMode.integrationOnly][
            "preeq_status"
        ][0]
        == [0, 1, 0]
    )
    assert (
        rdatas[amici.SteadyStateComputationMode.integrationOnly][
            "preeq_numsteps"
        ][0][0]
        == 0
    )

    assert np.all(
        rdatas[amici.SteadyStateComputationMode.newtonOnly]["preeq_status"][0]
        == [1, 0, 0]
    )
    assert (
        rdatas[amici.SteadyStateComputationMode.newtonOnly]["preeq_numsteps"][
            0
        ][0]
        > 0
    )

    # assert correct results
    for variable in ["llh", "sllh", "sx0", "sx_ss", "x_ss"]:
        assert_allclose(
            rdatas[stst_computation_modes[0]][variable],
            rdatas[stst_computation_modes[1]][variable],
            atol=1e-5,
            rtol=1e-5,
            err_msg=str(dict(variable=variable, sensi_meth=sensi_meth)),
        )


def test_simulation_errors(preeq_fixture):
    (
        model,
        solver,
        edata,
        edata_preeq,
        edata_presim,
        edata_sim,
        pscales,
        plists,
    ) = preeq_fixture

    solver.setSensitivityOrder(amici.SensitivityOrder.first)
    solver.setSensitivityMethodPreequilibration(
        amici.SensitivityMethod.forward
    )
    model.setSteadyStateSensitivityMode(
        amici.SteadyStateSensitivityMode.integrationOnly
    )
    solver.setMaxSteps(1)

    # exceeded maxsteps
    # preeq & posteq
    for e in [edata, edata_preeq]:
        rdata = amici.runAmiciSimulation(model, solver, e)
        assert rdata["status"] != amici.AMICI_SUCCESS
        assert rdata._swigptr.messages[0].severity == amici.LogSeverity_debug
        assert rdata._swigptr.messages[0].identifier == "EQUILIBRATION_FAILURE"
        assert (
            "exceeded maximum number of integration steps"
            in rdata._swigptr.messages[0].message
        )
        assert rdata._swigptr.messages[1].severity == amici.LogSeverity_error
        assert rdata._swigptr.messages[1].identifier == "OTHER"
        assert rdata._swigptr.messages[2].severity == amici.LogSeverity_debug
        assert rdata._swigptr.messages[2].identifier == "BACKTRACE"

    # too long simulations
    solver.setMaxSteps(int(1e4))
    solver.setRelativeToleranceSteadyState(0.0)
    solver.setAbsoluteToleranceSteadyState(0.0)
    # preeq & posteq
    for e in [edata_preeq, edata]:
        rdata = amici.runAmiciSimulation(model, solver, e)
        assert rdata["status"] != amici.AMICI_SUCCESS
        assert rdata._swigptr.messages[0].severity == amici.LogSeverity_debug
        assert (
            rdata._swigptr.messages[0].identifier
            == "CVODES:CVode:RHSFUNC_FAIL"
        )
        assert rdata._swigptr.messages[1].severity == amici.LogSeverity_debug
        assert rdata._swigptr.messages[1].identifier == "EQUILIBRATION_FAILURE"
        assert (
            "exceedingly long simulation time"
            in rdata._swigptr.messages[1].message
        )
        assert rdata._swigptr.messages[2].severity == amici.LogSeverity_error
        assert rdata._swigptr.messages[2].identifier == "OTHER"
        assert rdata._swigptr.messages[3].severity == amici.LogSeverity_debug
        assert rdata._swigptr.messages[3].identifier == "BACKTRACE"


def test_get_model_for_preeq(preeq_fixture):
    (
        model,
        solver,
        edata,
        edata_preeq,
        edata_presim,
        edata_sim,
        pscales,
        plists,
    ) = preeq_fixture
    model.setSteadyStateSensitivityMode(
        amici.SteadyStateSensitivityMode.integrationOnly
    )
    model_preeq = get_model_for_preeq(model, edata)
    # the exactly same settings are used, so results should match exactly
    rdata1 = amici.runAmiciSimulation(model_preeq, solver)
    rdata2 = amici.runAmiciSimulation(model, solver, edata_preeq)
    assert_equal(
        rdata1.x,
        rdata2.x,
    )
    assert_equal(
        rdata1.sx,
        rdata2.sx,
    )
