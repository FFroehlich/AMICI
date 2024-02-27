"""Tests related to amici.ExpData via Python"""
import amici
import numpy as np
from amici.testing import skip_on_valgrind
from test_sbml_import import model_units_module  # noqa: F401


@skip_on_valgrind
def test_edata_sensi_unscaling(model_units_module):  # noqa: F811
    """
    ExpData parameters should be used for unscaling initial state
    sensitivities.
    """
    parameters0 = (5, 5)
    parameters1 = (2, 2)

    sx0 = (3, 3, 3, 3)

    parameter_scales_log10 = [amici.ParameterScaling.log10.value] * len(
        parameters0
    )
    amici_parameter_scales_log10 = amici.parameterScalingFromIntVector(
        parameter_scales_log10
    )

    model = model_units_module.getModel()
    model.setTimepoints(np.linspace(0, 1, 3))
    model.setParameterScale(parameter_scales_log10)
    model.setParameters(parameters0)

    solver = model.getSolver()
    solver.setSensitivityOrder(amici.SensitivityOrder.first)

    edata0 = amici.ExpData(model)
    edata0.pscale = amici_parameter_scales_log10
    edata0.parameters = parameters0
    edata0.sx0 = sx0

    edata1 = amici.ExpData(model)
    edata1.pscale = amici_parameter_scales_log10
    edata1.parameters = parameters1
    edata1.sx0 = sx0

    rdata0 = amici.runAmiciSimulation(model, solver, edata0)
    rdata1 = amici.runAmiciSimulation(model, solver, edata1)

    # The initial state sensitivities are as specified.
    assert np.isclose(rdata0.sx0.flatten(), sx0).all()
    assert np.isclose(rdata1.sx0.flatten(), sx0).all()
