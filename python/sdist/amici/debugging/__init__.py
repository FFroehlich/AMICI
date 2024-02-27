"""Functions for debugging AMICI simulation failures."""
import amici
import numpy as np


def get_model_for_preeq(model: amici.Model, edata: amici.ExpData):
    """Get a model set-up to simulate the preequilibration condition as
    specified in `edata`.

    Useful for analyzing simulation errors during preequilibration.
    Simulating the returned model will reproduce the behavior of
    simulation-based preequilibration.

    Note that for models with events, the simulation results may differ.
    During preequilibration, event-handling is disabled. However, when
    simulating the returned model, event-handling will be enabled.
    For events triggered at fixed timepoints, this can be avoided by setting
    :meth:`t0 <amici.Model.setT0>` to a timepoints after the last trigger
    timepoint.

    :param model:
        The model for which *edata* was generated.
    :param edata:
        The experimental data object with a preequilibration condition
        specified.
    :return:
        A copy of *model* with the same parameters, initial states, ...
        as *amici_model* for the preequilibration condition.
        Output timepoints are set to ``[inf]`` and will have to be adjusted.
    """
    model = model.clone()
    model.setTimepoints([np.inf])
    model.setFixedParameters(edata.fixedParametersPreequilibration)
    if edata.pscale:
        model.setParameterScale(edata.pscale)
    if edata.parameters:
        model.setParameters(edata.parameters)
    if edata.plist:
        model.setParameterList(edata.plist)
    model.setInitialStates(edata.x0)
    # has to be set *after* parameter list/scale!
    model.setInitialStateSensitivities(edata.sx0)
    model.setT0(edata.tstart_)

    return model
