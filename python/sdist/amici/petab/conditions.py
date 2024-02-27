"""PEtab conditions to AMICI ExpDatas."""
import logging
import numbers
import warnings
from typing import Union
from collections.abc import Sequence

import amici
import numpy as np
import pandas as pd
import petab
from amici import AmiciModel
from petab.C import (
    MEASUREMENT,
    NOISE_PARAMETERS,
    OBSERVABLE_ID,
    PREEQUILIBRATION_CONDITION_ID,
    SIMULATION_CONDITION_ID,
    TIME,
)

from .parameter_mapping import (
    ParameterMapping,
    ParameterMappingForCondition,
    petab_to_amici_scale,
    scale_parameters_dict,
    unscale_parameters_dict,
)
from .util import get_states_in_condition_table

logger = logging.getLogger(__name__)

SingleParameterMapping = dict[str, Union[numbers.Number, str]]
SingleScaleMapping = dict[str, str]


def fill_in_parameters(
    edatas: list[amici.ExpData],
    problem_parameters: dict[str, numbers.Number],
    scaled_parameters: bool,
    parameter_mapping: ParameterMapping,
    amici_model: AmiciModel,
) -> None:
    """Fill fixed and dynamic parameters into the edatas (in-place).

    :param edatas:
        List of experimental datas :class:`amici.amici.ExpData` with
        everything except parameters filled.
    :param problem_parameters:
        Problem parameters as parameterId=>value dict. Only
        parameters included here will be set. Remaining parameters will
        be used as currently set in `amici_model`.
    :param scaled_parameters:
        If True, problem_parameters are assumed to be on the scale provided
        in the parameter mapping. If False, they are assumed
        to be in linear scale.
    :param parameter_mapping:
        Parameter mapping for all conditions.
    :param amici_model:
        AMICI model.
    """
    if unused_parameters := (
        set(problem_parameters.keys()) - parameter_mapping.free_symbols
    ):
        warnings.warn(
            "The following problem parameters were not used: "
            + str(unused_parameters),
            RuntimeWarning,
        )

    for edata, mapping_for_condition in zip(edatas, parameter_mapping):
        fill_in_parameters_for_condition(
            edata,
            problem_parameters,
            scaled_parameters,
            mapping_for_condition,
            amici_model,
        )


def fill_in_parameters_for_condition(
    edata: amici.ExpData,
    problem_parameters: dict[str, numbers.Number],
    scaled_parameters: bool,
    parameter_mapping: ParameterMappingForCondition,
    amici_model: AmiciModel,
) -> None:
    """Fill fixed and dynamic parameters into the edata for condition
    (in-place).

    :param edata:
        Experimental data object to fill parameters into.
    :param problem_parameters:
        Problem parameters as parameterId=>value dict. Only
        parameters included here will be set. Remaining parameters will
        be used as already set in `amici_model` and `edata`.
    :param scaled_parameters:
        If True, problem_parameters are assumed to be on the scale provided
        in the parameter mapping. If False, they
        are assumed to be in linear scale.
    :param parameter_mapping:
        Parameter mapping for current condition.
    :param amici_model:
        AMICI model
    """
    map_sim_var = parameter_mapping.map_sim_var
    scale_map_sim_var = parameter_mapping.scale_map_sim_var
    map_preeq_fix = parameter_mapping.map_preeq_fix
    scale_map_preeq_fix = parameter_mapping.scale_map_preeq_fix
    map_sim_fix = parameter_mapping.map_sim_fix
    scale_map_sim_fix = parameter_mapping.scale_map_sim_fix

    # Parameter mapping may contain parameter_ids as values, these *must*
    # be replaced

    def _get_par(model_par, value, mapping):
        """Replace parameter IDs in mapping dicts by values from
        problem_parameters where necessary"""
        if isinstance(value, str):
            try:
                # estimated parameter
                return problem_parameters[value]
            except KeyError:
                # condition table overrides must have been handled already,
                # e.g. by the PEtab parameter mapping, but parameters from
                # InitialAssignments may still be present.
                if mapping[value] == model_par:
                    # prevent infinite recursion
                    raise
                return _get_par(value, mapping[value], mapping)
        if model_par in problem_parameters:
            # user-provided
            return problem_parameters[model_par]
        # prevent nan-propagation in derivative
        if np.isnan(value):
            return 0.0
        # constant value
        return value

    map_preeq_fix = {
        key: _get_par(key, val, map_preeq_fix)
        for key, val in map_preeq_fix.items()
    }
    map_sim_fix = {
        key: _get_par(key, val, map_sim_fix)
        for key, val in map_sim_fix.items()
    }
    map_sim_var = {
        key: _get_par(key, val, dict(map_sim_fix, **map_sim_var))
        for key, val in map_sim_var.items()
    }

    # If necessary, (un)scale parameters
    if scaled_parameters:
        unscale_parameters_dict(map_preeq_fix, scale_map_preeq_fix)
        unscale_parameters_dict(map_sim_fix, scale_map_sim_fix)
    if not scaled_parameters:
        # We scale all parameters to the scale they are estimated on, and pass
        # that information to amici via edata.{parameters,pscale}.
        # The scaling is necessary to obtain correct derivatives.
        scale_parameters_dict(map_sim_var, scale_map_sim_var)
        # We can skip preequilibration parameters, because they are identical
        # with simulation parameters, and only the latter are used from here
        # on.

    ##########################################################################
    # variable parameters and parameter scale

    # parameter list from mapping dict
    parameters = [
        map_sim_var[par_id] for par_id in amici_model.getParameterIds()
    ]

    # scales list from mapping dict
    scales = [
        petab_to_amici_scale(scale_map_sim_var[par_id])
        for par_id in amici_model.getParameterIds()
    ]

    # plist
    plist = [
        ip
        for ip, par_id in enumerate(amici_model.getParameterIds())
        if isinstance(parameter_mapping.map_sim_var[par_id], str)
    ]

    if parameters:
        edata.parameters = np.asarray(parameters, dtype=float)

    if scales:
        edata.pscale = amici.parameterScalingFromIntVector(scales)

    if plist:
        edata.plist = plist

    ##########################################################################
    # fixed parameters preequilibration
    if map_preeq_fix:
        fixed_pars_preeq = [
            map_preeq_fix[par_id]
            for par_id in amici_model.getFixedParameterIds()
        ]
        edata.fixedParametersPreequilibration = fixed_pars_preeq

    ##########################################################################
    # fixed parameters simulation
    if map_sim_fix:
        fixed_pars_sim = [
            map_sim_fix[par_id]
            for par_id in amici_model.getFixedParameterIds()
        ]
        edata.fixedParameters = fixed_pars_sim


def create_parameterized_edatas(
    amici_model: AmiciModel,
    petab_problem: petab.Problem,
    problem_parameters: dict[str, numbers.Number],
    scaled_parameters: bool = False,
    parameter_mapping: ParameterMapping = None,
    simulation_conditions: Union[pd.DataFrame, dict] = None,
) -> list[amici.ExpData]:
    """Create list of :class:amici.ExpData objects with parameters filled in.

    :param amici_model:
        AMICI Model assumed to be compatible with ``petab_problem``.
    :param petab_problem:
        PEtab problem to work on.
    :param problem_parameters:
        Run simulation with these parameters. If ``None``, PEtab
        ``nominalValues`` will be used. To be provided as dict, mapping PEtab
        problem parameters to SBML IDs.
    :param scaled_parameters:
        If ``True``, ``problem_parameters`` are assumed to be on the scale
        provided in the PEtab parameter table and will be unscaled.
        If ``False``, they are assumed to be in linear scale.
    :param parameter_mapping:
        Optional precomputed PEtab parameter mapping for efficiency, as
        generated by :func:`create_parameter_mapping`.
    :param simulation_conditions:
        Result of :func:`petab.get_simulation_conditions`. Can be provided to
        save time if this has been obtained before.

    :return:
        List with one :class:`amici.amici.ExpData` per simulation condition,
        with filled in timepoints, data and parameters.
    """
    # number of amici simulations will be number of unique
    # (preequilibrationConditionId, simulationConditionId) pairs.
    # Can be optimized by checking for identical condition vectors.
    if simulation_conditions is None:
        simulation_conditions = (
            petab_problem.get_simulation_conditions_from_measurement_df()
        )

    # Get parameter mapping
    if parameter_mapping is None:
        from .parameter_mapping import create_parameter_mapping

        parameter_mapping = create_parameter_mapping(
            petab_problem=petab_problem,
            simulation_conditions=simulation_conditions,
            scaled_parameters=scaled_parameters,
            amici_model=amici_model,
        )

    # Generate ExpData with all condition-specific information
    edatas = create_edatas(
        amici_model=amici_model,
        petab_problem=petab_problem,
        simulation_conditions=simulation_conditions,
    )

    # Fill parameters in ExpDatas (in-place)
    fill_in_parameters(
        edatas=edatas,
        problem_parameters=problem_parameters,
        scaled_parameters=scaled_parameters,
        parameter_mapping=parameter_mapping,
        amici_model=amici_model,
    )

    return edatas


def create_edata_for_condition(
    condition: Union[dict, pd.Series],
    measurement_df: pd.DataFrame,
    amici_model: AmiciModel,
    petab_problem: petab.Problem,
    observable_ids: list[str],
) -> amici.ExpData:
    """Get :class:`amici.amici.ExpData` for the given PEtab condition.

    Sets timepoints, observed data and sigmas.

    :param condition:
        :class:`pandas.DataFrame` row with ``preequilibrationConditionId`` and
        ``simulationConditionId``.
    :param measurement_df:
        :class:`pandas.DataFrame` with measurements for the given condition.
    :param amici_model:
        AMICI model
    :param petab_problem:
        Underlying PEtab problem
    :param observable_ids:
        List of observable IDs

    :return:
        ExpData instance.
    """
    if amici_model.nytrue != len(observable_ids):
        raise AssertionError(
            "Number of AMICI model observables does not "
            "match number of PEtab observables."
        )

    # create an ExpData object
    edata = amici.ExpData(amici_model)
    edata.id = condition[SIMULATION_CONDITION_ID]
    if condition.get(PREEQUILIBRATION_CONDITION_ID):
        edata.id += "+" + condition.get(PREEQUILIBRATION_CONDITION_ID)
    ##########################################################################
    # enable initial parameters reinitialization

    states_in_condition_table = get_states_in_condition_table(
        petab_problem, condition=condition
    )
    if (
        condition.get(PREEQUILIBRATION_CONDITION_ID)
        and states_in_condition_table
    ):
        state_ids = amici_model.getStateIds()
        state_idx_reinitalization = [
            state_ids.index(s)
            for s, (v, v_preeq) in states_in_condition_table.items()
            if not np.isnan(v)
        ]
        edata.reinitialization_state_idxs_sim = state_idx_reinitalization
        logger.debug(
            "Enabling state reinitialization for condition "
            f"{condition.get(PREEQUILIBRATION_CONDITION_ID, '')} - "
            f"{condition.get(SIMULATION_CONDITION_ID)} "
            f"{states_in_condition_table}"
        )

    ##########################################################################
    # timepoints

    # find replicate numbers of time points
    timepoints_w_reps = _get_timepoints_with_replicates(
        df_for_condition=measurement_df
    )
    edata.setTimepoints(timepoints_w_reps)

    ##########################################################################
    # measurements and sigmas
    y, sigma_y = _get_measurements_and_sigmas(
        df_for_condition=measurement_df,
        timepoints_w_reps=timepoints_w_reps,
        observable_ids=observable_ids,
    )
    edata.setObservedData(y.flatten())
    edata.setObservedDataStdDev(sigma_y.flatten())

    return edata


def create_edatas(
    amici_model: AmiciModel,
    petab_problem: petab.Problem,
    simulation_conditions: Union[pd.DataFrame, dict] = None,
) -> list[amici.ExpData]:
    """Create list of :class:`amici.amici.ExpData` objects for PEtab problem.

    :param amici_model:
        AMICI model.
    :param petab_problem:
        Underlying PEtab problem.
    :param simulation_conditions:
        Result of :func:`petab.get_simulation_conditions`. Can be provided to
        save time if this has be obtained before.

    :return:
        List with one :class:`amici.amici.ExpData` per simulation condition,
        with filled in timepoints and data.
    """
    if simulation_conditions is None:
        simulation_conditions = (
            petab_problem.get_simulation_conditions_from_measurement_df()
        )

    observable_ids = amici_model.getObservableIds()

    measurement_groupvar = [SIMULATION_CONDITION_ID]
    if PREEQUILIBRATION_CONDITION_ID in simulation_conditions:
        measurement_groupvar.append(petab.PREEQUILIBRATION_CONDITION_ID)
    measurement_dfs = dict(
        list(
            petab_problem.measurement_df.fillna(
                {PREEQUILIBRATION_CONDITION_ID: ""}
            ).groupby(measurement_groupvar)
        )
    )

    edatas = []
    for _, condition in simulation_conditions.iterrows():
        # Create amici.ExpData for each simulation
        if PREEQUILIBRATION_CONDITION_ID in condition:
            measurement_index = (
                condition.get(SIMULATION_CONDITION_ID),
                condition.get(PREEQUILIBRATION_CONDITION_ID) or "",
            )
        else:
            measurement_index = (condition.get(SIMULATION_CONDITION_ID),)

        edata = create_edata_for_condition(
            condition=condition,
            amici_model=amici_model,
            measurement_df=measurement_dfs[measurement_index],
            petab_problem=petab_problem,
            observable_ids=observable_ids,
        )
        edatas.append(edata)

    return edatas


def _get_timepoints_with_replicates(
    df_for_condition: pd.DataFrame,
) -> list[numbers.Number]:
    """
    Get list of timepoints including replicate measurements

    :param df_for_condition:
        PEtab measurement table subset for a single condition.

    :return:
        Sorted list of timepoints, including multiple timepoints accounting
        for replicate measurements.
    """
    # create sorted list of all timepoints for which measurements exist
    timepoints = sorted(df_for_condition[TIME].unique().astype(float))

    # find replicate numbers of time points
    timepoints_w_reps = []
    for time in timepoints:
        # subselect for time
        df_for_time = df_for_condition[
            df_for_condition.time.astype(float) == time
        ]
        # rep number is maximum over rep numbers for observables
        n_reps = max(df_for_time.groupby([OBSERVABLE_ID, TIME]).size())
        # append time point n_rep times
        timepoints_w_reps.extend([time] * n_reps)

    return timepoints_w_reps


def _get_measurements_and_sigmas(
    df_for_condition: pd.DataFrame,
    timepoints_w_reps: Sequence[numbers.Number],
    observable_ids: Sequence[str],
) -> tuple[np.array, np.array]:
    """
    Get measurements and sigmas

    Generate arrays with measurements and sigmas in AMICI format from a
    PEtab measurement table subset for a single condition.

    :param df_for_condition:
        Subset of PEtab measurement table for one condition

    :param timepoints_w_reps:
        Timepoints for which there exist measurements, including replicates

    :param observable_ids:
        List of observable IDs for mapping IDs to indices.

    :return:
        arrays for measurement and sigmas
    """
    # prepare measurement matrix
    y = np.full(
        shape=(len(timepoints_w_reps), len(observable_ids)), fill_value=np.nan
    )
    # prepare sigma matrix
    sigma_y = y.copy()

    timepoints = sorted(df_for_condition[TIME].unique().astype(float))

    for time in timepoints:
        # subselect for time
        df_for_time = df_for_condition[df_for_condition[TIME] == time]
        time_ix_0 = timepoints_w_reps.index(time)

        # remember used time indices for each observable
        time_ix_for_obs_ix = {}

        # iterate over measurements
        for _, measurement in df_for_time.iterrows():
            # extract observable index
            observable_ix = observable_ids.index(measurement[OBSERVABLE_ID])

            # update time index for observable
            if observable_ix in time_ix_for_obs_ix:
                time_ix_for_obs_ix[observable_ix] += 1
            else:
                time_ix_for_obs_ix[observable_ix] = time_ix_0

            # fill observable and possibly noise parameter
            y[time_ix_for_obs_ix[observable_ix], observable_ix] = measurement[
                MEASUREMENT
            ]
            if isinstance(
                measurement.get(NOISE_PARAMETERS, None), numbers.Number
            ):
                sigma_y[
                    time_ix_for_obs_ix[observable_ix], observable_ix
                ] = measurement[NOISE_PARAMETERS]
    return y, sigma_y
