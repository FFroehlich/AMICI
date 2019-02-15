import pandas as pd
import numpy as np
import math
import copy

from .numpy import edataToNumPyArrays
import amici
from amici import ExpData


def getDataObservablesAsDataFrame(model, edata_list, by_id=False):
    """ 
    Write Observables from experimental data as DataFrame.

    Arguments:
        model: Model instance.
        edata_list: list of ExpData instances with experimental data.
        by_id: bool (optional, default = False)
            If True, uses observable ids as identifiers in dataframe,
            otherwise the possibly more descriptive observable names
            are used.

    Returns:
        pandas DataFrame with conditions and observables.

    Raises:

    """
    if isinstance(edata_list, amici.amici.ExpData):
        edata_list = [edata_list]

    # find observable column names using either parameter ids or names
    cols = _get_extended_observable_cols(model, by_id=by_id)

    df_edata = pd.DataFrame(columns=cols)
    for edata in edata_list:
        npdata = edataToNumPyArrays(edata)
        for i_time, timepoint in enumerate(edata.getTimepoints()):
            datadict = {
                'time': timepoint,
                'datatype': 'data'
            }
            for i_obs, obs in enumerate(_get_names_or_ids(
                    model, 'Observable', by_id=by_id)):
                datadict[obs] = npdata['observedData'][i_time, i_obs]
                datadict[obs + '_std'] = npdata['observedDataStdDev'][
                    i_time, i_obs]

            _fill_conditions_dict(datadict, model, edata, by_id=by_id)

            df_edata.loc[len(df_edata)] = datadict

    return df_edata


def getSimulationObservablesAsDataFrame(
        model, edata_list, rdata_list, by_id=False):
    """ Write Observables from simulation results as DataFrame

    Arguments:
        model: Model instance
        edata_list: list of ExpData instances with experimental data
        rdata_list: list of ReturnData instances corresponding to ExpData
        by_id: bool, optional (default = False)
            If True, ids are used as identifiers, otherwise the possibly more
            descriptive names.

    Returns:
        pandas DataFrame with conditions and observables

    Raises:

    """
    if isinstance(edata_list, amici.amici.ExpData):
        edata_list = [edata_list]

    cols = _get_extended_observable_cols(model, by_id=by_id)
    df_rdata = pd.DataFrame(columns=cols)

    for edata, rdata in zip(edata_list, rdata_list):
        for i_time, timepoint in enumerate(rdata['t']):
            datadict = {
                'time': timepoint,
                'datatype': 'simulation',
            }
            for i_obs, obs in enumerate(_get_names_or_ids(
                    model, 'Observable', by_id=by_id)):
                datadict[obs] = rdata['y'][i_time, i_obs]
                datadict[obs + '_std'] = rdata['sigmay'][i_time, i_obs]

            _fill_conditions_dict(datadict, model, edata, by_id=by_id)

            df_rdata.loc[len(df_rdata)] = datadict

    return df_rdata


def getSimulationStatesAsDataFrame(
        model, edata_list, rdata_list, by_id=False):
    """ Compute model residuals according to a list of ReturnData and ExpData

    Arguments:
        model: Model instance
        edata_list: list of ExpData instances with experimental data
        rdata_list: list of ReturnData instances corresponding to ExpData
        by_id: bool, optional (default = False)
            If True, ids are used as identifiers, otherwise the possibly more
            descriptive names.

    Returns:
        pandas DataFrame with conditions and observables

    Raises:

    """
    if isinstance(edata_list, amici.amici.ExpData):
        edata_list = [edata_list]
    if isinstance(rdata_list, amici.amici.ReturnData):
        rdata_list = [rdata_list]

    cols = _get_state_cols(model, by_id=by_id)
    df_rdata = pd.DataFrame(columns=cols)

    for edata, rdata in zip(edata_list, rdata_list):
        for i_time, timepoint in enumerate(rdata['t']):
            datadict = {
                'time': timepoint,
            }

            for i_state, state in enumerate(
                    _get_names_or_ids(model, 'State', by_id=by_id)):
                datadict[state] = rdata['x'][i_time, i_state]

            _fill_conditions_dict(datadict, model, edata, by_id=by_id)

            df_rdata.loc[len(df_rdata)] = datadict
    return df_rdata


def getResidualsAsDataFrame(model, edata_list, rdata_list, by_id=False):
    """ Convert a list of ExpData to pandas DataFrame

    Arguments:
        model: Model instance
        edata_list: list of ExpData instances with experimental data
        rdata_list: list of ReturnData instances corresponding to ExpData
        by_id: bool, optional (default = False)
            If True, ids are used as identifiers, otherwise the possibly more
            descriptive names.

    Returns:
        pandas DataFrame with conditions and observables

    Raises:

    """
    if isinstance(edata_list, amici.amici.ExpData):
        edata_list = [edata_list]
    if isinstance(rdata_list, amici.amici.ReturnData):
        rdata_list = [rdata_list]

    df_edata = getDataObservablesAsDataFrame(model, edata_list, by_id=by_id)
    df_rdata = getSimulationObservablesAsDataFrame(model, edata_list,
                                                   rdata_list, by_id=by_id)

    cols = _get_observable_cols(model, by_id=by_id)

    df_res = pd.DataFrame(columns=cols)

    for row in df_rdata.index:
        datadict = {
            'time': df_rdata.loc[row]['time'],
            't_presim': df_rdata.loc[row]['t_presim']
        }
        for obs in _get_names_or_ids(model, 'Observable', by_id=by_id):
            datadict[obs] = abs(
                (df_edata.loc[row][obs] - df_rdata.loc[row][obs]) /
                df_rdata.loc[row][obs + '_std'])
        for par in _get_names_or_ids(model, 'FixedParameter', by_id=by_id):
            datadict[par] = df_rdata.loc[row][par]
            datadict[par + '_preeq'] = df_rdata.loc[row][par + '_preeq']
            datadict[par + '_presim'] = df_rdata.loc[row][par + '_presim']
        df_res.loc[len(df_res)] = datadict

    return df_res


def _fill_conditions_dict(datadict, model, edata, by_id) -> dict:
    """ Helper function that fills in condition parameters from model and edata

    Arguments:
        datadict: dictionary in which condition parameters will be inserted
        as key value pairs
        model: Model instance
        edata: ExpData instance
        by_id: bool
            If True, ids are used as identifiers, otherwise the possibly more
            descriptive names.
    Returns:
        dictionary with filled condition parameters

    Raises:

    """
    datadict['t_presim'] = edata.t_presim

    for i_par, par in enumerate(_get_names_or_ids(model, 'FixedParameter', by_id=by_id)):
        if len(edata.fixedParameters):
            datadict[par] = edata.fixedParameters[i_par]
        else:
            datadict[par] = model.getFixedParameters()[i_par]

        if len(edata.fixedParametersPreequilibration):
            datadict[par + '_preeq'] = \
                edata.fixedParametersPreequilibration[i_par]
        else:
            datadict[par + '_preeq'] = math.nan

        if len(edata.fixedParametersPresimulation):
            datadict[par + '_presim'] = \
                edata.fixedParametersPresimulation[i_par]
        else:
            datadict[par + '_presim'] = math.nan
    return datadict


def _get_extended_observable_cols(model, by_id) -> list:
    """ Construction helper for extended observable dataframe headers

    Arguments:
        model: Model instance
        by_id: bool
            If True, ids are used as identifiers, otherwise the possibly more
            descriptive names.

    Returns:
        column names as list

    Raises:

    """
    return \
        ['time', 'datatype', 't_presim'] + \
        _get_names_or_ids(model, 'FixedParameter', by_id=by_id) + \
        [name + '_preeq' for name in
         _get_names_or_ids(model, 'FixedParameter', by_id=by_id)] + \
        [name + '_presim' for name in
         _get_names_or_ids(model, 'FixedParameter', by_id=by_id)] + \
        _get_names_or_ids(model, 'Observable', by_id=by_id) + \
        [name + '_std' for name in _get_names_or_ids(
            model, 'Observable', by_id=by_id)]


def _get_observable_cols(model, by_id):
    """ Construction helper for observable dataframe headers

    Arguments:
        model: Model instance
        by_id: bool
            If True, ids are used as identifiers, otherwise the possibly more
            descriptive names.

    Returns:
        column names as list

    Raises:

    """
    return \
        ['time', 't_presim'] + \
        _get_names_or_ids(model, 'FixedParameter', by_id=by_id) + \
        [name + '_preeq' for name in
         _get_names_or_ids(model, 'FixedParameter', by_id=by_id)] + \
        [name + '_presim' for name in
         _get_names_or_ids(model, 'FixedParameter', by_id=by_id)] + \
        _get_names_or_ids(model, 'Observable', by_id=by_id)


def _get_state_cols(model, by_id):
    """ Construction helper for state dataframe headers

    Arguments:
        model: Model instance
        by_id: bool
            If True, ids are used as identifiers, otherwise the possibly more
            descriptive names.

    Returns:
        column names as list

    Raises:

    """
    return \
        ['time', 't_presim'] + \
        _get_names_or_ids(model, 'FixedParameter', by_id=by_id) + \
        [name + '_preeq' for name in
         _get_names_or_ids(model, 'FixedParameter', by_id=by_id)] + \
        [name + '_presim' for name in
         _get_names_or_ids(model, 'FixedParameter', by_id=by_id)] + \
        _get_names_or_ids(model, 'State', by_id=by_id)


def _get_names_or_ids(model, variable, by_id):
    """ Obtains a unique list of identifiers for the specified variable
        first tries model.getVariableNames and then uses model.getVariableIds

    Arguments:
        model: Model instance
        variable: variable name
        by_id: bool
            If True, ids are used as identifiers, otherwise the possibly more
            descriptive names.

    Returns:
        column names as list

    Raises:

    """
    variable_options = ['Parameter', 'FixedParameter', 'Observable', 'State']
    if variable not in variable_options:
        raise ValueError('variable must be in ' + str(variable_options))
    namegetter = getattr(model, 'get' + variable + 'Names')
    idgetter = getattr(model, 'get' + variable + 'Ids')
    if not by_id and len(set(namegetter())) == len(namegetter()) \
            and model.hasObservableNames():
        return list(namegetter())
    elif model.hasObservableIds():
        return list(idgetter())
    else:
        raise RuntimeError('Model Observable Names are not unique and '
                           'Observable Ids are not set. ')


def _get_specialized_fixed_parameters(
        model, condition, overwrite, by_id) -> list:
    """ Copies values in condition and overwrites them according to key
    value pairs specified in overwrite

    Arguments:
        model: Model instance
        condition: dict/pd.Series containing FixedParameter values
        overwrite: dict specifying which values in condition are to be replaced
        by_id: bool
            If True, ids are used as identifiers, otherwise the possibly more
            descriptive names.

    Returns:
        overwritten FixedParameter as list

    Raises:

    """
    cond = copy.deepcopy(condition)
    for field in overwrite:
        cond[field] = overwrite[field]
    return [cond[name] for name in _get_names_or_ids(
        model, 'FixedParameter', by_id=by_id)]


def constructEdataFromDataFrame(df, model, condition):
    """ Constructs an ExpData instance according to the provided Model and DataFrame

    Arguments:
        df: pd.DataFrame with Observable Names/Ids as columns
            standard deviations may be specified by appending '_std' as suffix
        model: Model instance
        condition: pd.Series with FixedParameter Names/Ids as columns
            preequilibration conditions may be specified by appending '_preeq' as suffix
            presimulation conditions may be specified by appending '_presim' as suffix

    Returns:
        ExpData instance

    Raises:

    """
    edata = ExpData(model.get())

    # timepoints
    df = df.sort_values(by='time', ascending=True)
    edata.setTimepoints(df['time'].values)

    overwrite_preeq = {}
    overwrite_presim = {}
    for par in list(_get_names_or_ids(model, 'FixedParameter', by_id=False)):
        if par + '_preeq' in condition.keys() \
                and not math.isnan(condition[par + '_preeq']):
            overwrite_preeq[par] = condition[par + '_preeq']
        if par + '_presim' in condition.keys() \
                and not math.isnan(condition[par + '_presim']):
            overwrite_presim[par] = condition[par + '_presim']

    # fixedParameters
    edata.fixedParameters = \
        condition[_get_names_or_ids(model, 'FixedParameter', by_id=False)].values

    if any([overwrite_preeq[key] != condition[key] for key in
            overwrite_preeq.keys()]):
        edata.fixedParametersPreequilibration = \
            _get_specialized_fixed_parameters(model, condition,overwrite_preeq, by_id=False)
    elif len(overwrite_preeq.keys()):
        edata.fixedParametersPreequilibration = copy.deepcopy(
            edata.fixedParameters
        )


    if any([overwrite_presim[key] != condition[key] for key in
            overwrite_presim.keys()]):
        edata.fixedParametersPresimulation = _get_specialized_fixed_parameters(
            model, condition,overwrite_presim, by_id=False
        )
    elif len(overwrite_presim.keys()):
        edata.fixedParametersPresimulation = copy.deepcopy(
            edata.fixedParameters
        )

    if 't_presim' in condition.keys():
        edata.t_presim = condition['t_presim']

    # data
    for obs_index, obs in enumerate(_get_names_or_ids(model, 'Observable', by_id=False)):
        if obs in df.keys():
            edata.setObservedData(df[obs].values,
                                  obs_index)
        if obs + '_std' in df.keys():
            edata.setObservedDataStdDev(
                df[obs + '_std'].values,
                obs_index
            )

    return edata


def getEdataFromDataFrame(model, df):
    """ Constructs a ExpData instance according to the provided Model and DataFrame

    Arguments:
        df: pd.DataFrame with Observable Names/Ids, FixedParameter Names/Ids and time as columns
            standard deviations may be specified by appending '_std' as suffix
            preequilibration fixedParameters may be specified by appending '_preeq' as suffix
            presimulation fixedParameters may be specified by appending '_presim' as suffix
            presimulation time may be specified as 't_presim' column
        model: Model instance

    Returns:
        ExpData instance

    Raises:

    """
    edata_list = []
    # aggregate features that define a condition
    condition_parameters = _get_names_or_ids(model, 'FixedParameter', by_id=False)
    for par in _get_names_or_ids(model, 'FixedParameter', by_id=False):
        if par + '_preeq' in df.columns:
            condition_parameters.append(par + '_preeq')
        if par + '_presim' in df.columns:
            condition_parameters.append(par + '_presim')
    if 't_presim' in df.columns:
        condition_parameters.append('t_presim')
    conditions = df[condition_parameters].drop_duplicates()

    for row in conditions.iterrows():
        # subselect rows that match condition
        selected = np.ones((len(df),), dtype=bool)
        for par_label, par in row[1].iteritems():
            if math.isnan(par):
                selected = selected & np.isnan(df[par_label].values)
            else:
                selected = selected & (df[par_label] == par)
        edata_df = df[selected]

        edata_list.append(constructEdataFromDataFrame(edata_df, model, row[1]))

    return edata_list
