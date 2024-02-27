"""General helper functions for PEtab import.

Functions for PEtab import that are independent of the model format.
"""
import importlib
import logging
import os
import re
from pathlib import Path
from typing import Union

import amici
import pandas as pd
import petab
import sympy as sp
from petab.C import (
    CONDITION_NAME,
    ESTIMATE,
    NOISE_DISTRIBUTION,
    NOISE_FORMULA,
    OBSERVABLE_FORMULA,
    OBSERVABLE_NAME,
    OBSERVABLE_TRANSFORMATION,
)
from petab.parameters import get_valid_parameters_for_parameter_table
from sympy.abc import _clash

logger = logging.getLogger(__name__)


def get_observation_model(
    observable_df: pd.DataFrame,
) -> tuple[
    dict[str, dict[str, str]], dict[str, str], dict[str, Union[str, float]]
]:
    """
    Get observables, sigmas, and noise distributions from PEtab observation
    table in a format suitable for
    :meth:`amici.sbml_import.SbmlImporter.sbml2amici`.

    :param observable_df:
        PEtab observables table

    :return:
        Tuple of dicts with observables, noise distributions, and sigmas.
    """
    if observable_df is None:
        return {}, {}, {}

    observables = {}
    sigmas = {}

    nan_pat = r"^[nN]a[nN]$"
    for _, observable in observable_df.iterrows():
        oid = str(observable.name)
        # need to sanitize due to https://github.com/PEtab-dev/PEtab/issues/447
        name = re.sub(nan_pat, "", str(observable.get(OBSERVABLE_NAME, "")))
        formula_obs = re.sub(nan_pat, "", str(observable[OBSERVABLE_FORMULA]))
        formula_noise = re.sub(nan_pat, "", str(observable[NOISE_FORMULA]))
        observables[oid] = {"name": name, "formula": formula_obs}
        sigmas[oid] = formula_noise

    # PEtab does currently not allow observables in noiseFormula and AMICI
    #  cannot handle states in sigma expressions. Therefore, where possible,
    #  replace species occurring in error model definition by observableIds.
    replacements = {
        sp.sympify(observable["formula"], locals=_clash): sp.Symbol(
            observable_id
        )
        for observable_id, observable in observables.items()
    }
    for observable_id, formula in sigmas.items():
        repl = sp.sympify(formula, locals=_clash).subs(replacements)
        sigmas[observable_id] = str(repl)

    noise_distrs = petab_noise_distributions_to_amici(observable_df)

    return observables, noise_distrs, sigmas


def petab_noise_distributions_to_amici(
    observable_df: pd.DataFrame,
) -> dict[str, str]:
    """
    Map from the petab to the amici format of noise distribution
    identifiers.

    :param observable_df:
        PEtab observable table

    :return:
        dictionary of observable_id => AMICI noise-distributions
    """
    amici_distrs = {}
    for _, observable in observable_df.iterrows():
        amici_val = ""

        if (
            OBSERVABLE_TRANSFORMATION in observable
            and isinstance(observable[OBSERVABLE_TRANSFORMATION], str)
            and observable[OBSERVABLE_TRANSFORMATION]
        ):
            amici_val += observable[OBSERVABLE_TRANSFORMATION] + "-"

        if (
            NOISE_DISTRIBUTION in observable
            and isinstance(observable[NOISE_DISTRIBUTION], str)
            and observable[NOISE_DISTRIBUTION]
        ):
            amici_val += observable[NOISE_DISTRIBUTION]
        else:
            amici_val += "normal"
        amici_distrs[observable.name] = amici_val

    return amici_distrs


def petab_scale_to_amici_scale(scale_str: str) -> int:
    """Convert PEtab parameter scaling string to AMICI scaling integer"""

    if scale_str == petab.LIN:
        return amici.ParameterScaling_none
    if scale_str == petab.LOG:
        return amici.ParameterScaling_ln
    if scale_str == petab.LOG10:
        return amici.ParameterScaling_log10

    raise ValueError(f"Invalid parameter scale {scale_str}")


def _create_model_name(folder: Union[str, Path]) -> str:
    """
    Create a name for the model.
    Just re-use the last part of the folder.
    """
    return os.path.split(os.path.normpath(folder))[-1]


def _can_import_model(
    model_name: str, model_output_dir: Union[str, Path]
) -> bool:
    """
    Check whether a module of that name can already be imported.
    """
    # try to import (in particular checks version)
    try:
        with amici.add_path(model_output_dir):
            model_module = importlib.import_module(model_name)
    except ModuleNotFoundError:
        return False

    # no need to (re-)compile
    return hasattr(model_module, "getModel")


def get_fixed_parameters(
    petab_problem: petab.Problem,
    non_estimated_parameters_as_constants=True,
) -> list[str]:
    """
    Determine, set and return fixed model parameters.

    Non-estimated parameters and parameters specified in the condition table
    are turned into constants (unless they are overridden).
    Only global SBML parameters are considered. Local parameters are ignored.

    :param petab_problem:
        The PEtab problem instance

    :param non_estimated_parameters_as_constants:
        Whether parameters marked as non-estimated in PEtab should be
        considered constant in AMICI. Setting this to ``True`` will reduce
        model size and simulation times. If sensitivities with respect to those
        parameters are required, this should be set to ``False``.

    :return:
        list of IDs of parameters which are to be considered constant.
    """
    # if we have a parameter table, all parameters that are allowed to be
    #  listed in the parameter table, but are not marked as estimated, can be
    #  turned into AMICI constants
    # due to legacy API, we might not always have a parameter table, though
    fixed_parameters = set()
    if petab_problem.parameter_df is not None:
        all_parameters = get_valid_parameters_for_parameter_table(
            model=petab_problem.model,
            condition_df=petab_problem.condition_df,
            observable_df=petab_problem.observable_df
            if petab_problem.observable_df is not None
            else pd.DataFrame(columns=petab.OBSERVABLE_DF_REQUIRED_COLS),
            measurement_df=petab_problem.measurement_df
            if petab_problem.measurement_df is not None
            else pd.DataFrame(columns=petab.MEASUREMENT_DF_REQUIRED_COLS),
        )
        if non_estimated_parameters_as_constants:
            estimated_parameters = petab_problem.parameter_df.index.values[
                petab_problem.parameter_df[ESTIMATE] == 1
            ]
        else:
            # don't treat parameter table parameters as constants
            estimated_parameters = petab_problem.parameter_df.index.values
        fixed_parameters = set(all_parameters) - set(estimated_parameters)

    # Column names are model parameter IDs, compartment IDs or species IDs.
    # Thereof, all parameters except for any overridden ones should be made
    # constant.
    # (Could potentially still be made constant, but leaving them might
    # increase model reusability)

    # handle parameters in condition table
    condition_df = petab_problem.condition_df
    if condition_df is not None:
        logger.debug(f"Condition table: {condition_df.shape}")

        # remove overridden parameters (`object`-type columns)
        fixed_parameters.update(
            p
            for p in condition_df.columns
            # get rid of conditionName column
            if p != CONDITION_NAME
            # there is no parametric override
            # TODO: could check if the final overriding parameter is estimated
            #  or not, but for now, we skip the parameter if there is any kind
            #  of overriding
            if condition_df[p].dtype != "O"
            # p is a parameter
            and not petab_problem.model.is_state_variable(p)
        )

    # Ensure mentioned parameters exist in the model. Remove additional ones
    # from list
    for fixed_parameter in fixed_parameters.copy():
        # check global parameters
        if not petab_problem.model.has_entity_with_id(fixed_parameter):
            # TODO: could still exist as an output parameter?
            logger.warning(
                f"Column '{fixed_parameter}' used in condition "
                "table but not entity with the corresponding ID "
                "exists. Ignoring."
            )
            fixed_parameters.remove(fixed_parameter)

    return list(sorted(fixed_parameters))


def check_model(
    amici_model: amici.Model,
    petab_problem: petab.Problem,
) -> None:
    """Check that the model is consistent with the PEtab problem."""
    if petab_problem.parameter_df is None:
        return

    amici_ids_free = set(amici_model.getParameterIds())
    amici_ids = amici_ids_free | set(amici_model.getFixedParameterIds())

    petab_ids_free = set(
        petab_problem.parameter_df.loc[
            petab_problem.parameter_df[ESTIMATE] == 1
        ].index
    )

    amici_ids_free_required = petab_ids_free.intersection(amici_ids)

    if not amici_ids_free_required.issubset(amici_ids_free):
        raise ValueError(
            "The available AMICI model does not support estimating the "
            "following parameters. Please recompile the model and ensure "
            "that these parameters are not treated as constants. Deleting "
            "the current model might also resolve this. Parameters: "
            f"{amici_ids_free_required.difference(amici_ids_free)}"
        )
