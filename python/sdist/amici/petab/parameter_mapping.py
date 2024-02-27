from __future__ import annotations

"""
Parameter mapping
-----------------

When performing parameter inference, often parameters need to be mapped from
simulation to estimation parameters, and parameters can differ between
conditions. This can be handled using the `ParameterMapping`.

Note
~~~~

While the parameter mapping can be used directly with AMICI, it was developed
for usage together with PEtab, for which the whole workflow of generating
the mapping is automatized.
"""

import logging
import numbers
import re
from collections.abc import Sequence
from itertools import chain
from typing import Any, Union
from collections.abc import Collection, Iterator

import amici
import numpy as np
import pandas as pd
import petab
import sympy as sp
from amici.sbml_import import get_species_initial
from petab.C import *  # noqa: F403
from petab.C import (
    LIN,
    PARAMETER_SCALE,
    PREEQUILIBRATION_CONDITION_ID,
    SIMULATION_CONDITION_ID,
)
from petab.models import MODEL_TYPE_PYSB, MODEL_TYPE_SBML
from sympy.abc import _clash

from .. import AmiciModel
from . import PREEQ_INDICATOR_ID
from .util import get_states_in_condition_table

try:
    import pysb
except ImportError:
    pysb = None


logger = logging.getLogger(__name__)

SingleParameterMapping = dict[str, Union[numbers.Number, str]]
SingleScaleMapping = dict[str, str]


class ParameterMappingForCondition:
    """Parameter mapping for condition.

    Contains mappings for free parameters, fixed parameters, and fixed
    preequilibration parameters, both for parameters and scales.

    In the scale mappings, for each simulation parameter the scale
    on which the value is passed (and potentially gradients are to be
    returned) is given. In the parameter mappings, for each simulation
    parameter a corresponding optimization parameter (or a numeric value)
    is given.

    If a mapping is not passed, the parameter mappings are assumed to be empty,
    and if a scale mapping is not passed, all scales are set to linear.

    :param map_sim_var:
        Mapping for free simulation parameters.
    :param scale_map_sim_var:
        Scales for free simulation parameters.
    :param map_preeq_fix:
        Mapping for fixed preequilibration parameters.
    :param scale_map_preeq_fix:
        Scales for fixed preequilibration parameters.
    :param map_sim_fix:
        Mapping for fixed simulation parameters.
    :param scale_map_sim_fix:
        Scales for fixed simulation parameters.
    """

    def __init__(
        self,
        map_sim_var: SingleParameterMapping = None,
        scale_map_sim_var: SingleScaleMapping = None,
        map_preeq_fix: SingleParameterMapping = None,
        scale_map_preeq_fix: SingleScaleMapping = None,
        map_sim_fix: SingleParameterMapping = None,
        scale_map_sim_fix: SingleScaleMapping = None,
    ):
        if map_sim_var is None:
            map_sim_var = {}
        self.map_sim_var = map_sim_var

        if scale_map_sim_var is None:
            scale_map_sim_var = {key: LIN for key in map_sim_var}
        self.scale_map_sim_var = scale_map_sim_var

        if map_preeq_fix is None:
            map_preeq_fix = {}
        self.map_preeq_fix = map_preeq_fix

        if scale_map_preeq_fix is None:
            scale_map_preeq_fix = {key: LIN for key in map_preeq_fix}
        self.scale_map_preeq_fix = scale_map_preeq_fix

        if map_sim_fix is None:
            map_sim_fix = {}
        self.map_sim_fix = map_sim_fix

        if scale_map_sim_fix is None:
            scale_map_sim_fix = {key: LIN for key in map_sim_fix}
        self.scale_map_sim_fix = scale_map_sim_fix

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"map_sim_var={repr(self.map_sim_var)},"
            f"scale_map_sim_var={repr(self.scale_map_sim_var)},"
            f"map_preeq_fix={repr(self.map_preeq_fix)},"
            f"scale_map_preeq_fix={repr(self.scale_map_preeq_fix)},"
            f"map_sim_fix={repr(self.map_sim_fix)},"
            f"scale_map_sim_fix={repr(self.scale_map_sim_fix)})"
        )

    @property
    def free_symbols(self) -> set[str]:
        """Get IDs of all (symbolic) parameters present in this mapping"""
        return {
            p
            for p in chain(
                self.map_sim_var.values(),
                self.map_preeq_fix.values(),
                self.map_sim_fix.values(),
            )
            if isinstance(p, str)
        }


class ParameterMapping(Sequence):
    r"""Parameter mapping for multiple conditions.

    This can be used like a list of :class:`ParameterMappingForCondition`\ s.

    :param parameter_mappings:
        List of parameter mappings for specific conditions.
    """

    def __init__(
        self, parameter_mappings: list[ParameterMappingForCondition] = None
    ):
        super().__init__()
        if parameter_mappings is None:
            parameter_mappings = []
        self.parameter_mappings = parameter_mappings

    def __iter__(self):
        yield from self.parameter_mappings

    def __getitem__(
        self, item
    ) -> ParameterMapping | ParameterMappingForCondition:
        result = self.parameter_mappings[item]
        if isinstance(result, ParameterMappingForCondition):
            return result
        return ParameterMapping(result)

    def __len__(self):
        return len(self.parameter_mappings)

    def append(
        self, parameter_mapping_for_condition: ParameterMappingForCondition
    ):
        """Append a condition specific parameter mapping."""
        self.parameter_mappings.append(parameter_mapping_for_condition)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.parameter_mappings)})"

    @property
    def free_symbols(self) -> set[str]:
        """Get IDs of all (symbolic) parameters present in this mapping"""
        return set.union(*(mapping.free_symbols for mapping in self))


def petab_to_amici_scale(petab_scale: str) -> int:
    """Convert petab scale id to amici scale id."""
    if petab_scale == LIN:
        return amici.ParameterScaling_none
    if petab_scale == LOG10:
        return amici.ParameterScaling_log10
    if petab_scale == LOG:
        return amici.ParameterScaling_ln
    raise ValueError(f"PEtab scale not recognized: {petab_scale}")


def amici_to_petab_scale(amici_scale: int) -> str:
    """Convert amici scale id to petab scale id."""
    if amici_scale == amici.ParameterScaling_none:
        return LIN
    if amici_scale == amici.ParameterScaling_log10:
        return LOG10
    if amici_scale == amici.ParameterScaling_ln:
        return LOG
    raise ValueError(f"AMICI scale not recognized: {amici_scale}")


def scale_parameter(value: numbers.Number, petab_scale: str) -> numbers.Number:
    """Bring parameter from linear scale to target scale.

    :param value:
        Value to scale
    :param petab_scale:
        Target scale of ``value``

    :return:
        ``value`` on target scale
    """
    if petab_scale == LIN:
        return value
    if petab_scale == LOG10:
        return np.log10(value)
    if petab_scale == LOG:
        return np.log(value)
    raise ValueError(
        f"Unknown parameter scale {petab_scale}. "
        f"Must be from {(LIN, LOG, LOG10)}"
    )


def unscale_parameter(
    value: numbers.Number, petab_scale: str
) -> numbers.Number:
    """Bring parameter from scale to linear scale.

    :param value:
        Value to scale
    :param petab_scale:
        Target scale of ``value``

    :return:
        ``value`` on linear scale
    """
    if petab_scale == LIN:
        return value
    if petab_scale == LOG10:
        return np.power(10, value)
    if petab_scale == LOG:
        return np.exp(value)
    raise ValueError(
        f"Unknown parameter scale {petab_scale}. "
        f"Must be from {(LIN, LOG, LOG10)}"
    )


def scale_parameters_dict(
    value_dict: dict[Any, numbers.Number], petab_scale_dict: dict[Any, str]
) -> None:
    """
    Bring parameters from linear scale to target scale.

    Bring values in ``value_dict`` from linear scale to the scale
    provided in ``petab_scale_dict`` (in-place).
    Both arguments are expected to have the same length and matching keys.

    :param value_dict:
        Values to scale

    :param petab_scale_dict:
        Target scales of ``values``
    """
    if value_dict.keys() != petab_scale_dict.keys():
        raise AssertionError("Keys don't match.")

    for key, value in value_dict.items():
        value_dict[key] = scale_parameter(value, petab_scale_dict[key])


def unscale_parameters_dict(
    value_dict: dict[Any, numbers.Number], petab_scale_dict: dict[Any, str]
) -> None:
    """
    Bring parameters from target scale to linear scale.

    Bring values in ``value_dict`` from linear scale to the scale
    provided in ``petab_scale_dict`` (in-place).
    Both arguments are expected to have the same length and matching keys.

    :param value_dict:
        Values to scale

    :param petab_scale_dict:
        Target scales of ``values``
    """
    if value_dict.keys() != petab_scale_dict.keys():
        raise AssertionError("Keys don't match.")

    for key, value in value_dict.items():
        value_dict[key] = unscale_parameter(value, petab_scale_dict[key])


def create_parameter_mapping(
    petab_problem: petab.Problem,
    simulation_conditions: pd.DataFrame | list[dict],
    scaled_parameters: bool,
    amici_model: AmiciModel,
    **parameter_mapping_kwargs,
) -> ParameterMapping:
    """Generate AMICI specific parameter mapping.

    :param petab_problem:
        PEtab problem
    :param simulation_conditions:
        Result of :func:`petab.get_simulation_conditions`. Can be provided to
        save time if this has been obtained before.
    :param scaled_parameters:
        If ``True``, problem_parameters are assumed to be on the scale provided
        in the PEtab parameter table and will be unscaled. If ``False``, they
        are assumed to be in linear scale.
    :param amici_model:
        AMICI model.
    :param parameter_mapping_kwargs:
        Optional keyword arguments passed to
        :func:`petab.get_optimization_to_simulation_parameter_mapping`.
        To allow changing fixed PEtab problem parameters (``estimate=0``),
        use ``fill_fixed_parameters=False``.
    :return:
        List of the parameter mappings.
    """
    if simulation_conditions is None:
        simulation_conditions = (
            petab_problem.get_simulation_conditions_from_measurement_df()
        )
    if isinstance(simulation_conditions, list):
        simulation_conditions = pd.DataFrame(data=simulation_conditions)

    # Because AMICI globalizes all local parameters during model import,
    # we need to do that here as well to prevent parameter mapping errors
    # (PEtab does currently not care about SBML LocalParameters)
    if petab_problem.model.type_id == MODEL_TYPE_SBML:
        import libsbml

        if petab_problem.sbml_document:
            converter_config = (
                libsbml.SBMLLocalParameterConverter().getDefaultProperties()
            )
            petab_problem.sbml_document.convert(converter_config)
        else:
            logger.debug(
                "No petab_problem.sbml_document is set. Cannot "
                "convert SBML LocalParameters. If the model contains "
                "LocalParameters, parameter mapping will fail."
            )

    default_parameter_mapping_kwargs = {
        "warn_unmapped": False,
        "scaled_parameters": scaled_parameters,
        "allow_timepoint_specific_numeric_noise_parameters": not petab.lint.observable_table_has_nontrivial_noise_formula(
            petab_problem.observable_df
        ),
    }
    if parameter_mapping_kwargs is None:
        parameter_mapping_kwargs = {}

    prelim_parameter_mapping = (
        petab.get_optimization_to_simulation_parameter_mapping(
            condition_df=petab_problem.condition_df,
            measurement_df=petab_problem.measurement_df,
            parameter_df=petab_problem.parameter_df,
            observable_df=petab_problem.observable_df,
            mapping_df=petab_problem.mapping_df,
            model=petab_problem.model,
            simulation_conditions=simulation_conditions,
            **dict(
                default_parameter_mapping_kwargs, **parameter_mapping_kwargs
            ),
        )
    )

    parameter_mapping = ParameterMapping()
    for (_, condition), prelim_mapping_for_condition in zip(
        simulation_conditions.iterrows(), prelim_parameter_mapping
    ):
        mapping_for_condition = create_parameter_mapping_for_condition(
            prelim_mapping_for_condition, condition, petab_problem, amici_model
        )
        parameter_mapping.append(mapping_for_condition)

    return parameter_mapping


def create_parameter_mapping_for_condition(
    parameter_mapping_for_condition: petab.ParMappingDictQuadruple,
    condition: pd.Series | dict,
    petab_problem: petab.Problem,
    amici_model: AmiciModel,
) -> ParameterMappingForCondition:
    """Generate AMICI specific parameter mapping for condition.

    :param parameter_mapping_for_condition:
        Preliminary parameter mapping for condition.
    :param condition:
        :class:`pandas.DataFrame` row with ``preequilibrationConditionId`` and
        ``simulationConditionId``.
    :param petab_problem:
        Underlying PEtab problem.
    :param amici_model:
        AMICI model.

    :return:
        The parameter and parameter scale mappings, for fixed
        preequilibration, fixed simulation, and variable simulation
        parameters, and then the respective scalings.
    """
    (
        condition_map_preeq,
        condition_map_sim,
        condition_scale_map_preeq,
        condition_scale_map_sim,
    ) = parameter_mapping_for_condition
    logger.debug(f"PEtab mapping: {parameter_mapping_for_condition}")

    if len(condition_map_preeq) != len(condition_scale_map_preeq) or len(
        condition_map_sim
    ) != len(condition_scale_map_sim):
        raise AssertionError(
            "Number of parameters and number of parameter "
            "scales do not match."
        )
    if len(condition_map_preeq) and len(condition_map_preeq) != len(
        condition_map_sim
    ):
        logger.debug(f"Preequilibration parameter map: {condition_map_preeq}")
        logger.debug(f"Simulation parameter map: {condition_map_sim}")
        raise AssertionError(
            "Number of parameters for preequilbration "
            "and simulation do not match."
        )

    ##########################################################################
    # initial states
    # Initial states have been set during model import based on the SBML model.
    # If initial states were overwritten in the PEtab condition table, they are
    # applied here.
    # During model generation, parameters for initial concentrations and
    # respective initial assignments have been created for the
    # relevant species, here we add these parameters to the parameter mapping.
    # In absence of preequilibration this could also be handled via
    # ExpData.x0, but in the case of preequilibration this would not allow for
    # resetting initial states.

    if states_in_condition_table := get_states_in_condition_table(
        petab_problem, condition
    ):
        # set indicator fixed parameter for preeq
        # (we expect here, that this parameter was added during import and
        # that it was not added by the user with a different meaning...)
        if condition_map_preeq:
            condition_map_preeq[PREEQ_INDICATOR_ID] = 1.0
            condition_scale_map_preeq[PREEQ_INDICATOR_ID] = LIN

        condition_map_sim[PREEQ_INDICATOR_ID] = 0.0
        condition_scale_map_sim[PREEQ_INDICATOR_ID] = LIN

        for element_id, (
            value,
            preeq_value,
        ) in states_in_condition_table.items():
            # for preequilibration
            init_par_id = f"initial_{element_id}_preeq"
            if (
                condition_id := condition.get(PREEQUILIBRATION_CONDITION_ID)
            ) is not None:
                _set_initial_state(
                    petab_problem,
                    condition_id,
                    element_id,
                    init_par_id,
                    condition_map_preeq,
                    condition_scale_map_preeq,
                    preeq_value,
                )
            else:
                # need to set dummy value for preeq parameter anyways, as it
                #  is expected below (set to 0, not nan, because will be
                #  multiplied with indicator variable in initial assignment)
                condition_map_sim[init_par_id] = 0.0
                condition_scale_map_sim[init_par_id] = LIN

            # for simulation
            condition_id = condition[SIMULATION_CONDITION_ID]
            init_par_id = f"initial_{element_id}_sim"
            _set_initial_state(
                petab_problem,
                condition_id,
                element_id,
                init_par_id,
                condition_map_sim,
                condition_scale_map_sim,
                value,
            )

    ##########################################################################
    # separate fixed and variable AMICI parameters, because we may have
    # different fixed parameters for preeq and sim condition, but we cannot
    # have different variable parameters. without splitting,
    # merge_preeq_and_sim_pars_condition below may fail.
    # TODO: This can be done already in parameter mapping creation.
    variable_par_ids = amici_model.getParameterIds()
    fixed_par_ids = amici_model.getFixedParameterIds()

    condition_map_preeq_var, condition_map_preeq_fix = _subset_dict(
        condition_map_preeq, variable_par_ids, fixed_par_ids
    )

    (
        condition_scale_map_preeq_var,
        condition_scale_map_preeq_fix,
    ) = _subset_dict(
        condition_scale_map_preeq, variable_par_ids, fixed_par_ids
    )

    condition_map_sim_var, condition_map_sim_fix = _subset_dict(
        condition_map_sim, variable_par_ids, fixed_par_ids
    )

    condition_scale_map_sim_var, condition_scale_map_sim_fix = _subset_dict(
        condition_scale_map_sim, variable_par_ids, fixed_par_ids
    )

    logger.debug(
        "Fixed parameters preequilibration: " f"{condition_map_preeq_fix}"
    )
    logger.debug("Fixed parameters simulation: " f"{condition_map_sim_fix}")
    logger.debug(
        "Variable parameters preequilibration: " f"{condition_map_preeq_var}"
    )
    logger.debug("Variable parameters simulation: " f"{condition_map_sim_var}")

    petab.merge_preeq_and_sim_pars_condition(
        condition_map_preeq_var,
        condition_map_sim_var,
        condition_scale_map_preeq_var,
        condition_scale_map_sim_var,
        condition,
    )
    logger.debug(f"Merged: {condition_map_sim_var}")

    parameter_mapping_for_condition = ParameterMappingForCondition(
        map_preeq_fix=condition_map_preeq_fix,
        map_sim_fix=condition_map_sim_fix,
        map_sim_var=condition_map_sim_var,
        scale_map_preeq_fix=condition_scale_map_preeq_fix,
        scale_map_sim_fix=condition_scale_map_sim_fix,
        scale_map_sim_var=condition_scale_map_sim_var,
    )

    return parameter_mapping_for_condition


def _set_initial_state(
    petab_problem,
    condition_id,
    element_id,
    init_par_id,
    par_map,
    scale_map,
    value,
):
    value = petab.to_float_if_float(value)
    if pd.isna(value):
        if petab_problem.model.type_id == MODEL_TYPE_SBML:
            value = _get_initial_state_sbml(petab_problem, element_id)
        elif petab_problem.model.type_id == MODEL_TYPE_PYSB:
            value = _get_initial_state_pysb(petab_problem, element_id)

        try:
            value = float(value)
        except (ValueError, TypeError):
            if sp.nsimplify(value).is_Atom and (
                pysb is None or not isinstance(value, pysb.Component)
            ):
                # Get rid of multiplication with one
                value = sp.nsimplify(value)
            else:
                raise NotImplementedError(
                    "Cannot handle non-trivial initial state "
                    f"expression for {element_id}: {value}"
                )
            # this should be a parameter ID
            value = str(value)
        logger.debug(
            f"The species {element_id} has no initial value "
            f"defined for the condition {condition_id} in "
            "the PEtab conditions table. The initial value is "
            f"now set to {value}, which is the initial value "
            "defined in the SBML model."
        )
    par_map[init_par_id] = value
    if isinstance(value, float):
        # numeric initial state
        scale_map[init_par_id] = petab.LIN
    else:
        # parametric initial state
        scale_map[init_par_id] = petab_problem.parameter_df[
            PARAMETER_SCALE
        ].get(value, petab.LIN)


def _subset_dict(
    full: dict[Any, Any], *args: Collection[Any]
) -> Iterator[dict[Any, Any]]:
    """Get subset of dictionary based on provided keys

    :param full:
        Dictionary to subset
    :param args:
        Collections of keys to be contained in the different subsets

    :return:
        subsetted dictionary
    """
    for keys in args:
        yield {key: val for (key, val) in full.items() if key in keys}


def _get_initial_state_sbml(
    petab_problem: petab.Problem, element_id: str
) -> float | sp.Basic:
    import libsbml

    element = petab_problem.sbml_model.getElementBySId(element_id)
    type_code = element.getTypeCode()
    initial_assignment = petab_problem.sbml_model.getInitialAssignmentBySymbol(
        element_id
    )
    if initial_assignment:
        initial_assignment = sp.sympify(
            libsbml.formulaToL3String(initial_assignment.getMath()),
            locals=_clash,
        )
    if type_code == libsbml.SBML_SPECIES:
        value = (
            get_species_initial(element)
            if initial_assignment is None
            else initial_assignment
        )
    elif type_code == libsbml.SBML_PARAMETER:
        value = (
            element.getValue()
            if initial_assignment is None
            else initial_assignment
        )
    elif type_code == libsbml.SBML_COMPARTMENT:
        value = (
            element.getSize()
            if initial_assignment is None
            else initial_assignment
        )
    else:
        raise NotImplementedError(
            f"Don't know what how to handle {element_id} in "
            "condition table."
        )
    return value


def _get_initial_state_pysb(
    petab_problem: petab.Problem, element_id: str
) -> float | sp.Symbol:
    species_idx = int(re.match(r"__s(\d+)$", element_id)[1])
    species_pattern = petab_problem.model.model.species[species_idx]
    from pysb.pattern import match_complex_pattern

    value = next(
        (
            initial.value
            for initial in petab_problem.model.model.initials
            if match_complex_pattern(
                initial.pattern, species_pattern, exact=True
            )
        ),
        0.0,
    )
    if isinstance(value, pysb.Parameter):
        if value.name in petab_problem.parameter_df.index:
            value = value.name
        else:
            value = value.value

    return value
