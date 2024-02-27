"""
PySB-PEtab Import
-----------------
Import a model in the PySB-adapted :mod:`petab`
(https://github.com/PEtab-dev/PEtab) format into AMICI.
"""

import logging
import re
from pathlib import Path
from typing import Optional, Union

import petab
import pysb
import pysb.bng
import sympy as sp
from petab.C import CONDITION_NAME, NOISE_FORMULA, OBSERVABLE_FORMULA
from petab.models.pysb_model import PySBModel

from ..logging import get_logger, log_execution_time, set_log_level
from . import PREEQ_INDICATOR_ID
from .import_helpers import (
    get_fixed_parameters,
    petab_noise_distributions_to_amici,
)
from .util import get_states_in_condition_table

logger = get_logger(__name__, logging.WARNING)


def _add_observation_model(
    pysb_model: pysb.Model, petab_problem: petab.Problem
):
    """Extend PySB model by observation model as defined in the PEtab
    observables table"""

    # add any required output parameters
    local_syms = {
        sp.Symbol.__str__(comp): comp
        for comp in pysb_model.components
        if isinstance(comp, sp.Symbol)
    }
    for formula in [
        *petab_problem.observable_df[OBSERVABLE_FORMULA],
        *petab_problem.observable_df[NOISE_FORMULA],
    ]:
        sym = sp.sympify(formula, locals=local_syms)
        for s in sym.free_symbols:
            if not isinstance(s, pysb.Component):
                p = pysb.Parameter(str(s), 1.0)
                pysb_model.add_component(p)
                local_syms[sp.Symbol.__str__(p)] = p

    # add observables and sigmas to pysb model
    for observable_id, observable_formula, noise_formula in zip(
        petab_problem.observable_df.index,
        petab_problem.observable_df[OBSERVABLE_FORMULA],
        petab_problem.observable_df[NOISE_FORMULA],
    ):
        obs_symbol = sp.sympify(observable_formula, locals=local_syms)
        if observable_id in pysb_model.expressions.keys():
            obs_expr = pysb_model.expressions[observable_id]
        else:
            obs_expr = pysb.Expression(observable_id, obs_symbol)
            pysb_model.add_component(obs_expr)
        local_syms[observable_id] = obs_expr

        sigma_id = f"{observable_id}_sigma"
        sigma_symbol = sp.sympify(noise_formula, locals=local_syms)
        sigma_expr = pysb.Expression(sigma_id, sigma_symbol)
        pysb_model.add_component(sigma_expr)
        local_syms[sigma_id] = sigma_expr


def _add_initialization_variables(
    pysb_model: pysb.Model, petab_problem: petab.Problem
):
    """Add initialization variables to the PySB model to support initial
    conditions specified in the PEtab condition table.

    To parameterize initial states, we currently need initial assignments.
    If they occur in the condition table, we create a new parameter
    initial_${speciesID}. Feels dirty and should be changed (see also #924).
    """

    initial_states = get_states_in_condition_table(petab_problem)
    fixed_parameters = []
    if initial_states:
        # add preequilibration indicator variable
        # NOTE: would only be required if we actually have preequilibration
        #  adding it anyways. can be optimized-out later
        if PREEQ_INDICATOR_ID in [c.name for c in pysb_model.components]:
            raise AssertionError(
                "Model already has a component with ID "
                f"{PREEQ_INDICATOR_ID}. Cannot handle "
                "species and compartments in condition table "
                "then."
            )
        preeq_indicator = pysb.Parameter(PREEQ_INDICATOR_ID)
        pysb_model.add_component(preeq_indicator)
        # Can only reset parameters after preequilibration if they are fixed.
        fixed_parameters.append(PREEQ_INDICATOR_ID)
        logger.debug(
            "Adding preequilibration indicator constant "
            f"{PREEQ_INDICATOR_ID}"
        )
    logger.debug(f"Adding initial assignments for {initial_states.keys()}")

    for assignee_id in initial_states:
        init_par_id_preeq = f"initial_{assignee_id}_preeq"
        init_par_id_sim = f"initial_{assignee_id}_sim"
        for init_par_id in [init_par_id_preeq, init_par_id_sim]:
            if init_par_id in [c.name for c in pysb_model.components]:
                raise ValueError(
                    "Cannot create parameter for initial assignment "
                    f"for {assignee_id} because an entity named "
                    f"{init_par_id} exists already in the model."
                )
            p = pysb.Parameter(init_par_id)
            pysb_model.add_component(p)

        species_idx = int(re.match(r"__s(\d+)$", assignee_id)[1])
        # use original model here since that's what was used to generate
        # the ids in initial_states
        species_pattern = petab_problem.model.model.species[species_idx]

        # species pattern comes from the _original_ model, but we only want
        # to modify pysb_model, so we have to reconstitute the pattern using
        # pysb_model
        for c in pysb_model.components:
            globals()[c.name] = c
        species_pattern = pysb.as_complex_pattern(eval(str(species_pattern)))

        from pysb.pattern import match_complex_pattern

        formula = pysb.Expression(
            f"initial_{assignee_id}_formula",
            preeq_indicator * pysb_model.parameters[init_par_id_preeq]
            + (1 - preeq_indicator) * pysb_model.parameters[init_par_id_sim],
        )
        pysb_model.add_component(formula)

        for initial in pysb_model.initials:
            if match_complex_pattern(
                initial.pattern, species_pattern, exact=True
            ):
                logger.debug(
                    "The PySB model has an initial defined for species "
                    f"{assignee_id}, but this species  also has an initial "
                    "value defined in the PEtab  condition table. The SBML "
                    "initial assignment will be overwritten to handle "
                    "preequilibration and  initial values specified by the "
                    "PEtab problem."
                )
                initial.value = formula
                break
        else:
            # No initial in the pysb model, so add one
            init = pysb.Initial(species_pattern, formula)
            pysb_model.add_component(init)

    return fixed_parameters


@log_execution_time("Importing PEtab model", logger)
def import_model_pysb(
    petab_problem: petab.Problem,
    model_output_dir: Optional[Union[str, Path]] = None,
    verbose: Optional[Union[bool, int]] = True,
    model_name: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Create AMICI model from PySB-PEtab problem

    :param petab_problem:
        PySB PEtab problem

    :param model_output_dir:
        Directory to write the model code to. Will be created if doesn't
        exist. Defaults to current directory.

    :param verbose:
        Print/log extra information.

    :param model_name:
        Name of the generated model module

    :param kwargs:
        Additional keyword arguments to be passed to
        :func:`amici.pysb_import.pysb2amici`.
    """
    set_log_level(logger, verbose)

    logger.info("Importing model ...")

    if not isinstance(petab_problem.model, PySBModel):
        raise ValueError("Not a PySB model")

    # need to create a copy here as we don't want to modify the original
    pysb.SelfExporter.cleanup()
    og_export = pysb.SelfExporter.do_export
    pysb.SelfExporter.do_export = False
    pysb_model = pysb.Model(
        base=petab_problem.model.model,
        name=petab_problem.model.model_id,
    )

    _add_observation_model(pysb_model, petab_problem)
    # generate species for the _original_ model
    pysb.bng.generate_equations(petab_problem.model.model)
    fixed_parameters = _add_initialization_variables(pysb_model, petab_problem)
    pysb.SelfExporter.do_export = og_export

    # check condition table for supported features, important to use pysb_model
    # here, as we want to also cover output parameters
    model_parameters = [p.name for p in pysb_model.parameters]
    condition_species_parameters = get_states_in_condition_table(
        petab_problem, return_patterns=True
    )
    for x in petab_problem.condition_df.columns:
        if x == CONDITION_NAME:
            continue

        x = petab.mapping.resolve_mapping(petab_problem.mapping_df, x)

        # parameters
        if x in model_parameters:
            continue

        # species/pattern
        if x in condition_species_parameters:
            continue

        raise NotImplementedError(
            "For PySB PEtab import, only model parameters and species, but "
            "not compartments are allowed in the condition table. Offending "
            f"column: {x}"
        )

    constant_parameters = (
        get_fixed_parameters(petab_problem) + fixed_parameters
    )

    if petab_problem.observable_df is None:
        observables = None
        sigmas = None
        noise_distrs = None
    else:
        observables = [
            expr.name
            for expr in pysb_model.expressions
            if expr.name in petab_problem.observable_df.index
        ]

        sigmas = {obs_id: f"{obs_id}_sigma" for obs_id in observables}

        noise_distrs = petab_noise_distributions_to_amici(
            petab_problem.observable_df
        )

    from amici.pysb_import import pysb2amici

    pysb2amici(
        model=pysb_model,
        output_dir=model_output_dir,
        model_name=model_name,
        verbose=True,
        observables=observables,
        sigmas=sigmas,
        constant_parameters=constant_parameters,
        noise_distributions=noise_distrs,
        **kwargs,
    )
