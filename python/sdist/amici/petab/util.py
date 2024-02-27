"""Various helper functions for working with PEtab problems."""
import re
from typing import TYPE_CHECKING, Union

import libsbml
import pandas as pd
import petab
from petab.C import PREEQUILIBRATION_CONDITION_ID, SIMULATION_CONDITION_ID
from petab.mapping import resolve_mapping
from petab.models import MODEL_TYPE_PYSB, MODEL_TYPE_SBML

if TYPE_CHECKING:
    pysb = None


def get_states_in_condition_table(
    petab_problem: petab.Problem,
    condition: Union[dict, pd.Series] = None,
    return_patterns: bool = False,
) -> dict[str, tuple[Union[float, str, None], Union[float, str, None]]]:
    """Get states and their initial condition as specified in the condition table.

    Returns: Dictionary: ``stateId -> (initial condition simulation, initial condition preequilibration)``
    """
    if petab_problem.model.type_id not in (MODEL_TYPE_SBML, MODEL_TYPE_PYSB):
        raise NotImplementedError()

    species_check_funs = {
        MODEL_TYPE_SBML: lambda x: _element_is_sbml_state(
            petab_problem.sbml_model, x
        ),
        MODEL_TYPE_PYSB: lambda x: _element_is_pysb_pattern(
            petab_problem.model.model, x
        ),
    }
    states = {
        resolve_mapping(petab_problem.mapping_df, col): (None, None)
        if condition is None
        else (
            petab_problem.condition_df.loc[
                condition[SIMULATION_CONDITION_ID], col
            ],
            petab_problem.condition_df.loc[
                condition[PREEQUILIBRATION_CONDITION_ID], col
            ]
            if PREEQUILIBRATION_CONDITION_ID in condition
            else None,
        )
        for col in petab_problem.condition_df.columns
        if species_check_funs[petab_problem.model.type_id](
            resolve_mapping(petab_problem.mapping_df, col)
        )
    }

    if petab_problem.model.type_id == MODEL_TYPE_PYSB:
        if return_patterns:
            return states
        import pysb.pattern

        if not petab_problem.model.model.species:
            import pysb.bng

            pysb.bng.generate_equations(petab_problem.model.model)

        try:
            spm = pysb.pattern.SpeciesPatternMatcher(
                model=petab_problem.model.model
            )
        except NotImplementedError:
            raise NotImplementedError(
                "Requires https://github.com/pysb/pysb/pull/570. "
                "To use this functionality, update pysb via "
                "`pip install git+https://github.com/FFroehlich/pysb@fix_pattern_matching`"
            )

        # expose model components as variables so we can evaluate patterns
        for c in petab_problem.model.model.components:
            globals()[c.name] = c

        states = {
            f"__s{ix}": value
            for pattern, value in states.items()
            for ix in spm.match(eval(pattern), index=True, exact=True)
        }
    return states


def _element_is_pysb_pattern(model: "pysb.Model", element: str) -> bool:
    """Check if element is a pysb pattern"""
    if match := re.match(r"[a-zA-Z_][\w_]*\(", element):
        return match[0][:-1] in [m.name for m in model.monomers]
    return False


def _element_is_sbml_state(sbml_model: libsbml.Model, sbml_id: str) -> bool:
    """Does the element with ID `sbml_id` correspond to a state variable?"""
    if sbml_model.getCompartment(sbml_id) is not None:
        return True
    if sbml_model.getSpecies(sbml_id) is not None:
        return True
    if (
        rule := sbml_model.getRuleByVariable(sbml_id)
    ) is not None and rule.getTypeCode() == libsbml.SBML_RATE_RULE:
        return True

    return False
