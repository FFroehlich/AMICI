import logging
import math
import os
import tempfile
from itertools import chain
from pathlib import Path
from typing import Optional, Union
from warnings import warn

import amici
import libsbml
import pandas as pd
import petab
import sympy as sp
from _collections import OrderedDict
from amici.logging import log_execution_time, set_log_level
from petab.models import MODEL_TYPE_SBML
from sympy.abc import _clash

from . import PREEQ_INDICATOR_ID
from .import_helpers import (
    check_model,
    get_fixed_parameters,
    get_observation_model,
)
from .util import get_states_in_condition_table

logger = logging.getLogger(__name__)


@log_execution_time("Importing PEtab model", logger)
def import_model_sbml(
    sbml_model: Union[str, Path, "libsbml.Model"] = None,
    condition_table: Optional[Union[str, Path, pd.DataFrame]] = None,
    observable_table: Optional[Union[str, Path, pd.DataFrame]] = None,
    measurement_table: Optional[Union[str, Path, pd.DataFrame]] = None,
    petab_problem: petab.Problem = None,
    model_name: Optional[str] = None,
    model_output_dir: Optional[Union[str, Path]] = None,
    verbose: Optional[Union[bool, int]] = True,
    allow_reinit_fixpar_initcond: bool = True,
    validate: bool = True,
    non_estimated_parameters_as_constants=True,
    output_parameter_defaults: Optional[dict[str, float]] = None,
    discard_sbml_annotations: bool = False,
    **kwargs,
) -> amici.SbmlImporter:
    """
    Create AMICI model from PEtab problem

    :param sbml_model:
        PEtab SBML model or SBML file name.
        Deprecated, pass ``petab_problem`` instead.

    :param condition_table:
        PEtab condition table. If provided, parameters from there will be
        turned into AMICI constant parameters (i.e. parameters w.r.t. which
        no sensitivities will be computed).
        Deprecated, pass ``petab_problem`` instead.

    :param observable_table:
        PEtab observable table. Deprecated, pass ``petab_problem`` instead.

    :param measurement_table:
        PEtab measurement table. Deprecated, pass ``petab_problem`` instead.

    :param petab_problem:
        PEtab problem.

    :param model_name:
        Name of the generated model. If model file name was provided,
        this defaults to the file name without extension, otherwise
        the SBML model ID will be used.

    :param model_output_dir:
        Directory to write the model code to. Will be created if doesn't
        exist. Defaults to current directory.

    :param verbose:
        Print/log extra information.

    :param allow_reinit_fixpar_initcond:
        See :class:`amici.de_export.ODEExporter`. Must be enabled if initial
        states are to be reset after preequilibration.

    :param validate:
        Whether to validate the PEtab problem

    :param non_estimated_parameters_as_constants:
        Whether parameters marked as non-estimated in PEtab should be
        considered constant in AMICI. Setting this to ``True`` will reduce
        model size and simulation times. If sensitivities with respect to those
        parameters are required, this should be set to ``False``.

    :param output_parameter_defaults:
        Optional default parameter values for output parameters introduced in
        the PEtab observables table, in particular for placeholder parameters.
        dictionary mapping parameter IDs to default values.

    :param discard_sbml_annotations:
        Discard information contained in AMICI SBML annotations (debug).

    :param kwargs:
        Additional keyword arguments to be passed to
        :meth:`amici.sbml_import.SbmlImporter.sbml2amici`.

    :return:
        The created :class:`amici.sbml_import.SbmlImporter` instance.
    """
    from petab.models.sbml_model import SbmlModel

    set_log_level(logger, verbose)

    logger.info("Importing model ...")

    if any([sbml_model, condition_table, observable_table, measurement_table]):
        warn(
            "The `sbml_model`, `condition_table`, `observable_table`, and "
            "`measurement_table` arguments are deprecated and will be "
            "removed in a future version. Use `petab_problem` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if petab_problem:
            raise ValueError(
                "Must not pass a `petab_problem` argument in "
                "combination with any of `sbml_model`, "
                "`condition_table`, `observable_table`, or "
                "`measurement_table`."
            )

        petab_problem = petab.Problem(
            model=SbmlModel(sbml_model)
            if isinstance(sbml_model, libsbml.Model)
            else SbmlModel.from_file(sbml_model),
            condition_df=petab.get_condition_df(condition_table),
            observable_df=petab.get_observable_df(observable_table),
        )

    if petab_problem.observable_df is None:
        raise NotImplementedError(
            "PEtab import without observables table "
            "is currently not supported."
        )

    assert isinstance(petab_problem.model, SbmlModel)

    if validate:
        logger.info("Validating PEtab problem ...")
        petab.lint_problem(petab_problem)

    # Model name from SBML ID or filename
    if model_name is None:
        if not (model_name := petab_problem.model.sbml_model.getId()):
            if not isinstance(sbml_model, (str, Path)):
                raise ValueError(
                    "No `model_name` was provided and no model "
                    "ID was specified in the SBML model."
                )
            model_name = os.path.splitext(os.path.split(sbml_model)[-1])[0]

    if model_output_dir is None:
        model_output_dir = os.path.join(
            os.getcwd(), f"{model_name}-amici{amici.__version__}"
        )

    logger.info(
        f"Model name is '{model_name}'.\n"
        f"Writing model code to '{model_output_dir}'."
    )

    # Create a copy, because it will be modified by SbmlImporter
    sbml_doc = petab_problem.model.sbml_model.getSBMLDocument().clone()
    sbml_model = sbml_doc.getModel()

    show_model_info(sbml_model)

    sbml_importer = amici.SbmlImporter(
        sbml_model,
        discard_annotations=discard_sbml_annotations,
    )
    sbml_model = sbml_importer.sbml

    allow_n_noise_pars = (
        not petab.lint.observable_table_has_nontrivial_noise_formula(
            petab_problem.observable_df
        )
    )
    if (
        petab_problem.measurement_df is not None
        and petab.lint.measurement_table_has_timepoint_specific_mappings(
            petab_problem.measurement_df,
            allow_scalar_numeric_noise_parameters=allow_n_noise_pars,
        )
    ):
        raise ValueError(
            "AMICI does not support importing models with timepoint specific "
            "mappings for noise or observable parameters. Please flatten "
            "the problem and try again."
        )

    if petab_problem.observable_df is not None:
        observables, noise_distrs, sigmas = get_observation_model(
            petab_problem.observable_df
        )
    else:
        observables = noise_distrs = sigmas = None

    logger.info(f"Observables: {len(observables)}")
    logger.info(f"Sigmas: {len(sigmas)}")

    if len(sigmas) != len(observables):
        raise AssertionError(
            f"Number of provided observables ({len(observables)}) and sigmas "
            f"({len(sigmas)}) do not match."
        )

    # TODO: adding extra output parameters is currently not supported,
    #  so we add any output parameters to the SBML model.
    #  this should be changed to something more elegant
    # <BeginWorkAround>
    formulas = chain(
        (val["formula"] for val in observables.values()), sigmas.values()
    )
    output_parameters = OrderedDict()
    for formula in formulas:
        # we want reproducible parameter ordering upon repeated import
        free_syms = sorted(
            sp.sympify(formula, locals=_clash).free_symbols,
            key=lambda symbol: symbol.name,
        )
        for free_sym in free_syms:
            sym = str(free_sym)
            if (
                sbml_model.getElementBySId(sym) is None
                and sym != "time"
                and sym not in observables
            ):
                output_parameters[sym] = None
    logger.debug(
        "Adding output parameters to model: "
        f"{list(output_parameters.keys())}"
    )
    output_parameter_defaults = output_parameter_defaults or {}
    if extra_pars := (
        set(output_parameter_defaults) - set(output_parameters.keys())
    ):
        raise ValueError(
            f"Default output parameter values were given for {extra_pars}, "
            "but they those are not output parameters."
        )

    for par in output_parameters.keys():
        _add_global_parameter(
            sbml_model=sbml_model,
            parameter_id=par,
            value=output_parameter_defaults.get(par, 0.0),
        )
    # <EndWorkAround>

    # TODO: to parameterize initial states or compartment sizes, we currently
    #  need initial assignments. if they occur in the condition table, we
    #  create a new parameter initial_${speciesOrCompartmentID}.
    #  feels dirty and should be changed (see also #924)
    # <BeginWorkAround>

    initial_states = get_states_in_condition_table(petab_problem)
    fixed_parameters = []
    if initial_states:
        # add preequilibration indicator variable
        # NOTE: would only be required if we actually have preequilibration
        #  adding it anyways. can be optimized-out later
        if sbml_model.getParameter(PREEQ_INDICATOR_ID) is not None:
            raise AssertionError(
                "Model already has a parameter with ID "
                f"{PREEQ_INDICATOR_ID}. Cannot handle "
                "species and compartments in condition table "
                "then."
            )
        indicator = sbml_model.createParameter()
        indicator.setId(PREEQ_INDICATOR_ID)
        indicator.setName(PREEQ_INDICATOR_ID)
        # Can only reset parameters after preequilibration if they are fixed.
        fixed_parameters.append(PREEQ_INDICATOR_ID)
        logger.debug(
            "Adding preequilibration indicator "
            f"constant {PREEQ_INDICATOR_ID}"
        )
    logger.debug(f"Adding initial assignments for {initial_states.keys()}")
    for assignee_id in initial_states:
        init_par_id_preeq = f"initial_{assignee_id}_preeq"
        init_par_id_sim = f"initial_{assignee_id}_sim"
        for init_par_id in [init_par_id_preeq, init_par_id_sim]:
            if sbml_model.getElementBySId(init_par_id) is not None:
                raise ValueError(
                    "Cannot create parameter for initial assignment "
                    f"for {assignee_id} because an entity named "
                    f"{init_par_id} exists already in the model."
                )
            init_par = sbml_model.createParameter()
            init_par.setId(init_par_id)
            init_par.setName(init_par_id)
        assignment = sbml_model.getInitialAssignment(assignee_id)
        if assignment is None:
            assignment = sbml_model.createInitialAssignment()
            assignment.setSymbol(assignee_id)
        else:
            logger.debug(
                "The SBML model has an initial assignment defined "
                f"for model entity {assignee_id}, but this entity "
                "also has an initial value defined in the PEtab "
                "condition table. The SBML initial assignment will "
                "be overwritten to handle preequilibration and "
                "initial values specified by the PEtab problem."
            )
        formula = (
            f"{PREEQ_INDICATOR_ID} * {init_par_id_preeq} "
            f"+ (1 - {PREEQ_INDICATOR_ID}) * {init_par_id_sim}"
        )
        math_ast = libsbml.parseL3Formula(formula)
        assignment.setMath(math_ast)
    # <EndWorkAround>

    fixed_parameters.extend(
        _get_fixed_parameters_sbml(
            petab_problem=petab_problem,
            non_estimated_parameters_as_constants=non_estimated_parameters_as_constants,
        )
    )

    logger.debug(f"Fixed parameters are {fixed_parameters}")
    logger.info(f"Overall fixed parameters: {len(fixed_parameters)}")
    logger.info(
        "Variable parameters: "
        + str(len(sbml_model.getListOfParameters()) - len(fixed_parameters))
    )

    # Create Python module from SBML model
    sbml_importer.sbml2amici(
        model_name=model_name,
        output_dir=model_output_dir,
        observables=observables,
        constant_parameters=fixed_parameters,
        sigmas=sigmas,
        allow_reinit_fixpar_initcond=allow_reinit_fixpar_initcond,
        noise_distributions=noise_distrs,
        verbose=verbose,
        **kwargs,
    )

    if kwargs.get(
        "compile",
        amici._get_default_argument(sbml_importer.sbml2amici, "compile"),
    ):
        # check that the model extension was compiled successfully
        model_module = amici.import_model_module(model_name, model_output_dir)
        model = model_module.getModel()
        check_model(amici_model=model, petab_problem=petab_problem)

    return sbml_importer


def show_model_info(sbml_model: "libsbml.Model"):
    """Log some model quantities"""

    logger.info(f"Species: {len(sbml_model.getListOfSpecies())}")
    logger.info(
        "Global parameters: " + str(len(sbml_model.getListOfParameters()))
    )
    logger.info(f"Reactions: {len(sbml_model.getListOfReactions())}")


# TODO - remove?!
def species_to_parameters(
    species_ids: list[str], sbml_model: "libsbml.Model"
) -> list[str]:
    """
    Turn a SBML species into parameters and replace species references
    inside the model instance.

    :param species_ids:
        list of SBML species ID to convert to parameters with the same ID as
        the replaced species.

    :param sbml_model:
        SBML model to modify

    :return:
        list of IDs of species which have been converted to parameters
    """
    transformables = []

    for species_id in species_ids:
        species = sbml_model.getSpecies(species_id)

        if species.getHasOnlySubstanceUnits():
            logger.warning(
                f"Ignoring {species.getId()} which has only substance units."
                " Conversion not yet implemented."
            )
            continue

        if math.isnan(species.getInitialConcentration()):
            logger.warning(
                f"Ignoring {species.getId()} which has no initial "
                "concentration. Amount conversion not yet implemented."
            )
            continue

        transformables.append(species_id)

    # Must not remove species while iterating over getListOfSpecies()
    for species_id in transformables:
        species = sbml_model.removeSpecies(species_id)
        par = sbml_model.createParameter()
        par.setId(species.getId())
        par.setName(species.getName())
        par.setConstant(True)
        par.setValue(species.getInitialConcentration())
        par.setUnits(species.getUnits())

    # Remove from reactants and products
    for reaction in sbml_model.getListOfReactions():
        for species_id in transformables:
            # loop, since removeX only removes one instance
            while reaction.removeReactant(species_id):
                # remove from reactants
                pass
            while reaction.removeProduct(species_id):
                # remove from products
                pass
            while reaction.removeModifier(species_id):
                # remove from modifiers
                pass

    return transformables


def _add_global_parameter(
    sbml_model: libsbml.Model,
    parameter_id: str,
    parameter_name: str = None,
    constant: bool = False,
    units: str = "dimensionless",
    value: float = 0.0,
) -> libsbml.Parameter:
    """Add new global parameter to SBML model

    Arguments:
        sbml_model: SBML model
        parameter_id: ID of the new parameter
        parameter_name: Name of the new parameter
        constant: Is parameter constant?
        units: SBML unit ID
        value: parameter value

    Returns:
        The created parameter
    """
    if parameter_name is None:
        parameter_name = parameter_id

    p = sbml_model.createParameter()
    p.setId(parameter_id)
    p.setName(parameter_name)
    p.setConstant(constant)
    p.setValue(value)
    p.setUnits(units)
    return p


def _get_fixed_parameters_sbml(
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
    if not petab_problem.model.type_id == MODEL_TYPE_SBML:
        raise ValueError("Not an SBML model.")
    # initial concentrations for species or initial compartment sizes in
    # condition table will need to be turned into fixed parameters

    # if there is no initial assignment for that species, we'd need
    # to create one. to avoid any naming collision right away, we don't
    # allow that for now

    # we can't handle them yet
    compartments = [
        col
        for col in petab_problem.condition_df
        if petab_problem.model.sbml_model.getCompartment(col) is not None
    ]
    if compartments:
        raise NotImplementedError(
            "Can't handle initial compartment sizes "
            "at the moment. Consider creating an "
            f"initial assignment for {compartments}"
        )

    fixed_parameters = get_fixed_parameters(
        petab_problem, non_estimated_parameters_as_constants
    )

    # exclude targets of rules or initial assignments
    sbml_model = petab_problem.model.sbml_model
    for fixed_parameter in fixed_parameters.copy():
        # check global parameters
        if sbml_model.getInitialAssignmentBySymbol(
            fixed_parameter
        ) or sbml_model.getRuleByVariable(fixed_parameter):
            fixed_parameters.remove(fixed_parameter)

    return list(sorted(fixed_parameters))


def _create_model_output_dir_name(
    sbml_model: "libsbml.Model", model_name: Optional[str] = None
) -> Path:
    """
    Find a folder for storing the compiled amici model.
    If possible, use the sbml model id, otherwise create a random folder.
    The folder will be located in the `amici_models` subfolder of the current
    folder.
    """
    BASE_DIR = Path("amici_models").absolute()
    BASE_DIR.mkdir(exist_ok=True)
    # try model_name
    if model_name:
        return BASE_DIR / model_name

    # try sbml model id
    if sbml_model_id := sbml_model.getId():
        return BASE_DIR / sbml_model_id

    # create random folder name
    return Path(tempfile.mkdtemp(dir=BASE_DIR))
