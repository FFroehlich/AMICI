"""
PEtab Import
------------
Import a model in the :mod:`petab` (https://github.com/PEtab-dev/PEtab) format
into AMICI.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Union

import amici
import petab
from petab.models import MODEL_TYPE_PYSB, MODEL_TYPE_SBML

from ..logging import get_logger
from .import_helpers import _can_import_model, _create_model_name, check_model
from .sbml_import import import_model_sbml

try:
    from .pysb_import import import_model_pysb
except ModuleNotFoundError:
    # pysb not available
    import_model_pysb = None


__all__ = ["import_petab_problem"]

logger = get_logger(__name__, logging.WARNING)


def import_petab_problem(
    petab_problem: petab.Problem,
    model_output_dir: Union[str, Path, None] = None,
    model_name: str = None,
    compile_: bool = None,
    non_estimated_parameters_as_constants=True,
    **kwargs,
) -> "amici.Model":
    """
    Create an AMICI model for a PEtab problem.

    :param petab_problem:
        A petab problem containing all relevant information on the model.

    :param model_output_dir:
        Directory to write the model code to. It will be created if it doesn't
        exist. Defaults to current directory.

    :param model_name:
        Name of the generated model module. Defaults to the ID of the model
        or the model file name without the extension.

    :param compile_:
        If ``True``, the model will be compiled. If ``False``, the model will
        not be compiled. If ``None``, the model will be compiled if it cannot
        be imported.

    :param non_estimated_parameters_as_constants:
        Whether parameters marked as non-estimated in PEtab should be
        considered constant in AMICI. Setting this to ``True`` will reduce
        model size and simulation times. If sensitivities with respect to those
        parameters are required, this should be set to ``False``.

    :param kwargs:
        Additional keyword arguments to be passed to
        :meth:`amici.sbml_import.SbmlImporter.sbml2amici` or
        :func:`amici.pysb_import.pysb2amici`, depending on the model type.

    :return:
        The imported model.
    """
    if "force_compile" in kwargs:
        if kwargs["force_compile"]:
            compile_ = True
            del kwargs["force_compile"]
        warn(
            "The `force_compile` option is deprecated, please use the "
            "new `compile_` option, which also supports 'do not compile'.",
            DeprecationWarning,
            stacklevel=2,
        )

    if petab_problem.model.type_id not in (MODEL_TYPE_SBML, MODEL_TYPE_PYSB):
        raise NotImplementedError(
            "Unsupported model type " + petab_problem.model.type_id
        )

    if petab_problem.mapping_df is not None:
        # It's partially supported. Remove at your own risk...
        raise NotImplementedError(
            "PEtab v2.0.0 mapping tables are not yet supported."
        )

    model_name = model_name or petab_problem.model.model_id

    if petab_problem.model.type_id == MODEL_TYPE_PYSB and model_name is None:
        model_name = petab_problem.pysb_model.name
    elif model_name is None and model_output_dir:
        model_name = _create_model_name(model_output_dir)

    # generate folder and model name if necessary
    if model_output_dir is None:
        if petab_problem.model.type_id == MODEL_TYPE_PYSB:
            raise ValueError("Parameter `model_output_dir` is required.")

        from .sbml_import import _create_model_output_dir_name

        model_output_dir = _create_model_output_dir_name(
            petab_problem.sbml_model, model_name
        )
    else:
        model_output_dir = os.path.abspath(model_output_dir)

    # create folder
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)

    # check if compilation necessary
    if compile_ or (
        compile_ is None
        and not _can_import_model(model_name, model_output_dir)
    ):
        # check if folder exists
        if os.listdir(model_output_dir) and not compile_:
            raise ValueError(
                f"Cannot compile to {model_output_dir}: not empty. "
                "Please assign a different target or set `compile_` to `True`."
            )

        # remove folder if exists
        if os.path.exists(model_output_dir):
            shutil.rmtree(model_output_dir)

        logger.info(f"Compiling model {model_name} to {model_output_dir}.")
        # compile the model
        if petab_problem.model.type_id == MODEL_TYPE_PYSB:
            import_model_pysb(
                petab_problem,
                model_name=model_name,
                model_output_dir=model_output_dir,
                **kwargs,
            )
        else:
            import_model_sbml(
                petab_problem=petab_problem,
                model_name=model_name,
                model_output_dir=model_output_dir,
                non_estimated_parameters_as_constants=non_estimated_parameters_as_constants,
                **kwargs,
            )

    # import model
    model_module = amici.import_model_module(model_name, model_output_dir)
    model = model_module.getModel()
    check_model(amici_model=model, petab_problem=petab_problem)

    logger.info(
        f"Successfully loaded model {model_name} " f"from {model_output_dir}."
    )

    return model


# for backwards compatibility
import_model = import_model_sbml
