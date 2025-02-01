"""
JAX Export
----------
This module provides all necessary functionality to specify an ordinary
differential equation model and generate executable jax simulation code.
The user generally won't have to directly call any function from this module
as this will be done by
:py:func:`amici.pysb_import.pysb2jax`,
:py:func:`amici.sbml_import.SbmlImporter.sbml2jax` and
:py:func:`amici.petab_import.import_model`.
"""

from __future__ import annotations
import logging
import os
from pathlib import Path

import sympy as sp

from amici import (
    amiciModulePath,
)

from amici._codegen.template import apply_template
from amici.jax.jaxcodeprinter import AmiciJaxCodePrinter
from amici.jax.model import JAXModel
from amici.de_model import DEModel
from amici.de_export import is_valid_identifier
from amici.import_utils import (
    strip_pysb,
)
from amici.logging import get_logger, log_execution_time, set_log_level
from amici.sympy_utils import (
    _custom_pow_eval_derivative,
    _monkeypatched,
)

#: python log manager
logger = get_logger(__name__, logging.ERROR)


def _jax_variable_assignments(
    model: DEModel, sym_names: tuple[str, ...]
) -> dict:
    return {
        f"{sym_name.upper()}_SYMS": "".join(
            str(strip_pysb(s)) + ", " for s in model.sym(sym_name)
        )
        if model.sym(sym_name)
        else "_"
        for sym_name in sym_names
    }


def _jax_variable_equations(
    model: DEModel,
    code_printer: AmiciJaxCodePrinter,
    eq_names: tuple[str, ...],
    subs: dict,
    indent: int = 8,
) -> dict:
    return {
        f"{eq_name.upper()}_EQ": "\n".join(
            code_printer._get_sym_lines(
                (str(strip_pysb(s)) for s in model.sym(eq_name)),
                model.eq(eq_name).subs(subs),
                indent,
            )
        )[indent:]  # remove indent for first line
        for eq_name in eq_names
    }


def _jax_return_variables(
    model: DEModel,
    eq_names: tuple[str, ...],
) -> dict:
    return {
        f"{eq_name.upper()}_RET": _jnp_array_str(
            strip_pysb(s) for s in model.sym(eq_name)
        )
        if model.sym(eq_name)
        else "jnp.array([])"
        for eq_name in eq_names
    }


def _jax_variable_ids(model: DEModel, sym_names: tuple[str, ...]) -> dict:
    return {
        f"{sym_name.upper()}_IDS": "".join(
            f'"{strip_pysb(s)}", ' for s in model.sym(sym_name)
        )
        if model.sym(sym_name)
        else "tuple()"
        for sym_name in sym_names
    }


def _jnp_array_str(array) -> str:
    elems = ", ".join(str(s) for s in array)

    return f"jnp.array([{elems}])"


class ODEExporter:
    """
    The ODEExporter class generates AMICI jax files for a model as
    defined in symbolic expressions.

    :ivar model:
        DE definition

    :ivar verbose:
        more verbose output if True

    :ivar model_name:
        name of the model that will be used for compilation

    :ivar model_path:
        path to the generated model specific files

    :ivar _code_printer:
        Code printer to generate JAX code
    """

    def __init__(
        self,
        ode_model: DEModel,
        outdir: Path | str | None = None,
        verbose: bool | int | None = False,
        model_name: str | None = "model",
    ):
        """
        Generate AMICI jax files for the ODE provided to the constructor.

        :param ode_model:
            DE model definition

        :param outdir:
            see :meth:`amici.de_export.DEExporter.set_paths`

        :param verbose:
            verbosity level for logging, ``True``/``False`` default to
            :data:`logging.Error`/:data:`logging.DEBUG`

        :param model_name:
            name of the model to be used during code generation
        """
        set_log_level(logger, verbose)

        self.verbose: bool = logger.getEffectiveLevel() <= logging.DEBUG

        self.model_path: Path = Path()

        self.set_name(model_name)
        self.set_paths(outdir)

        self.model: DEModel = ode_model

        self._code_printer = AmiciJaxCodePrinter()

    @log_execution_time("generating jax code", logger)
    def generate_model_code(self) -> None:
        """
        Generates the jax code for the loaded model
        """
        with _monkeypatched(
            sp.Pow, "_eval_derivative", _custom_pow_eval_derivative
        ):
            self._prepare_model_folder()
            self._generate_jax_code()

    def _prepare_model_folder(self) -> None:
        """
        Create model directory or remove all files if the output directory
        already exists.
        """
        self.model_path.mkdir(parents=True, exist_ok=True)

        for file in self.model_path.glob("*"):
            if file.is_file():
                file.unlink()

    @log_execution_time("generating jax code", logger)
    def _generate_jax_code(self) -> None:
        eq_names = (
            "xdot",
            "w",
            "x0",
            "y",
            "sigmay",
            "Jy",
            "x_solver",
            "x_rdata",
            "total_cl",
        )
        sym_names = (
            "p",
            "np",
            "op",
            "x",
            "tcl",
            "w",
            "my",
            "y",
            "sigmay",
            "x_rdata",
        )

        indent = 8

        # replaces Heaviside variables with corresponding functions
        subs_heaviside = dict(
            zip(
                self.model.sym("h"),
                [sp.Heaviside(x) for x in self.model.eq("root")],
                strict=True,
            )
        )
        # replaces observables with a generic my variable
        subs_observables = dict(
            zip(
                self.model.sym("my"),
                [sp.Symbol("my")] * len(self.model.sym("my")),
                strict=True,
            )
        )
        subs = subs_heaviside | subs_observables

        tpl_data = {
            # assign named variable using corresponding algebraic formula (function body)
            **_jax_variable_equations(
                self.model, self._code_printer, eq_names, subs, indent
            ),
            # create jax array from concatenation of named variables
            **_jax_return_variables(self.model, eq_names),
            # assign named variables from a jax array
            **_jax_variable_assignments(self.model, sym_names),
            # tuple of variable names (ids as they are unique)
            **_jax_variable_ids(self.model, ("p", "k", "y", "x_rdata")),
            **{
                "MODEL_NAME": self.model_name,
                # keep track of the API version that the model was generated with so we
                # can flag conflicts in the future
                "MODEL_API_VERSION": f"'{JAXModel.MODEL_API_VERSION}'",
            },
        }

        apply_template(
            Path(amiciModulePath) / "jax" / "jax.template.py",
            self.model_path / "__init__.py",
            tpl_data,
        )

    def set_paths(self, output_dir: str | Path | None = None) -> None:
        """
        Set output paths for the model and create if necessary

        :param output_dir:
            relative or absolute path where the generated model
            code is to be placed. If ``None``, this will default to
            ``amici-{self.model_name}`` in the current working directory.
            will be created if it does not exist.

        """
        if output_dir is None:
            output_dir = Path(os.getcwd()) / f"amici-{self.model_name}"

        self.model_path = Path(output_dir).resolve()
        self.model_path.mkdir(parents=True, exist_ok=True)

    def set_name(self, model_name: str) -> None:
        """
        Sets the model name

        :param model_name:
            name of the model (may only contain upper and lower case letters,
            digits and underscores, and must not start with a digit)
        """
        if not is_valid_identifier(model_name):
            raise ValueError(
                f"'{model_name}' is not a valid model name. "
                "Model name may only contain upper and lower case letters, "
                "digits and underscores, and must not start with a digit."
            )

        self.model_name = model_name
