"""
C++ Export
----------
This module provides all necessary functionality to specify a differential
equation model and generate executable C++ simulation code.
The user generally won't have to directly call any function from this module
as this will be done by
:py:func:`amici.pysb_import.pysb2amici`,
:py:func:`amici.sbml_import.SbmlImporter.sbml2amici` and
:py:func:`amici.petab_import.import_model`.
"""

from __future__ import annotations
import copy
import logging
import os
import re
import shutil
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Literal,
)

import sympy as sp

from . import (
    __commit__,
    __version__,
    amiciModulePath,
    amiciSrcPath,
    amiciSwigPath,
    splines,
)
from ._codegen.cxx_functions import (
    _FunctionInfo,
    functions,
    sparse_functions,
    nobody_functions,
    sensi_functions,
    sparse_sensi_functions,
    event_functions,
    event_sensi_functions,
    multiobs_functions,
)
from ._codegen.model_class import (
    get_function_extern_declaration,
    get_sunindex_extern_declaration,
    get_model_override_implementation,
    get_sunindex_override_implementation,
    get_state_independent_event_intializer,
)
from ._codegen.template import apply_template
from .cxxcodeprinter import (
    AmiciCxxCodePrinter,
    get_switch_statement,
)
from .de_model import DEModel
from .de_model_components import *
from .import_utils import (
    strip_pysb,
)
from .logging import get_logger, log_execution_time, set_log_level
from .compile import build_model_extension
from .sympy_utils import (
    _custom_pow_eval_derivative,
    _monkeypatched,
    smart_is_zero_matrix,
)

if TYPE_CHECKING:
    pass


# Template for model simulation main.cpp file
CXX_MAIN_TEMPLATE_FILE = os.path.join(amiciSrcPath, "main.template.cpp")
# Template for model/swig/CMakeLists.txt
SWIG_CMAKE_TEMPLATE_FILE = os.path.join(
    amiciSwigPath, "CMakeLists_model.cmake"
)
# Template for model/CMakeLists.txt
MODEL_CMAKE_TEMPLATE_FILE = os.path.join(
    amiciSrcPath, "CMakeLists.template.cmake"
)

IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z_]\w*$")

#: list of equations that have ids which may not be unique
non_unique_id_symbols = ["x_rdata", "y"]

#: custom c++ function replacements
CUSTOM_FUNCTIONS = [
    {
        "sympy": "polygamma",
        "c++": "boost::math::polygamma",
        "include": "#include <boost/math/special_functions/polygamma.hpp>",
        "build_hint": "Using polygamma requires libboost-math header files.",
    },
    {"sympy": "Heaviside", "c++": "amici::heaviside"},
    {"sympy": "DiracDelta", "c++": "amici::dirac"},
]

#: python log manager
logger = get_logger(__name__, logging.ERROR)


class DEExporter:
    """
    The DEExporter class generates AMICI C++ files for a model as
    defined in symbolic expressions.

    :ivar model:
        DE definition

    :ivar verbose:
        more verbose output if True

    :ivar assume_pow_positivity:
        if set to true, a special pow function is
        used to avoid problems with state variables that may become negative
        due to numerical errors

    :ivar compiler:
        Absolute path to the compiler executable to be used to build the Python
        extension, e.g. ``/usr/bin/clang``.

    :ivar functions:
        carries C++ function signatures and other specifications

    :ivar model_name:
        name of the model that will be used for compilation

    :ivar model_path:
        path to the generated model specific files

    :ivar model_swig_path:
        path to the generated swig files

    :ivar allow_reinit_fixpar_initcond:
        indicates whether reinitialization of
        initial states depending on fixedParameters is allowed for this model

    :ivar _build_hints:
        If the given model uses special functions, this set contains hints for
        model building.

    :ivar _code_printer:
        Code printer to generate C++ code

    :ivar generate_sensitivity_code:
        Specifies whether code for sensitivity computation is to be generated

    .. note::
        When importing large models (several hundreds of species or
        parameters), import time can potentially be reduced by using multiple
        CPU cores. This is controlled by setting the ``AMICI_IMPORT_NPROCS``
        environment variable to the number of parallel processes that are to be
        used (default: 1). Note that for small models this may (slightly)
        increase import times.
    """

    def __init__(
        self,
        de_model: DEModel,
        outdir: Path | str | None = None,
        verbose: bool | int | None = False,
        assume_pow_positivity: bool | None = False,
        compiler: str | None = None,
        allow_reinit_fixpar_initcond: bool | None = True,
        generate_sensitivity_code: bool | None = True,
        model_name: str | None = "model",
        hybridisation: dict | None = None,
    ):
        """
        Generate AMICI C++ files for the DE provided to the constructor.

        :param de_model:
            DE model definition

        :param outdir:
            see :meth:`amici.de_export.DEExporter.set_paths`

        :param verbose:
            verbosity level for logging, ``True``/``False`` default to
            :data:`logging.Error`/:data:`logging.DEBUG`

        :param assume_pow_positivity:
            if set to true, a special pow function is
            used to avoid problems with state variables that may become
            negative due to numerical errors

        :param compiler: Absolute path to the compiler executable to be used
            to build the Python extension, e.g. ``/usr/bin/clang``.

        :param allow_reinit_fixpar_initcond:
            see :class:`amici.de_export.DEExporter`

        :param generate_sensitivity_code:
            specifies whether code required for sensitivity computation will be
            generated

        :param model_name:
            name of the model to be used during code generation
        """
        set_log_level(logger, verbose)

        self.verbose: bool = logger.getEffectiveLevel() <= logging.DEBUG
        self.assume_pow_positivity: bool = assume_pow_positivity
        self.compiler: str = compiler

        self.model_path: str = ""
        self.model_swig_path: str = ""

        self.set_name(model_name)
        self.set_paths(outdir)

        self._code_printer = AmiciCxxCodePrinter()
        for fun in CUSTOM_FUNCTIONS:
            self._code_printer.known_functions[fun["sympy"]] = fun["c++"]

        # Signatures and properties of generated model functions (see
        # include/amici/model.h for details)
        self.model: DEModel = de_model
        self._code_printer.known_functions.update(
            splines.spline_user_functions(
                self.model._splines, self._get_index("p")
            )
        )

        # To only generate a subset of functions, apply subselection here
        self.functions: dict[str, _FunctionInfo] = copy.deepcopy(functions)

        self.allow_reinit_fixpar_initcond: bool = allow_reinit_fixpar_initcond
        self._build_hints = set()
        self.generate_sensitivity_code: bool = generate_sensitivity_code
        self.hybridisation = hybridisation

    @log_execution_time("generating cpp code", logger)
    def generate_model_code(self) -> None:
        """
        Generates the native C++ code for the loaded model and a Matlab
        script that can be run to compile a mex file from the C++ code
        """
        with _monkeypatched(
            sp.Pow, "_eval_derivative", _custom_pow_eval_derivative
        ):
            self._prepare_model_folder()
            self._generate_c_code()
            self._generate_m_code()

    @log_execution_time("compiling cpp code", logger)
    def compile_model(self) -> None:
        """
        Compiles the generated code it into a simulatable module
        """
        build_model_extension(
            package_dir=self.model_path,
            compiler=self.compiler,
            verbose=self.verbose,
            extra_msg="\n".join(self._build_hints),
        )

    def _prepare_model_folder(self) -> None:
        """
        Create model directory or remove all files if the output directory
        already exists.
        """
        os.makedirs(self.model_path, exist_ok=True)

        for file in os.listdir(self.model_path):
            file_path = os.path.join(self.model_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    def _generate_c_code(self) -> None:
        """
        Create C++ code files for the model based on
        :attribute:`DEExporter.model`.
        """
        for func_name, func_info in self.functions.items():
            if (
                func_name in sensi_functions + sparse_sensi_functions
                and not self.generate_sensitivity_code
            ):
                continue

            if func_info.generate_body:
                dec = log_execution_time(f"writing {func_name}.cpp", logger)
                dec(self._write_function_file)(func_name)

        for name in self.model.sym_names():
            # only generate for those that have nontrivial implementation,
            # check for both basic variables (not in functions) and function
            # computed values
            if (
                (
                    name in self.functions
                    and not self.functions[name].body
                    and name not in nobody_functions
                )
                or name not in self.functions
            ) and len(self.model.sym(name)) == 0:
                continue
            self._write_index_files(name)

        self._write_wrapfunctions_cpp()
        self._write_wrapfunctions_header()
        self._write_model_header_cpp()
        self._write_c_make_file()
        self._write_swig_files()
        self._write_module_setup()
        _write_gitignore(Path(self.model_path))

        shutil.copy(
            CXX_MAIN_TEMPLATE_FILE, os.path.join(self.model_path, "main.cpp")
        )

    def _generate_m_code(self) -> None:
        """
        Create a Matlab script for compiling code files to a mex file
        """

        # Second order code is not yet implemented. Once this is done,
        # those variables will have to be replaced by
        # "self.model.<var>true()", or the corresponding "model.self.o2flag"
        nxtrue_rdata = self.model.num_states_rdata()
        nytrue = self.model.num_obs()
        nztrue = self.model.num_eventobs()
        o2flag = 0

        lines = [
            "% This compile script was automatically created from"
            " Python SBML import.",
            "% If mex compiler is set up within MATLAB, it can be run"
            " from MATLAB ",
            "% in order to compile a mex-file from the Python"
            " generated C++ files.",
            "",
            f"modelName = '{self.model_name}';",
            "amimodel.compileAndLinkModel(modelName, '', [], [], [], []);",
            f"amimodel.generateMatlabWrapper({nxtrue_rdata}, "
            f"{nytrue}, {self.model.num_par()}, "
            f"{self.model.num_const()}, {nztrue}, {o2flag}, ...",
            "    [], ['simulate_' modelName '.m'], modelName, ...",
            "    'lin', 1, 1);",
        ]

        # write compile script (for mex)
        compile_script = os.path.join(self.model_path, "compileMexFile.m")
        with open(compile_script, "w") as fileout:
            fileout.write("\n".join(lines))

    def _get_index(self, name: str) -> dict[sp.Symbol, int]:
        """
        Compute indices for a symbolic array.
        :param name:
            key in self.model._syms for which to obtain the index.
        :return:
            a dictionary of symbol/index pairs.
        """
        if name in self.model.sym_names():
            if name in sparse_functions:
                symbols = self.model.sparsesym(name)
            else:
                symbols = self.model.sym(name).T
        else:
            raise ValueError(f"Unknown symbolic array: {name}")

        return {
            strip_pysb(symbol).name: index
            for index, symbol in enumerate(symbols)
        }

    def _write_index_files(self, name: str) -> None:
        """
        Write index file for a symbolic array.

        :param name:
            key in ``self.model._syms`` for which the respective file should
            be written
        """
        if name not in self.model.sym_names():
            raise ValueError(f"Unknown symbolic array: {name}")

        symbols = (
            self.model.sparsesym(name)
            if name in sparse_functions
            else self.model.sym(name).T
        )
        if not len(symbols):
            return

        # flatten multiobs
        if isinstance(next(iter(symbols), None), list):
            symbols = [symbol for obs in symbols for symbol in obs]

        lines = []
        for index, symbol in enumerate(symbols):
            symbol_name = strip_pysb(symbol)
            if str(symbol) == "0":
                continue
            if str(symbol_name) == "":
                raise ValueError(f'{name} contains a symbol called ""')
            lines.append(f"#define {symbol_name} {name}[{index}]")
            if name == "stau":
                # we only need a single macro, as all entries have the same symbol
                break

        filename = os.path.join(self.model_path, f"{name}.h")
        with open(filename, "w") as fileout:
            fileout.write("\n".join(lines))

    def _write_function_file(self, function: str) -> None:
        """
        Generate equations and write the C++ code for the function
        ``function``.

        :param function:
            name of the function to be written (see ``self.functions``)
        """

        # first generate the equations to make sure we have everything we
        # need in subsequent steps
        if function in sparse_functions:
            equations = self.model.sparseeq(function)
        elif (
            not self.allow_reinit_fixpar_initcond
            and function == "sx0_fixedParameters"
        ):
            # Not required. Will create empty function body.
            equations = sp.Matrix()
        elif function == "create_splines":
            # nothing to do
            pass
        else:
            equations = self.model.eq(function)

        # function body
        if function == "create_splines":
            body = self._get_create_splines_body()
        else:
            body = self._get_function_body(function, equations)
        if not body:
            return

        # colptrs / rowvals for sparse matrices
        if function in sparse_functions:
            lines = self._generate_function_index(function, "colptrs")
            lines.extend(self._generate_function_index(function, "rowvals"))
            lines.append("\n\n")
        else:
            lines = []

        # function header
        lines.extend(
            [
                '#include "amici/symbolic_functions.h"',
                '#include "amici/defines.h"',
                '#include "sundials/sundials_types.h"',
                "",
                "#include <gsl/gsl-lite.hpp>",
                "#include <algorithm>",
                "",
            ]
        )
        if function == "create_splines":
            lines += [
                '#include "amici/splinefunctions.h"',
                "#include <vector>",
            ]

        func_info = self.functions[function]

        # extract symbols that need definitions from signature
        # don't add includes for files that won't be generated.
        # Unfortunately we cannot check for `self.functions[sym].body`
        # here since it may not have been generated yet.
        for sym in re.findall(
            r"const (?:realtype|double) \*([\w]+)[0]*(?:,|$)",
            func_info.arguments(self.model.is_ode()),
        ):
            if sym not in self.model.sym_names():
                continue

            if sym in sparse_functions:
                iszero = smart_is_zero_matrix(self.model.sparseeq(sym))
            elif sym in self.functions:
                iszero = smart_is_zero_matrix(self.model.eq(sym))
            else:
                iszero = len(self.model.sym(sym)) == 0

            if iszero and not (
                (sym == "y" and "Jy" in function)
                or (
                    sym == "w"
                    and "xdot" in function
                    and len(self.model.sym(sym))
                )
            ):
                continue

            lines.append(f'#include "{sym}.h"')

        # include return symbols
        if (
            function in self.model.sym_names()
            and function not in non_unique_id_symbols
        ):
            lines.append(f'#include "{function}.h"')

        lines.extend(
            [
                "",
                "namespace amici {",
                f"namespace model_{self.model_name} {{",
                "",
                f"{func_info.return_type} {function}_{self.model_name}"
                f"({func_info.arguments(self.model.is_ode())}){{",
            ]
        )

        if self.assume_pow_positivity and func_info.assume_pow_positivity:
            pow_rx = re.compile(r"(^|\W)std::pow\(")
            body = [
                # execute this twice to catch cases where the ending '(' would
                #  be the starting (^|\W) for the following match
                pow_rx.sub(
                    r"\1amici::pos_pow(",
                    pow_rx.sub(r"\1amici::pos_pow(", line),
                )
                for line in body
            ]

        self.functions[function].body = body

        lines += body
        lines.extend(
            [
                "}",
                "",
                f"}} // namespace model_{self.model_name}",
                "} // namespace amici\n",
            ]
        )

        # check custom functions
        for fun in CUSTOM_FUNCTIONS:
            if "include" in fun and any(fun["c++"] in line for line in lines):
                if "build_hint" in fun:
                    self._build_hints.add(fun["build_hint"])
                lines.insert(0, fun["include"])

        # if not body is None:
        filename = os.path.join(self.model_path, f"{function}.cpp")
        with open(filename, "w") as fileout:
            fileout.write("\n".join(lines))

    def _generate_function_index(
        self, function: str, indextype: Literal["colptrs", "rowvals"]
    ) -> list[str]:
        """
        Generate equations and C++ code for the function ``function``.

        :param function:
            name of the function to be written (see ``self.functions``)

        :param indextype:
            type of index {'colptrs', 'rowvals'}

        :returns:
            The code lines for the respective function index file
        """
        if indextype == "colptrs":
            values = self.model.colptrs(function)
            setter = "indexptrs"
        elif indextype == "rowvals":
            values = self.model.rowvals(function)
            setter = "indexvals"
        else:
            raise ValueError(
                "Invalid value for indextype, must be colptrs or "
                f"rowvals: {indextype}"
            )

        # function signature
        if function in multiobs_functions:
            signature = f"(SUNMatrixWrapper &{function}, int index)"
        else:
            signature = f"(SUNMatrixWrapper &{function})"

        lines = [
            '#include "amici/sundials_matrix_wrapper.h"',
            '#include "sundials/sundials_types.h"',
            "",
            "#include <array>",
            "#include <algorithm>",
            "",
            "namespace amici {",
            f"namespace model_{self.model_name} {{",
            "",
        ]

        # Generate static array with indices
        if len(values):
            static_array_name = f"{function}_{indextype}_{self.model_name}_"
            if function in multiobs_functions:
                # list of index vectors
                lines.append(
                    "static constexpr std::array<std::array<sunindextype, "
                    f"{len(values[0])}>, {len(values)}> "
                    f"{static_array_name} = {{{{"
                )
                lines.extend(
                    [
                        "    {" + ", ".join(map(str, index_vector)) + "}, "
                        for index_vector in values
                    ]
                )
                lines.append("}};")
            else:
                # single index vector
                lines.extend(
                    [
                        "static constexpr std::array<sunindextype, "
                        f"{len(values)}> {static_array_name} = {{",
                        "    " + ", ".join(map(str, values)),
                        "};",
                    ]
                )

        lines.extend(
            [
                "",
                f"void {function}_{indextype}_{self.model_name}{signature}{{",
            ]
        )

        if len(values):
            if function in multiobs_functions:
                lines.append(
                    f"    {function}.set_{setter}"
                    f"(gsl::make_span({static_array_name}[index]));"
                )
            else:
                lines.append(
                    f"    {function}.set_{setter}"
                    f"(gsl::make_span({static_array_name}));"
                )

        lines.extend(
            [
                "}" "",
                f"}} // namespace model_{self.model_name}",
                "} // namespace amici\n",
            ]
        )

        return lines

    def _get_function_body(
        self, function: str, equations: sp.Matrix
    ) -> list[str]:
        """
        Generate C++ code for body of function ``function``.

        :param function:
            name of the function to be written (see ``self.functions``)

        :param equations:
            symbolic definition of the function body

        :return:
            generated C++ code
        """
        lines = []

        if len(equations) == 0 or (
            isinstance(equations, sp.Matrix | sp.ImmutableDenseMatrix)
            and min(equations.shape) == 0
        ):
            # dJydy is a list
            return lines

        if not self.allow_reinit_fixpar_initcond and function in {
            "sx0_fixedParameters",
            "x0_fixedParameters",
        }:
            return lines

        if function == "sx0_fixedParameters":
            # here we only want to overwrite values where x0_fixedParameters
            # was applied

            lines.extend(
                [
                    # Keep list of indices of fixed parameters occurring in x0
                    "    static const std::array<int, "
                    + str(len(self.model._x0_fixedParameters_idx))
                    + "> _x0_fixedParameters_idxs = {",
                    "        "
                    + ", ".join(
                        str(x) for x in self.model._x0_fixedParameters_idx
                    ),
                    "    };",
                    "",
                    # Set all parameters that are to be reset to 0, so that the
                    #  switch statement below only needs to handle non-zero entries
                    #  (which usually reduces file size and speeds up
                    #  compilation significantly).
                    "    for(auto idx: reinitialization_state_idxs) {",
                    "        if(std::find(_x0_fixedParameters_idxs.cbegin(), "
                    "_x0_fixedParameters_idxs.cend(), idx) != "
                    "_x0_fixedParameters_idxs.cend())\n"
                    "            sx0_fixedParameters[idx] = 0.0;",
                    "    }",
                ]
            )

            cases = {}
            for ipar in range(self.model.num_par()):
                expressions = []
                for index, formula in zip(
                    self.model._x0_fixedParameters_idx,
                    equations[:, ipar],
                    strict=True,
                ):
                    if not formula.is_zero:
                        expressions.extend(
                            [
                                f"if(std::find("
                                "reinitialization_state_idxs.cbegin(), "
                                f"reinitialization_state_idxs.cend(), {index}) != "
                                "reinitialization_state_idxs.cend())",
                                f"    {function}[{index}] = "
                                f"{self._code_printer.doprint(formula)};",
                            ]
                        )
                cases[ipar] = expressions
            lines.extend(get_switch_statement("ip", cases, 1))

        elif function == "x0_fixedParameters":
            for index, formula in zip(
                self.model._x0_fixedParameters_idx, equations, strict=True
            ):
                lines.append(
                    f"    if(std::find(reinitialization_state_idxs.cbegin(), "
                    f"reinitialization_state_idxs.cend(), {index}) != "
                    "reinitialization_state_idxs.cend())\n        "
                    f"{function}[{index}] = "
                    f"{self._code_printer.doprint(formula)};"
                )

        elif function in event_functions:
            cases = {
                ie: self._code_printer._get_sym_lines_array(
                    equations[ie], function, 0
                )
                for ie in range(self.model.num_events())
                if not smart_is_zero_matrix(equations[ie])
            }
            lines.extend(get_switch_statement("ie", cases, 1))

        elif function in event_sensi_functions:
            outer_cases = {}
            for ie, inner_equations in enumerate(equations):
                inner_lines = []
                inner_cases = {
                    ipar: self._code_printer._get_sym_lines_array(
                        inner_equations[:, ipar], function, 0
                    )
                    for ipar in range(self.model.num_par())
                    if not smart_is_zero_matrix(inner_equations[:, ipar])
                }
                inner_lines.extend(get_switch_statement("ip", inner_cases, 0))
                outer_cases[ie] = copy.copy(inner_lines)
            lines.extend(get_switch_statement("ie", outer_cases, 1))

        elif (
            function in sensi_functions
            and equations.shape[1] == self.model.num_par()
        ):
            cases = {
                ipar: self._code_printer._get_sym_lines_array(
                    equations[:, ipar], function, 0
                )
                for ipar in range(self.model.num_par())
                if not smart_is_zero_matrix(equations[:, ipar])
            }
            lines.extend(get_switch_statement("ip", cases, 1))
        elif function in multiobs_functions:
            if function == "dJydy":
                cases = {
                    iobs: self._code_printer._get_sym_lines_array(
                        equations[iobs], function, 0
                    )
                    for iobs in range(self.model.num_obs())
                    if not smart_is_zero_matrix(equations[iobs])
                }
            else:
                cases = {
                    iobs: self._code_printer._get_sym_lines_array(
                        equations[:, iobs], function, 0
                    )
                    for iobs in range(equations.shape[1])
                    if not smart_is_zero_matrix(equations[:, iobs])
                }
            if function.startswith(("Jz", "dJz", "Jrz", "dJrz")):
                iterator = "iz"
            else:
                iterator = "iy"
            lines.extend(get_switch_statement(iterator, cases, 1))

        elif (
            function in self.model.sym_names()
            and function not in non_unique_id_symbols
        ):
            if function in sparse_functions:
                symbols = list(map(sp.Symbol, self.model.sparsesym(function)))
            else:
                symbols = self.model.sym(function)

            if function in ("w", "dwdw", "dwdx", "dwdp"):
                # Split into a block of static and dynamic expressions.
                if len(static_idxs := self.model.static_indices(function)) > 0:
                    tmp_symbols = sp.Matrix(
                        [[symbols[i]] for i in static_idxs]
                    )
                    tmp_equations = sp.Matrix(
                        [equations[i] for i in static_idxs]
                    )
                    tmp_lines = self._code_printer._get_sym_lines_symbols(
                        tmp_symbols,
                        tmp_equations,
                        function,
                        8,
                        static_idxs,
                    )
                    if tmp_lines:
                        lines.extend(
                            [
                                "    // static expressions",
                                "    if (include_static) {",
                                *tmp_lines,
                                "    }",
                            ]
                        )

                # dynamic expressions
                if len(dynamic_idxs := self.model.dynamic_indices(function)):
                    tmp_symbols = sp.Matrix(
                        [[symbols[i]] for i in dynamic_idxs]
                    )
                    tmp_equations = sp.Matrix(
                        [equations[i] for i in dynamic_idxs]
                    )

                    tmp_lines = self._code_printer._get_sym_lines_symbols(
                        tmp_symbols,
                        tmp_equations,
                        function,
                        4,
                        dynamic_idxs,
                    )
                    if tmp_lines:
                        lines.append("\n    // dynamic expressions")
                        lines.extend(tmp_lines)

            else:
                lines += self._code_printer._get_sym_lines_symbols(
                    symbols, equations, function, 4
                )

        else:
            lines += self._code_printer._get_sym_lines_array(
                equations, function, 4
            )

        return [line for line in lines if line]

    def _get_create_splines_body(self):
        if not self.model._splines:
            return ["    return {};"]

        ind4 = " " * 4
        ind8 = " " * 8

        body = ["return {"]
        for ispl, spline in enumerate(self.model._splines):
            if isinstance(spline.nodes, splines.UniformGrid):
                nodes = (
                    f"{ind8}{{{spline.nodes.start}, {spline.nodes.stop}}}, "
                )
            else:
                nodes = f"{ind8}{{{', '.join(map(str, spline.nodes))}}}, "

            # vector with the node values
            values = (
                f"{ind8}{{{', '.join(map(str, spline.values_at_nodes))}}}, "
            )
            # vector with the slopes
            if spline.derivatives_by_fd:
                slopes = f"{ind8}{{}},"
            else:
                slopes = f"{ind8}{{{', '.join(map(str, spline.derivatives_at_nodes))}}},"

            body.extend(
                [
                    f"{ind4}HermiteSpline(",
                    nodes,
                    values,
                    slopes,
                ]
            )

            bc_to_cpp = {
                None: "SplineBoundaryCondition::given, ",
                "zeroderivative": "SplineBoundaryCondition::zeroDerivative, ",
                "natural": "SplineBoundaryCondition::natural, ",
                "zeroderivative+natural": "SplineBoundaryCondition::naturalZeroDerivative, ",
                "periodic": "SplineBoundaryCondition::periodic, ",
            }
            for bc in spline.bc:
                try:
                    body.append(ind8 + bc_to_cpp[bc])
                except KeyError:
                    raise ValueError(
                        f"Unknown boundary condition '{bc}' "
                        "found in spline object"
                    )
            extrapolate_to_cpp = {
                None: "SplineExtrapolation::noExtrapolation, ",
                "polynomial": "SplineExtrapolation::polynomial, ",
                "constant": "SplineExtrapolation::constant, ",
                "linear": "SplineExtrapolation::linear, ",
                "periodic": "SplineExtrapolation::periodic, ",
            }
            for extr in spline.extrapolate:
                try:
                    body.append(ind8 + extrapolate_to_cpp[extr])
                except KeyError:
                    raise ValueError(
                        f"Unknown extrapolation '{extr}' "
                        "found in spline object"
                    )
            line = ind8
            line += "true, " if spline.derivatives_by_fd else "false, "
            line += (
                "true, "
                if isinstance(spline.nodes, splines.UniformGrid)
                else "false, "
            )
            line += "true" if spline.logarithmic_parametrization else "false"
            body.append(line)
            body.append(f"{ind4}),")

        body.append("};")
        return ["    " + line for line in body]

    def _write_wrapfunctions_cpp(self) -> None:
        """
        Write model-specific 'wrapper' file (``wrapfunctions.cpp``).
        """
        template_data = {"MODELNAME": self.model_name}
        apply_template(
            os.path.join(amiciSrcPath, "wrapfunctions.template.cpp"),
            os.path.join(self.model_path, "wrapfunctions.cpp"),
            template_data,
        )

    def _write_wrapfunctions_header(self) -> None:
        """
        Write model-specific header file (``wrapfunctions.h``).
        """
        template_data = {"MODELNAME": str(self.model_name)}
        apply_template(
            os.path.join(amiciSrcPath, "wrapfunctions.template.h"),
            os.path.join(self.model_path, "wrapfunctions.h"),
            template_data,
        )

    def _write_model_header_cpp(self) -> None:
        """
        Write model-specific header and cpp file (MODELNAME.{h,cpp}).
        """
        model_type = "ODE" if self.model.is_ode() else "DAE"
        tpl_data = {
            "MODEL_TYPE_LOWER": model_type.lower(),
            "MODEL_TYPE_UPPER": model_type,
            "MODELNAME": self.model_name,
            "NX_RDATA": self.model.num_states_rdata(),
            "NXTRUE_RDATA": self.model.num_states_rdata(),
            "NX_SOLVER": self.model.num_states_solver(),
            "NXTRUE_SOLVER": self.model.num_states_solver(),
            "NX_SOLVER_REINIT": self.model.num_state_reinits(),
            "NY": self.model.num_obs(),
            "NYTRUE": self.model.num_obs(),
            "NZ": self.model.num_eventobs(),
            "NZTRUE": self.model.num_eventobs(),
            "NEVENT": self.model.num_events(),
            "NEVENT_SOLVER": self.model.num_events_solver(),
            "NOBJECTIVE": "1",
            "NSPL": len(self.model._splines),
            "NW": len(self.model.sym("w")),
            "NDWDP": len(
                self.model.sparsesym(
                    "dwdp", force_generate=self.generate_sensitivity_code
                )
            ),
            "NDWDX": len(self.model.sparsesym("dwdx")),
            "NDWDW": len(self.model.sparsesym("dwdw")),
            "NDXDOTDW": len(self.model.sparsesym("dxdotdw")),
            "NDXDOTDP_EXPLICIT": len(
                self.model.sparsesym(
                    "dxdotdp_explicit",
                    force_generate=self.generate_sensitivity_code,
                )
            ),
            "NDXDOTDX_EXPLICIT": len(self.model.sparsesym("dxdotdx_explicit")),
            "NDJYDY": f"std::vector<int>{{{','.join(str(len(x)) for x in self.model.sparsesym('dJydy'))}}}",
            "NDXRDATADXSOLVER": len(self.model.sparsesym("dx_rdatadx_solver")),
            "NDXRDATADTCL": len(self.model.sparsesym("dx_rdatadtcl")),
            "NDTOTALCLDXRDATA": len(self.model.sparsesym("dtotal_cldx_rdata")),
            "UBW": self.model.num_states_solver(),
            "LBW": self.model.num_states_solver(),
            "NP": self.model.num_par(),
            "NK": self.model.num_const(),
            "O2MODE": "amici::SecondOrderMode::none",
            # using code printer ensures proper handling of nan/inf
            "PARAMETERS": self._code_printer.doprint(self.model.val("p"))[
                1:-1
            ],
            "FIXED_PARAMETERS": self._code_printer.doprint(
                self.model.val("k")
            )[1:-1],
            "PARAMETER_NAMES_INITIALIZER_LIST": self._get_symbol_name_initializer_list(
                "p"
            ),
            "STATE_NAMES_INITIALIZER_LIST": self._get_symbol_name_initializer_list(
                "x_rdata"
            ),
            "FIXED_PARAMETER_NAMES_INITIALIZER_LIST": self._get_symbol_name_initializer_list(
                "k"
            ),
            "OBSERVABLE_NAMES_INITIALIZER_LIST": self._get_symbol_name_initializer_list(
                "y"
            ),
            "OBSERVABLE_TRAFO_INITIALIZER_LIST": "\n".join(
                f"ObservableScaling::{trafo.value}, // y[{idx}]"
                for idx, trafo in enumerate(
                    self.model.get_observable_transformations()
                )
            ),
            "EXPRESSION_NAMES_INITIALIZER_LIST": self._get_symbol_name_initializer_list(
                "w"
            ),
            "PARAMETER_IDS_INITIALIZER_LIST": self._get_symbol_id_initializer_list(
                "p"
            ),
            "STATE_IDS_INITIALIZER_LIST": self._get_symbol_id_initializer_list(
                "x_rdata"
            ),
            "FIXED_PARAMETER_IDS_INITIALIZER_LIST": self._get_symbol_id_initializer_list(
                "k"
            ),
            "OBSERVABLE_IDS_INITIALIZER_LIST": self._get_symbol_id_initializer_list(
                "y"
            ),
            "EXPRESSION_IDS_INITIALIZER_LIST": self._get_symbol_id_initializer_list(
                "w"
            ),
            "STATE_IDXS_SOLVER_INITIALIZER_LIST": ", ".join(
                str(idx)
                for idx, state in enumerate(self.model.states())
                if not state.has_conservation_law()
            ),
            "REINIT_FIXPAR_INITCOND": AmiciCxxCodePrinter.print_bool(
                self.allow_reinit_fixpar_initcond
            ),
            "AMICI_VERSION_STRING": __version__,
            "AMICI_COMMIT_STRING": __commit__,
            "W_RECURSION_DEPTH": self.model._w_recursion_depth,
            "QUADRATIC_LLH": AmiciCxxCodePrinter.print_bool(
                self.model._has_quadratic_nllh
            ),
            "ROOT_INITIAL_VALUES": ", ".join(
                map(
                    lambda event: AmiciCxxCodePrinter.print_bool(
                        event.get_initial_value()
                    ),
                    self.model.events(),
                )
            ),
            "Z2EVENT": ", ".join(map(str, self.model._z2event)),
            "STATE_INDEPENDENT_EVENTS": get_state_independent_event_intializer(
                self.model.events()
            ),
            "ID": ", ".join(
                str(float(isinstance(s, DifferentialState)))
                for s in self.model.states()
                if not s.has_conservation_law()
            ),
        }

        for func_name, func_info in self.functions.items():
            if func_name in nobody_functions:
                continue

            if not func_info.body:
                tpl_data[f"{func_name.upper()}_DEF"] = ""

                if (
                    func_name in sensi_functions + sparse_sensi_functions
                    and not self.generate_sensitivity_code
                ):
                    impl = ""
                else:
                    impl = get_model_override_implementation(
                        func_name,
                        self.model_name,
                        self.model.is_ode(),
                        nobody=True,
                    )

                tpl_data[f"{func_name.upper()}_IMPL"] = impl

                if func_name in sparse_functions:
                    for indexfield in ["colptrs", "rowvals"]:
                        if (
                            func_name in sparse_sensi_functions
                            and not self.generate_sensitivity_code
                        ):
                            impl = ""
                        else:
                            impl = get_sunindex_override_implementation(
                                func_name,
                                self.model_name,
                                indexfield,
                                nobody=True,
                            )
                        tpl_data[
                            f"{func_name.upper()}_{indexfield.upper()}_DEF"
                        ] = ""
                        tpl_data[
                            f"{func_name.upper()}_{indexfield.upper()}_IMPL"
                        ] = impl
                continue

            tpl_data[f"{func_name.upper()}_DEF"] = (
                get_function_extern_declaration(
                    func_name, self.model_name, self.model.is_ode()
                )
            )
            tpl_data[f"{func_name.upper()}_IMPL"] = (
                get_model_override_implementation(
                    func_name, self.model_name, self.model.is_ode()
                )
            )
            if func_name in sparse_functions:
                tpl_data[f"{func_name.upper()}_COLPTRS_DEF"] = (
                    get_sunindex_extern_declaration(
                        func_name, self.model_name, "colptrs"
                    )
                )
                tpl_data[f"{func_name.upper()}_COLPTRS_IMPL"] = (
                    get_sunindex_override_implementation(
                        func_name, self.model_name, "colptrs"
                    )
                )
                tpl_data[f"{func_name.upper()}_ROWVALS_DEF"] = (
                    get_sunindex_extern_declaration(
                        func_name, self.model_name, "rowvals"
                    )
                )
                tpl_data[f"{func_name.upper()}_ROWVALS_IMPL"] = (
                    get_sunindex_override_implementation(
                        func_name, self.model_name, "rowvals"
                    )
                )

        if self.model.num_states_solver() == self.model.num_states_rdata():
            tpl_data["X_RDATA_DEF"] = ""
            tpl_data["X_RDATA_IMPL"] = ""

        tpl_data = {k: str(v) for k, v in tpl_data.items()}

        apply_template(
            os.path.join(amiciSrcPath, "model_header.template.h"),
            os.path.join(self.model_path, f"{self.model_name}.h"),
            tpl_data,
        )

        apply_template(
            os.path.join(amiciSrcPath, "model.template.cpp"),
            os.path.join(self.model_path, f"{self.model_name}.cpp"),
            tpl_data,
        )

    def _get_symbol_name_initializer_list(self, name: str) -> str:
        """
        Get SBML name initializer list for vector of names for the given
        model entity

        :param name:
            any key present in ``self.model._syms``

        :return:
            Template initializer list of names
        """
        return "\n".join(
            f'"{symbol}", // {name}[{idx}]'
            for idx, symbol in enumerate(self.model.name(name))
        )

    def _get_symbol_id_initializer_list(self, name: str) -> str:
        """
        Get C++ initializer list for vector of names for the given model
        entity

        :param name:
            any key present in ``self.model._syms``

        :return:
            Template initializer list of ids
        """
        return "\n".join(
            f'"{self._code_printer.doprint(symbol)}", // {name}[{idx}]'
            for idx, symbol in enumerate(self.model.sym(name))
        )

    def _write_c_make_file(self):
        """Write CMake ``CMakeLists.txt`` file for this model."""
        sources = "\n".join(
            sorted(
                f + " "
                for f in os.listdir(self.model_path)
                if f.endswith(
                    (".cpp", ".h"),
                )
                and f != "main.cpp"
            )
        )

        template_data = {
            "MODELNAME": self.model_name,
            "SOURCES": sources,
            "AMICI_VERSION": __version__,
        }
        apply_template(
            MODEL_CMAKE_TEMPLATE_FILE,
            Path(self.model_path, "CMakeLists.txt"),
            template_data,
        )

    def _write_swig_files(self) -> None:
        """Write SWIG interface files for this model."""
        Path(self.model_swig_path).mkdir(exist_ok=True)
        template_data = {"MODELNAME": self.model_name}
        apply_template(
            Path(amiciSwigPath, "modelname.template.i"),
            Path(self.model_swig_path, self.model_name + ".i"),
            template_data,
        )
        shutil.copy(
            SWIG_CMAKE_TEMPLATE_FILE,
            Path(self.model_swig_path, "CMakeLists.txt"),
        )

    def _write_module_setup(self) -> None:
        """
        Create a setuptools ``setup.py`` file for compile the model module.
        """

        template_data = {
            "MODELNAME": self.model_name,
            "AMICI_VERSION": __version__,
            "PACKAGE_VERSION": "0.1.0",
        }
        apply_template(
            Path(amiciModulePath, "setup.template.py"),
            Path(self.model_path, "setup.py"),
            template_data,
        )
        apply_template(
            Path(amiciModulePath, "MANIFEST.template.in"),
            Path(self.model_path, "MANIFEST.in"),
            {},
        )
        # write __init__.py for the model module
        Path(self.model_path, self.model_name).mkdir(exist_ok=True)

        apply_template(
            Path(amiciModulePath, "__init__.template.py"),
            Path(self.model_path, self.model_name, "__init__.py"),
            template_data,
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
            output_dir = os.path.join(os.getcwd(), f"amici-{self.model_name}")

        self.model_path = os.path.abspath(output_dir)
        self.model_swig_path = os.path.join(self.model_path, "swig")

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


def is_valid_identifier(x: str) -> bool:
    """
    Check whether `x` is a valid identifier for conditions, parameters,
    observables... . Identifiers may only contain upper and lower case letters,
    digits and underscores, and must not start with a digit.

    :param x:
        string to check

    :return:
        ``True`` if valid, ``False`` otherwise
    """

    return IDENTIFIER_PATTERN.match(x) is not None


def _write_gitignore(dest_dir: Path) -> None:
    """Write .gitignore file.

    Generate a `.gitignore` file to ignore a model directory.

    :param dest_dir:
        Path to the directory to write the `.gitignore` file to.
    """
    dest_dir.mkdir(exist_ok=True, parents=True)

    with open(dest_dir / ".gitignore", "w") as f:
        f.write("**")
