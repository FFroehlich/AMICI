"""C++ code generation"""
import itertools
import os
import re
from typing import Optional
from collections.abc import Sequence
from collections.abc import Iterable

import sympy as sp
from sympy.codegen.rewriting import Optimization, optimize
from sympy.printing.cxx import CXX11CodePrinter
from sympy.utilities.iterables import numbered_symbols
from toposort import toposort


class AmiciCxxCodePrinter(CXX11CodePrinter):
    """
    C++ code printer

    Attributes
    ----------
    extract_cse:
        Whether to extract common subexpression during code printing.
        Currently controlled by environment variable ``AMICI_EXTRACT_CSE``.
    optimizations:
        Iterable of :class:`sympy.codegen.rewriting.Optimization`s to optimize
        generated code (e.g. :data:`sympy.codegen.rewriting.Optimization` for
        optimizations, such as ``log(1 + x)`` --> ``logp1(x)``).
        Applying these optimizations is potentially quite costly.
    """

    optimizations: Iterable[Optimization] = ()

    def __init__(self):
        """Create code printer"""
        super().__init__()

        # extract common subexpressions in matrix functions?
        self.extract_cse = os.getenv("AMICI_EXTRACT_CSE", "0").lower() in (
            "1",
            "on",
            "true",
        )

        # Floating-point optimizations
        # e.g., log(1 + x) --> logp1(x)
        if self.optimizations:
            self._fpoptimizer = lambda x: optimize(x, self.optimizations)
        else:
            self._fpoptimizer = None

    def doprint(self, expr: sp.Expr, assign_to: Optional[str] = None) -> str:
        if self._fpoptimizer:
            if isinstance(expr, list):
                expr = list(map(self._fpoptimizer, expr))
            else:
                expr = self._fpoptimizer(expr)

        try:
            # floating point
            code = super().doprint(expr, assign_to)
            code = re.sub(r"(^|\W)M_PI(\W|$)", r"\1amici::pi\2", code)

            return code
        except TypeError as e:
            raise ValueError(
                f'Encountered unsupported function in expression "{expr}"'
            ) from e

    def _print_min_max(self, expr, cpp_fun: str, sympy_fun):
        # C++ doesn't like mixing int and double for arguments for min/max,
        #  therefore, we just always convert to float
        arg0 = (
            sp.Float(expr.args[0]) if expr.args[0].is_number else expr.args[0]
        )
        if len(expr.args) == 1:
            return self._print(arg0)
        return "{}{}({}, {})".format(
            self._ns,
            cpp_fun,
            self._print(arg0),
            self._print(sympy_fun(*expr.args[1:])),
        )

    def _print_Min(self, expr):
        from sympy.functions.elementary.miscellaneous import Min

        return self._print_min_max(expr, "min", Min)

    def _print_Max(self, expr):
        from sympy.functions.elementary.miscellaneous import Max

        return self._print_min_max(expr, "max", Max)

    def _get_sym_lines_array(
        self, equations: sp.Matrix, variable: str, indent_level: int
    ) -> list[str]:
        """
        Generate C++ code for assigning symbolic terms in symbols to C++ array
        `variable`.

        :param equations:
            vectors of symbolic expressions

        :param variable:
            name of the C++ array to assign to

        :param indent_level:
            indentation level (number of leading blanks)

        :return:
            C++ code as list of lines
        """
        return [
            " " * indent_level + f"{variable}[{index}] = "
            f"{self.doprint(math)};"
            for index, math in enumerate(equations)
            if math not in [0, 0.0]
        ]

    def _get_sym_lines_symbols(
        self,
        symbols: sp.Matrix,
        equations: sp.Matrix,
        variable: str,
        indent_level: int,
        indices: Optional[Sequence[int]] = None,
    ) -> list[str]:
        """
        Generate C++ code for where array elements are directly replaced with
        their corresponding macro symbol

        :param symbols:
            vectors of symbols that equations are assigned to

        :param equations:
            vectors of expressions

        :param variable:
            name of the C++ array to assign to, only used in comments

        :param indent_level:
            indentation level (number of leading blanks)

        :param indices:
            Optional custom indices corresponding to entries in `symbols`.
            Only used for comments.

        :return:
            C++ code as list of lines
        """
        assert len(symbols) == len(equations)
        if indices is None:
            indices = range(len(symbols))
        else:
            assert len(indices) == len(symbols)

        indent = " " * indent_level

        def format_regular_line(symbol, math, index):
            return (
                f"{indent}{self.doprint(symbol)} = {self.doprint(math)};"
                f"  // {variable}[{index}]".replace("\n", "\n" + indent)
            )

        if self.extract_cse:
            # Extract common subexpressions
            cse_sym_prefix = "__amici_cse_"
            symbol_generator = numbered_symbols(
                cls=sp.Symbol, prefix=cse_sym_prefix
            )
            replacements, reduced_exprs = sp.cse(
                equations,
                symbols=symbol_generator,
                order="none",
                list=False,
            )
            if replacements:
                # we need toposort to handle the dependencies of extracted
                #  subexpressions
                expr_dict = dict(
                    itertools.chain(zip(symbols, reduced_exprs), replacements)
                )
                sorted_symbols = toposort(
                    {
                        identifier: {
                            s
                            for s in definition.free_symbols
                            if s in expr_dict
                        }
                        for (identifier, definition) in expr_dict.items()
                    }
                )
                symbol_to_idx = {
                    sym: idx for idx, sym in zip(indices, symbols)
                }

                def format_line(symbol: sp.Symbol):
                    math = expr_dict[symbol]
                    if str(symbol).startswith(cse_sym_prefix):
                        return (
                            f"{indent}const realtype "
                            f"{self.doprint(symbol)} "
                            f"= {self.doprint(math)};"
                        )
                    elif math not in [0, 0.0]:
                        return format_regular_line(
                            symbol, math, symbol_to_idx[symbol]
                        )

                return [
                    line
                    for symbol_group in sorted_symbols
                    for symbol in sorted(symbol_group, key=str)
                    if (line := format_line(symbol))
                ]

        return [
            format_regular_line(sym, math, index)
            for index, sym, math in zip(indices, symbols, equations)
            if math not in [0, 0.0]
        ]

    @staticmethod
    def print_bool(expr) -> str:
        """Print the boolean value of the given expression"""
        return "true" if bool(expr) else "false"


def get_switch_statement(
    condition: str,
    cases: dict[int, list[str]],
    indentation_level: Optional[int] = 0,
    indentation_step: Optional[str] = " " * 4,
):
    """
    Generate code for a C++ switch statement.

    Generate code for a C++ switch statement with a ``break`` after each case.

    :param condition:
        Condition for switch

    :param cases:
        Cases as dict with expressions as keys and statement as
        list of strings

    :param indentation_level:
        indentation level

    :param indentation_step:
        indentation whitespace per level

    :return:
        Code for switch expression as list of strings
    """
    if not cases:
        return []

    indent0 = indentation_level * indentation_step
    indent1 = (indentation_level + 1) * indentation_step
    indent2 = (indentation_level + 2) * indentation_step

    # try to find redundant statements and collapse those cases
    # map statements to case expressions
    cases_map: dict[tuple[str, ...], list[str]] = {}
    for expression, statements in cases.items():
        if statements:
            statement_code = tuple(
                [
                    *(f"{indent2}{statement}" for statement in statements),
                    f"{indent2}break;",
                ]
            )
            case_code = f"{indent1}case {expression}:"

            cases_map[statement_code] = cases_map.get(statement_code, []) + [
                case_code
            ]

    if not cases_map:
        return []

    return [
        f"{indent0}switch({condition}) {{",
        *(
            code
            for codes in cases_map.items()
            for code in itertools.chain.from_iterable(reversed(codes))
        ),
        indent0 + "}",
    ]


def csc_matrix(
    matrix: sp.Matrix,
    rownames: list[sp.Symbol],
    colnames: list[sp.Symbol],
    identifier: Optional[int] = 0,
    pattern_only: Optional[bool] = False,
) -> tuple[list[int], list[int], sp.Matrix, list[str], sp.Matrix]:
    """
    Generates the sparse symbolic identifiers, symbolic identifiers,
    sparse matrix, column pointers and row values for a symbolic
    variable

    :param matrix:
        dense matrix to be sparsified

    :param rownames:
        ids of the variable of which the derivative is computed (assuming
        matrix is the jacobian)

    :param colnames:
        ids of the variable with respect to which the derivative is computed
        (assuming matrix is the jacobian)

    :param identifier:
        additional identifier that gets appended to symbol names to
        ensure their uniqueness in outer loops

    :param pattern_only:
        flag for computing sparsity pattern without whole matrix

    :return:
        symbol_col_ptrs, symbol_row_vals, sparse_list, symbol_list,
        sparse_matrix
    """
    idx = 0
    nrows, ncols = matrix.shape

    if not pattern_only:
        sparse_matrix = sp.zeros(nrows, ncols)
    symbol_list = []
    sparse_list = []
    symbol_col_ptrs = []
    symbol_row_vals = []

    for col in range(ncols):
        symbol_col_ptrs.append(idx)
        for row in range(nrows):
            if matrix[row, col] == 0:
                continue

            symbol_row_vals.append(row)
            idx += 1
            symbol_name = f"d{rownames[row].name}" f"_d{colnames[col].name}"
            if identifier:
                symbol_name += f"_{identifier}"
            symbol_list.append(symbol_name)
            if pattern_only:
                continue

            sparse_matrix[row, col] = sp.Symbol(symbol_name, real=True)
            sparse_list.append(matrix[row, col])

    if idx == 0:
        symbol_col_ptrs = []  # avoid bad memory access for empty matrices
    else:
        symbol_col_ptrs.append(idx)

    if pattern_only:
        sparse_matrix = None
    else:
        sparse_list = sp.Matrix(sparse_list)

    return (
        symbol_col_ptrs,
        symbol_row_vals,
        sparse_list,
        symbol_list,
        sparse_matrix,
    )
